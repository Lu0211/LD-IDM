import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy.stats import norm
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== Global Hyperparameters (strictly aligned with the paper) =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vehicle dynamics parameters Table2
V_DES = 15.0          # Desired speed m/s
ACC_MIN = -9.0        # Minimum acceleration (maximum deceleration) m/s²
ACC_MAX = 2.0         # Maximum acceleration m/s²
B_COMF = 2.0          # Comfortable deceleration m/s²
J_COMF = 2.0          # Comfort jerk m/s^3
G_MIN = 2.0           # Minimum safe gap m
T_GAP = 1.5           # Desired time gap s
T_LIM =15.0
W_GAP =0.004
W_JERK =0.5
G_MAX = 200.0         # Upper bound for gap truncation

# DDPG parameters Table1
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.95
TAU = 0.001           # soft update
BUFFER_SIZE = int(1e5)
BATCH_SIZE =32
OU_THETA =0.15
OU_SIGMA =0.2

# Training settings
EPISODES = 800
EPISODE_STEPS = 500   # 500 steps per episode, paper setting
DT = 0.1              # Simulation time step 100ms

# File path (use the same dataset as your LSTM)
DATA_PATH = "data/car_following_pairs_data13_new.xlsx"
MODEL_SAVE_DIR = "output_rl_ddpg"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ------------------------------
# 1. Ornstein‑Uhlenbeck process (generate leader trajectory for RL training env, Eq.(14)(15) in paper)
# ------------------------------
def ou_process(n_steps, mu=7.5, theta=0.132, sigma=3.847, dt=0.1, v_min=0, v_max=16.6):
    """Generate leader velocity sequence for RL training environment"""
    v = np.zeros(n_steps)
    v[0] = np.random.uniform(0, V_DES)
    for i in range(1, n_steps):
        dw = np.random.normal(0, np.sqrt(dt))
        v[i] = v[i-1] + theta*(mu - v[i-1])*dt + sigma * dw
    v = np.clip(v, v_min, v_max)
    return v

# OU noise for DDPG exploration
class OUNoise:
    def __init__(self, size, theta=OU_THETA, sigma=OU_SIGMA):
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.reset()

    def reset(self):
        self.state = np.zeros(self.size)

    def sample(self):
        x = self.state
        dx = -self.theta * x + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state

# Replay Buffer [independent class, each policy instantiates one]
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, s, a, r, s_next, done):
        self.memory.append((s,a,r,s_next,done))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        s = torch.from_numpy(np.vstack([b[0] for b in batch])).float().to(DEVICE)
        a = torch.from_numpy(np.vstack([b[1] for b in batch])).float().to(DEVICE)
        r = torch.from_numpy(np.vstack([b[2] for b in batch])).float().to(DEVICE)
        s2 = torch.from_numpy(np.vstack([b[3] for b in batch])).float().to(DEVICE)
        d = torch.from_numpy(np.vstack([b[4] for b in batch])).float().to(DEVICE)
        return s,a,r,s2,d

    def __len__(self):
        return len(self.memory)

# ------------------------------
# 2. Actor‑Critic Network: two independent policies: free-driving and car-following, Fig.7 in paper
# ------------------------------
class ActorFree(nn.Module):
    """Free‑Driving Policy Actor: input [normalized ego speed, normalized ego acceleration], output tanh [-1,1] action"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

class CriticFree(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2+1,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )
    def forward(self, s,a):
        x = torch.cat([s,a],dim=1)
        return self.net(x)

class ActorFollow(nn.Module):
    """Car‑Following Policy Actor, 4-dimensional state input, 2 hidden layers with 32 neurons"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

class CriticFollow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4+1,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
    def forward(self, s,a):
        x = torch.cat([s,a],dim=1)
        return self.net(x)

# ------------------------------
# 3. Modular DDPG Agent [Fixed: dual buffers]
# ------------------------------
class ModularDDPGAgent:
    def __init__(self):
        # ========= Free Driving Policy =========
        self.actor_free = ActorFree().to(DEVICE)
        self.actor_free_target = ActorFree().to(DEVICE)
        self.critic_free = CriticFree().to(DEVICE)
        self.critic_free_target = CriticFree().to(DEVICE)
        # copy weights
        for t_param, param in zip(self.actor_free_target.parameters(), self.actor_free.parameters()):
            t_param.data.copy_(param.data)
        for t_param, param in zip(self.critic_free_target.parameters(), self.critic_free.parameters()):
            t_param.data.copy_(param.data)

        # ========= Car Following Policy =========
        self.actor_follow = ActorFollow().to(DEVICE)
        self.actor_follow_target = ActorFollow().to(DEVICE)
        self.critic_follow = CriticFollow().to(DEVICE)
        self.critic_follow_target = CriticFollow().to(DEVICE)
        for t_param, param in zip(self.actor_follow_target.parameters(), self.actor_follow.parameters()):
            t_param.data.copy_(param.data)
        for t_param, param in zip(self.critic_follow_target.parameters(), self.critic_follow.parameters()):
            t_param.data.copy_(param.data)

        # optimizer
        self.opt_actor_free = optim.Adam(self.actor_free.parameters(), lr=LR_ACTOR)
        self.opt_critic_free = optim.Adam(self.critic_free.parameters(), lr=LR_CRITIC)
        self.opt_actor_follow = optim.Adam(self.actor_follow.parameters(), lr=LR_ACTOR)
        self.opt_critic_follow = optim.Adam(self.critic_follow.parameters(), lr=LR_CRITIC)

        self.noise = OUNoise(size=1)
        # ========== Key Fix: two independent replay buffers ==========
        self.buffer_free = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.buffer_follow = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

    def act_free(self, s_free, add_noise=True):
        """Free-driving policy output action [-1,1]"""
        s = torch.from_numpy(s_free).float().to(DEVICE).unsqueeze(0)
        self.actor_free.eval()
        with torch.no_grad():
            a = self.actor_free(s).cpu().data.numpy()
        self.actor_free.train()
        if add_noise:
            a += self.noise.sample()
        return np.clip(a, -1,1)

    def act_follow(self, s_follow, add_noise=True):
        """Car-following policy output action [-1,1]"""
        s = torch.from_numpy(s_follow).float().to(DEVICE).unsqueeze(0)
        self.actor_follow.eval()
        with torch.no_grad():
            a = self.actor_follow(s).cpu().data.numpy()
        self.actor_follow.train()
        if add_noise:
            a += self.noise.sample()
        return np.clip(a, -1,1)

    def action_map(self, a_tanh):
        """Map tanh action in [-1,1] to real acceleration [ACC_MIN, ACC_MAX]"""
        raw = abs(ACC_MIN)* a_tanh
        acc = np.clip(raw, ACC_MIN, ACC_MAX)
        return acc

    def arbitrate(self, a_free_tanh, a_follow_tanh):
        """Synchronized modular arbiter: min function, Section 2.2 in paper"""
        a_combined_tanh = min(a_free_tanh.item(), a_follow_tanh.item())
        return a_combined_tanh

    def soft_update(self, local_net, target_net):
        for t, l in zip(target_net.parameters(), local_net.parameters()):
            t.data.copy_(TAU * l.data + (1.0-TAU)*t.data)

    def learn_free(self):
        """Update Free‑Driving Policy, sample only from free buffer"""
        if len(self.buffer_free) < BATCH_SIZE:
            return
        s,a,r,s_next,done = self.buffer_free.sample()
        # critic loss
        a_next = self.actor_free_target(s_next)
        q_target = r + GAMMA * self.critic_free_target(s_next, a_next) * (1-done)
        q_pred = self.critic_free(s,a)
        loss_critic = nn.MSELoss()(q_pred, q_target.detach())
        self.opt_critic_free.zero_grad()
        loss_critic.backward()
        self.opt_critic_free.step()
        # actor loss
        a_pred = self.actor_free(s)
        loss_actor = -self.critic_free(s, a_pred).mean()
        self.opt_actor_free.zero_grad()
        loss_actor.backward()
        self.opt_actor_free.step()
        self.soft_update(self.actor_free, self.actor_free_target)
        self.soft_update(self.critic_free, self.critic_free_target)

    def learn_follow(self):
        """Update Car‑Following Policy, sample only from follow buffer"""
        if len(self.buffer_follow) < BATCH_SIZE:
            return
        s,a,r,s_next,done = self.buffer_follow.sample()
        a_next = self.actor_follow_target(s_next)
        q_target = r + GAMMA * self.critic_follow_target(s_next, a_next) * (1-done)
        q_pred = self.critic_follow(s,a)
        loss_critic = nn.MSELoss()(q_pred, q_target.detach())
        self.opt_critic_follow.zero_grad()
        loss_critic.backward()
        self.opt_critic_follow.step()

        a_pred = self.actor_follow(s)
        loss_actor = -self.critic_follow(s, a_pred).mean()
        self.opt_actor_follow.zero_grad()
        loss_actor.backward()
        self.opt_actor_follow.step()
        self.soft_update(self.actor_follow, self.actor_follow_target)
        self.soft_update(self.critic_follow, self.critic_follow_target)

    def save(self, path_prefix):
        torch.save({
            "actor_free":self.actor_free.state_dict(),
            "critic_free":self.critic_free.state_dict(),
            "actor_follow":self.actor_follow.state_dict(),
            "critic_follow":self.critic_follow.state_dict()
        }, f"{path_prefix}.pth")

    def load(self, path_prefix):
        ckpt = torch.load(f"{path_prefix}.pth", map_location=DEVICE)
        self.actor_free.load_state_dict(ckpt["actor_free"])
        self.critic_free.load_state_dict(ckpt["critic_free"])
        self.actor_follow.load_state_dict(ckpt["actor_follow"])
        self.critic_follow.load_state_dict(ckpt["critic_follow"])

# ------------------------------
# 4. Simulation Environment & Reward Function (fully reproduce all reward formulas in Section 2.5 of paper)
# ------------------------------
# ------------------------------
# 4. Simulation Environment & Reward Function [Fixed array out-of-bounds]
# ------------------------------
class CarFollowingSimEnv:
    def __init__(self, leader_speed_seq):
        self.leader_speed_seq = leader_speed_seq
        self.n_step_total = len(leader_speed_seq)
        self.reset()

    def reset(self):
        # Initialize state
        self.t = 0
        self.v_f = np.random.uniform(0, V_DES)   # Follower speed
        self.acc_f = 0.0                         # Follower acceleration
        self.gap = 120.0                         # Initial gap 120m, paper setting
        return self.get_obs()

    def get_obs(self):
        """Return observation vectors for two policies: s_free(2D), s_follow(4D), Eq.(3)(4) in paper"""
        # Fix: prevent t out of range
        t_safe = min(self.t, self.n_step_total -1)
        v_l = self.leader_speed_seq[t_safe]

        s_free = np.array([
            self.v_f / V_DES,
            (self.acc_f - ACC_MIN)/(ACC_MAX - ACC_MIN)
        ], dtype=np.float32)

        gap_clipped = np.clip(self.gap, 0, G_MAX)
        delta_v = v_l - self.v_f
        s_follow = np.array([
            self.v_f / V_DES,
            (self.acc_f - ACC_MIN)/(ACC_MAX - ACC_MIN),
            delta_v / V_DES,
            gap_clipped / G_MAX
        ], dtype=np.float32)
        return s_free, s_follow

    def step(self, acc_command):
        """
        acc_command: real acceleration m/s²
        return: s_free_next, s_follow_next, reward_free, reward_follow, done
        """
        # Fix: check max steps at entry, directly return done
        if self.t >= self.n_step_total:
            return None, None, 0.0, 0.0, True

        t_safe = min(self.t, self.n_step_total -1)
        v_l = self.leader_speed_seq[t_safe]
        idx_next = min(self.t+1, self.n_step_total-1)

        # Vehicle dynamics Euler integration, Eq.(12)(13) in paper
        acc_f_new = np.clip(acc_command, ACC_MIN, ACC_MAX)
        jerk = (acc_f_new - self.acc_f)/DT
        v_f_new = self.v_f + acc_f_new * DT
        v_f_new = max(0, v_f_new)

        delta_x = 0.5*(self.v_f + v_f_new)*DT - 0.5*(v_l + self.leader_speed_seq[idx_next])*DT
        self.gap = self.gap + delta_x
        self.gap = max(0.0, self.gap)

        self.v_f = v_f_new
        self.acc_f = acc_f_new
        self.t +=1

        # done condition: episode ends or collision
        done = bool(self.t >= self.n_step_total or self.gap <= 0.01)

        if done:
            # terminate episode, no next observation
            return None, None, 0.0, 0.0, True

        s_free_next, s_follow_next = self.get_obs()

        # ========== Calculate reward components, Section 2.5 ==========
        # r_speed
        if self.v_f <= V_DES:
            r_speed = self.v_f / V_DES
        else:
            r_speed = V_DES / self.v_f

        # r_jerk
        r_jerk = -(jerk / J_COMF)**2

        # r_safe
        b_kin = 0.0
        t_safe_cur = min(self.t-1, self.n_step_total-1)
        v_l_cur = self.leader_speed_seq[t_safe_cur]
        if self.v_f > v_l_cur and self.gap>1e-6:
            b_kin = (self.v_f - v_l_cur)**2 / self.gap
        indicator = 1.0 if b_kin > B_COMF else 0.0
        r_safe = -np.tanh( (b_kin - B_COMF)/ (-ACC_MIN) ) * indicator

        # r_gap
        g_opt = self.v_f * T_GAP + G_MIN
        g_var = 0.5 * g_opt
        g_lim = self.v_f * T_LIM + 2*G_MIN
        g_star = (g_opt + g_lim)/2
        phi0 = norm.pdf(0)
        if self.gap < g_star:
            r_gap = norm.pdf((self.gap - g_opt)/g_var)/ phi0
        else:
            r_gap = norm.pdf((self.gap - g_opt)/g_var)/ phi0 * (1 - (self.gap - g_star)/(g_lim - g_star))

        # Free-driving reward and car-following reward, Eq.(10)(11)
        reward_free = r_speed + W_JERK * r_jerk
        reward_follow = r_safe + W_GAP * r_gap + W_JERK * r_jerk

        return s_free_next, s_follow_next, reward_free, reward_follow, done

# ------------------------------
# 5. RL Training Loop [Fixed: add samples to corresponding buffers]
# ------------------------------
def train_rl():
    agent = ModularDDPGAgent()
    print(f"Start training modular DDPG car-following model, device:{DEVICE}")
    for ep in range(EPISODES):
        lv_seq = ou_process(n_steps=EPISODE_STEPS)
        env = CarFollowingSimEnv(lv_seq)
        s_free, s_follow = env.reset()
        agent.noise.reset()
        ep_r_free = 0.0
        ep_r_follow =0.0
        for st in range(EPISODE_STEPS):
            # Two policies output tanh actions separately
            a_free_tanh = agent.act_free(s_free)
            a_follow_tanh = agent.act_follow(s_follow)
            a_comb_tanh = agent.arbitrate(a_free_tanh, a_follow_tanh)
            acc_real = agent.action_map(a_comb_tanh)
            s_free_n, s_follow_n, r_free, r_follow, done = env.step(acc_real)

            if done:
                break

            # =========Fix: store samples into respective independent buffers=========
            agent.buffer_free.add(s_free, a_free_tanh, r_free, s_free_n, np.array([done]))
            agent.buffer_follow.add(s_follow, a_follow_tanh, r_follow, s_follow_n, np.array([done]))

            agent.learn_free()
            agent.learn_follow()

            s_free, s_follow = s_free_n, s_follow_n
            ep_r_free += r_free
            ep_r_follow += r_follow
            if done:
                break
        if (ep+1) % 50 == 0:
            agent.save(os.path.join(MODEL_SAVE_DIR, "ddpg_carfollow"))
            print(f"Episode {ep+1:4d} | R_free:{ep_r_free:.2f} | R_follow:{ep_r_follow:.2f} | Checkpoint saved")
    agent.save(os.path.join(MODEL_SAVE_DIR, "ddpg_carfollow_final"))
    print("✅ RL training completed!")
    return agent

# ------------------------------
# 6. Offline test on your real dataset (use your Excel dataset)
# ------------------------------
def test_on_real_dataset(agent):
    """Test simulation on your own car_following_pairs_data13_new.xlsx dataset
    Use real leader trajectory, RL agent acts as follower, predict acceleration and compare with ground-truth follower data
    """
    print("\nLoad real dataset for testing: ", DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    df = df.sort_values(["leader_id","follower_id","frame"]).reset_index(drop=True)
    pairs = df[["leader_id","follower_id"]].drop_duplicates().values.tolist()
    print(f"Number of car-following pairs in test set:{len(pairs)}")

    eval_records = []
    for lid, fid in pairs[:10]: # Only take first 10 pairs for demonstration, can be modified
        pair_df = df[(df["leader_id"]==lid)&(df["follower_id"]==fid)].copy()
        lv = pair_df[pair_df["role"]=="LV"].sort_values("frame").reset_index(drop=True)
        fv_real = pair_df[pair_df["role"]=="FV"].sort_values("frame").reset_index(drop=True)
        minlen = min(len(lv), len(fv_real))
        if minlen < 20:
            continue
        # Extract real leader velocity sequence as environment input
        leader_speed_real = lv["xVelocity"].values[:minlen]
        env = CarFollowingSimEnv(leader_speed_real)
        s_free, s_follow = env.reset()
        for t in range(minlen):
            a_free_tanh = agent.act_free(s_free, add_noise=False)
            a_follow_tanh = agent.act_follow(s_follow, add_noise=False)
            a_comb_tanh = agent.arbitrate(a_free_tanh, a_follow_tanh)
            acc_pred = agent.action_map(a_comb_tanh)
            s_free_n, s_follow_n, _, _, done = env.step(acc_pred)
            # Record
            eval_records.append({
                "leader_id":lid,
                "follower_id":fid,
                "time_step":t,
                "v_leader": leader_speed_real[t],
                "v_follower_rl": env.v_f,
                "gap_rl": env.gap,
                "acc_rl_pred": acc_pred,
                "acc_real": fv_real.iloc[t]["xAcceleration"]
            })
            s_free, s_follow = s_free_n, s_follow_n
            if done:
                break
    df_eval = pd.DataFrame(eval_records)
    out_csv = os.path.join(MODEL_SAVE_DIR, "rl_test_result.csv")
    df_eval.to_csv(out_csv, index=False)
    mse_acc = np.mean((df_eval["acc_rl_pred"] - df_eval["acc_real"])**2)
    print(f"Test on real dataset finished, predicted acceleration MSE={mse_acc:.4f}, output file:{out_csv}")
    return df_eval

if __name__ == "__main__":
    # Training
    rl_agent = train_rl()
    # Test on your dataset
    test_df = test_on_real_dataset(rl_agent)
