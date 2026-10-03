import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 【Must be fully consistent with hyperparameters in training code】 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vehicle dynamics parameters Table2 (identical to training code)
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

# DDPG network hyperparameters
LR_ACTOR = 0.001
LR_CRITIC = 0.001
GAMMA = 0.95
TAU = 0.001
BUFFER_SIZE = int(1e5)
BATCH_SIZE =32
OU_THETA =0.15
OU_SIGMA =0.2

# ========= Test configuration (aligned with your LSTM test) =========
DATA_PATH = "data/car_following_pairs_data13_new.xlsx"
# DATA_PATH = "data/car_following_pairs_data01_new.xlsx"
MODEL_WEIGHT = "output_rl_ddpg/ddpg_carfollow_final.pth"
SAVE_PRED_PATH = "output_rl_ddpg/DDPG_prediction_results_data13.xlsx"
ERROR_RESULT_PATH = "output_rl_ddpg/DDPG_evaluation_metrics_data13.xlsx"
TEST_DT = 0.04   # Real time step of HighD dataset
MIN_SEQ_LEN =0  # Filter short car-following segments

# ------------------------------
# Network definition (fully copied from training code, do not modify)
# ------------------------------
class ActorFree(nn.Module):
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
# Agent class (only keep inference functions, remove buffer and training learn part)
# ------------------------------
class ModularDDPGAgent:
    def __init__(self):
        # Free‑Driving
        self.actor_free = ActorFree().to(DEVICE)
        self.critic_free = CriticFree().to(DEVICE)
        # Car‑Following
        self.actor_follow = ActorFollow().to(DEVICE)
        self.critic_follow = CriticFollow().to(DEVICE)

    def act_free(self, s_free, add_noise=False):
        s = torch.from_numpy(s_free).float().to(DEVICE).unsqueeze(0)
        self.actor_free.eval()
        with torch.no_grad():
            a = self.actor_free(s).cpu().data.numpy()
        return np.clip(a, -1,1)

    def act_follow(self, s_follow, add_noise=False):
        s = torch.from_numpy(s_follow).float().to(DEVICE).unsqueeze(0)
        self.actor_follow.eval()
        with torch.no_grad():
            a = self.actor_follow(s).cpu().data.numpy()
        return np.clip(a, -1,1)

    def action_map(self, a_tanh):
        raw = abs(ACC_MIN)* a_tanh
        acc = np.clip(raw, ACC_MIN, ACC_MAX)
        return acc

    def arbitrate(self, a_free_tanh, a_follow_tanh):
        a_combined_tanh = min(a_free_tanh.item(), a_follow_tanh.item())
        return a_combined_tanh

    def load(self, path_prefix):
        ckpt = torch.load(f"{path_prefix}.pth", map_location=DEVICE)
        self.actor_free.load_state_dict(ckpt["actor_free"])
        self.critic_free.load_state_dict(ckpt["critic_free"])
        self.actor_follow.load_state_dict(ckpt["actor_follow"])
        self.critic_follow.load_state_dict(ckpt["critic_follow"])

    def eval(self):
        self.actor_free.eval()
        self.actor_follow.eval()

# ------------------------------
# Test simulation environment (adapted to TEST_DT=0.04, only for inference, no reward)
# ------------------------------
class CarFollowingTestEnv:
    def __init__(self, leader_speed_seq, dt):
        self.leader_speed_seq = leader_speed_seq
        self.n_step_total = len(leader_speed_seq)
        self.dt = dt
        # 【Fix】Remove self.reset() here; reset is called externally with parameters

    def reset(self, init_v_f, init_gap):
        self.t = 0
        self.v_f = init_v_f
        self.acc_f = 0.0
        self.gap = init_gap
        return self.get_obs()

    def get_obs(self):
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
        """One closed-loop simulation step, return next observation, no reward"""
        if self.t >= self.n_step_total:
            return None, None, True
        acc_f_new = np.clip(acc_command, ACC_MIN, ACC_MAX)
        v_f_new = self.v_f + acc_f_new * self.dt
        v_f_new = max(0.0, v_f_new)

        idx_next = min(self.t+1, self.n_step_total-1)
        v_l_curr = self.leader_speed_seq[self.t]
        v_l_next = self.leader_speed_seq[idx_next]
        delta_x = 0.5*(self.v_f + v_f_new)*self.dt - 0.5*(v_l_curr + v_l_next)*self.dt
        self.gap = self.gap + delta_x
        self.gap = max(0.0, self.gap)

        self.v_f = v_f_new
        self.acc_f = acc_f_new
        self.t += 1
        done = bool(self.t >= self.n_step_total or self.gap <= 0.001)
        if done:
            return None, None, True
        s_free_next, s_follow_next = self.get_obs()
        return s_free_next, s_follow_next, done

# ===================== Main test pipeline =====================
def main():
    # Load model
    agent = ModularDDPGAgent()
    agent.load(os.path.splitext(MODEL_WEIGHT)[0])
    agent.eval()
    print(f"✅ DDPG model loaded, device:{DEVICE}")

    # Read dataset
    df = pd.read_excel(DATA_PATH)
    df = df.sort_values(["leader_id", "follower_id", "frame"]).reset_index(drop=True)
    pairs = df[["leader_id", "follower_id"]].drop_duplicates().values.tolist()

    all_results = []
    error_list = []
    print("\nStart closed-loop test for DDPG-RL model...\n")

    for lid, fid in pairs:
        pair_df = df[(df["leader_id"] == lid) & (df["follower_id"] == fid)].copy()
        lv = pair_df[pair_df["role"] == "LV"].sort_values("frame").reset_index(drop=True)
        fv = pair_df[pair_df["role"] == "FV"].sort_values("frame").reset_index(drop=True)

        min_len = min(len(lv), len(fv))
        if min_len < MIN_SEQ_LEN:
            continue

        lv = lv.iloc[:min_len]
        fv = fv.iloc[:min_len]

        # Ground-truth data
        lv_spd = lv["xVelocity"].values
        fv_real_spd = fv["xVelocity"].values
        fv_real_gap = fv["dhw"].values
        fv_real_acc = fv["xAcceleration"].values

        # ---------- Closed-loop simulation initialization: use real initial states ----------
        init_v_f = fv_real_spd[0]
        init_gap = fv_real_gap[0]
        env = CarFollowingTestEnv(lv_spd, dt=TEST_DT)
        s_free, s_follow = env.reset(init_v_f=init_v_f, init_gap=init_gap)

        pred_acc_list = []
        pred_spd_list = []
        pred_gap_list = []

        for _ in range(min_len):
            a_free_tanh = agent.act_free(s_free, add_noise=False)
            a_follow_tanh = agent.act_follow(s_follow, add_noise=False)
            a_comb_tanh = agent.arbitrate(a_free_tanh, a_follow_tanh)
            acc_pred = agent.action_map(a_comb_tanh)

            pred_acc_list.append(acc_pred)
            pred_spd_list.append(env.v_f)
            pred_gap_list.append(env.gap)

            s_free, s_follow, done = env.step(acc_pred)
            if done:
                break

        n_pred = len(pred_acc_list)
        # Truncate to align with ground truth
        real_acc = fv_real_acc[:n_pred]
        real_spd = fv_real_spd[:n_pred]
        real_gap = fv_real_gap[:n_pred]

        pred_acc = np.array(pred_acc_list)
        pred_spd = np.array(pred_spd_list)
        pred_gap = np.array(pred_gap_list)

        # Save result DataFrame
        fv_result = fv.iloc[:n_pred].copy()
        fv_result["pred_acc"] = pred_acc
        fv_result["pred_speed"] = pred_spd
        fv_result["pred_gap"] = pred_gap
        all_results.append(fv_result)

        # Calculate metrics, fully consistent with LSTM
        acc_mse = np.mean((pred_acc - real_acc) ** 2)
        vel_mae = np.mean(np.abs(pred_spd - real_spd))
        vel_rmse = np.sqrt(np.mean((pred_spd - real_spd) ** 2))
        vel_mape = np.mean(np.abs((pred_spd - real_spd) / (real_spd + 1e-6))) * 100
        gap_mae = np.mean(np.abs(pred_gap - real_gap))

        error_list.append({
            "leader_id": lid,
            "follower_id": fid,
            "acc_mse": acc_mse,
            "vel_mae": vel_mae,
            "vel_rmse": vel_rmse,
            "vel_mape(%)": vel_mape,
            "gap_mae": gap_mae,
            "frames": n_pred
        })
        print(f"✅ Test completed: {lid} ← {fid}")

    # Save all prediction results
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_excel(SAVE_PRED_PATH, index=False)

    error_df = pd.DataFrame(error_list)
    error_df.to_excel(ERROR_RESULT_PATH, index=False)

    # Print overall statistics
    print("\n" + "=" * 70)
    print("📊 DDPG‑RL Model Overall Evaluation Metrics (Closed-loop Simulation)")
    print(f"Valid car-following pairs: {len(error_df)}")
    print(f"Average Acceleration MSE    = {error_df['acc_mse'].mean():.6f}")
    print(f"Average Velocity MAE        = {error_df['vel_mae'].mean():.6f}")
    print(f"Average Velocity RMSE       = {error_df['vel_rmse'].mean():.6f}")
    print(f"Average Velocity MAPE(%)    = {error_df['vel_mape(%)'].mean():.6f}")
    print(f"Average Gap MAE            = {error_df['gap_mae'].mean():.6f}")
    print("=" * 70)
    print(f"\n✅ Prediction results saved to: {SAVE_PRED_PATH}")
    print(f"✅ Evaluation metrics saved to: {ERROR_RESULT_PATH}")

if __name__ == "__main__":
    main()
