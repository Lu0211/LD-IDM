import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== Configuration: identical to training code =====================
TIME_STEP = 1
input_size = 3
hidden_size = 128
num_layers = 2

# Paths
DATA_PATH = "data/car_following_pairs_data13_new.xlsx"
# DATA_PATH = "data/car_following_pairs_data01_new.xlsx"

MODEL_PATH = "output_LSTM/lstm_car_following_model.h5"
SCALER_PATH = "output_LSTM/lstm_scaler.npy"
SAVE_PRED_PATH = "output_LSTM/LSTM_prediction_results.xlsx"
ERROR_RESULT_PATH = "output_LSTM/LSTM_evaluation_metrics.xlsx"

# ===================== 1. Load model architecture (must match training) =====================
class LSTMCarFollow(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ===================== 2. Load model & normalization scaler =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMCarFollow(input_size=3, hidden_size=128, num_layers=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load normalization parameters
min_, scale_ = np.load(SCALER_PATH, allow_pickle=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.min_ = min_
scaler.scale_ = scale_

# ===================== 3. Load test dataset =====================
df = pd.read_excel(DATA_PATH)
df = df.sort_values(["leader_id", "follower_id", "frame"]).reset_index(drop=True)
pairs = df[["leader_id", "follower_id"]].drop_duplicates().values.tolist()

# ===================== 4. Predict for each car-following pair =====================
all_results = []
error_list = []

print("Start LSTM model testing...\n")

with torch.no_grad():
    for lid, fid in pairs:
        pair_df = df[(df["leader_id"] == lid) & (df["follower_id"] == fid)].copy()
        lv = pair_df[pair_df["role"] == "LV"].sort_values("frame").reset_index(drop=True)
        fv = pair_df[pair_df["role"] == "FV"].sort_values("frame").reset_index(drop=True)

        min_len = min(len(lv), len(fv))
        if min_len < TIME_STEP + 1:
            continue

        lv = lv.iloc[:min_len]
        fv = fv.iloc[:min_len]

        # Ground truth values
        lv_spd = lv["xVelocity"].values
        fv_spd = fv["xVelocity"].values
        gap = fv["dhw"].values
        acc_real = fv["xAcceleration"].values

        # Build sequence samples
        X_seq = []
        frames_used = []
        for i in range(TIME_STEP, min_len):
            seq_x = np.stack([
                lv_spd[i-TIME_STEP:i],
                fv_spd[i-TIME_STEP:i],
                gap[i-TIME_STEP:i]
            ], axis=1)
            X_seq.append(seq_x)
            frames_used.append(i)

        X_seq = np.array(X_seq)
        X_reshaped = X_seq.reshape(-1, 3)
        X_scaled = scaler.transform(X_reshaped).reshape(X_seq.shape)

        # Model prediction
        tensor_X = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        pred_acc = model(tensor_X).cpu().numpy().flatten()

        # Align ground truth
        real_acc = acc_real[TIME_STEP:min_len]
        real_spd = fv_spd[TIME_STEP:min_len]
        real_gap = gap[TIME_STEP:min_len]

        # Integrate to obtain speed (critical step!)
        dt = 0.04
        pred_spd = np.zeros_like(pred_acc)
        pred_spd[0] = fv_spd[TIME_STEP]
        for t in range(1, len(pred_spd)):
            pred_spd[t] = pred_spd[t-1] + pred_acc[t] * dt

        # Integrate to obtain gap distance
        pred_gap = np.zeros_like(pred_acc)
        pred_gap[0] = gap[TIME_STEP]
        for t in range(1, len(pred_gap)):
            pred_gap[t] = pred_gap[t-1] + (lv_spd[TIME_STEP + t] - pred_spd[t]) * dt

        # Save prediction results
        fv_result = fv.iloc[TIME_STEP:min_len].copy()
        fv_result["pred_acc"] = pred_acc
        fv_result["pred_speed"] = pred_spd
        fv_result["pred_gap"] = pred_gap
        all_results.append(fv_result)

        # ===================== Calculate evaluation metrics =====================
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
            "frames": len(pred_acc)
        })

        print(f"✅ Test completed: {lid} ← {fid}")

# ===================== 5. Save prediction outputs =====================
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_excel(SAVE_PRED_PATH, index=False)

error_df = pd.DataFrame(error_list)
error_df.to_excel(ERROR_RESULT_PATH, index=False)

# ===================== 6. Print overall metrics =====================
print("\n" + "=" * 70)
print("📊 LSTM Overall Evaluation Metrics (invalid pairs filtered)")
print(f"Valid car-following pairs: {len(error_df)}")
print(f"Average Acceleration MSE    = {error_df['acc_mse'].mean():.6f}")
print(f"Average Velocity MAE        = {error_df['vel_mae'].mean():.6f}")
print(f"Average Velocity RMSE       = {error_df['vel_rmse'].mean():.6f}")
print(f"Average Velocity MAPE(%)    = {error_df['vel_mape(%)'].mean():.6f}")
print(f"Average Gap MAE             = {error_df['gap_mae'].mean():.6f}")
print("=" * 70)
print(f"\n✅ Prediction results saved to: {SAVE_PRED_PATH}")
print(f"✅ Evaluation metrics saved to: {ERROR_RESULT_PATH}")
