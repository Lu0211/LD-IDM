import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. Hyperparameter Configuration =====================
TIME_STEP = 25  # Time step: use previous 25 frames to predict the next frame
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
TEST_SIZE = 0.2

# File paths
DATA_PATH = "data/car_following_pairs_data13_new.xlsx"
MODEL_SAVE_PATH = "output_LSTM/lstm_car_following_model.h5"
SCALER_SAVE_PATH = "output_LSTM/lstm_scaler.npy"

# ===================== 2. Load Dataset =====================
print("Loading dataset...")
df = pd.read_excel(DATA_PATH)
df = df.sort_values(["leader_id", "follower_id", "frame"]).reset_index(drop=True)

pairs = df[["leader_id", "follower_id"]].drop_duplicates().values.tolist()
print(f"The dataset contains {len(pairs)} car-following pairs")

# Construct time-series samples
sequences_X = []
sequences_y = []

for lid, fid in pairs:
    pair_df = df[(df["leader_id"] == lid) & (df["follower_id"] == fid)].copy()
    lv = pair_df[pair_df["role"] == "LV"].sort_values("frame").reset_index(drop=True)
    fv = pair_df[pair_df["role"] == "FV"].sort_values("frame").reset_index(drop=True)

    min_len = min(len(lv), len(fv))
    if min_len < TIME_STEP + 1:
        continue

    lv = lv.iloc[:min_len]
    fv = fv.iloc[:min_len]

    # Input: leader velocity, follower velocity, gap distance
    lv_spd = lv["xVelocity"].values
    fv_spd = fv["xVelocity"].values
    gap = fv["dhw"].values
    fv_acc = fv["xAcceleration"].values

    # Build time series
    for i in range(TIME_STEP, min_len):
        seq_x = np.stack([
            lv_spd[i-TIME_STEP : i],
            fv_spd[i-TIME_STEP : i],
            gap[i-TIME_STEP : i]
        ], axis=1)
        seq_y = fv_acc[i]

        sequences_X.append(seq_x)
        sequences_y.append(seq_y)

X = np.array(sequences_X)
y = np.array(sequences_y)
print(f"Sample construction completed: samples={len(X)}, input shape={X.shape}")

# ===================== 3. Normalization =====================
n_features = X.shape[2]
scaler = MinMaxScaler((0, 1))
X_reshaped = X.reshape(-1, n_features)
X_scaled = scaler.fit_transform(X_reshaped).reshape(X.shape)

# Save normalization parameters
np.save(SCALER_SAVE_PATH, [scaler.min_, scaler.scale_])

# ===================== 4. Train / Test Split =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, shuffle=True, random_state=42
)

# ===================== 5. PyTorch Dataset Definition =====================
class CarFollowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(CarFollowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(CarFollowDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# ===================== 6. LSTM Model Definition =====================
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
        out = out[:, -1, :]  # Take output at the last time step
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ===================== 7. Model Initialization =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMCarFollow(input_size=3).to(device)
criterion = nn.MSELoss()  # MSE loss for acceleration prediction
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Using device: {device}")
print(model)

# ===================== 8. Training Process =====================
print("\nStart training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        outputs = model(bx)
        loss = criterion(outputs, by)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * bx.size(0)

    train_loss /= len(train_loader.dataset)

    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            outputs = model(bx)
            loss = criterion(outputs, by)
            test_loss += loss.item() * bx.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f}")

# ===================== 9. Save Model =====================
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("\n🎉 Training finished!")
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"✅ Normalization parameters saved to: {SCALER_SAVE_PATH}")
