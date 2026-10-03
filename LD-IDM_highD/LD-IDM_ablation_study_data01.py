import pandas as pd
import numpy as np
import math
import os
import sys

# ===================== 【Configuration】 =====================
INPUT_PATH = "data/sensitivity_analysis_data01.xlsx"
OUTPUT_PATH = "ablation_study_data01/LD-IDM_lateral_deviation.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_psychology.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_occlusion.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_lane_position.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_no_psychology.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_no_occlusion.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM_no_lane_position.xlsx"
# OUTPUT_PATH = "ablation_study_data01/LD-IDM.xlsx"

# ===================== Ensure output directory exists =====================
# output_dir = os.path.dirname(OUTPUT_PATH)
# if output_dir and not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#     print(f"📁 Output directory created: {output_dir}")

# Basic IDM parameters
s0 = 2.641  # Minimum safe distance
T = 1.265   # Safe time headway
a_max = 0.776  # Maximum acceleration
b = 4.452    # Comfortable deceleration
v0 = 29.675  # Desired speed
delta = 4.0  # Acceleration exponent

alpha = 0.6675
beta = 0.5847
gamma = 1.1630
delta_offset = 0.2774
k1 = 1.2886
k2 = 0.3561
psi = 1.2587

# alpha = 0.5969
# beta = 0.8958
# gamma = 1.2441
# delta_offset = 0.1235
# k1 = 1.0749
# k2 = 0.7559
# psi = 1.0943

width = 1.8
dt = 0.04  # Simulation time step (globally reused)

# ===================== 1. Load real-world data =====================
print("Loading real car-following pair data...")
df = pd.read_excel(INPUT_PATH)
df = df.sort_values(by=["leader_id", "follower_id", "frame"]).reset_index(drop=True)

# Get all car-following pairs
pairs = df[["leader_id", "follower_id"]].drop_duplicates()
print(f"Loaded {len(pairs)} car-following pairs, starting IDM simulation...")

simulation_results = []

# ===================== 2. Run IDM simulation for each car-following pair =====================
for idx, (leader_id, follower_id) in enumerate(pairs.values, 1):
    pair_data = df[(df["leader_id"] == leader_id) & (df["follower_id"] == follower_id)].copy()
    leader = pair_data[pair_data["role"] == "LV"].sort_values("frame").reset_index(drop=True)
    follower = pair_data[pair_data["role"] == "FV"].sort_values("frame").reset_index(drop=True)

    if len(leader) != len(follower):
        print(f"⏭️ Skip {leader_id} ← {follower_id} | Data missing / length mismatch")
        continue

    n_frames = len(leader)
    if n_frames < 2:
        print(f"⏭️ Skip {leader_id} ← {follower_id} | Insufficient frames")
        continue

    # Extract leader vehicle arrays (preload to avoid repeated indexing inside loop)
    leader_x_arr = leader["x"].values
    leader_v_arr = leader["xVelocity"].values
    leader_accel_arr = leader["xAcceleration"].values  # Leader acceleration sequence for trend analysis
    leader_width_arr = leader["width"].values
    leader_height_arr = leader["height"].values
    leader_y_arr = leader["y"].values

    # Extract fixed follower parameters
    length_follower = follower["width"].iloc[0]
    width_follower = follower["height"].iloc[0]
    wf = width_follower / 2
    lane_id = follower["laneId"].iloc[0]
    y_real = follower["y"].values  # Keep y coordinate as ground truth throughout

    # Initialize simulation states (only first frame uses real data; subsequent states are recursively predicted)
    x_sim = [follower["x"].iloc[0]]
    v_sim = [follower["xVelocity"].iloc[0]]
    a_sim = [follower["xAcceleration"].iloc[0]]
    s_sim = [follower["dhw"].iloc[0]]  # Initial gap uses real dhw

    # ===================== Core: IDM iterative calculation (with leader acceleration trend adjustment) =====================
    for i in range(0, n_frames - 1):
        # ---------------------- 1. Fetch current ground-truth leader data ----------------------
        x_lead = leader_x_arr[i]
        v_lead = leader_v_arr[i]
        length_lead = leader_width_arr[i]
        width_lead = leader_height_arr[i]
        wl = width_lead / 2
        leader_y = leader_y_arr[i]
        follower_y = y_real[i]

        # Lane parameter setup
        if lane_id == 5:
            lane_width = 3.96
            y0 = 22.98
            y = follower_y + wf
            leader_y = leader_y + wl
        elif lane_id == 6:
            lane_width = 3.84
            y0 = 26.88
            y = follower_y + wf
            leader_y = leader_y + wl

        # ---------------------- 2. Fetch current simulated follower state ----------------------
        v_curr = v_sim[-1]
        x_curr = x_sim[-1]
        s_curr = s_sim[-1]  # Current simulated gap from previous iteration
        last_accel = a_sim[-1]  # Previous acceleration for adjustment

        # ---------------------- 3. Compute core model parameters (preserve original custom logic) ----------------------
        # dv = v_lead - v_curr  # Speed difference (leader - follower)
        dv = v_curr - v_lead
        delta_y = abs(leader_y - y)  # Lateral deviation

        # Deviation angle calculation (division-by-zero protection)
        if s_curr <= 1e-3:
            deviation_angle = math.pi / 2
        else:
            deviation_angle = math.atan2(delta_y, s_curr)

        # Lateral influence coefficient w2 (original logic retained)
        if s_curr < 150 and s_curr > 1e-3:
            w2 = (math.atan2(delta_y + wl, s_curr) - math.atan2(delta_y - wl, s_curr)) / (
                2 * math.atan2(wl, s_curr)
            )
        else:
            w2 = 0

        # Lane-offset coefficient w3 (original activation logic retained)
        if delta_y <= 0.1 or (y - y0 <= 0.1 and y - y0 >= 0) or (y - y0 >= -0.1 and y - y0 <= 0):
            w3 = 0
        elif lane_id == 6 and y - y0 > 0.1:
            w3 = 1.2
        elif lane_id == 5 and y - y0 < -0.1:
            w3 = 1.5
        else:
            w3 = 1

        # Lateral deviation coefficient w1 (division-by-zero protection)
        denominator_w1 = max(1e-3, lane_width - width_lead)
        w1 = math.pow(delta_y / denominator_w1, psi)

        # Total offset factor
        # LD-IDM
        # OffsetFactor = beta * w1 + gamma * w2 + w3 * delta_offset
        # lateral_deviation
        OffsetFactor = 0
        # psychology
        # OffsetFactor = beta * w1
        # visual occlusion
        # OffsetFactor = gamma * w2
        # lane position
        # OffsetFactor = w3 * delta_offset
        # no psychology
        # OffsetFactor = gamma * w2 + w3 * delta_offset
        # no visual occlusion
        # OffsetFactor = beta * w1 + w3 * delta_offset
        # no lane position
        # OffsetFactor = beta * w1 + gamma * w2

        # Desired gap s_star
        # LD-IDM
        # s_star = s0 + max(0, v_curr * T + (v_curr * dv) / (2 * np.sqrt(a_max * b))) * (0 + alpha * OffsetFactor)
        # Other ablation cases
        s_star = s0 + max(0, v_curr * T + (v_curr * dv) / (2 * np.sqrt(a_max * b))) * (1 + alpha * OffsetFactor)

        # Perceived gap and resistance coefficient (original logic retained)
        Vp = 1 / (1 + k2 * math.tan(deviation_angle))
        s_perceived = (s_curr / math.cos(deviation_angle)) * Vp
        R_thea = 1 + k1 * (math.sin(deviation_angle) ** 2)

        # Base IDM acceleration (before trend adjustment)
        base_acc = (a_max * (1 - (v_curr / v0) ** delta - (s_star / s_perceived) ** 2)) / R_thea

        # ---------------------- 4. Core: adjust acceleration based on leader motion trend ----------------------
        # Judge leader motion trend: use base acceleration directly for i=0 with no previous frame
        if i == 0:
            final_acc = last_accel
        else:
            current_leader_accel = leader_accel_arr[i]
            prev_leader_accel = leader_accel_arr[i - 1]

            # Trend classification (strictly follows reference logic)
            if current_leader_accel > 0 and current_leader_accel > prev_leader_accel:
                trend = "increasing acceleration"
            elif current_leader_accel > 0 and current_leader_accel < prev_leader_accel:
                trend = "decreasing acceleration"
            elif current_leader_accel < 0 and current_leader_accel < prev_leader_accel:
                trend = "increasing deceleration"
            elif current_leader_accel < 0 and current_leader_accel > prev_leader_accel:
                trend = "decreasing deceleration"
            elif current_leader_accel == prev_leader_accel:
                trend = "constant acceleration"
            else:
                trend = "no obvious trend"

            # Adjust acceleration according to trend
            if trend == "constant acceleration":
                # Leader constant acceleration: follower keeps previous acceleration
                # For ablation cases
                final_acc = max(0.06, last_accel)
                # LD-IDM
                # final_acc = max(0.1, last_accel)
            elif trend in ["increasing acceleration", "decreasing deceleration", "decreasing acceleration", "increasing deceleration"]:
                # Non-constant leader acceleration: use base acceleration, then clamp based on leader acceleration sign
                leader_accel = current_leader_accel
                if leader_accel >= 0:
                    # Leader accelerating: lower bound 0.3, upper bound limited by previous acceleration
                    final_acc = max(0.3, min(last_accel, base_acc))
                else:
                    # Leader decelerating:
                    # For ablation cases
                    final_acc = min(0.1, min(last_accel, base_acc))
                    # LD-IDM
                    # final_acc = min(0.1, max(last_accel, base_acc))
            else:
                # No clear trend, directly use base acceleration
                final_acc = base_acc

        # ---------------------- 5. Numerical integration for state update (fixed integration bug) ----------------------
        # Velocity update with boundary protection: no reverse movement, capped by desired speed
        v_new = v_curr + final_acc * dt
        v_new = max(0.0, v_new)
        v_new = min(v_new, v0)

        # Position update
        # For ablation cases
        x_new = x_curr + v_curr * dt + 0.5 * final_acc * dt ** 2

        # Gap update
        s_new = max(s0, leader_x_arr[i + 1] - x_new - length_follower)

        # ---------------------- 6. Store frame results ----------------------
        a_sim.append(final_acc)
        v_sim.append(v_new)
        x_sim.append(x_new)
        s_sim.append(s_new)

    # Replace follower ground-truth data with simulated values
    follower_sim = follower.copy()
    follower_sim["x"] = x_sim
    follower_sim["xVelocity"] = v_sim
    follower_sim["xAcceleration"] = a_sim
    follower_sim["dhw"] = s_sim
    follower_sim["y"] = y_real  # Retain ground-truth y coordinate

    # Merge leader and simulated follower data
    result_pair = pd.concat([leader, follower_sim], ignore_index=True)
    simulation_results.append(result_pair)

    print(f"✅ Completed pair {idx}/{len(pairs)}: {leader_id} ← {follower_id} | Total {n_frames} frames")

# ===================== 3. Save final simulation dataset =====================
if simulation_results:
    final_df = pd.concat(simulation_results, ignore_index=True)
    final_df = final_df.sort_values(by=["leader_id", "follower_id", "frame"])
    final_df.to_excel(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("🎉 IDM simulation dataset generation finished!")
    print(f"📊 Total car-following pairs: {len(pairs)}")
    print(f"📊 Total data rows: {len(final_df)}")
    print(f"📁 Simulation file saved to: {OUTPUT_PATH}")
    print("=" * 60)
else:
    print("\n❌ No simulation data generated")
