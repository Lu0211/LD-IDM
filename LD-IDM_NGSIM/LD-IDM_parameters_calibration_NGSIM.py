import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import os

# Set font to ensure proper display
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

class IDMModel:
    def __init__(self, s0=4.44, T=1.06, a=1.3, b=1.94):
        # IDM basic parameters
        self.s0 = s0  # Minimum safe gap
        self.T = T  # Safe time headway
        self.a = a  # Maximum acceleration
        self.b = b  # Comfortable deceleration

        # Lateral offset risk parameters
        self.alpha = 0.6  # Lateral offset influence coefficient
        self.beta = 1.2  # Lateral offset weight
        self.gamma = 1.0  # Visual occlusion weight
        self.delta = 0.2  # Lane position weight
        self.kr = 0.8
        self.fan = 1.2
        self.kv = 0.5

        # Vehicle and lane parameters
        self.vehicle_width = 1.8  # Vehicle width (m)
        self.lane_width = 3.7  # Default lane width (m), will be adjusted based on actual lane widths

    def calculate_offset_factor(self, d, occlusion_factor, lane_position_factor):
        """Calculate lateral offset risk factor"""
        normalized_offset = math.pow(math.fabs(d) / (self.lane_width - self.vehicle_width), self.fan)
        offset_component = self.beta * normalized_offset
        occlusion_component = self.gamma * occlusion_factor
        lane_position_component = self.delta * lane_position_factor
        return offset_component + occlusion_component + lane_position_component

    def calculate_occlusion_factor(self, d, L):
        """Calculate visual occlusion factor"""
        return (math.atan2(abs(d) + self.vehicle_width / 2, L) - math.atan2(abs(d) - self.vehicle_width / 2, L)) / (
                2 * math.atan2(self.vehicle_width, 2 * L))

    def calculate_lane_position_factor(self, lane_position):
        """Calculate lane position factor"""
        if lane_position == "center":
            return 0.5
        elif lane_position == "right_overtake":
            return 1.2
        elif lane_position == "left_overtake":
            return 1.5
        else:
            return 1.0  # Default centered

    def calculate_desired_gap(self, v, delta_v, d, lane_position, lane_width=None):
        """Calculate desired safe gap considering lateral offset"""
        # Use current lane width if provided
        current_lane_width = lane_width if lane_width is not None else self.lane_width

        occlusion_factor = self.calculate_occlusion_factor(d, max(1.0, v * self.T))
        lane_position_factor = self.calculate_lane_position_factor(lane_position)
        offset_factor = self.calculate_offset_factor(d, occlusion_factor, lane_position_factor)

        s_star = self.s0 + max(0, v * self.T + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
        s_star = s_star * (1 + self.alpha * offset_factor)
        return s_star

    def calculate_acceleration(self, v, delta_v, gap, d, lane_position, lane_width=None):
        """Calculate IDM acceleration"""
        # Use current lane width if provided
        current_lane_width = lane_width if lane_width is not None else self.lane_width

        s_star = self.calculate_desired_gap(v, delta_v, d, lane_position, current_lane_width)

        # Speed term
        speed_term = (v / 23.54) ** 4  # Assume speed limit around 110km/h (30m/s)
        deviation_angle = math.atan2(d, gap)
        vp = 1 / (1 + self.kv * math.tan(deviation_angle))
        # Gap term
        gap_term = (s_star / max((gap / math.cos(deviation_angle)) * vp, 0.1)) ** 2
        R = 1 + self.kr * (math.sin(deviation_angle) ** 2)

        # Final acceleration
        acceleration = (self.a * (1 - speed_term - gap_term)) / R
        return acceleration


class NGSIMDataLoader:
    def __init__(self, tracks_file, static_file, meta_file):
        self.tracks_file = tracks_file
        self.static_file = static_file
        self.meta_file = meta_file
        self.tracks = None
        self.tracks_meta = None
        self.recording_meta = None

        # Set lane information according to provided information (E0_0 to E0_6)
        # Lane widths from image information, lane center Y coordinates from provided data
        self.lane_info = {
            "E0_0": {"width": 3.86, "center_y": 1.50},
            "E0_1": {"width": 3.76, "center_y": 5.31},
            "E0_2": {"width": 3.70, "center_y": 9.04},
            "E0_3": {"width": 3.73, "center_y": 12.75},
            "E0_4": {"width": 4.20, "center_y": 16.62},
            "E0_5": {"width": 4.70, "center_y": 21.22},
            "E0_6": {"width": 6.05, "center_y": 27.04}
        }

        self.frame_rate = 10.0  # NGSIM dataset typically 10Hz

    def load_data(self):
        """Load NGSIM dataset"""
        try:
            self.tracks = pd.read_csv(self.tracks_file)
            self.tracks_meta = pd.read_csv(self.static_file)
            # Try to read recordingMeta file to get frame rate
            try:
                self.recording_meta = pd.read_csv(self.meta_file)
                if 'frameRate' in self.recording_meta.columns:
                    self.frame_rate = float(self.recording_meta['frameRate'].values[0])
                    print(f"Got frame rate from recording metadata: {self.frame_rate} Hz")
                else:
                    print(f"frameRate column not found in recording metadata, using default frame rate: {self.frame_rate} Hz")
            except Exception as e:
                print(f"Cannot read recording metadata: {e}, using default frame rate: {self.frame_rate} Hz")

            print(f"Successfully loaded NGSIM data")
            return True
        except FileNotFoundError as e:
            print(f"Cannot find data files: {e}")
            return False

    def get_vehicle_trajectory(self, vehicle_id):
        """Get trajectory data for specific vehicle"""
        if self.tracks is None:
            return None

        vehicle_traj = self.tracks[self.tracks['id'] == vehicle_id]
        if vehicle_traj.empty:
            return None

        # Extract relevant information
        frames = vehicle_traj['frame'].values

        # Check if timestamp_ms column exists, otherwise calculate time using frame and frame rate
        if 'timestamp_ms' in vehicle_traj.columns:
            times = vehicle_traj['timestamp_ms'].values / 1000.0  # Convert to seconds
        else:
            times = frames / self.frame_rate  # Calculate time using frame rate

        # Adapt to possible column names used by NGSIM
        x_col = 'x' if 'x' in vehicle_traj.columns else 'global_x'
        y_col = 'y' if 'y' in vehicle_traj.columns else 'global_y'
        # vel_col = 'xVelocity' if 'xVelocity' in vehicle_traj.columns else 'speed'
        # acc_col = 'xAcceleration' if 'xAcceleration' in vehicle_traj.columns else 'acceleration'
        # Modified
        vel_col = 'v_Vel'  # Directly use speed column name v_Vel from data
        acc_col = 'v_Acc'

        x_positions = vehicle_traj[x_col].values
        y_positions = vehicle_traj[y_col].values
        velocities = vehicle_traj[vel_col].values
        accelerations = vehicle_traj[acc_col].values if acc_col in vehicle_traj.columns else np.zeros_like(velocities)

        # Calculate lane positions
        lane_positions = []
        for i, y in enumerate(y_positions):
            # Find which lane the vehicle is in
            lane_id = self._find_lane_by_position(y)

            if lane_id:
                lane_data = self.lane_info.get(lane_id, {})
                lane_width = lane_data.get("width", 3.7)  # Use actual width for each lane
                lane_center = lane_data.get("center_y", 0)
                lateral_offset = y - lane_center

                # Determine if in leftmost or rightmost lane (overtaking lane)
                lane_index = int(lane_id.split("_")[1])
                is_leftmost_lane = (lane_index == 6)  # Leftmost lane
                is_rightmost_lane = (lane_index == 0)  # Rightmost lane (E0_6)

                if abs(lateral_offset) < 0.1:  # More precise centering judgment
                    position_type = "center"
                else:
                    position_type = "normal"

                # If in leftmost lane and deviating left, or rightmost lane and deviating right, mark as overtaking state
                if is_leftmost_lane and lateral_offset > 0:
                    position_type = "left_overtake"
                if is_rightmost_lane and lateral_offset < 0:
                    position_type = "right_overtake"
            else:
                # Default values
                lane_width = 3.7
                lane_center = 0
                lateral_offset = 0
                position_type = "center"
                is_leftmost_lane = False
                is_rightmost_lane = False

            lane_positions.append({
                'lane_id': lane_id,
                'lane_width': lane_width,
                'lateral_offset': lateral_offset,
                'position_type': position_type,
                'is_leftmost_lane': is_leftmost_lane,
                'is_rightmost_lane': is_rightmost_lane
            })

        return {
            'frames': frames,
            'times': times,
            'x_positions': x_positions,
            'y_positions': y_positions,
            'velocities': velocities,
            'accelerations': accelerations,
            'lane_positions': lane_positions
        }

    def _find_lane_by_position(self, y_position):
        """Determine which lane the vehicle is in based on y coordinate"""
        min_distance = float('inf')
        best_lane = None

        for lane_id, lane_data in self.lane_info.items():
            center_y = lane_data["center_y"]
            lane_width = lane_data["width"]
            # Calculate distance to lane center
            distance = abs(y_position - center_y)

            # Check if within lane width range
            if distance < min_distance and distance < lane_width / 2:
                min_distance = distance
                best_lane = lane_id

        return best_lane

    def get_leading_vehicle(self, ego_id, frame):
        """Get leader vehicle information for specific frame"""
        if self.tracks is None:
            return None

        ego_data = self.tracks[(self.tracks['id'] == ego_id) & (self.tracks['frame'] == frame)]
        if ego_data.empty:
            return None

        # Adapt to possible column names used by NGSIM
        y_col = 'y' if 'y' in ego_data.columns else 'global_y'
        x_col = 'x' if 'x' in ego_data.columns else 'global_x'

        ego_y = ego_data[y_col].values[0]
        ego_lane_id = self._find_lane_by_position(ego_y)
        ego_x = ego_data[x_col].values[0]

        if not ego_lane_id:
            return None

        # Find vehicles in same lane
        same_lane_vehicles = self.tracks[
            (self.tracks['frame'] == frame) &
            (self.tracks['id'] != ego_id) &
            (self.tracks[x_col] > ego_x)
            ]

        if same_lane_vehicles.empty:
            return None

        # Find closest leader vehicle
        leading_vehicle = None
        min_distance = float('inf')

        for _, vehicle in same_lane_vehicles.iterrows():
            vehicle_lane_id = self._find_lane_by_position(vehicle[y_col])
            if vehicle_lane_id == ego_lane_id:
                distance = vehicle[x_col] - ego_x
                if distance < min_distance:
                    min_distance = distance
                    leading_vehicle = vehicle

        if leading_vehicle is not None:
            # vel_col = 'xVelocity' if 'xVelocity' in leading_vehicle else 'speed'
            # Modified
            vel_col = 'v_Vel'  # Directly use speed column name v_Vel from data
            return {
                'id': leading_vehicle['id'],
                'x': leading_vehicle[x_col],
                'y': leading_vehicle[y_col],
                'velocity': leading_vehicle[vel_col]
            }

        return None

    def get_valid_vehicle_ids(self, min_frames=400):
        """Get vehicle IDs that never changed lanes and have total frames greater than specified value"""
        if self.tracks is None:
            return []

        valid_ids = []
        vehicle_ids = self.tracks['id'].unique()

        for vehicle_id in vehicle_ids:
            # Get vehicle trajectory
            vehicle_data = self.get_vehicle_trajectory(vehicle_id)
            if vehicle_data is None:
                continue

            # Check if total frames are sufficient
            if len(vehicle_data['frames']) < min_frames:
                continue

            # Check if there was any lane changing
            lane_ids = [lp['lane_id'] for lp in vehicle_data['lane_positions'] if lp['lane_id'] is not None]
            if len(lane_ids) == 0:
                continue

            # If all frames are in same lane, consider no lane changing occurred
            if len(set(lane_ids)) == 1:
                valid_ids.append(vehicle_id)

        print(f"Found {len(valid_ids)} qualified vehicles (no lane changes and frames ≥ {min_frames})")
        return valid_ids


class IDMParameterCalibrator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.idm_model = IDMModel()
        self.bounds = [
            (0.3, 1.0),  # alpha
            (0.5, 1.5),  # beta
            (0.8, 1.5),  # gamma
            (0.1, 0.3),  # delta
            (0.5, 1.5),  # kr
            (1.0, 1.5),  # fan
            (0.3, 0.8)  # kv
        ]
        self.maxiter = 10
        self.popsize = 8

    def simulate_vehicle(self, vehicle_data, params):
        """Simulate vehicle trajectory using IDM model"""
        # Set model parameters
        self.idm_model.alpha, self.idm_model.beta, self.idm_model.gamma, self.idm_model.delta, self.idm_model.kr, self.idm_model.fan, self.idm_model.kv = params

        # Extract vehicle data
        times = vehicle_data['times']
        dt = times[1] - times[0] if len(times) > 1 else 0.1

        # Initialize
        sim_velocities = np.zeros_like(times)
        sim_velocities[0] = vehicle_data['velocities'][0]
        sim_positions = np.zeros_like(times)
        sim_positions[0] = vehicle_data['x_positions'][0]

        # Simulate each time step
        for i in range(len(times) - 1):
            current_velocity = sim_velocities[i]
            current_position = sim_positions[i]

            # Get leader vehicle information - fixed spelling error (leadinging -> leading)
            leading_vehicle = self.data_loader.get_leading_vehicle(
                ego_id=vehicle_data['id'],
                frame=vehicle_data['frames'][i]
            )

            if leading_vehicle is not None:
                # Calculate relative velocity and gap
                delta_v = current_velocity - leading_vehicle['velocity']
                gap = leading_vehicle['x'] - current_position

                # Get lane position information
                lane_info = vehicle_data['lane_positions'][i]
                lane_position = lane_info['position_type']
                lateral_offset = lane_info['lateral_offset']
                lane_width = lane_info['lane_width']  # Use actual width of current lane

                # Calculate acceleration, pass current lane width
                acceleration = self.idm_model.calculate_acceleration(
                    current_velocity, delta_v, gap, lateral_offset, lane_position, lane_width
                )
            else:
                # No leader vehicle, use free flow acceleration
                acceleration = self.idm_model.a * (1 - (current_velocity / 23.54) ** 4)

            # Update velocity and position
            sim_velocities[i + 1] = sim_velocities[i] + acceleration * dt
            sim_positions[i + 1] = sim_positions[i] + sim_velocities[i] * dt + 0.5 * acceleration * dt ** 2

        return {
            'times': times,
            'velocities': sim_velocities,
            'positions': sim_positions,
            'accelerations': [0.0] + [(sim_velocities[i + 1] - sim_velocities[i]) / dt for i in
                                      range(len(sim_velocities) - 1)]
        }

    def objective_function(self, params):
        """Optimization objective function: calculate error between simulation results and real data"""
        total_error = 0.0
        valid_vehicles = 0

        # Get qualified vehicle IDs (no lane changes and frames ≥ 300)
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=400)

        if not valid_vehicle_ids:
            print("No qualified vehicle data found")
            return float('inf')

        # Sample from qualified vehicles
        sample_size = min(3, len(valid_vehicle_ids))
        vehicle_ids = np.random.choice(valid_vehicle_ids, sample_size, replace=False)

        print(
            f"Current parameters: α={params[0]:.2f}, β={params[1]:.2f}, γ={params[2]:.2f}, δ={params[3]:.2f}, kr={params[4]:.2f}, fan={params[5]:.2f}, kv={params[6]:.2f}")
        print(f"Calibrating parameters for {len(vehicle_ids)} qualified vehicles...")

        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle_data = self.data_loader.get_vehicle_trajectory(vehicle_id)
            if vehicle_data is None:
                continue

            # Add vehicle ID for use in function
            vehicle_data['id'] = vehicle_id

            # Simulate vehicle behavior
            sim_result = self.simulate_vehicle(vehicle_data, params)

            # Calculate error (weighted error of velocity and position)
            vel_error = np.mean((np.array(sim_result['velocities']) - np.array(vehicle_data['velocities'])) ** 2)
            pos_error = np.mean((np.array(sim_result['positions']) - np.array(vehicle_data['x_positions'])) ** 2)

            # Weighted combination of errors
            error = 0.7 * vel_error + 0.3 * pos_error
            total_error += error
            valid_vehicles += 1

            if i % 2 == 0:  # Print progress every 2 vehicles processed
                print(f"  Processing progress: {i + 1}/{len(vehicle_ids)}")

        if valid_vehicles == 0:
            return float('inf')

        return total_error / valid_vehicles

    def calibrate_parameters(self):
        """Calibrate parameters using differential evolution algorithm"""
        print("Starting IDM model parameter calibration...")

        # Check if there are qualified vehicles
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=400)
        if not valid_vehicle_ids:
            print("Insufficient qualified vehicle data for calibration")
            return None

        result = differential_evolution(
            self.objective_function,
            self.bounds,
            strategy='best1bin',
            popsize=self.popsize,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.7,
            maxiter=self.maxiter,
            workers=1,  # Use single process
        )

        # Set optimal parameters
        self.idm_model.alpha, self.idm_model.beta, self.idm_model.gamma, self.idm_model.delta, self.idm_model.kr, self.idm_model.fan, self.idm_model.kv = result.x

        print("Calibration completed!")
        print(
            f"Optimal parameters: α={result.x[0]:.4f}, β={result.x[1]:.4f}, γ={result.x[2]:.4f}, δ={result.x[3]:.4f}, kr={result.x[4]:.4f}, fan={result.x[5]:.4f}, kv={result.x[6]:.4f}")
        print(f"Final error: {result.fun:.4f}")

        return result.x

    def evaluate_model(self, best_params=None):
        """Evaluate model performance on test data"""
        if best_params is not None:
            self.idm_model.alpha, self.idm_model.beta, self.idm_model.gamma, self.idm_model.delta, self.idm_model.kr, self.idm_model.fan, self.idm_model.kv = best_params

        # Get qualified vehicle IDs (no lane changes and frames ≥ 300)
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=400)
        if not valid_vehicle_ids:
            print("Insufficient qualified vehicle data for evaluation")
            return

        # Select some vehicles for evaluation (different from calibration vehicles)
        sample_size = min(5, len(valid_vehicle_ids))
        test_vehicle_ids = np.random.choice(valid_vehicle_ids, sample_size, replace=False)

        for vehicle_id in test_vehicle_ids:
            vehicle_data = self.data_loader.get_vehicle_trajectory(vehicle_id)
            if vehicle_data is None:
                continue

            vehicle_data['id'] = vehicle_id

            # Simulate vehicle behavior
            sim_result = self.simulate_vehicle(vehicle_data, best_params)

            # Plot results
            self.plot_comparison(vehicle_data, sim_result, vehicle_id)

    def plot_comparison(self, real_data, sim_data, vehicle_id):
        """Plot comparison between real data and simulation results"""
        plt.figure(figsize=(15, 10))

        # Velocity comparison
        plt.subplot(2, 1, 1)
        plt.plot(real_data['times'], real_data['velocities'], 'b-', label='Real velocity')
        plt.plot(sim_data['times'], sim_data['velocities'], 'r--', label='Simulated velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Vehicle {vehicle_id} Velocity Comparison')
        plt.legend()
        plt.grid(True)

        # Position comparison
        plt.subplot(2, 1, 2)
        plt.plot(real_data['times'], real_data['x_positions'], 'b-', label='Real position')
        plt.plot(sim_data['times'], sim_data['positions'], 'r--', label='Simulated position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title(f'Vehicle {vehicle_id} Position Comparison')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        # plt.savefig(f'vehicle_{vehicle_id}_comparison.png')
        plt.savefig(f'calibration_png/vehicle_{vehicle_id}_comparison.png')
        plt.close()


# Usage example
if __name__ == "__main__":
    # Define data file paths
    tracks_file = "data/track_NGSIM_data-0500-0515.csv"
    static_file = "data/static_NGSIM_data-0500-0515.csv"
    meta_file = "data/meta_NGSIM_data-0500-0515.csv"

    # Initialize data loader
    data_loader = NGSIMDataLoader(tracks_file, static_file, meta_file)

    # Load data
    if data_loader.load_data():
        # Initialize calibrator
        calibrator = IDMParameterCalibrator(data_loader)

        # Calibrate parameters
        best_params = calibrator.calibrate_parameters()

        # If optimal parameters found, evaluate model
        if best_params is not None:
            calibrator.evaluate_model(best_params)

            # Save optimal parameters
            with open('LD-IDM_parameters_calibrated_NGSIM.txt', 'w') as f:
                f.write(
                    f"α={best_params[0]:.4f}, β={best_params[1]:.4f}, γ={best_params[2]:.4f}, δ={best_params[3]:.4f}, kr={best_params[4]:.4f}, fan={best_params[5]:.4f}, kv={best_params[6]:.4f}")