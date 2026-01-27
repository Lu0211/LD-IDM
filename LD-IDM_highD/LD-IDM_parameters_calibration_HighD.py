import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


class IDMModel:
    def __init__(self, s0=2.0, T=1.6, a=1.5, b=1.67):
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
        self.lane_width = 3.7  # Lane width (m)

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

    def calculate_desired_gap(self, v, delta_v, d, lane_position):
        """Calculate desired safe gap considering lateral offset"""
        occlusion_factor = self.calculate_occlusion_factor(d, max(1.0, v * self.T))
        lane_position_factor = self.calculate_lane_position_factor(lane_position)
        offset_factor = self.calculate_offset_factor(d, occlusion_factor, lane_position_factor)

        s_star = self.s0 + max(0, v * self.T + (v * delta_v) / (2 * np.sqrt(self.a * self.b)))
        s_star = s_star * (1 + self.alpha * offset_factor)
        return s_star

    def calculate_acceleration(self, v, delta_v, gap, d, lane_position):
        """Calculate IDM acceleration"""
        s_star = self.calculate_desired_gap(v, delta_v, d, lane_position)

        # Speed term
        speed_term = (v / 33.3) ** 4  # Assume speed limit ~110km/h (30m/s)
        deviation_angle = math.atan2(d, gap)
        vp = 1/(1+self.kv*math.tan(deviation_angle))
        # Gap term
        gap_term = (s_star / max((gap / math.cos(deviation_angle))*vp, 0.1)) ** 2
        R = 1 + self.kr * (math.sin(deviation_angle) ** 2)

        # Final acceleration
        acceleration = (self.a * (1 - speed_term - gap_term)) / R
        return acceleration


class HighDDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.tracks = None
        self.tracks_meta = None
        self.recording_meta = None
        self.lane_info = {
            "E0_0": {"width": 3.90, "center_y": 32.48},
            "E0_1": {"width": 3.67, "center_y": 36.265},
            "E0_2": {"width": 4.01, "center_y": 40.105}
        }  # Store lane information
        self.frame_rate = 25.0  # Default frame rate 25Hz

    def load_data(self, recording_id):
        """Load HighD dataset for specific recording"""
        track_file = os.path.join(self.data_path, f"{recording_id}_tracks.csv")
        track_meta_file = os.path.join(self.data_path, f"{recording_id}_tracksMeta.csv")
        recording_meta_file = os.path.join(self.data_path, f"{recording_id}_recordingMeta.csv")

        try:
            self.tracks = pd.read_csv(track_file)
            self.tracks_meta = pd.read_csv(track_meta_file)
            # Try to read recordingMeta file to get frame rate
            try:
                self.recording_meta = pd.read_csv(recording_meta_file)
                if 'frameRate' in self.recording_meta.columns:
                    self.frame_rate = float(self.recording_meta['frameRate'].values[0])
                    print(f"Retrieved frame rate from recording metadata: {self.frame_rate} Hz")
                else:
                    print(f"frameRate column not found in recording metadata, using default frame rate: {self.frame_rate} Hz")
            except Exception as e:
                print(f"Cannot read recording metadata: {e}, using default frame rate: {self.frame_rate} Hz")

            print(f"Successfully loaded data for recording {recording_id}")
            return True
        except FileNotFoundError:
            print(f"Data files for recording {recording_id} not found")
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

        x_positions = vehicle_traj['x'].values
        y_positions = vehicle_traj['y'].values
        velocities = vehicle_traj['xVelocity'].values
        accelerations = vehicle_traj['xAcceleration'].values

        # Calculate lane position (based on SUMO lane definition)
        lane_positions = []
        for i, y in enumerate(y_positions):
            # Find vehicle's lane
            lane_id = self._find_lane_by_position(y)

            if lane_id:
                lane_data = self.lane_info.get(lane_id, {})
                lane_width = lane_data.get("width", 3.7)  # Default value
                lane_center = lane_data.get("center_y", 0)
                lateral_offset = y - lane_center

                # Determine if in leftmost lane (overtaking lane)
                is_overtaking_lane = (lane_id.endswith("_2"))  # Assume leftmost lane index is 0
                is_overtaking_right_lane = (lane_id.endswith("_0"))  # Assume leftmost lane index is 0

                if abs(lateral_offset) == 0:  # More precise center determination
                    position_type = "center"
                else:
                    position_type = "normal"

                # If in overtaking lane and shifted left, mark as "left_overtake"
                if is_overtaking_lane and lateral_offset > 0:
                    position_type = "left_overtake"
                if is_overtaking_right_lane and lateral_offset < 0:
                    position_type = "right_overtake"
            else:
                # Default values
                lane_width = 3.7
                lane_center = 0
                lateral_offset = 0
                position_type = "center"
                is_overtaking_lane = False

            lane_positions.append({
                'lane_id': lane_id,
                'lane_width': lane_width,
                'lateral_offset': lateral_offset,
                'position_type': position_type,
                'is_overtaking_lane': is_overtaking_lane
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
        """Determine vehicle's lane based on y coordinate"""
        min_distance = float('inf')
        best_lane = None

        for lane_id, lane_data in self.lane_info.items():
            center_y = lane_data["center_y"]
            distance = abs(y_position - center_y)

            if distance < min_distance and distance < lane_data["width"] / 2:
                min_distance = distance
                best_lane = lane_id

        return best_lane

    def get_leading_vehicle(self, ego_id, frame):
        """Get leading vehicle information for specific frame"""
        if self.tracks is None:
            return None

        ego_data = self.tracks[(self.tracks['id'] == ego_id) & (self.tracks['frame'] == frame)]
        if ego_data.empty:
            return None

        ego_y = ego_data['y'].values[0]
        ego_lane_id = self._find_lane_by_position(ego_y)
        ego_x = ego_data['x'].values[0]

        if not ego_lane_id:
            return None

        # Find vehicles in same lane
        same_lane_vehicles = self.tracks[
            (self.tracks['frame'] == frame) &
            (self.tracks['id'] != ego_id) &
            (self.tracks['x'] > ego_x)
            ]

        if same_lane_vehicles.empty:
            return None

        # Find closest leading vehicle
        leading_vehicle = None
        min_distance = float('inf')

        for _, vehicle in same_lane_vehicles.iterrows():
            vehicle_lane_id = self._find_lane_by_position(vehicle['y'])
            if vehicle_lane_id == ego_lane_id:
                distance = vehicle['x'] - ego_x
                if distance < min_distance:
                    min_distance = distance
                    leading_vehicle = vehicle

        if leading_vehicle is not None:
            return {
                'id': leading_vehicle['id'],
                'x': leading_vehicle['x'],
                'y': leading_vehicle['y'],
                'velocity': leading_vehicle['xVelocity']
            }

        return None

    def get_valid_vehicle_ids(self, min_frames=300):
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

            # Check if vehicle changed lanes
            lane_ids = [lp['lane_id'] for lp in vehicle_data['lane_positions'] if lp['lane_id'] is not None]
            if len(lane_ids) == 0:
                continue

            # If all frames are in same lane, consider no lane change occurred
            if len(set(lane_ids)) == 1:
                valid_ids.append(vehicle_id)

        print(f"Found {len(valid_ids)} eligible vehicles (no lane changes and frames ≥ {min_frames})")
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
            (0.3, 0.8)   # kv
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

        # Simulate each timestep
        for i in range(len(times) - 1):
            current_velocity = sim_velocities[i]
            current_position = sim_positions[i]

            # Get leading vehicle information
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

                # Calculate acceleration
                acceleration = self.idm_model.calculate_acceleration(
                    current_velocity, delta_v, gap, lateral_offset, lane_position
                )
            else:
                # No leading vehicle, use free flow acceleration
                acceleration = self.idm_model.a * (1 - (current_velocity / 33.3) ** 4)

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

        # Get eligible vehicle IDs (no lane changes and frames ≥ 240)
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=300)

        if not valid_vehicle_ids:
            print("No eligible vehicle data found")
            return float('inf')

        # Sample from eligible vehicles
        sample_size = min(3, len(valid_vehicle_ids))
        vehicle_ids = np.random.choice(valid_vehicle_ids, sample_size, replace=False)

        print(
            f"Current parameters: α={params[0]:.2f}, β={params[1]:.2f}, γ={params[2]:.2f}, δ={params[3]:.2f}, kr={params[4]:.2f}, fan={params[5]:.2f}, kv={params[6]:.2f}")
        print(f"Calibrating parameters for {len(vehicle_ids)} eligible vehicles...")

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

            # Weighted combined error
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

        # Check if there are eligible vehicles
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=300)
        if not valid_vehicle_ids:
            print("Insufficient eligible vehicle data for calibration")
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

        # Get eligible vehicle IDs (no lane changes and frames ≥ 240)
        valid_vehicle_ids = self.data_loader.get_valid_vehicle_ids(min_frames=300)
        if not valid_vehicle_ids:
            print("Insufficient eligible vehicle data for evaluation")
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
        plt.savefig(f'vehicle_{vehicle_id}_comparison.png')
        plt.close()


# Usage example
if __name__ == "__main__":
    # Initialize data loader, specify your dataset path
    data_loader = HighDDataLoader(data_path="data")

    # Load data for recording 13
    if data_loader.load_data(recording_id=13):
        # Initialize calibrator
        calibrator = IDMParameterCalibrator(data_loader)

        # Calibrate parameters
        best_params = calibrator.calibrate_parameters()

        # If optimal parameters found, evaluate model
        if best_params is not None:
            calibrator.evaluate_model(best_params)

            # Save optimal parameters
            with open('LD-IDM_parameters_calibrated_HighD.txt', 'w') as f:
                f.write(
                    f"α={best_params[0]:.4f}, β={best_params[1]:.4f}, γ={best_params[2]:.4f}, δ={best_params[3]:.4f}, kr={best_params[4]:.4f}, fan={best_params[5]:.4f}, kv={best_params[6]:.4f}")