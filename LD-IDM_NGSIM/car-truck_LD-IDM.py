import random
from collections import deque

import numpy as np
import traci
import math
import os
import sys
import pandas as pd
from datetime import datetime

# New driving behavior analysis class
class DriverBehaviorAnalyzer:
    def __init__(self):
        # Driving behavior analysis parameters
        self.speed_history = deque(maxlen=30)  # Store last 30 speed samples
        self.acceleration_history = deque(maxlen=30)  # Store last 30 acceleration samples
        self.deviation_history = deque(maxlen=30)  # Store last 30 lane deviation angle samples
        self.following_distance_history = deque(maxlen=30)  # Store last 30 following distance samples

        # Driving style thresholds
        self.acceleration_variance_threshold = 1.5  # Acceleration variance threshold
        self.deviation_angle_threshold = 0.1  # Lane deviation angle threshold
        self.following_distance_threshold = 15.0  # Following distance threshold
        self.speed_variance_threshold = 5.0  # Speed variance threshold

        # Safety parameters
        self.s0 = 6.28  # Minimum safe distance
        self.T = 1.23  # Safe time headway
        self.a_max = 1.28  # Maximum acceleration 0.73
        self.b = 2.36  # Comfortable deceleration
        self.v0 = 25.97  # Desired speed
        self.delta = 4.0  # Acceleration exponent


    def collect_data(self, speed, acceleration, deviation_angle, following_distance):
        """Collect driving data"""
        self.speed_history.append(speed)
        self.acceleration_history.append(acceleration)
        self.deviation_history.append(deviation_angle)
        self.following_distance_history.append(following_distance)

    def analyze_behavior(self):
        """Analyze driving behavior and return behavioral characteristics"""
        if len(self.acceleration_history) < 10:
            return {"is_enough_data": False}

        # Calculate various driving behavior indicators
        acceleration_variance = np.var(self.acceleration_history)
        speed_variance = np.var(self.speed_history)
        avg_deviation = np.mean(np.abs(list(self.deviation_history)))
        avg_following_distance = np.mean(self.following_distance_history)

        # Calculate behavioral characteristic scores (0-1, 1 means most conservative)
        acceleration_score = max(0, min(1, 1 - acceleration_variance / self.acceleration_variance_threshold))
        speed_score = max(0, min(1, 1 - speed_variance / self.speed_variance_threshold))
        deviation_score = max(0, min(1, 1 - avg_deviation / self.deviation_angle_threshold))
        following_distance_score = max(0, min(1, avg_following_distance / self.following_distance_threshold))

        # Comprehensive score (simple average)
        # overall_score = (acceleration_score + speed_score + deviation_score + following_distance_score) / 4
        overall_score = (0.3*acceleration_score + 0.5*speed_score + 0.1*deviation_score + 0.1*following_distance_score)

        return {
            "is_enough_data": True,
            "acceleration_variance": acceleration_variance,
            "speed_variance": speed_variance,
            "avg_deviation": avg_deviation,
            "avg_following_distance": avg_following_distance,
            "acceleration_score": acceleration_score,
            "speed_score": speed_score,
            "deviation_score": deviation_score,
            "following_distance_score": following_distance_score,
            "overall_score": overall_score
        }

    def determine_driver_type(self):
        """Determine driver type based on behavior analysis results"""
        analysis = self.analyze_behavior()
        if not analysis["is_enough_data"]:
            return "insufficient_data", 0.5

        score = analysis["overall_score"]

        if score >= 0.7:
            return "conservative", score
        elif score >= 0.4:
            return "moderate", score
        else:
            return "aggressive", score

    def calculate_fan(self, actual_distance, delta_y, deviation_angle, v, delta_v):
        """Calculate fan value based on driver type"""
        driver_type, score = self.determine_driver_type()

        # Adjust parameters based on driver type
        if driver_type == "conservative":
            fan = 0.7 + 0.3 * score  # Conservative drivers have fan close to 0.8
        elif driver_type == "aggressive":
            fan = 0.3 - 0.3 * score  # Aggressive drivers have fan close to 0.8
        else:
            fan = 0.5  # Moderate drivers use default value

        return fan


# Create output data list
output_data = []

STEP_LENGTH = 0.01  # Simulation step length (seconds)
# Control time parameters
OVERRIDE_START_TIME = 0.0  # Start control time (seconds)
OVERRIDE_DURATION = 95.0  # Control duration (seconds)

def get_idm_acceleration(vehicle_id):
    """Get acceleration calculated by SUMO internal IDM model"""
    # Define default parameter values in case of retrieval failure
    default_params = {
        "idm.maxSpeed": 25.97,  # Default desired speed (200 km/h)
        "idm.safeTimeHeadway": 1.23,  # Default safe time headway
        "idm.minGap": 6.28,  # Default minimum gap
        "idm.accel": 1.28,  # Default maximum acceleration
        "idm.decel": 2.36,  # Default comfortable deceleration
        "idm.delta": 4.0  # Default acceleration exponent
    }

    # Get IDM model parameters, use default values to handle empty strings or exceptions
    def get_param(param_name):
        try:
            value = traci.vehicle.getParameter(vehicle_id, param_name)
            return float(value) if value else default_params[param_name]
        except (ValueError, traci.TraCIException):
            return default_params[param_name]

    v0 = get_param("idm.maxSpeed")
    T = get_param("idm.safeTimeHeadway")
    s0 = get_param("idm.minGap")
    a = get_param("idm.accel")
    b = get_param("idm.decel")
    delta = get_param("idm.delta")

    # Get current vehicle state
    v = traci.vehicle.getSpeed(vehicle_id)  # Current speed

    # Get leader vehicle information
    leader = traci.vehicle.getLeader(vehicle_id, dist=150)

    if leader is not None:
        s = leader[1]  # Distance to leader vehicle
        vl = traci.vehicle.getSpeed(leader[0])  # Leader vehicle speed
        delta_v = v - vl  # Speed difference
        # Calculate desired gap
        s_star = s0 + max(0, v * T + (v * delta_v) / (2 * math.sqrt(a * b)))
        # Calculate IDM acceleration
        idm_accel = a * (1 - math.pow(v / v0, delta) - (s_star / max(s, 0.1)) ** 2)
    else:
        # When no leader vehicle, follow real dataset acceleration
        idm_accel = a * (1 - math.pow(v / v0, delta))

    return idm_accel

# Define center Y coordinates for each lane (based on network file)
LANE_CENTER_Y = {
    0: 1.50,   # Lane 0 center Y coordinate
    1: 5.31,   # Lane 1 center Y coordinate
    2: 9.04,   # Lane 2 center Y coordinate
    3: 12.75,  # Lane 3 center Y coordinate
    4: 16.62,  # Lane 4 center Y coordinate
    5: 21.22,  # Lane 5 center Y coordinate
    6: 27.04   # Lane 6 center Y coordinate
}

# Store lateral offset state for each vehicle
vehicle_lateral_offsets = {}  # Record current lateral offset for each vehicle
# Lateral position setting function (reduce code duplication)
def set_vehicle_lateral_position(vehicle_id, real_y_position):
    try:
        # Get current lane of vehicle
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        lane_width = traci.lane.getWidth(lane_id)

        # Determine lane type, get lane center Y coordinate
        try:
            # Extract lane index from lane_id (assume format "E0_0", "E1_1", etc.)
            lane_index = int(lane_id.split("_")[-1])
            lane_center_y = LANE_CENTER_Y[lane_index]
        except (ValueError, IndexError):
            print(f"Warning: Invalid lane ID '{lane_id}', using default value")
            lane_center_y = 14.8  # Default value, take network center Y coordinate

        # Calculate offset relative to lane center
        target_offset = real_y_position - lane_center_y
        vehicle_lateral_offsets[vehicle_id] = target_offset

        # Set vehicle lateral position
        traci.vehicle.setLateralLanePosition(vehicle_id, target_offset)
    except traci.TraCIException as e:
        print(f"Error setting lateral position for vehicle {vehicle_id}: {e}")
        # Alternative method: use moveToXY method
        x, _ = traci.vehicle.getPosition(vehicle_id)  # Get current X coordinate
        traci.vehicle.moveToXY(
            vehicle_id,
            edgeID=traci.vehicle.getRoadID(vehicle_id),
            laneIndex=traci.vehicle.getLaneIndex(vehicle_id),
            x=x,
            y=real_y_position,  # Directly use Y coordinate from real data
            angle=traci.vehicle.getAngle(vehicle_id),
            keepRoute=True
        )


# Dictionary to store leader vehicle information (ID → historical data)
leader_history = {}


def get_leader_acceleration(vehicle_id, time_step):
    """Get leader vehicle acceleration and change trend"""
    # Get leader vehicle information (leader_id, distance)
    leader_info = traci.vehicle.getLeader(vehicle_id, dist=100)

    if leader_info is None:
        return None, None, None  # No leader vehicle

    leader_id, distance = leader_info

    # Get leader vehicle current speed
    current_speed = traci.vehicle.getSpeed(leader_id)

    # Update leader vehicle historical data
    if leader_id not in leader_history:
        leader_history[leader_id] = {"speeds": [current_speed], "times": [time_step]}
        return None, None, None  # First record, cannot calculate acceleration

    # Store latest speed and time
    leader_history[leader_id]["speeds"].append(current_speed)
    leader_history[leader_id]["times"].append(time_step)

    # Only keep last two data points
    if len(leader_history[leader_id]["speeds"]) > 2:
        leader_history[leader_id]["speeds"] = leader_history[leader_id]["speeds"][-2:]
        leader_history[leader_id]["times"] = leader_history[leader_id]["times"][-2:]

    # Calculate acceleration (m/s²)
    if len(leader_history[leader_id]["speeds"]) == 2:
        delta_v = leader_history[leader_id]["speeds"][1] - leader_history[leader_id]["speeds"][0]
        delta_t = leader_history[leader_id]["times"][1] - leader_history[leader_id]["times"][0]

        if delta_t > 0:
            current_accel = delta_v / delta_t

            # Get previous acceleration (if available)
            prev_accel = leader_history[leader_id].get("last_accel", None)
            leader_history[leader_id]["last_accel"] = current_accel

            # Determine change trend
            trend = None
            if prev_accel is not None:
                if current_accel > 0 and current_accel > prev_accel:
                    trend = "acceleration increasing"
                elif current_accel > 0 and current_accel < prev_accel:
                    trend = "acceleration decreasing"
                elif current_accel < 0 and current_accel < prev_accel:  # Note negative sign, smaller value means larger deceleration
                    trend = "deceleration increasing"
                elif current_accel < 0 and current_accel > prev_accel:
                    trend = "deceleration decreasing"
                elif current_accel == 0:
                    trend = "constant speed"

            return leader_id, current_accel, trend

    return leader_id, None, None


def run_simulation(config_file: str, real_data_file: str, sumo_binary: str = "sumo-gui"):
    # Check if SUMO_HOME environment variable is set
    if 'SUMO_HOME' not in os.environ:
        print("Please set SUMO_HOME environment variable to point to SUMO installation directory")
        sys.exit(1)
    # Add SUMO tools directory to system path
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    # Check if configuration file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"SUMO configuration file does not exist: {config_file}")
    # Check if real data file exists
    if not os.path.exists(real_data_file):
        raise FileNotFoundError(f"Real data file does not exist: {real_data_file}")

    # Read real vehicle data
    real_data = pd.read_csv(real_data_file)
    print(f"Loaded real data: {len(real_data)} records")

    # Ensure data contains necessary columns
    required_columns = ['frame', 'id', 'speed', 'acceleration','y_position']
    for col in required_columns:
        if col not in real_data.columns:
            raise ValueError(f"Real data missing necessary column: {col}")

    # Build SUMO command
    sumo_cmd = [sumo_binary, "-c", config_file, "--step-length", str(STEP_LENGTH)]
    print(f"Preparing to start SUMO simulation: {' '.join(sumo_cmd)}")
    simulation_time = -0.01

    # Create output directory (if not exists)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output file initialization, save to output directory
    output_file_path = os.path.join(output_dir, f"info_idm_vehicle_data_2025_06_18.csv")
    with open(output_file_path, 'w') as output_file:
        output_file.write(
            "frame,time,vehicle_id,speed,idm_acceleration,specified_acceleration,position_x,y_position,heading,gap,override_active\n")
        leader_id = None  # Leader vehicle ID
        real_data_index = 0  # Current real data index being used
        real_data_index_notleader = 0 # Current real data index being used for non-leaders
        real_leader_id = 466  # Leader ID in real dataset
        real_follower_id = 478
        leader_real_data = real_data[real_data['id'] == real_leader_id]
        if leader_real_data.empty:
            raise ValueError(f"Cannot find records for vehicle ID {real_leader_id} in real data")
        # Filter data for follower vehicles from real data
        follower_real_data = real_data[real_data['id'] == real_follower_id]
        if follower_real_data.empty:
            raise ValueError(f"Cannot find records for vehicle ID {real_follower_id} in real data")
        try:
            # Start SUMO and connect TraCI
            traci.start(sumo_cmd)
            # Get total simulation time steps
            sim_end_time = 7200  # Can be modified as needed
            override_active = False
            # Main simulation loop
            step = 0
            # frame = 150  # 316
            frame = 447  #466
            # Define mapping relationship between vehicle IDs and initial Y coordinates
            vehicle_initial_ys = {
                # "1488": 16.57,
                # "1499": 16.35,
                # "445": 9.79,
                # "466": 8.8,
                # "628": 9.79,
                # "631": 9.27,
                # "751": 9.63,
                # "755": 9.34,
                # "478": 9.27,
                # "489": 9.1,
                # "492": 9.58,
                # "637": 9.56,
                # "645": 8.8,
                # "651": 9.18,
                # "767": 9.49,
                # "775": 9.65,
                # "785": 9.87,
                # "1185": 10.17,
                # "1193": 9.41,
                # "768": 12.86,
                # "776": 12.91,
                "466": 8.8,
                "478": 9.27,
            }
            # # Store lateral offset state for each vehicle
            # vehicle_lateral_offsets = {}  # Record current lateral offset for each vehicle
            vehicle_analyzers = {}  # Create a DriverBehaviorAnalyzer instance for each vehicle
            while step < sim_end_time:
                # Execute one simulation step
                traci.simulationStep()
                simulation_time += 0.01
                if step % 10 == 0:
                    frame += 1
                # Add your control logic here
                # For example: get vehicle information, control traffic lights, etc.
                # Dynamically get IDs of all current vehicles
                all_vehicle_ids = traci.vehicle.getIDList()
                # Select vehicle with largest x-coordinate as leader
                if all_vehicle_ids:
                    # Get positions of all vehicles and find vehicle with maximum x
                    max_x = -float('inf')
                    new_leader_id = None
                    for vehicle_id in all_vehicle_ids:
                        traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                        pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
                        if pos_x > max_x:
                            max_x = pos_x
                            new_leader_id = vehicle_id
                    # If new leader detected, update leader ID
                    if new_leader_id != leader_id:
                        leader_id = new_leader_id

                # Control leader vehicle to drive according to real data
                if leader_id and leader_id in traci.vehicle.getIDList() and real_data_index < len(leader_real_data):
                    # Get real data corresponding to current frame
                    current_real_data = leader_real_data.iloc[real_data_index]
                    # Ensure real data ID matches current leader ID (for verification only)
                    real_id = current_real_data['id']
                    # Set leader vehicle speed
                    real_speed = current_real_data['speed']  # Unit: m/s
                    traci.vehicle.setSpeed(leader_id, real_speed)
                    # Set leader vehicle lateral position
                    set_vehicle_lateral_position(leader_id, current_real_data['y_position'])
                    # Record data and update index only every 4 frames
                    if step % 10 == 0:
                        # print(f"Frame {frame}: Set leader {leader_id} speed to {real_speed:.2f} m/s (real data ID: {real_id})")
                        # Increase real data index
                        real_data_index += 1
                        if real_data_index >= len(leader_real_data):
                            print("Used up all real data for leader vehicle")
                # Set lateral position for all vehicles (including leader)
                for vehicle_id in all_vehicle_ids:
                    if vehicle_id != leader_id and real_data_index_notleader < len(follower_real_data):
                        # print('len()',len(follower_real_data))
                        # Leader vehicle uses leader dataset
                        current_real_data = follower_real_data.iloc[real_data_index_notleader]
                        set_vehicle_lateral_position(vehicle_id, current_real_data['y_position'])
                        if step % 10 == 0:
                            real_data_index_notleader += 1
                    # elif vehicle_id != leader_id and real_data_index_notleader >= len(follower_real_data):
                    elif real_data_index >= len(leader_real_data) or real_data_index_notleader >= len(follower_real_data):
                        # Vehicles without available real data use default behavior
                        # Get vehicle current lane and lateral position
                        lane_id = traci.vehicle.getLaneID(vehicle_id)
                        lane_width = traci.lane.getWidth(lane_id)
                        # Determine lane type, get lane center Y coordinate
                        try:
                            # Extract lane index from lane_id (assume format "E0_0", "E1_1", etc.)
                            lane_index = int(lane_id.split("_")[-1])
                            lane_center_y = LANE_CENTER_Y[lane_index]
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid lane ID '{lane_id}', using default value")
                            # lane_center_y = 14.8  # Default value, take network center Y coordinate
                        if step % 10 == 0:
                            real_data_index += 1
                            real_data_index_notleader += 1
                        # Lateral offset calculation logic
                        if step < 10 and vehicle_id in vehicle_initial_ys:
                            # if step < 10 and vehicle_id == '191':
                            # First 10 steps: use predefined Y coordinates
                            target_y = vehicle_initial_ys[vehicle_id]
                            target_offset = target_y - lane_center_y
                            vehicle_lateral_offsets[vehicle_id] = target_offset
                        elif step % 10 == 0:  # Every 10 steps (starting from step 20): major adjustment
                            if vehicle_id in vehicle_lateral_offsets:
                                # Get previous offset and calculate actual Y coordinate
                                prev_offset = vehicle_lateral_offsets[vehicle_id]
                                prev_actual_y = lane_center_y + prev_offset
                                # Calculate random adjustment amount based on lane width
                                max_adjustment = lane_width / 2 * random.uniform(0, 0.01)
                                adjustment = random.choice([-max_adjustment, max_adjustment])
                                # Adjust based on actual Y coordinate
                                new_actual_y = prev_actual_y + adjustment
                                # Calculate new offset and store
                                target_offset = new_actual_y - lane_center_y
                                vehicle_lateral_offsets[vehicle_id] = target_offset
                                # print(f'Step {step}: Vehicle {vehicle_id} large adj Y: {new_actual_y:.3f} (prev: {prev_actual_y:.3f}, adj: {adjustment:.3f})')
                            else:
                                # Edge case handling
                                max_offset = lane_width / 2 * random.uniform(0, 0.01)
                                target_offset = random.choice([-max_offset, max_offset])
                                vehicle_lateral_offsets[vehicle_id] = target_offset
                        else:
                            # Maintain previous offset state
                            if vehicle_id in vehicle_lateral_offsets:
                                target_offset = vehicle_lateral_offsets[vehicle_id]
                            else:
                                target_offset = 0.0
                                vehicle_lateral_offsets[vehicle_id] = target_offset
                        try:
                            # Set vehicle lateral position
                            traci.vehicle.setLateralLanePosition(vehicle_id, target_offset)
                        except traci.TraCIException as e:
                            print(f"Error setting lateral position for vehicle {vehicle_id}: {e}")
                            # Alternative method: use moveToXY method
                            x, _ = traci.vehicle.getPosition(vehicle_id)
                            new_y = lane_center_y + target_offset
                            print(f'Fallback to moveToXY: new_y={new_y}')
                            traci.vehicle.moveToXY(
                                vehicle_id,
                                edgeID=traci.vehicle.getRoadID(vehicle_id),
                                laneIndex=traci.vehicle.getLaneIndex(vehicle_id),
                                x=x,
                                y=new_y,
                                angle=traci.vehicle.getAngle(vehicle_id),
                                keepRoute=True
                            )

                for vehicle_id in all_vehicle_ids:
                    # Get vehicle actual position (for verification)
                    actual_pos = traci.vehicle.getPosition(vehicle_id)
                    # actual_offset = actual_pos[1] - lane_center_y
                    # Get basic vehicle information
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    position = traci.vehicle.getPosition(vehicle_id)
                    x, y = position[0], position[1]  # Current vehicle coordinates
                    heading = traci.vehicle.getAngle(vehicle_id)

                    if step % 10 == 0:
                        # Get IDM model calculated acceleration
                        idm_accel = get_idm_acceleration(vehicle_id)
                        # print('idm_accel', idm_accel)
                        # Determine if custom control is activated
                        override_active = (
                                OVERRIDE_START_TIME <= simulation_time < OVERRIDE_START_TIME + OVERRIDE_DURATION
                        )
                        specified_acceleration = idm_accel  # Default use IDM acceleration
                        # Initialize leader vehicle information
                        leader_x, leader_y = None, None
                        delta_x, delta_y = 0, 0
                        deviation_angle = 0
                        # Get leader vehicle information
                        lead_vehicle = traci.vehicle.getLeader(vehicle_id)
                        print('vehicle_id, lead_vehicle:', vehicle_id, lead_vehicle)
                        gap = lead_vehicle[1] if lead_vehicle is not None else float('inf')
                        if lead_vehicle is not None:
                            leader_id = lead_vehicle[0]
                            leader_pos = traci.vehicle.getPosition(leader_id)
                            leader_accel = traci.vehicle.getAcceleration(leader_id)
                            # follower_pos = traci.vehicle.getPosition(vehicle_id)
                            leader_x, leader_y = leader_pos[0], leader_pos[1]
                            # follower_x, follower_y = follower_pos[0], follower_pos[1]
                            # Get vehicle length
                            leader_length = traci.vehicle.getLength(leader_id)
                            print('leader_length:', leader_length)
                            # follower_length = traci.vehicle.getLength(vehicle_id)
                            print('x, y:', x, y)
                            print('leader_x, leader_y:', leader_x, leader_y)
                            # Calculate coordinate differences
                            delta_x = leader_x - x
                            delta_y = leader_y - y
                            delta_y = abs(delta_y)
                            print('delta_x, delta_y:', delta_x, delta_y)
                            actual_distance = max(0.1, delta_x - leader_length)  # Avoid zero distance
                            s = lead_vehicle[1]  # Distance to leader vehicle
                            print('actual_distance, s:', actual_distance, s)
                            # Calculate lateral deviation angle (radians)
                            if delta_x != 0:  # Avoid division by zero
                                deviation_angle = math.atan2(delta_y, actual_distance)
                            else:
                                deviation_angle = math.pi / 2 if delta_y > 0 else -math.pi / 2
                            # Convert angle to degrees (for recording only)
                            deviation_angle_deg = math.degrees(deviation_angle)
                            # Only override acceleration when control is active
                            if override_active:
                                # Get vehicle speed and speed difference
                                v = speed
                                leader_speed = traci.vehicle.getSpeed(leader_id)
                                delta_v = v - leader_speed
                                s0 = 6.28
                                a_max = 1.28  # 0.73
                                T0 = 1.23
                                b = 2.36
                                v0 = 25.97
                                delta = 4.0
                                lambda_coeff = 1.5
                                k_style = 0.5
                                # Get vehicle type
                                vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                                # Get width of this type (unit: meters)
                                width = traci.vehicletype.getWidth(vehicle_type)
                                print('vehicle_id: width:', vehicle_id, width)

                                alpha = 0.4739
                                gamma = 0.9706
                                beta = 0.7192
                                delta_offset = 0.2090
                                deviation_angle = math.atan2(delta_y, actual_distance)
                                if actual_distance <= 150:
                                    OcclusionFactor = (math.atan2(delta_y + width / 2, actual_distance) - math.atan2(
                                        delta_y - width / 2, actual_distance)) / (
                                                              2 * math.atan2(width, 2 * actual_distance))
                                else:
                                    OcclusionFactor = 0
                                print('OcclusionFactor:', OcclusionFactor)
                                # lane_id = traci.vehicle.getLaneID(vehicle_id)
                                # print('lane_id',lane_id)
                                try:
                                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                                    print('lane_id', lane_id)
                                except Exception as e:
                                    print(f"Failed to get lane ID: {e}")
                                    lane_id = None  # Or other default value
                                lane_num = int(lane_id.split('_')[1])  # Split string and take second part
                                # Determine lane type, get lane center Y coordinate
                                try:
                                    # Extract lane index from lane_id (assume format "E0_0", "E1_1", etc.)
                                    lane_index = int(lane_id.split("_")[-1])
                                    lane_center_y = LANE_CENTER_Y[lane_index]
                                except (ValueError, IndexError):
                                    print(f"Warning: Invalid lane ID '{lane_id}', using default value")
                                    lane_center_y = 15  # Default value, take network center Y coordinate
                                print('lane_center_y', lane_center_y)
                                if delta_y == 0:
                                    LanePositionFactor = 0.0
                                elif lane_num == 6 and y - lane_center_y > 0:  # If overtaking lane and deviating left
                                    LanePositionFactor = 1.5
                                elif lane_num == 0 and y - lane_center_y < 0:  # If rightmost lane and deviating right
                                    LanePositionFactor = 1.2
                                else:
                                    LanePositionFactor = 1

                                # Collect driving data
                                if vehicle_id not in vehicle_analyzers:
                                    vehicle_analyzers[vehicle_id] = DriverBehaviorAnalyzer()
                                analyzer = vehicle_analyzers[vehicle_id]
                                acceleration = traci.vehicle.getAcceleration(vehicle_id)
                                analyzer.collect_data(speed, acceleration, deviation_angle, actual_distance)
                                # Calculate fan value: driver style parameter (following driver visual sensitivity coefficient, conservative: 1, aggressive: 0)
                                fan = analyzer.calculate_fan(actual_distance, delta_y, deviation_angle, v, delta_v)
                                print('fan:', fan)

                                # Calculate driver perceived gap
                                psi = 1.4652
                                kv = 0.5096
                                kr = 0.8479
                                # Three lanes
                                # OffsetDistance = math.pow(delta_y / (3.76-width),psi)  # 2 lanes, rightmost lane(1)
                                OffsetDistance = math.pow(delta_y / (3.7 - width), psi)  # 3 lanes(2)
                                # OffsetDistance = math.pow(delta_y / (3.73 - width), psi)  # 4 lanes(3)
                                # OffsetDistance = math.pow(delta_y / (4.2 - width), psi)  # 5 lanes(4)

                                Vp = 1 / (1 + kv * math.tan(deviation_angle))
                                s_perceived = (actual_distance / math.cos(deviation_angle)) * Vp
                                OffsetFactor = beta * OffsetDistance + gamma * OcclusionFactor + LanePositionFactor * delta_offset
                                R_thea = 1 + kr * (math.sin(deviation_angle) ** 2)


                                T = T0
                                # Get leader vehicle information
                                leader_id, accel, trend = get_leader_acceleration(vehicle_id, step)
                                # if trend == "acceleration increasing" or trend == "acceleration decreasing":
                                # if leader_accel > 0:
                                # if real_accel > 0:
                                real_accel = real_data.iloc[frame]['acceleration']
                                print('lead_acceleration:', leader_accel)
                                # Smooth acceleration (avoid sudden changes)
                                if 'last_accel' in locals():
                                    if abs(accel - last_accel) > 1:
                                        smooth_factor = 0.96
                                    else:
                                        smooth_factor = 0.6
                                    specified_acceleration = smooth_factor * specified_acceleration + (1 - smooth_factor) * last_accel


                                # Record current acceleration for next frame smoothing
                                last_accel = specified_acceleration
                                if trend == "constant speed":
                                    specified_acceleration = last_accel
                                else:
                                    if trend == "acceleration increasing" or trend == "deceleration decreasing":
                                        # Leader acceleration capability enhanced or deceleration capability weakened, reduce safe distance
                                        s_star = s0 + max(0, v * T + (v * delta_v) / (2 * math.sqrt(a_max * b))) * (1 + alpha * OffsetFactor)
                                        specified_acceleration = a_max * (1 - math.pow(v / v0, delta) - math.pow(s_star / s_perceived,2)) / R_thea
                                        if leader_accel >= 0:
                                            specified_acceleration = max(-0.8, min(last_accel,specified_acceleration))  # max(0.1, specified_acceleration)
                                        else:
                                            specified_acceleration = min(-0.8, min(last_accel, specified_acceleration))
                                    else:
                                        s_star = s0 + max(0, v * T + (v * delta_v) / (2 * math.sqrt(a_max * b))) * (1 + alpha * OffsetFactor)
                                        specified_acceleration = a_max * (1 - math.pow(v / v0, delta) - math.pow(s_star / s_perceived,2)) / R_thea
                                        if leader_accel >= 0:
                                            specified_acceleration = max(-0.5, min(last_accel,specified_acceleration))  # max(0.1, specified_acceleration)
                                        else:
                                            specified_acceleration = min(-0.8, min(last_accel, specified_acceleration))  # min(-0.1, specified_acceleration)

                                # Collect driving data
                                if vehicle_id not in vehicle_analyzers:
                                    vehicle_analyzers[vehicle_id] = DriverBehaviorAnalyzer()
                                analyzer = vehicle_analyzers[vehicle_id]
                                acceleration = traci.vehicle.getAcceleration(vehicle_id)
                                analyzer.collect_data(speed, acceleration, deviation_angle, actual_distance)
                                # Calculate fan value
                                fan = analyzer.calculate_fan(actual_distance, delta_y, deviation_angle, v, delta_v)
                                print('fan:', fan)

                                # Set specified acceleration for vehicle
                                traci.vehicle.setAcceleration(vehicle_id, specified_acceleration,duration=STEP_LENGTH)
                                # # Calculate target speed based on acceleration
                                print('real_accel,specified_acceleration:',real_accel,specified_acceleration)
                                target_speed = speed + specified_acceleration * STEP_LENGTH * 10
                                if vehicle_id is not leader_id:
                                    traci.vehicle.setSpeed(vehicle_id, target_speed)
                                # Print debug information (only when control is active)
                                print(f"Time: {simulation_time:.2f}, Vehicle: {vehicle_id}, "
                                      f"IDM Accel: {idm_accel:.2f}, Lateral Angle: {deviation_angle_deg:.2f}°, "
                                      f"Specified Accel: {specified_acceleration:.2f}")
                        else:
                            # Check if frame index is within real_data range
                            if frame-447 < len(real_data):
                                real_accel = real_data.iloc[frame]['acceleration']  # Unit: m/s²
                                specified_acceleration = real_accel
                                print('real_accel: specified_acceleration:', real_accel, specified_acceleration)
                            else:
                                print(f"Warning: Frame {frame} exceeds real data range (total frames: {len(real_data)})")
                                # Can choose to use last frame data or other default behavior
                                if len(real_data) > 0:
                                    real_accel = real_data.iloc[-1]['acceleration']
                                    specified_acceleration = real_accel
                            # Set acceleration
                            traci.vehicle.setAcceleration(vehicle_id, specified_acceleration, duration=STEP_LENGTH)
                            # Calculate target speed based on acceleration
                            target_speed = speed + specified_acceleration * STEP_LENGTH * 10
                            if vehicle_id is not leader_id:
                                traci.vehicle.setSpeed(vehicle_id, target_speed)
                            # Print debug information (only when control is active)
                            print(f"Time: {simulation_time:.2f}, Vehicle: {vehicle_id}, "
                                  f"Specified Accel: {specified_acceleration:.2f}")
                        output_data.append({
                            'frame': frame,
                            'acceleration': specified_acceleration,
                            'id': vehicle_id,
                            'time': simulation_time,
                            'speed': speed,
                            'idm_acceleration': idm_accel,
                            'position_x': position[0],
                            'y_position': position[1],
                            'heading': heading,
                            # 'gap': gap,
                            'override_active': override_active
                        })

                # Increment simulation step
                step += 1
            # Close TraCI connection
            traci.close()
            print("SUMO simulation completed successfully")

            # Check data before saving
            if output_data:
                print(f"Collected data entries: {len(output_data)}")
                # Check data structure
                print(f"Data sample: {output_data[0]}")

                # Create DataFrame and save
                df = pd.DataFrame(output_data)
                print(f"DataFrame column names: {df.columns.tolist()}")
                print(f"DataFrame shape: {df.shape}")

                # Ensure acceleration column exists
                if 'acceleration' in df.columns:
                    print(
                        f"Acceleration column statistics: mean={df['acceleration'].mean():.4f}, max={df['acceleration'].max():.4f}, min={df['acceleration'].min():.4f}")
                else:
                    print("Warning: Acceleration column not in DataFrame!")

                # Save data
                df.to_excel("output/fcd_LD-IDM/478.xlsx", index=False)
                print("Data saved to output/fcd_LD-IDM/478.xlsx")
            else:
                print("Warning: No data collected!")

        except Exception as e:
            print(f"Error running SUMO simulation: {e}")
            # Ensure connection is closed on error
            if traci.isConnected():
                traci.close()
        finally:
            # Close connection
            if traci.isConnected():
                traci.close()


if __name__ == "__main__":
    # Configuration parameters
    SUMO_CONFIG_FILE = "simulation.sumo.cfg"  # SUMO configuration file path
    SUMO_BINARY = "sumo-gui"  # SUMO executable, use "sumo" to run in background
    REAL_DATA_FILE = "ngsim_selected_tracks.csv"  # Real data file path
    # Run simulation
    run_simulation(SUMO_CONFIG_FILE, REAL_DATA_FILE, SUMO_BINARY)

    # Save data as Excel
    if output_data:
        df = pd.DataFrame(output_data)
        excel_path = os.path.join("output", "info_idm_fcd_with_leader_follower.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Data saved to {excel_path}")
    else:
        print("No data to save")