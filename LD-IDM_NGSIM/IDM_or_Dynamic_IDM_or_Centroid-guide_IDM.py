import traci
import os
import sys
import time
import pandas as pd
from sumolib import checkBinary

# Create output data list
output_data = []


def run_sumo_simulation(config_file: str, real_data_file: str, sumo_binary: str = "sumo-gui"):
    """
    Connect to and control SUMO simulation using TraCI

    Parameters:
        config_file: Path to SUMO configuration file
        real_data_file: Path to CSV file containing real vehicle trajectories
        sumo_binary: Path to SUMO executable, defaults to "sumo-gui" to launch GUI
    """
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
    required_columns = ['frame', 'id', 'speed', 'acceleration']
    for col in required_columns:
        if col not in real_data.columns:
            raise ValueError(f"Real data missing necessary column: {col}")

    # Build SUMO command
    sumo_cmd = [sumo_binary, "-c", config_file]

    print(f"Preparing to start SUMO simulation: {' '.join(sumo_cmd)}")

    # Create output directory (if not exists)
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_time = 0.0
    leader_id = None  # Leader vehicle ID
    real_data_index = 0  # Current real data index being used
    real_leader_id = 459  # Leader ID in real dataset

    # Filter leader vehicle data from real data
    leader_real_data = real_data[real_data['id'] == real_leader_id]
    if leader_real_data.empty:
        raise ValueError(f"Cannot find records for vehicle ID {real_leader_id} in real data")

    print(f"Prepared {len(leader_real_data)} records for real leader vehicle {real_leader_id}")

    try:
        # Start SUMO and connect TraCI
        traci.start(sumo_cmd)
        # Get total simulation time steps
        sim_end_time = 8500  # Can be modified as needed
        # Main simulation loop
        step = 0
        # frame = 175  # 328 (176-1)
        # frame = 151  # 323
        # frame = 5  # 294
        # frame = 33  # 305
        # frame = 150  # 316
        # frame = 168  # 321
        # frame = 364  # 406
        # frame = 1508 #838
        # frame = 1945  # 943
        frame = 491  # 459
        # frame = 4213  # 1488
        # frame = 396  # 445
        # frame = 922  # 628
        # frame = 1295  # 751
        # frame = 491  # 478
        # frame = 511  #489
        # frame = 982 #637
        # frame = 1009  #645
        # frame = 1374  # 767
        # frame = 1398  # 775
        # frame = 2984  #1185
        # frame = 1286  # 768
        # frame = 447  #466
        while step < sim_end_time:
            # Execute one simulation step
            traci.simulationStep()
            simulation_time += 0.01
            # Get all vehicle IDs
            vehicle_ids = traci.vehicle.getIDList()

            # Select vehicle with largest x-coordinate as leader
            if vehicle_ids:
                # Get positions of all vehicles and find vehicle with maximum x
                max_x = -float('inf')
                new_leader_id = None

                for vehicle_id in vehicle_ids:
                    pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
                    if pos_x > max_x:
                        max_x = pos_x
                        new_leader_id = vehicle_id
                # If new leader detected, update leader ID
                if new_leader_id != leader_id:
                    leader_id = new_leader_id
                    print(f"Frame {frame}: Updated leader vehicle ID to {leader_id}, position x={max_x:.2f}")

            # Control leader vehicle to drive according to real data
            if leader_id and leader_id in traci.vehicle.getIDList() and real_data_index < len(leader_real_data):
                # Get real data corresponding to current frame
                current_real_data = leader_real_data.iloc[real_data_index]
                # Ensure real data ID matches current leader ID (for verification only)
                real_id = current_real_data['id']
                # Set leader vehicle speed
                real_speed = current_real_data['speed']  # Unit: m/s
                traci.vehicle.setSpeed(leader_id, real_speed)
                # Record data and update index only every 10 frames
                if step % 10 == 0:
                    # frame += 1
                    print(f"Frame {frame}: Set leader {leader_id} speed to {real_speed:.2f} m/s (real data ID: {real_id})")
                    # Increase real data index
                    real_data_index += 1
                    if real_data_index >= len(leader_real_data):
                        print("Used up all real data for leader vehicle")

            # Record all vehicle data every 4 frames
            if step % 10 == 0:
                frame += 1
                # Get all vehicle IDs
                vehicle_ids = traci.vehicle.getIDList()
                # Iterate through each vehicle to get leader and follower information
                for vehicle_id in vehicle_ids:
                    traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                    # Get current lane
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    # Get leader vehicle information (vehicle ID, distance)
                    leader = traci.vehicle.getLeader(vehicle_id, dist=500)  # Search 1000m ahead
                    leader_id = leader[0] if leader else None
                    leader_dist = leader[1] if leader else None

                    # Get follower vehicle information (calculated through other vehicles on lane)
                    follower_id = None
                    follower_dist = None
                    # Get all vehicles on current lane
                    lane_vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                    # Calculate follower distance
                    vehicle_pos = traci.vehicle.getLanePosition(vehicle_id)
                    for other_vehicle in lane_vehicles:
                        if other_vehicle == vehicle_id:
                            continue
                        other_pos = traci.vehicle.getLanePosition(other_vehicle)
                        dist = vehicle_pos - other_pos

                        # If other vehicle is behind current vehicle and closest
                        if dist > 0 and (follower_dist is None or dist < follower_dist):
                            follower_id = other_vehicle
                            follower_dist = dist

                    # Get acceleration value and check if valid
                    acceleration = traci.vehicle.getAcceleration(vehicle_id)

                    # Record data
                    # traci.vehicle.setSpeed("14", real_speed)
                    output_data.append({
                        'frame': frame,
                        'acceleration': acceleration,
                        'id': vehicle_id,
                        'leader_id': leader_id,
                        'leader_dist': leader_dist,
                        'follower_id': follower_id,
                        'follower_dist': follower_dist,
                        'speed': traci.vehicle.getSpeed(vehicle_id),
                        'position_x': traci.vehicle.getPosition(vehicle_id)[0],
                        'position_y': traci.vehicle.getPosition(vehicle_id)[1]
                    })

                    # Print data structure once every 100 steps for debugging
                    if step % 100 == 0 and vehicle_id == vehicle_ids[0]:
                        print(f"Debug: Data structure example - {output_data[-1]}")
            # Increment simulation step
            step += 1
        # Close TraCI connection
        traci.close()
        print("SUMO simulation completed successfully")

        # Check data before saving
        if output_data:
            print(f"Collected data entries: {len(output_data)}")
            # Check data structure
            print(f"Data example: {output_data[0]}")

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
           # car-car:
           # df.to_excel("output/fcd_IDM/321.xlsx", index=False)
           # print("Data saved to output/fcd_IDM/321.xlsx")
           # df.to_excel("output/fcd_Dynamic_IDM/321.xlsx", index=False)
           # print("Data saved to output/fcd_Dynamic_IDM/321.xlsx")
           # df.to_excel("output/fcd_centroid-guided_IDM/321.xlsx", index=False)
           # print("Data saved to output/fcd_centroid-guided_IDM/321.xlsx")

            # car-truck
            # df.to_excel("output/fcd_IDM/478.xlsx", index=False)
            # print("Data saved to output/fcd_IDM/478.xlsx")
            # df.to_excel("output/fcd_Dynamic_IDM/478.xlsx", index=False)
            # print("Data saved to output/fcd_Dynamic_IDM/478.xlsx")
            # df.to_excel("output/fcd_centroid-guided_IDM/478.xlsx", index=False)
            # print("Data saved to output/fcd_centroid-guided_IDM/478.xlsx")

            # truck-car
            # df.to_excel("output/fcd_IDM/493.xlsx", index=False)
            # print("Data saved to output/fcd_IDM/493.xlsx")
            # df.to_excel("output/fcd_Dynamic_IDM/493.xlsx", index=False)
            # print("Data saved to output/fcd_Dynamic_IDM/493.xlsx")
            df.to_excel("output/fcd_centroid-guided_IDM/493.xlsx", index=False)
            print("Data saved to output/fcd_centroid-guided_IDM/493.xlsx")
        else:
            print("Warning: No data collected!")

    except Exception as e:
        print(f"Error running SUMO simulation: {e}")
        # Ensure connection is closed on error
        if traci.isConnected():
            traci.close()


if __name__ == "__main__":
    # Configuration parameters
    SUMO_CONFIG_FILE = "simulation.sumo.cfg"  # SUMO configuration file path
    REAL_DATA_FILE = "ngsim_selected_tracks.csv"  # Real data file path
    SUMO_BINARY = "sumo-gui"  # SUMO executable, use "sumo" to run in background

    # Start SUMO simulation
    run_sumo_simulation(SUMO_CONFIG_FILE, REAL_DATA_FILE, SUMO_BINARY)