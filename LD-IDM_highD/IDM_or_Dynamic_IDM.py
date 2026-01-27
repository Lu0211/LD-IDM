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
    Connect and control SUMO simulation using TraCI

    Parameters:
        config_file: Path to SUMO configuration file
        real_data_file: Path to CSV file containing real vehicle trajectory data
        sumo_binary: SUMO executable path, defaults to "sumo-gui" to launch GUI
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
            raise ValueError(f"Missing required column in real data: {col}")

    # Build SUMO command
    sumo_cmd = [sumo_binary, "-c", config_file]

    print(f"Preparing to launch SUMO simulation: {' '.join(sumo_cmd)}")

    # Create output directory if it doesn't exist
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_time = 0.0
    leader_id = None  # Leader vehicle ID
    real_data_index = 0  # Current real data index
    real_leader_id = 1181  # Leader ID in real dataset

    # Filter real data for leader vehicle
    leader_real_data = real_data[real_data['id'] == real_leader_id]
    if leader_real_data.empty:
        raise ValueError(f"No records found for vehicle ID {real_leader_id} in real data")

    print(f"Prepared {len(leader_real_data)} records for real leader vehicle {real_leader_id}")

    try:
        # Start SUMO and connect TraCI
        traci.start(sumo_cmd)
        # Get total simulation timesteps
        sim_end_time = 5000  # Can be modified as needed
        # Simulation main loop
        step = 0
        # Friday:data23
        # frame = 3913  #217
        # frame = 8419  #462
        # frame = 10734  # 574
        # frame = 455  #45
        # frame = 827  # 62
        # frame = 1845  #99

        # morning:data01
        # frame = 7717  # 377
        # frame = 19887  # 935
        # frame = 6014  # 293
        # frame = 19149  # 908
        # frame = 5936  # 286
        # frame = 4163  # 191
        # frame = 0  # 3,6
        # frame = 6549 #329
        # frame = 6568 #331
        # frame = 6597  #332
        # frame = 637 #35
        # frame = 13677  # 658
        # frame = 13750  # 660
        # frame = 1835  # 98
        # frame = 2521  # 121
        # frame = 3228  #144
        # frame = 4210  #196
        # frame = 5136  # 249
        # frame = 2651  #124
        # frame = 5353  # 259
        # frame = 8271  #411
        # frame = 9099  # 456
        # frame = 9606  #476

        # noon:data06
        # frame = 400  #26
        # frame = 2341  #114
        # frame = 11477  #547
        frame = 25655  # 1181
        # frame = 28339  # 1303
        # frame = 8512 # 423
        # frame = 490  # 28

        # night:data13
        # frame = 13  # 46
        # frame = 59  # 52
        # frame = 71  # 54
        # frame = 1621  # 235
        # frame = 1682  # 242
        # frame = 1714  # 245
        # frame = 1812  # 260
        # frame = 1838  # 264
        # frame = 1870  # 269
        # frame = 1960  # 280
        while step < sim_end_time:
            # Execute one simulation step
            traci.simulationStep()
            simulation_time += 0.01
            # Get all vehicle IDs
            vehicle_ids = traci.vehicle.getIDList()

            # Select vehicle with maximum x-coordinate as leader
            if vehicle_ids:
                # Get positions of all vehicles and find vehicle with maximum x
                max_x = -float('inf')
                new_leader_id = None

                for vehicle_id in vehicle_ids:
                    pos_x, pos_y = traci.vehicle.getPosition(vehicle_id)
                    if pos_x > max_x:
                        max_x = pos_x
                        new_leader_id = vehicle_id
                # Update leader ID if new leader detected
                if new_leader_id != leader_id:
                    leader_id = new_leader_id
                    print(f"Frame {frame}: Updated leader vehicle ID to {leader_id}, position x={max_x:.2f}")

            # Control leader vehicle to follow real data
            if leader_id and leader_id in traci.vehicle.getIDList() and real_data_index < len(leader_real_data):
                # Get real data corresponding to current frame
                current_real_data = leader_real_data.iloc[real_data_index]
                # Ensure real data ID matches current leader ID (for verification only)
                real_id = current_real_data['id']
                # Set leader vehicle speed
                real_speed = current_real_data['speed']  # Unit: m/s
                traci.vehicle.setSpeed(leader_id, real_speed)
                # Record data and update index only every 4 frames
                if step % 4 == 0:
                    # frame += 1
                    print(f"Frame {frame}: Set leader {leader_id} speed to {real_speed:.2f} m/s (Real data ID: {real_id})")
                    # Increment real data index
                    real_data_index += 1
                    if real_data_index >= len(leader_real_data):
                        print("All real data for leader vehicle has been used")

            # Record all vehicle data every 4 frames
            if step % 4 == 0:
                frame += 1
            if step % 20 == 0:
                # Get all vehicle IDs
                vehicle_ids = traci.vehicle.getIDList()
                # Iterate through each vehicle to get leader and follower information
                for vehicle_id in vehicle_ids:
                    traci.vehicle.setLaneChangeMode(vehicle_id, 0)
                    # Get current lane
                    lane_id = traci.vehicle.getLaneID(vehicle_id)
                    # Get leader information (vehicle ID, distance)
                    leader = traci.vehicle.getLeader(vehicle_id, dist=500)  # Search 1000m ahead
                    leader_id = leader[0] if leader else None
                    leader_dist = leader[1] if leader else None

                    # Get follower information (calculated through other vehicles on lane)
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

                    # Get acceleration value and check validity
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

                    # Print data structure every 100 steps for debugging
                    if step % 100 == 0 and vehicle_id == vehicle_ids[0]:
                        print(f"Debug: Data structure example - {output_data[-1]}")
            # Increment simulation step
            step += 1
        # Close TraCI connection
        traci.close()
        print("SUMO simulation completed successfully")

        # Check data before saving
        if output_data:
            print(f"Number of collected data entries: {len(output_data)}")
            # Check data structure
            print(f"Data example: {output_data[0]}")

            # Create DataFrame and save
            df = pd.DataFrame(output_data)
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame shape: {df.shape}")

            # Ensure acceleration column exists
            if 'acceleration' in df.columns:
                print(f"Acceleration column statistics: mean={df['acceleration'].mean():.2f}, max={df['acceleration'].max():.2f}, min={df['acceleration'].min():.2f}")
            else:
                print("Warning: Acceleration column not found in DataFrame!")

            # Save data
            # df.to_excel("output/fcd_IDM/1184.xlsx", index=False)
            # print("Data saved to output/fcd_IDM/1184.xlsx")
            df.to_excel("output/fcd_Dynamic_IDM/1184.xlsx", index=False)
            print("Data saved to output/fcd_Dynamic_IDM/1184.xlsx")
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
    REAL_DATA_FILE = "selected_tracks.csv"  # Real data file path
    SUMO_BINARY = "sumo-gui"  # SUMO executable, use "sumo" to run in background

    # Launch SUMO simulation
    run_sumo_simulation(SUMO_CONFIG_FILE, REAL_DATA_FILE, SUMO_BINARY)