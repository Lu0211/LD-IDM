import pandas as pd
import numpy as np
import os

# Configuration parameters
TRACK_FILE = "data/01_tracks.csv"  # Replace with actual trajectory file path
STATIC_FILE = "data/01_tracksMeta.csv"  # Replace with actual static info file path
OUTPUT_FILE = "selected_tracks.csv"
VEHICLE_IDS = [144, 154]  # Vehicle IDs to extract


def read_highd_data(track_file, static_file):
    """Read HighD dataset trajectory and static information files"""
    try:
        # Read trajectory data
        track_data = pd.read_csv(track_file)
        # Read static data
        static_data = pd.read_csv(static_file)
        return track_data, static_data
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None, None
    except Exception as e:
        print(f"Error: Error occurred while reading files - {e}")
        return None, None


def extract_vehicle_data(track_data, static_data, vehicle_ids):
    """Extract information for specified vehicles from data"""
    if track_data is None or static_data is None:
        return None

    # Extract trajectory data for specified vehicles
    selected_tracks = track_data[track_data['id'].isin(vehicle_ids)].copy()

    # Extract static information for vehicles
    selected_static = static_data[static_data['id'].isin(vehicle_ids)].copy()

    # Create a dictionary mapping vehicle ID to its class
    vehicle_class_map = dict(zip(selected_static['id'], selected_static['class']))

    # Add vehicle class column
    selected_tracks['class'] = selected_tracks['id'].map(vehicle_class_map)

    return selected_tracks


def adjust_for_sumo_reference_point(df, static_data, reference_point='front'):
    """
    Adjust coordinates according to SUMO reference point
    reference_point: 'center' or 'front'
    """
    result_df = df.copy()

    # Create a dictionary mapping vehicle ID to its length
    vehicle_length_map = dict(zip(static_data['id'], static_data['width']))

    if reference_point == 'front':
        # Add vehicle length to x-coordinate for each vehicle
        result_df['x_sumo'] = df.apply(lambda row: row['x'] + vehicle_length_map[row['id']], axis=1)
        # Keep y-coordinate unchanged and add offset
        result_df['y_sumo'] = df['y'] + 0.9
    else:
        # Geometric center is already the default reference point
        result_df['x_sumo'] = df['x']
        result_df['y_sumo'] = df['y']

    # Calculate resultant speed (if dataset has xVelocity and yVelocity columns)
    result_df['speed'] = (df['xVelocity'])
    result_df['yVelocity'] = (df['yVelocity'])
    # if 'xVelocity' in df.columns and 'yVelocity' in df.columns:
    #     result_df['speed'] = np.sqrt(df['xVelocity'] ** 2 + df['yVelocity'] ** 2)

    return result_df


def format_float_columns(df):
    """Format floating-point columns in DataFrame to two decimal places"""
    # Create an empty dictionary to store columns that need formatting and their formatting functions
    float_formatters = {}

    # Iterate through all columns of DataFrame
    for col in df.columns:
        # Check if column data type is float
        if pd.api.types.is_float_dtype(df[col]):
            # Add formatting function for this column, keeping two decimal places
            float_formatters[col] = lambda x: f"{x:.2f}"

    # Apply formatting functions
    if float_formatters:
        return df.round(2)  # Using pandas round function is more efficient
    return df


def process_highd_data(track_file, static_file, output_file, vehicle_ids):
    """Process HighD data and output converted CSV file"""
    # Read data
    track_data, static_data = read_highd_data(track_file, static_file)
    if track_data is None or static_data is None:
        return False

    # Extract data for specified vehicles
    selected_tracks = extract_vehicle_data(track_data, static_data, vehicle_ids)
    if selected_tracks is None or selected_tracks.empty:
        print("Error: No data found for specified vehicles")
        return False

    # Ensure dataset contains x and y columns
    if 'x' not in selected_tracks.columns or 'y' not in selected_tracks.columns:
        raise KeyError("Coordinate columns not found in dataset, please check if column names are 'x' and 'y'")

    # Adjust reference point (from rear axle center to front bumper center)
    # Note: The width column in HighD is actually vehicle length
    # We will rename it to length when outputting
    adjusted_tracks = adjust_for_sumo_reference_point(selected_tracks, static_data, reference_point='front')

    # Format floating-point columns to two decimal places
    formatted_tracks = format_float_columns(adjusted_tracks)

    # Select required columns and rename them
    output_columns = {
        'id': 'id',
        'frame': 'frame',
        'x_sumo': 'x_position',
        'y_sumo': 'y_position',
        'speed': 'speed',  # Add speed column
        'yVelocity': 'yVelocity',
        'xAcceleration': 'acceleration',
        'yAcceleration': 'yAcceleration',
        'width': 'length',  # Rename width to length
        'precedingId': 'preceding_vehicle_id',
        'followingId': 'following_vehicle_id',
        'laneId': 'lane_id',
        'class': 'vehicle_class',
        'dhw': 'distance_to_preceding',
        'thw': 'time_to_preceding',
        'ttc': 'time_to_collision'
    }

    # Ensure all required columns exist
    available_columns = set(formatted_tracks.columns)
    output_columns = {k: v for k, v in output_columns.items() if k in available_columns}

    # Select and rename columns
    output_data = formatted_tracks[list(output_columns.keys())].rename(columns=output_columns)

    # Save results
    try:
        output_data.to_csv(output_file, index=False)
        print(f"Data has been successfully processed and saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error: Error occurred while saving file - {e}")
        return False


if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process data
    process_highd_data(TRACK_FILE, STATIC_FILE, OUTPUT_FILE, VEHICLE_IDS)