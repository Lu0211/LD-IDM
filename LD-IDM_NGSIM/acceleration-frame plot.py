import pandas as pd
import matplotlib.pyplot as plt
import os


def compare_five_datasets_acceleration(original_file, idm_file, dynamic_idm_file,
                                       ld_idm_file, dynamic_ld_idm_file,
                                       output_dir='comparison_acceleration_fourmodel'):
    """Compare acceleration of five datasets (original data, IDM, Dynamic_IDM, LD-IDM, Dynamic_LD-IDM),
    displayed as scatter plots, extract one point every 5 rows, only compare within original data's frame range"""
    # Read five datasets
    try:
        df_original = pd.read_csv(original_file)
        df_idm = pd.read_excel(idm_file)
        df_dynamic_idm = pd.read_excel(dynamic_idm_file)
        df_ld_idm = pd.read_excel(ld_idm_file)
        df_dynamic_ld_idm = pd.read_excel(dynamic_ld_idm_file)
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all vehicle IDs (use IDs from original data as baseline)
    try:
        vehicle_ids = df_original['id'].unique()
    except KeyError:
        print("Missing 'id' column in original data")
        return

    # Set font to ensure proper display of Chinese and English
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue
    plt.rcParams['font.family'] = 'Times New Roman'

    # Style configuration - set different scatter styles for five models for distinction
    scatter_config = {
        'original': {'color': 'blue', 'marker': 'o', 'label': 'ActualData', 's': 6},  # s is point size
        'idm': {'color': 'green', 'marker': 'o', 'label': 'IDM', 's': 8},
        'dynamic_idm': {'color': 'orange', 'marker': 'D', 'label': 'Dynamic IDM(2024)', 's': 8},
        'ld_idm': {'color': 'red', 'marker': 's', 'label': 'LD-IDM', 's': 8},
        'dynamic_ld_idm': {'color': 'mediumpurple', 'marker': '^', 'label': 'centroid-guided IDM(2025)', 's': 8},
    }

    # Create comparison charts for each vehicle
    for vehicle_id in vehicle_ids:
        # Filter vehicle data
        try:
            original_data = df_original[df_original['id'] == vehicle_id].sort_values('frame')
            idm_data = df_idm[df_idm['id'] == vehicle_id].sort_values('frame')
            dynamic_idm_data = df_dynamic_idm[df_dynamic_idm['id'] == vehicle_id].sort_values('frame')
            ld_idm_data = df_ld_idm[df_ld_idm['id'] == vehicle_id].sort_values('frame')
            dynamic_ld_idm_data = df_dynamic_ld_idm[df_dynamic_ld_idm['id'] == vehicle_id].sort_values('frame')
        except KeyError as e:
            print(f"Error processing vehicle {vehicle_id}: Missing column {e}")
            continue

        # Check if data is empty
        if original_data.empty:
            print(f"No original data for vehicle {vehicle_id}, skipping")
            continue

        # Get maximum frame value from original data, used to truncate other model data
        max_original_frame = original_data['frame'].max()

        # Check if acceleration column exists and truncate data to original data's frame range
        required_columns = ['acceleration', 'frame']
        valid_data = True

        # Collect first frame of each model for this vehicle
        first_frames = []
        # First frame of original data
        if not original_data.empty:
            first_frames.append(original_data['frame'].min())
        # First frame of IDM model
        if not idm_data.empty:
            first_frames.append(idm_data['frame'].min())
        # First frame of Dynamic IDM model
        if not dynamic_idm_data.empty:
            first_frames.append(dynamic_idm_data['frame'].min())
        # First frame of LD-IDM model
        if not ld_idm_data.empty:
            first_frames.append(ld_idm_data['frame'].min())
        # First frame of Dynamic LD-IDM model
        if not dynamic_ld_idm_data.empty:
            first_frames.append(dynamic_ld_idm_data['frame'].min())

        # Find earliest frame among all models as unified starting point
        if first_frames:
            earliest_frame = max(first_frames)
            # Truncate original data to start from earliest frame
            original_data = original_data[original_data['frame'] >= earliest_frame]

        # Truncate original data and sample one point every 5 rows
        original_data = original_data[(original_data['frame'] >= earliest_frame) & (original_data['frame'] <= max_original_frame)]
        original_sampled = original_data.iloc[::12]  # Take one point every 5 rows

        # Extract and print first frame value for each model
        print(f"\nStarting frame information for vehicle {vehicle_id}:")
        print('earliest_frame:',earliest_frame)
        if not original_data.empty:
            print(f"  Original data first frame: {original_data['frame'].min()}")
        else:
            print(f"  Original data: No data")

        if not idm_data.empty:
            print(f"  IDM model first frame: {idm_data['frame'].min()}")
        else:
            print(f"  IDM model: No data")

        # Truncate IDM data and sample
        if 'frame' in idm_data.columns:
            idm_data = idm_data[(idm_data['frame'] >= earliest_frame) & (idm_data['frame'] <= max_original_frame)]
            idm_sampled = idm_data.iloc[::12]  # Take one point every 5 rows
        else:
            print(f"IDM data missing 'frame' column, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate Dynamic IDM data and sample
        if 'frame' in dynamic_idm_data.columns:
            dynamic_idm_data = dynamic_idm_data[(dynamic_idm_data['frame'] >= earliest_frame) & (dynamic_idm_data['frame'] <= max_original_frame)]
            dynamic_idm_sampled = dynamic_idm_data.iloc[::12]  # Take one point every 5 rows
        else:
            print(f"Dynamic IDM data missing 'frame' column, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate LD-IDM data and sample
        if 'frame' in ld_idm_data.columns:
            ld_idm_data = ld_idm_data[(ld_idm_data['frame'] >= earliest_frame) & (ld_idm_data['frame'] <= max_original_frame)]
            ld_idm_sampled = ld_idm_data.iloc[::12]  # Take one point every 5 rows
        else:
            print(f"LD-IDM data missing 'frame' column, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate Dynamic LD-IDM data and sample
        if 'frame' in dynamic_ld_idm_data.columns:
            dynamic_ld_idm_data = dynamic_ld_idm_data[(dynamic_ld_idm_data['frame'] >= earliest_frame) & (dynamic_ld_idm_data['frame'] <= max_original_frame)]
            dynamic_ld_idm_sampled = dynamic_ld_idm_data.iloc[::12]  # Take one point every 5 rows
        else:
            print(f"Dynamic LD-IDM data missing 'frame' column, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Check if all data has acceleration column
        for df, name in [(original_sampled, 'Original data'), (idm_sampled, 'IDM data'),
                         (dynamic_idm_sampled, 'Dynamic_IDM data'), (ld_idm_sampled, 'LD-IDM data'),
                         (dynamic_ld_idm_sampled, 'Dynamic_LD-IDM data')]:
            if 'acceleration' not in df.columns:
                print(f"{name} missing 'acceleration' column, failed to process vehicle {vehicle_id}")
                valid_data = False
                break

        if not valid_data:
            print(f"Vehicle {vehicle_id} skipped due to incomplete data")
            continue

        # Output frame numbers corresponding to sampling points for each model
        print(f"\nSampling point frame counts for vehicle {vehicle_id}:")
        print(f"  Original data sampled frames: {original_sampled['frame'].tolist()}")
        print(f"  IDM model sampled frames: {idm_sampled['frame'].tolist()}")
        print(f"  Dynamic IDM model sampled frames: {dynamic_idm_sampled['frame'].tolist()}")
        print(f"  LD-IDM model sampled frames: {ld_idm_sampled['frame'].tolist()}")
        print(f"  Dynamic LD-IDM model sampled frames: {dynamic_ld_idm_sampled['frame'].tolist()}")
        # Create single acceleration comparison scatter plot
        plt.figure(figsize=(5, 5))

        # Plot original data scatter points and line
        plt.scatter(original_sampled['frame'], original_sampled['acceleration'],
                    color=scatter_config['original']['color'],
                    marker=scatter_config['original']['marker'],
                    label=scatter_config['original']['label'],
                    s=scatter_config['original']['s'])
        # Add connecting lines
        plt.plot(original_sampled['frame'], original_sampled['acceleration'],
                 color=scatter_config['original']['color'], linestyle='-', linewidth=0.8)

        # Plot IDM data scatter points and line
        plt.scatter(idm_sampled['frame'], idm_sampled['acceleration'],
                    color=scatter_config['idm']['color'],
                    marker=scatter_config['idm']['marker'],
                    label=scatter_config['idm']['label'],
                    s=scatter_config['idm']['s'])
        # Add connecting lines
        plt.plot(idm_sampled['frame'], idm_sampled['acceleration'],
                 color=scatter_config['idm']['color'], linestyle='-', linewidth=0.8)

        # Plot Dynamic IDM data scatter points and line
        plt.scatter(dynamic_idm_sampled['frame'], dynamic_idm_sampled['acceleration'],
                    color=scatter_config['dynamic_idm']['color'],
                    marker=scatter_config['dynamic_idm']['marker'],
                    label=scatter_config['dynamic_idm']['label'],
                    s=scatter_config['dynamic_idm']['s'])
        # Add connecting lines
        plt.plot(dynamic_idm_sampled['frame'], dynamic_idm_sampled['acceleration'],
                 color=scatter_config['dynamic_idm']['color'], linestyle='-', linewidth=0.8)

        # Plot LD-IDM data scatter points and line
        plt.scatter(ld_idm_sampled['frame'], ld_idm_sampled['acceleration'],
                    color=scatter_config['ld_idm']['color'],
                    marker=scatter_config['ld_idm']['marker'],
                    label=scatter_config['ld_idm']['label'],
                    s=scatter_config['ld_idm']['s'])
        # Add connecting lines
        plt.plot(ld_idm_sampled['frame'], ld_idm_sampled['acceleration'],
                 color=scatter_config['ld_idm']['color'], linestyle='-', linewidth=0.8)

        # Plot Dynamic LD-IDM data scatter points and line
        plt.scatter(dynamic_ld_idm_sampled['frame'], dynamic_ld_idm_sampled['acceleration'],
                    color=scatter_config['dynamic_ld_idm']['color'],
                    marker=scatter_config['dynamic_ld_idm']['marker'],
                    label=scatter_config['dynamic_ld_idm']['label'],
                    s=scatter_config['dynamic_ld_idm']['s'])
        # Add connecting lines
        plt.plot(dynamic_ld_idm_sampled['frame'], dynamic_ld_idm_sampled['acceleration'],
                 color=scatter_config['dynamic_ld_idm']['color'], linestyle='-', linewidth=0.8)

        plt.title(f'FV: car{vehicle_id}  acceleration-frame scatter graph', fontsize=10)
        plt.xlabel('frame', fontsize=12)
        plt.ylabel('acceleration (m/s²)', fontsize=12)
        # plt.legend(loc='upper left', fontsize=6)
        plt.legend(fontsize=6)
        plt.grid(True)

        # Set x-axis range to ensure only showing up to maximum frame of original data
        plt.xlim(original_sampled['frame'].min(), max_original_frame)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        try:
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_acceleration_scatter.png'), dpi=300)
        except Exception as e:
            print(f"Error saving chart for vehicle {vehicle_id}: {e}")

        plt.close()

    print(f"Acceleration scatter comparison charts saved to {output_dir} directory")


if __name__ == "__main__":
    # Define all data file paths
    original_file = 'ngsim_selected_tracks.csv'  # Original data
    # car-car
    # idm_file = 'output/fcd_IDM/321.xlsx'  # IDM model data
    # dynamic_idm_file = 'output/fcd_Dynamic_IDM/321.xlsx'  # Dynamic_IDM model data
    # ld_idm_file = 'output/fcd_LD-IDM/321.xlsx'  # LD-IDM model data
    # dynamic_ld_idm_file = 'output/fcd_centroid-guided_IDM/321.xlsx'  # centroid-guided-IDM model data

    # truck-car
    idm_file = 'output/fcd_IDM/493.xlsx'  # IDM data
    dynamic_idm_file = 'output/fcd_Dynamic_IDM/493.xlsx'  # Dynamic_IDM data
    ld_idm_file = 'output/fcd_LD-IDM/493.xlsx'  # LD-IDM data
    dynamic_ld_idm_file = 'output/fcd_centroid-guided_IDM/493.xlsx'  # centroid-guided-IDM data

    # car-truck
    # idm_file = 'output/fcd_IDM/478.xlsx'  # IDM model data
    # dynamic_idm_file = 'output/fcd_Dynamic_IDM/478.xlsx'  # Dynamic_IDM model data
    # ld_idm_file = 'output/fcd_LD-IDM/478.xlsx'  # LD-IDM model data
    # dynamic_ld_idm_file = 'output/fcd_centroid-guided_IDM/478.xlsx'  # centroid-guided-IDM model data

    # Execute comparison
    compare_five_datasets_acceleration(original_file, idm_file, dynamic_idm_file,
                                       ld_idm_file, dynamic_ld_idm_file)