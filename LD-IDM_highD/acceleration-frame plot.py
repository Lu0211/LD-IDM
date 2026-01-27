# Final code for connecting scattered points
import pandas as pd
import matplotlib.pyplot as plt
import os


def compare_five_datasets_acceleration(original_file, idm_file, dynamic_idm_file,
                                       lda_idm_file, dynamic_lda_idm_file,
                                       output_dir='comparison_acceleration_fourmodel/data01_comparison_plots'):
    """Compare acceleration of five datasets (original data, IDM, Dynamic_IDM, LD-IDM, Dynamic_LD-IDM),
    display as scatter plot, extract one point every 5 rows, only compare within original data frame range"""
    # Read five datasets
    try:
        df_original = pd.read_csv(original_file)
        df_idm = pd.read_excel(idm_file)
        df_dynamic_idm = pd.read_excel(dynamic_idm_file)
        df_lda_idm = pd.read_excel(lda_idm_file)
        df_dynamic_lda_idm = pd.read_excel(dynamic_lda_idm_file)
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all vehicle IDs (use IDs from original data as reference)
    try:
        vehicle_ids = df_original['id'].unique()
    except KeyError:
        print("Missing 'id' column in original data")
        return

    # Set font to ensure normal display of English and numbers
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue
    plt.rcParams['font.family'] = 'Times New Roman'

    # Style configuration - set different scatter styles for five models for distinction
    scatter_config = {
        'original': {'color': 'blue', 'marker': 'o', 'label': 'ActualData', 's': 6},  # s is point size
        'idm': {'color': 'green', 'marker': 'o', 'label': 'IDM', 's': 8},
        'dynamic_idm': {'color': 'orange', 'marker': 'D', 'label': 'Dynamic IDM', 's': 8},
        'lda_idm': {'color': 'mediumpurple', 'marker': 's', 'label': 'LD-IDM', 's': 8},
        'dynamic_lda_idm': {'color': 'red', 'marker': '^', 'label': 'Dynamic LD-IDM', 's': 8},
        }

    # Create comparison plots for each vehicle
    for vehicle_id in vehicle_ids:
        # Filter vehicle data
        try:
            original_data = df_original[df_original['id'] == vehicle_id].sort_values('frame')
            idm_data = df_idm[df_idm['id'] == vehicle_id].sort_values('frame')
            dynamic_idm_data = df_dynamic_idm[df_dynamic_idm['id'] == vehicle_id].sort_values('frame')
            lda_idm_data = df_lda_idm[df_lda_idm['id'] == vehicle_id].sort_values('frame')
            dynamic_lda_idm_data = df_dynamic_lda_idm[df_dynamic_lda_idm['id'] == vehicle_id].sort_values('frame')
        except KeyError as e:
            print(f"Error processing vehicle {vehicle_id}: missing column {e}")
            continue

        # Check if data is empty
        if original_data.empty:
            print(f"No original data for vehicle {vehicle_id}, skip")
            continue

        # Get maximum frame value of original data, used to truncate other model data
        max_original_frame = original_data['frame'].max()

        # Check if acceleration column exists and truncate data to original data frame range
        required_columns = ['acceleration', 'frame']
        valid_data = True

        # Collect first frame values for this vehicle from all models
        first_frames = []
        # Original data first frame
        if not original_data.empty:
            first_frames.append(original_data['frame'].min())
        # IDM model first frame
        if not idm_data.empty:
            first_frames.append(idm_data['frame'].min())
        # Dynamic IDM model first frame
        if not dynamic_idm_data.empty:
            first_frames.append(dynamic_idm_data['frame'].min())
        # LD-IDM model first frame
        if not lda_idm_data.empty:
            first_frames.append(lda_idm_data['frame'].min())
        # Dynamic LD-IDM model first frame
        if not dynamic_lda_idm_data.empty:
            first_frames.append(dynamic_lda_idm_data['frame'].min())

        # Find earliest frame among all models as unified starting point
        if first_frames:
            earliest_frame = max(first_frames)
            # Truncate original data to start from earliest frame
            original_data = original_data[original_data['frame'] >= earliest_frame]

        # Truncate original data and sample one point every 5 rows
        original_data = original_data[(original_data['frame'] >= earliest_frame) & (original_data['frame'] <= max_original_frame)]
        original_sampled = original_data.iloc[::10]  # Take one point every 5 rows

        # Extract and print first frame values for each model
        print(f"\nStarting frame information for vehicle {vehicle_id}:")
        print('earliest_frame:',earliest_frame)
        if not original_data.empty:
            print(f"  Original data first frame: {original_data['frame'].min()}")
        else:
            print(f"  Original data: no data")

        if not idm_data.empty:
            print(f"  IDM model first frame: {idm_data['frame'].min()}")
        else:
            print(f"  IDM model: no data")

        # Truncate IDM data and sample
        if 'frame' in idm_data.columns:
            idm_data = idm_data[(idm_data['frame'] >= earliest_frame) & (idm_data['frame'] <= max_original_frame)]
            idm_sampled = idm_data.iloc[::2]  # Take one point every 5 rows
        else:
            print(f"Missing 'frame' column in IDM data, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate Dynamic IDM data and sample
        if 'frame' in dynamic_idm_data.columns:
            dynamic_idm_data = dynamic_idm_data[(dynamic_idm_data['frame'] >= earliest_frame) & (dynamic_idm_data['frame'] <= max_original_frame)]
            dynamic_idm_sampled = dynamic_idm_data.iloc[::2]  # Take one point every 5 rows
        else:
            print(f"Missing 'frame' column in Dynamic IDM data, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate LD-IDM data and sample
        if 'frame' in lda_idm_data.columns:
            lda_idm_data = lda_idm_data[(lda_idm_data['frame'] >= earliest_frame) & (lda_idm_data['frame'] <= max_original_frame)]
            lda_idm_sampled = lda_idm_data.iloc[::2]  # Take one point every 5 rows
        else:
            print(f"Missing 'frame' column in LD-IDM data, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Truncate Dynamic LD-IDM data and sample
        if 'frame' in dynamic_lda_idm_data.columns:
            dynamic_lda_idm_data = dynamic_lda_idm_data[(dynamic_lda_idm_data['frame'] >= earliest_frame) & (dynamic_lda_idm_data['frame'] <= max_original_frame)]
            dynamic_lda_idm_sampled = dynamic_lda_idm_data.iloc[::2]  # Take one point every 5 rows
        else:
            print(f"Missing 'frame' column in Dynamic LD-IDM data, failed to process vehicle {vehicle_id}")
            valid_data = False

        # Check if all data have acceleration column
        for df, name in [(original_sampled, 'Original data'), (idm_sampled, 'IDM data'),
                         (dynamic_idm_sampled, 'Dynamic_IDM data'), (lda_idm_sampled, 'LD-IDM data'),
                         (dynamic_lda_idm_sampled, 'Dynamic_LD-IDM data')]:
            if 'acceleration' not in df.columns:
                print(f"Missing 'acceleration' column in {name}, failed to process vehicle {vehicle_id}")
                valid_data = False
                break

        if not valid_data:
            print(f"Vehicle {vehicle_id} skipped due to incomplete data")
            continue

        # Output frame numbers corresponding to sampling points for each model
        print(f"\nSampling frame numbers for vehicle {vehicle_id}:")
        print(f"  Original data sampling frames: {original_sampled['frame'].tolist()}")
        print(f"  IDM model sampling frames: {idm_sampled['frame'].tolist()}")
        print(f"  Dynamic IDM model sampling frames: {dynamic_idm_sampled['frame'].tolist()}")
        print(f"  LD-IDM model sampling frames: {lda_idm_sampled['frame'].tolist()}")
        print(f"  Dynamic LD-IDM model sampling frames: {dynamic_lda_idm_sampled['frame'].tolist()}")
        # Create single acceleration comparison scatter plot
        plt.figure(figsize=(5, 5))

        # Plot original data scatter points and line
        plt.scatter(original_sampled['frame'], original_sampled['acceleration'],
                    color=scatter_config['original']['color'],
                    marker=scatter_config['original']['marker'],
                    label=scatter_config['original']['label'],
                    s=scatter_config['original']['s'])
        # Add line connection
        plt.plot(original_sampled['frame'], original_sampled['acceleration'],
                 color=scatter_config['original']['color'], linestyle='-', linewidth=0.8)

        # Plot IDM data scatter points and line
        plt.scatter(idm_sampled['frame'], idm_sampled['acceleration'],
                    color=scatter_config['idm']['color'],
                    marker=scatter_config['idm']['marker'],
                    label=scatter_config['idm']['label'],
                    s=scatter_config['idm']['s'])
        # Add line connection
        plt.plot(idm_sampled['frame'], idm_sampled['acceleration'],
                 color=scatter_config['idm']['color'], linestyle='-', linewidth=0.8)

        # Plot Dynamic IDM data scatter points and line
        plt.scatter(dynamic_idm_sampled['frame'], dynamic_idm_sampled['acceleration'],
                    color=scatter_config['dynamic_idm']['color'],
                    marker=scatter_config['dynamic_idm']['marker'],
                    label=scatter_config['dynamic_idm']['label'],
                    s=scatter_config['dynamic_idm']['s'])
        # Add line connection
        plt.plot(dynamic_idm_sampled['frame'], dynamic_idm_sampled['acceleration'],
                 color=scatter_config['dynamic_idm']['color'], linestyle='-', linewidth=0.8)

        # Plot LD-IDM data scatter points and line
        plt.scatter(lda_idm_sampled['frame'], lda_idm_sampled['acceleration'],
                    color=scatter_config['lda_idm']['color'],
                    marker=scatter_config['lda_idm']['marker'],
                    label=scatter_config['lda_idm']['label'],
                    s=scatter_config['lda_idm']['s'])
        # Add line connection
        plt.plot(lda_idm_sampled['frame'], lda_idm_sampled['acceleration'],
                 color=scatter_config['lda_idm']['color'], linestyle='-', linewidth=0.8)

        # Plot Dynamic LD-IDM data scatter points and line
        plt.scatter(dynamic_lda_idm_sampled['frame'], dynamic_lda_idm_sampled['acceleration'],
                    color=scatter_config['dynamic_lda_idm']['color'],
                    marker=scatter_config['dynamic_lda_idm']['marker'],
                    label=scatter_config['dynamic_lda_idm']['label'],
                    s=scatter_config['dynamic_lda_idm']['s'])
        # Add line connection
        plt.plot(dynamic_lda_idm_sampled['frame'], dynamic_lda_idm_sampled['acceleration'],
                 color=scatter_config['dynamic_lda_idm']['color'], linestyle='-', linewidth=0.8)

        plt.title(f'FV: car{vehicle_id}  acceleration-frame scatter graph', fontsize=10)
        plt.xlabel('frame', fontsize=12)
        plt.ylabel('acceleration (m/s²)', fontsize=12)
        # plt.legend(loc='upper left', fontsize=6)
        plt.legend(fontsize=6)
        plt.grid(True)

        # Set x-axis range to ensure only show up to maximum frame of original data
        plt.xlim(original_sampled['frame'].min(), max_original_frame)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        try:
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_acceleration_scatter.png'), dpi=300)
        except Exception as e:
            print(f"Error saving chart for vehicle {vehicle_id}: {e}")

        plt.close()

    print(f"Acceleration scatter comparison plots saved to {output_dir} directory")

if __name__ == "__main__":
    # Define all data file paths
    original_file = 'selected_tracks.csv'  # Original data
    idm_file = 'output/fcd_IDM/154.xlsx'  # IDM data
    dynamic_idm_file = 'output/fcd_Dynamic_IDM/154.xlsx'  # Dynamic_IDM data
    lda_idm_file = 'output/fcd_LD-IDM/154.xlsx'  # LD-IDM data
    dynamic_lda_idm_file = 'output/fcd_Dynamic_LD-IDM/154.xlsx'  # Dynamic_LD-IDM data

    # Execute comparison
    compare_five_datasets_acceleration(original_file, idm_file, dynamic_idm_file,
                                       lda_idm_file, dynamic_lda_idm_file)