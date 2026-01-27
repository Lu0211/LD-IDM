import pandas as pd
import matplotlib.pyplot as plt
import os


def compare_five_datasets(original_file, idm_file, dynamic_idm_file,
                          ld_idm_file, dynamic_ld_idm_file,
                          output_dir='comparison_speed_fourmodel'):
    """Compare five datasets (raw data, IDM, Dynamic_IDM, LD-IDM, Dynamic_LD-IDM)"""
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

    # Get all vehicle IDs (use IDs from raw data as baseline)
    try:
        vehicle_ids = df_original['id'].unique()
    except KeyError:
        print("Missing 'id' column in raw data")
        return

    # Set font to ensure proper display of Chinese and English
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
    plt.rcParams['font.family'] = 'Times New Roman'

    # Color and line style configuration - set different styles for five models for distinction
    line_config = {
        'original': {'color': 'blue', 'linestyle': '-', 'label': 'ActualData', 'linewidth': 1},
        'idm': {'color': 'green', 'linestyle': '--', 'label': 'IDM', 'linewidth': 1},
        'dynamic_idm': {'color': 'orange', 'linestyle': '--', 'label': 'Dynamic IDM(2024)', 'linewidth': 1},
        'dynamic_ld_idm': {'color': 'mediumpurple', 'linestyle': '--', 'label': 'Centroid-guided IDM(2025)', 'linewidth': 1},
        'ld_idm': {'color': 'red', 'linestyle': '--', 'label': 'LD-IDM', 'linewidth': 1},
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
            print(f"No raw data for vehicle {vehicle_id}, skipping")
            continue

        # Create single speed comparison chart
        plt.figure(figsize=(5, 5))
        # Plot speed comparison
        plt.subplot(2, 1, 1)

        # Plot speed comparison
        plt.plot(original_data['frame'], original_data['speed'],
                 color=line_config['original']['color'],
                 linestyle=line_config['original']['linestyle'],
                 label=line_config['original']['label'],
                 linewidth=line_config['original']['linewidth'])

        plt.plot(idm_data['frame'], idm_data['speed'],
                 color=line_config['idm']['color'],
                 linestyle=line_config['idm']['linestyle'],
                 label=line_config['idm']['label'],
                 linewidth=line_config['idm']['linewidth'])

        plt.plot(dynamic_idm_data['frame'], dynamic_idm_data['speed'],
                 color=line_config['dynamic_idm']['color'],
                 linestyle=line_config['dynamic_idm']['linestyle'],
                 label=line_config['dynamic_idm']['label'],
                 linewidth=line_config['dynamic_idm']['linewidth'])

        plt.plot(dynamic_ld_idm_data['frame'], dynamic_ld_idm_data['speed'],
                 color=line_config['dynamic_ld_idm']['color'],
                 linestyle=line_config['dynamic_ld_idm']['linestyle'],
                 label=line_config['dynamic_ld_idm']['label'],
                 linewidth=line_config['dynamic_ld_idm']['linewidth'])

        plt.plot(ld_idm_data['frame'], ld_idm_data['speed'],
                 color=line_config['ld_idm']['color'],
                 linestyle=line_config['ld_idm']['linestyle'],
                 label=line_config['ld_idm']['label'],
                 linewidth=line_config['ld_idm']['linewidth'])

        # plt.title(f'vehicle {vehicle_id} speed-frame comparison chart', fontsize=12)
        # plt.xlabel('Frame (frame)', fontsize=10)
        # plt.ylabel('Speed (m/s)', fontsize=10)
        # plt.legend(loc='best', fontsize=9)
        # plt.grid(True, linestyle='--', alpha=0.7)

        plt.title(f'FV: truck{vehicle_id}  speed-frame graph', fontsize=10)
        plt.xlabel('frame', fontsize=10)
        plt.ylabel('speed(m/s)', fontsize=10)
        # plt.legend(loc='upper left', fontsize=6)
        plt.legend(fontsize=6)
        plt.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save chart
        try:
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_speed_comparison.png'), dpi=300)
        except Exception as e:
            print(f"Error saving chart for vehicle {vehicle_id}: {e}")

        plt.close()

    print(f"Speed comparison charts saved to {output_dir} directory")


if __name__ == "__main__":
    # Define all data file paths
    original_file = 'ngsim_selected_tracks.csv'  # Raw data
    #car-car
    # idm_file = 'output/IDM_data.xlsx'  # IDM model data
    # dynamic_idm_file = 'output/Dynamic_idm_data.xlsx'  # Dynamic_IDM model data
    # ld_idm_file = 'output/LD-IDM_data.xlsx'  # LD-IDM model data
    # dynamic_ld_idm_file = 'output/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM model data

    #car-truck
    # idm_file = 'output/car-truck/IDM_data.xlsx'  # IDM model data
    # dynamic_idm_file = 'output/car-truck/Dynamic_idm_data.xlsx'  # Dynamic_IDM model data
    # ld_idm_file = 'output/car-truck/LD-IDM_data.xlsx'  # LD-IDM model data
    # dynamic_ld_idm_file = 'output/car-truck/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM model data

    # truck-car
    idm_file = 'output/truck-car/IDM_data.xlsx'  # IDM model data
    dynamic_idm_file = 'output/truck-car/Dynamic_idm_data.xlsx'  # Dynamic_IDM model data
    ld_idm_file = 'output/truck-car/LD-IDM_data.xlsx'  # LD-IDM model data
    dynamic_ld_idm_file = 'output/truck-car/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM model data

    # Execute comparison
    compare_five_datasets(original_file, idm_file, dynamic_idm_file,
                          ld_idm_file, dynamic_ld_idm_file)