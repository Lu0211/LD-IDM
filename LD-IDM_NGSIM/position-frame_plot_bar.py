import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Set font to ensure proper display
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue
plt.rcParams['font.family'] = 'Times New Roman'


def read_five_datasets(idm_path, dynamic_idm_path, ld_idm_path, centroid_guided_IDM_path, raw_path):
    """Read five datasets: four models and raw data"""
    # Model configuration
    models = {
        'idm': {'path': idm_path, 'name': 'IDM', 'color': 'green', 'dark_color': 'darkgreen'},
        'dynamic_idm': {'path': dynamic_idm_path, 'name': 'Dynamic IDM(2024)', 'color': 'blueviolet',
                        'dark_color': 'darkviolet'},
        'ld_idm': {'path': ld_idm_path, 'name': 'LD-IDM', 'color': 'red', 'dark_color': 'orangered'},
        'centroid-guided_IDM': {'path': centroid_guided_IDM_path, 'name': 'Centroid-guided IDM(2025)', 'color': 'orange',
                            'dark_color': 'darkorange'}
    }

    # Read all model data
    model_data = {}
    try:
        for key, model in models.items():
            if not os.path.exists(model['path']):
                raise FileNotFoundError(f"{model['name']} model data file does not exist: {model['path']}")
            model_data[key] = pd.read_excel(model['path'])
    except Exception as e:
        print(f"Error reading model data: {e}")
        return None, None

    # Read raw data
    try:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file does not exist: {raw_path}")
        raw_data = pd.read_csv(raw_path)
    except Exception as e:
        print(f"Error reading raw data: {e}")
        return None, None

    return model_data, models, raw_data


def calculate_mae(model_data, raw_data):
    """Calculate MAE (Mean Absolute Error) for X and Y coordinates between model data and raw data"""
    if len(model_data) == 0 or len(raw_data) == 0:
        return None, None

    # Ensure both datasets have same length, take the smaller length
    min_len = min(len(model_data), len(raw_data))
    model_data = model_data.iloc[:min_len]
    raw_data = raw_data.iloc[:min_len]

    # Calculate MAE
    x_mae = abs(model_data['PositionX'] - raw_data['x_position']).mean()
    y_mae = abs(model_data['PositionY'] - raw_data['y_position']).mean()

    return x_mae, y_mae


def get_frame_column(data):
    """Get frame data column name"""
    frame_columns = ['Frame', 'frame', 'FrameID', 'frame_id']
    for col in frame_columns:
        if col in data.columns:
            return col
    return None


def plot_vehicle_comparison(vehicle_id, model_data, models, raw_data, output_folder):
    """Plot bar chart comparison for five datasets for a single vehicle, showing X and Y coordinates separately"""
    # Check if frame data column exists
    frame_cols = {}
    for key, data in model_data.items():
        frame_col = get_frame_column(data)
        if not frame_col:
            print(f"Warning: Vehicle ID {vehicle_id} {models[key]['name']} data missing frame data, cannot plot chart")
            return
        frame_cols[key] = frame_col

    raw_frame_col = get_frame_column(raw_data)
    if not raw_frame_col:
        print(f"Warning: Vehicle ID {vehicle_id} raw data missing frame data, cannot plot chart")
        return

    # Ensure data has at least 2 points
    for key, data in model_data.items():
        if len(data) < 2:
            print(f"Warning: Insufficient data points for vehicle ID {vehicle_id} {models[key]['name']}, chart may be incomplete")
            return

    if len(raw_data) < 2:
        print(f"Warning: Insufficient data points for vehicle ID {vehicle_id} raw data, chart may be incomplete")
        return

    # Find all common frames across datasets for comparison
    all_frames = [set(data[frame_cols[key]]) for key, data in model_data.items()]
    all_frames.append(set(raw_data[raw_frame_col]))
    common_frames = set.intersection(*all_frames)

    if not common_frames:
        print(f"Warning: No common frame data found for vehicle ID {vehicle_id}, cannot plot chart")
        return

    # Sort by frame
    common_frames = sorted(common_frames)

    # If too many frames, sample to avoid overcrowding
    if len(common_frames) > 15:  # Reduce sampling count for five models to avoid crowding
        sample_interval = len(common_frames) // 15  # Represents number of bar charts
        common_frames = common_frames[::sample_interval]
        print(f"Too many frame data for vehicle ID {vehicle_id}, sampled to show {len(common_frames)} frames")

    # Extract data for each model at common frames
    data = {'raw': {'y': [], 'x': []}}
    for key in model_data.keys():
        data[key] = {'y': [], 'x': []}

    for f in common_frames:
        # Raw data
        data['raw']['y'].append(raw_data[raw_data[raw_frame_col] == f]['y_position'].values[0])
        data['raw']['x'].append(raw_data[raw_data[raw_frame_col] == f]['x_position'].values[0])

        # Model data
        for key, model in models.items():
            data[key]['y'].append(model_data[key][model_data[key][frame_cols[key]] == f]['PositionY'].values[0])
            data[key]['x'].append(model_data[key][model_data[key][frame_cols[key]] == f]['PositionX'].values[0])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    bar_width = 0.15  # Bar chart width, five models need narrower bars
    index = np.arange(len(common_frames))  # Frame indices

    # Data order: raw data, IDM, Dynamic_IDM, LD-IDM, centroid-guided_IDM
    order = ['idm', 'dynamic_idm', 'centroid-guided_IDM', 'ld_idm', 'raw']

    # Plot Y coordinate bar chart (upper part)
    for i, key in enumerate(order):
        if key == 'raw':
            # Raw data
            ax1.bar(index + i * bar_width - 2 * bar_width, data[key]['y'], bar_width,
                    alpha=0.7, color='blue', label='ActualData')
        else:
            # Model data
            ax1.bar(index + i * bar_width - 2 * bar_width, data[key]['y'], bar_width,
                    alpha=0.7, color=models[key]['color'], label=models[key]['name'])

    ax1.set_ylabel('y (m)', fontsize=14)
    ax1.set_title(f'FV: truck{vehicle_id} position_y comparison', fontsize=14)
    # ax1.legend(loc='best', fontsize=11)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Set Y coordinate range
    all_y_values = [val for key in data for val in data[key]['y']]
    y_min, y_max = min(all_y_values), max(all_y_values)
    ax1.set_ylim(y_min - 0.2, y_max + 0.5)

    # Plot X coordinate bar chart (lower part)
    for i, key in enumerate(order):
        if key == 'raw':
            # Raw data
            ax2.bar(index + i * bar_width - 2 * bar_width, data[key]['x'], bar_width,
                    alpha=0.7, color='royalblue', label='ActualData')
        else:
            # Model data
            ax2.bar(index + i * bar_width - 2 * bar_width, data[key]['x'], bar_width,
                    alpha=0.7, color=models[key]['dark_color'], label=models[key]['name'])
    # ax2.set_ylabel('x (m)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('frame', fontsize=15)
    ax2.set_ylabel('x (m)', fontsize=14)
    # ax2.set_title(f'car{vehicle_id} position_x comparison', fontsize=14)
    ax2.set_title(f'FV: truck{vehicle_id} position_x comparison', fontsize=14)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Set X coordinate range
    all_x_values = [val for key in data for val in data[key]['x']]
    x_min, x_max = min(all_x_values), max(all_x_values)
    ax2.set_ylim(x_min - 0.2, x_max + 0.5)

    # Set X-axis ticks and labels
    ax2.set_xticks(index)
    ax2.set_xticklabels(common_frames, rotation=45, ha='right', fontsize=11)

    # Adjust layout
    plt.tight_layout()

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save image
    output_path = os.path.join(output_folder, f'vehicle_{vehicle_id}_bar_comparison.png')
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar comparison chart for vehicle ID {vehicle_id} saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image for vehicle ID {vehicle_id}: {e}")
    finally:
        plt.close()  # Close figure, release memory


def plot_comparison(model_data, models, raw_data, output_folder):
    """Group by vehicle ID and plot bar comparison charts for five datasets, while calculating MAE and performance improvement"""
    if model_data is None or raw_data is None:
        print("Incomplete data, cannot plot comparison charts")
        return

    # Check if required data columns exist
    required_cols_model = ['PositionX', 'PositionY']
    required_cols_raw = ['x_position', 'y_position']

    for key, data in model_data.items():
        for col in required_cols_model:
            if col not in data.columns:
                print(f"{models[key]['name']} data missing required column: {col}")
                return

    for col in required_cols_raw:
        if col not in raw_data.columns:
            print(f"Raw data missing required column: {col}")
            return

    # Check if frame data exists
    for key, data in model_data.items():
        if not get_frame_column(data):
            print(f"{models[key]['name']} data missing frame-related columns")
            return

    if not get_frame_column(raw_data):
        print("Raw data missing frame-related columns")
        return

    # Try to identify vehicle ID columns
    possible_id_columns = ['VehicleID', 'vehicle_id', 'ID', 'id', 'CarID']
    id_cols = {}
    raw_id_col = None

    for col in possible_id_columns:
        for key, data in model_data.items():
            if col in data.columns and key not in id_cols:
                id_cols[key] = col
        if col in raw_data.columns and raw_id_col is None:
            raw_id_col = col

    # Check if all models found ID column
    for key in model_data.keys():
        if key not in id_cols:
            print(f"Cannot identify vehicle ID column in {models[key]['name']} data")
            return

    if raw_id_col is None:
        print("Cannot identify vehicle ID column in raw data")
        return

    # Get all unique vehicle IDs
    vehicle_ids = [set(data[id_cols[key]].unique()) for key, data in model_data.items()]
    vehicle_ids.append(set(raw_data[raw_id_col].unique()))
    common_vehicle_ids = set.intersection(*vehicle_ids)

    if not common_vehicle_ids:
        print("No common vehicle IDs found across five datasets")
        return

    print(f"Found {len(common_vehicle_ids)} common vehicle IDs across all five datasets")

    # Store MAE results and performance improvements for all vehicles
    mae_results = []

    # Plot comparison charts and calculate MAE for each vehicle
    for vehicle_id in common_vehicle_ids:
        # Get all data for this vehicle, starting from second point (ignore first point)
        vehicle_data = {}
        valid = True

        for key, data in model_data.items():
            veh_data = data[data[id_cols[key]] == vehicle_id].iloc[1:]
            if len(veh_data) < 2:
                print(f"Insufficient data points for vehicle ID {vehicle_id} {models[key]['name']}, skipping")
                valid = False
                break
            vehicle_data[key] = veh_data

        if not valid:
            continue

        raw_vehicle_data = raw_data[raw_data[raw_id_col] == vehicle_id].iloc[1:]
        if len(raw_vehicle_data) < 2:
            print(f"Insufficient raw data points for vehicle ID {vehicle_id}, skipping")
            continue

        # Calculate MAE between each model and raw data
        model_mae = {}
        valid_mae = True

        for key in model_data.keys():
            x_mae, y_mae = calculate_mae(vehicle_data[key], raw_vehicle_data)
            if x_mae is None or y_mae is None:
                print(f"Cannot calculate MAE for vehicle ID {vehicle_id} {models[key]['name']}, skipping")
                valid_mae = False
                break
            model_mae[key] = (x_mae, y_mae)

        if not valid_mae:
            continue

        # Store MAE of each model relative to raw data
        metrics = {'VehicleID': vehicle_id}

        # Add MAE for all models
        for key in model_data.keys():
            metrics[f'{key}_X_MAE'] = model_mae[key][0]
            metrics[f'{key}_Y_MAE'] = model_mae[key][1]
            metrics[f'{key}_Total_MAE'] = (model_mae[key][0] + model_mae[key][1]) / 2

        # Calculate error improvement of other models relative to IDM (IDM error - other model error)
        # Positive values indicate other models outperform IDM, negative values indicate worse performance
        for key in ['dynamic_idm', 'ld_idm', 'centroid-guided_IDM']:
            metrics[f'{key}_vs_IDM_X'] = model_mae['idm'][0] - model_mae[key][0]
            metrics[f'{key}_vs_IDM_Y'] = model_mae['idm'][1] - model_mae[key][1]
            metrics[f'{key}_vs_IDM_Total'] = ((model_mae['idm'][0] + model_mae['idm'][1]) / 2) - \
                                             ((model_mae[key][0] + model_mae[key][1]) / 2)

        mae_results.append(metrics)

        # Plot and save five-dataset comparison chart
        plot_vehicle_comparison(vehicle_id, vehicle_data, models, raw_vehicle_data, output_folder)

    # If MAE results exist, save to Excel
    if mae_results:
        mae_df = pd.DataFrame(mae_results)

        # Calculate overall statistics
        stats = ['Mean', 'Maximum', 'Minimum', 'Standard Deviation']
        overall_stats = pd.DataFrame({'Statistic': stats})

        # Add MAE statistics for each model (relative to raw data)
        for key in model_data.keys():
            overall_stats[f'{key}_X_MAE'] = [
                mae_df[f'{key}_X_MAE'].mean(),
                mae_df[f'{key}_X_MAE'].max(),
                mae_df[f'{key}_X_MAE'].min(),
                mae_df[f'{key}_X_MAE'].std()
            ]
            overall_stats[f'{key}_Y_MAE'] = [
                mae_df[f'{key}_Y_MAE'].mean(),
                mae_df[f'{key}_Y_MAE'].max(),
                mae_df[f'{key}_Y_MAE'].min(),
                mae_df[f'{key}_Y_MAE'].std()
            ]
            overall_stats[f'{key}_Total_MAE'] = [
                mae_df[f'{key}_Total_MAE'].mean(),
                mae_df[f'{key}_Total_MAE'].max(),
                mae_df[f'{key}_Total_MAE'].min(),
                mae_df[f'{key}_Total_MAE'].std()
            ]

        # Add improvement statistics relative to IDM
        for key in ['dynamic_idm', 'ld_idm', 'centroid-guided_IDM']:
            for metric in ['X', 'Y', 'Total']:
                overall_stats[f'{key}_vs_IDM_{metric}'] = [
                    mae_df[f'{key}_vs_IDM_{metric}'].mean(),
                    mae_df[f'{key}_vs_IDM_{metric}'].max(),
                    mae_df[f'{key}_vs_IDM_{metric}'].min(),
                    mae_df[f'{key}_vs_IDM_{metric}'].std()
                ]

        # Create Excel writer object
        excel_path = os.path.join(output_folder, 'car-truck_four_models_mae_statistics.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Add float_format parameter to keep 2 decimal places
            mae_df.to_excel(writer, sheet_name='Vehicle MAE Details', index=False, float_format="%.2f")
            overall_stats.to_excel(writer, sheet_name='Overall Statistics', index=False, float_format="%.2f")

            # Adjust column widths
            for sheet_name in ['Vehicle MAE Details', 'Overall Statistics']:
                sheet = writer.sheets[sheet_name]
                for i, col in enumerate(sheet.columns):
                    max_len = max(len(str(x)) for x in sheet.col_values(i + 1))
                    sheet.column_dimensions[chr(65 + i)].width = min(max_len + 2, 25)

        print(f"MAE statistics saved to: {excel_path}")
        return mae_df
    else:
        print("Insufficient data to calculate MAE")
        return None


def main():
    """Main function"""
    # File paths
    raw_path = "ngsim_selected_tracks.csv"  # Raw data file path
    output_folder = "comparison_position_fourmodel"  # Output image folder
    #car-car
    # idm_path = "output/outputxml_data.xlsx"  # IDM model data path
    # dynamic_idm_path = "output/dynamic_idm_data.xlsx"  # Dynamic_IDM model data path
    # ld_idm_path = "output/info_idm_default_outputxml_data.xlsx"  # LD-IDM model data path
    # centroid_guided_IDM_path = "output/centroid-guided_IDM_data.xlsx"  # centroid-guided_IDM model data path

    # car-truck
    idm_path = "output/car-truck/outputxml_data.xlsx"  # IDM model data path
    dynamic_idm_path = "output/car-truck/dynamic_idm_data.xlsx"  # Dynamic_IDM model data path
    ld_idm_path = "output/car-truck/info_idm_default_outputxml_data.xlsx"  # LD-IDM model data path
    centroid_guided_IDM_path = "output/car-truck/centroid-guided_IDM_data.xlsx"  # centroid-guided_IDM model data path

    # truck-car
    # idm_path = "output/truck-car/outputxml_data.xlsx"  # IDM model data path
    # dynamic_idm_path = "output/truck-car/dynamic_idm_data.xlsx"  # Dynamic_IDM model data path
    # ld_idm_path = "output/truck-car/info_idm_default_outputxml_data.xlsx"  # LD-IDM model data path
    # centroid_guided_IDM_path = "output/truck-car/centroid-guided_IDM_data.xlsx"  # centroid-guided_IDM model data path

    # Read five datasets
    model_data, models, raw_data = read_five_datasets(
        idm_path, dynamic_idm_path, ld_idm_path, centroid_guided_IDM_path, raw_path
    )

    # Group by vehicle ID and plot five-dataset comparison charts, while calculating MAE
    mae_results = plot_comparison(model_data, models, raw_data, output_folder)

    # Print overall MAE statistics and performance improvements
    if mae_results is not None and not mae_results.empty:
        print("\n===== MAE Statistics for Each Model Relative to Raw Data =====")
        for key in model_data.keys():
            print(f"{models[key]['name']} Average X coordinate MAE: {mae_results[f'{key}_X_MAE'].mean():.4f}")
            print(f"{models[key]['name']} Average Y coordinate MAE: {mae_results[f'{key}_Y_MAE'].mean():.4f}")
            print(f"{models[key]['name']} Overall average MAE: {mae_results[f'{key}_Total_MAE'].mean():.4f}\n")

        print("\n===== Error Improvement of Other Models Compared to IDM Model =====")
        print("(Positive values indicate smaller error than IDM, negative values indicate larger error than IDM)")
        for key in ['dynamic_idm', 'ld_idm', 'centroid-guided_IDM']:
            print(f"{models[key]['name']} Error improvement relative to IDM in X coordinate: {mae_results[f'{key}_vs_IDM_X'].mean():.4f}")
            print(f"{models[key]['name']} Error improvement relative to IDM in Y coordinate: {mae_results[f'{key}_vs_IDM_Y'].mean():.4f}")
            print(f"{models[key]['name']} Overall error improvement relative to IDM: {mae_results[f'{key}_vs_IDM_Total'].mean():.4f}\n")


if __name__ == "__main__":
    main()