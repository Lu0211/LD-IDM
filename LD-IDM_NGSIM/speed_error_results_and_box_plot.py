import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Set font to ensure proper display
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue
plt.rcParams['font.family'] = 'Times New Roman'


def calculate_error_and_plot_per_vehicle(idm_file, dynamic_idm_file, ld_idm_file, centroid_guided_idm_file,
                                         ground_truth_file,
                                         output_dir="speed_error_results_and_box_plot/error_analysis_fourmodel"):
    """
    Calculate speed errors between five models and raw data per vehicle ID and plot images

    Parameters:
    idm_file: Excel file path for IDM model output
    dynamic_idm_file: Excel file path for Dynamic_IDM model output
    ld_idm_file: Excel file path for LD-IDM model output
    dynamic_ld_idm_file: Excel file path for Dynamic_LD-IDM model output
    ground_truth_file: CSV file path for raw ground truth data
    output_dir: Directory for output images
    """
    # Create output directory (if not exists)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    try:
        idm_data = pd.read_excel(idm_file)
        dynamic_idm_data = pd.read_excel(dynamic_idm_file)
        ld_idm_data = pd.read_excel(ld_idm_file)
        centroid_guided_idm_data = pd.read_excel(centroid_guided_idm_file)
        ground_truth = pd.read_csv(ground_truth_file)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Model configuration - for unified processing of multiple models
    models = {
        'idm': {'name': 'IDM', 'data': idm_data, 'color': 'green', 'linestyle': '-'},
        'dynamic_idm': {'name': 'Dynamic IDM(2024)', 'data': dynamic_idm_data, 'color': 'blue', 'linestyle': '--'},
        'ld_idm': {'name': 'LD-IDM', 'data': ld_idm_data, 'color': 'yellow', 'linestyle': '-'},
        'centroid_guided_idm': {'name': 'Centroid-guided IDM(2025)', 'data': centroid_guided_idm_data, 'color': 'red',
                                'linestyle': '--'}
    }

    # Check if data columns exist
    required_columns = ['frame', 'id', 'speed']
    # Check all model data and raw data
    all_dfs = [ground_truth] + [model['data'] for model in models.values()]
    all_names = ['ActualData'] + [model['name'] for model in models.values()]

    for df, name in zip(all_dfs, all_names):
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"{name} data missing required columns: {', '.join(missing_cols)}")
            return

    # Ensure data is sorted by frame
    ground_truth.sort_values('frame', inplace=True)
    for model in models.values():
        model['data'].sort_values('frame', inplace=True)

    # Rename column names to avoid conflicts
    ground_truth = ground_truth.rename(columns={'speed': 'speed_ground_truth', 'id': 'vehicle_id', 'frame': 'frame'})

    # Merge each model with raw data
    merged_data = {}
    for model_key, model in models.items():
        renamed_data = model['data'].rename(columns={
            'speed': f'speed_{model_key}',
            'id': 'vehicle_id',
            'frame': 'frame'
        })
        merged = pd.merge(ground_truth, renamed_data, on=['frame', 'vehicle_id'], how='left')
        merged[f'{model_key}_error'] = merged['speed_ground_truth'] - merged[f'speed_{model_key}']
        merged_data[model_key] = merged

    # Get all vehicle IDs
    vehicle_ids = ground_truth['vehicle_id'].unique()

    # Store error metrics and detailed error data for all vehicles
    all_metrics = []
    error_data = []  # For storing all error data to plot box plots

    # Plot speed comparison and error analysis charts for each vehicle
    for vehicle_id in vehicle_ids:
        # Extract all model data for this vehicle
        vehicle_data = {}
        valid = True

        for model_key, model in models.items():
            veh_data = merged_data[model_key][merged_data[model_key]['vehicle_id'] == vehicle_id].dropna()
            if veh_data.empty:
                print(f"No matching records for vehicle {vehicle_id} in {model['name']} dataset, skipping")
                valid = False
                break
            vehicle_data[model_key] = veh_data

        # Skip this vehicle if any model data is missing
        if not valid:
            continue

        # Get raw data
        original_data = ground_truth[ground_truth['vehicle_id'] == vehicle_id]

        # Save error data for box plots
        for model_key, model in models.items():
            for error in vehicle_data[model_key][f'{model_key}_error']:
                error_data.append({
                    'vehicle_id': vehicle_id,
                    'model': model['name'],
                    'error': error
                })

        # Calculate error metrics for each model
        model_metrics = {}
        for model_key, model in models.items():
            metrics = calculate_error_metrics(
                vehicle_data[model_key]['speed_ground_truth'],
                vehicle_data[model_key][f'speed_{model_key}']
            )
            model_metrics[model_key] = metrics

        # Record metrics
        metrics = {'vehicle_id': vehicle_id}
        # Add error metrics for each model
        for model_key, model in models.items():
            for metric_name, value in model_metrics[model_key].items():
                metrics[f'{model_key}_{metric_name}'] = value

        # Calculate improvements relative to IDM
        for metric in ['MAE', 'RMSE', 'MAPE']:
            metrics[f'ld_idm_improvement_{metric}'] = (
                (model_metrics['idm'][metric] - model_metrics['ld_idm'][metric])
                / model_metrics['idm'][metric] * 100
                if model_metrics['idm'][metric] != 0 else 0
            )
            metrics[f'centroid_guided_idm_improvement_{metric}'] = (
                (model_metrics['idm'][metric] - model_metrics['centroid_guided_idm'][metric])
                / model_metrics['idm'][metric] * 100
                if model_metrics['idm'][metric] != 0 else 0
            )
            metrics[f'dynamic_idm_improvement_{metric}'] = (
                (model_metrics['idm'][metric] - model_metrics['dynamic_idm'][metric])
                / model_metrics['idm'][metric] * 100
                if model_metrics['idm'][metric] != 0 else 0
            )

        all_metrics.append(metrics)

        # Plot images
        plot_vehicle_error_analysis(vehicle_id, original_data, vehicle_data, model_metrics, models, output_dir)

    # Save error metrics for all vehicles to CSV
    if all_metrics:
        # Round all numeric columns to 2 decimal places
        metrics_df = pd.DataFrame(all_metrics).round(2)
        metrics_df.to_csv(os.path.join(output_dir, 'car-truck_493_all_vehicles_error_metrics.csv'), index=False)

        # Convert error data to DataFrame
        error_df = pd.DataFrame(error_data)

        # Plot error distribution box plots for all vehicles
        plot_all_vehicles_boxplot_comparison(metrics_df, error_df, models, output_dir)

    print(f"Error analysis completed, results saved to {output_dir}")
    return all_metrics


def calculate_error_metrics(ground_truth, predicted):
    """Calculate error metrics"""
    error = ground_truth - predicted
    metrics = {
        'MAE': np.mean(np.abs(error)),  # Mean Absolute Error
        'RMSE': np.sqrt(np.mean(error ** 2)),  # Root Mean Square Error
        'MAPE': np.mean(np.abs(error / ground_truth)) * 100,  # Mean Absolute Percentage Error
    }
    return metrics


def plot_vehicle_error_analysis(vehicle_id, original_data, vehicle_data, model_metrics, models, output_dir):
    """Plot error analysis chart for a single vehicle"""
    plt.figure(figsize=(15, 15))

    # 1. Speed comparison chart
    plt.subplot(3, 2, 1)
    # Plot raw data
    plt.plot(original_data['frame'], original_data['speed_ground_truth'],
             'k-', label='ActualData', linewidth=2)

    # Plot each model data
    for model_key, model in models.items():
        plt.plot(vehicle_data[model_key]['frame'], vehicle_data[model_key][f'speed_{model_key}'],
                 color=model['color'], linestyle=model['linestyle'],
                 label=model['name'], alpha=0.8, linewidth=1)

    plt.title(f'vehicle {vehicle_id} speed comparison')
    plt.xlabel('frame')
    plt.ylabel('speed (m/s)')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Error comparison chart
    plt.subplot(3, 2, 2)
    for model_key, model in models.items():
        plt.plot(vehicle_data[model_key]['frame'], vehicle_data[model_key][f'{model_key}_error'],
                 color=model['color'], linestyle=model['linestyle'],
                 label=f'{model["name"]} error', alpha=0.8)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'vehicle {vehicle_id} speed error comparison')
    plt.xlabel('frame')
    plt.ylabel('speed_error (m/s)')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3-6. Error distribution histograms for each model
    for i, (model_key, model) in enumerate(models.items(), 3):
        plt.subplot(3, 2, i)
        plt.hist(vehicle_data[model_key][f'{model_key}_error'], bins=20,
                 color=model['color'], alpha=0.6, label=f'{model["name"]} error')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        plt.title(f'{model["name"]} model error distribution')
        plt.xlabel('speed_error (m/s)')
        plt.ylabel('frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add error metrics text
        metrics_text = (f"MAE: {model_metrics[model_key]['MAE']:.2f} m/s\n"
                        f"RMSE: {model_metrics[model_key]['RMSE']:.2f} m/s\n"
                        f"MAPE: {model_metrics[model_key]['MAPE']:.2f}%")
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8), fontsize=9,
                 verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_all_vehicles_boxplot_comparison(metrics_df, error_df, models, output_dir):
    """Plot error distribution box plot comparison for all vehicles"""
    # 2. MAE comparison box plot
    # plt.subplot(2, 2, 2)
    # Create single speed comparison chart
    plt.figure(figsize=(3.5, 5.5))
    plt.subplot(2, 1, 1)

    ordered_model_keys = ['idm', 'ld_idm', 'dynamic_idm', 'centroid_guided_idm']
    # Prepare MAE data in new order
    # mae_data = [metrics_df[f'{model_key}_MAE'] for model_key in ordered_model_keys]
    mae_data = [metrics_df[f'{model_key}_RMSE'] for model_key in ordered_model_keys]
    # Plot box plot, set uniform width
    bp = plt.boxplot(mae_data, widths=0.3, patch_artist=True)  # Ensure widths parameter is fixed value

    # Set color for each box in new order
    ordered_models = [models[key] for key in ordered_model_keys]
    # for patch, model in zip(bp['boxes'], ordered_models):
    #     patch.set_facecolor(model['color'])
    #     patch.set_alpha(0.6)  # Box color intensity

    # Set color for each box in new order
    # Colors adjusted to: light green, light yellow, light blue, light purple
    colors = ['green', 'gold', 'blue', 'indigo']
    # colors = ['green', 'yellow', 'blue', 'red']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Box color intensity
        # patch.set_edgecolor('none')  # Remove box border
        # patch.set_edgecolor(color)  # Border color same as box
        patch.set_linewidth(0.4)  # Make border lines thinner

    # Set median line to red
    for median in bp['medians']:
        median.set_color('red')
        median.set_linewidth(0.4)
    # Add median value annotations on the right side of boxes
    for i, median in enumerate(bp['medians']):
        # Get y-coordinate of median
        median_y = median.get_ydata()[0]
        # Add text annotation on the right side of box, x-coordinate is box position + 0.15
        plt.text(i + 1 + 0.2, median_y, f'{median_y:.2f}',
                 horizontalalignment='left',
                 verticalalignment='center',
                 color='red', fontsize=5)

    # Adjust x-axis range, shorten left and right margins
    # Get number of boxes
    n_boxes = len(ordered_model_keys)
    # Set x-axis range, left and right margins set to 0.5 (adjustable)
    plt.xlim(0.5, n_boxes + 0.5)

    plt.title('RMSE distributions of models', fontsize=6)
    # Set axis ticks and labels using new model order
    plt.xticks(range(1, len(ordered_model_keys) + 1), [model['name'] for model in ordered_models], fontsize=5)
    plt.yticks(fontsize=5)  # Set y-axis tick font size
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'car-truck_493_all_vehicles_error_comparison_boxplot.png'), dpi=300,
                bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Configure file paths
    ground_truth_file = 'ngsim_selected_tracks.csv'  # Raw ground truth data
    # car-car
    # idm_file = 'output/IDM_data.xlsx'  # IDM data
    # dynamic_idm_file = 'output/Dynamic_idm_data.xlsx'  # Dynamic_IDM data
    # ld_idm_file = 'output/LD-IDM_data.xlsx'  # LD-IDM data
    # centroid_guided_idm_file = 'output/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM data

    # car-truck
    # idm_file = 'output/car-truck/IDM_data.xlsx'  # IDM data
    # dynamic_idm_file = 'output/car-truck/Dynamic_idm_data.xlsx'  # Dynamic_IDM data
    # ld_idm_file = 'output/car-truck/LD-IDM_data.xlsx'  # LD-IDM data
    # centroid_guided_idm_file = 'output/car-truck/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM data

    # truck-car
    idm_file = 'output/truck-car/IDM_data.xlsx'  # IDM data
    dynamic_idm_file = 'output/truck-car/Dynamic_idm_data.xlsx'  # Dynamic_IDM data
    ld_idm_file = 'output/truck-car/LD-IDM_data.xlsx'  # LD-IDM data
    centroid_guided_idm_file = 'output/truck-car/Centroid-guided_IDM_data.xlsx'  # centroid-guided-IDM data

    # Execute error analysis
    all_metrics = calculate_error_and_plot_per_vehicle(idm_file, dynamic_idm_file, ld_idm_file,
                                                       centroid_guided_idm_file, ground_truth_file)