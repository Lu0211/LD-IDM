import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def compare_position_speed(original_file, idm_file, dynamic_idm_file,
                           ld_idm_file, dynamic_ld_idm_file,
                           output_dir='comparison_position_speed/data01_position_speed_heatmaps'):
    """Compare position-speed relationships across five datasets, with longitudinal position as x-axis and speed as y-axis, using heatmaps to show density distribution"""
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

    # Get all vehicle IDs (using IDs from original data as reference)
    try:
        vehicle_ids = df_original['id'].unique()
    except KeyError:
        print("Missing 'id' column in original data")
        return

    # Set font to ensure proper display
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams['axes.unicode_minus'] = False  # Solve negative sign display issue

    # Create unified color mapping (gradient from light yellow to dark blue)
    colors = [(1, 1, 0.8),  # Light yellow (low values)
              (0.6, 0.9, 0.9),  # Light blue-green
              (0, 0.5, 0.8),  # Blue
              (0, 0.2, 0.5)]  # Dark blue (high values)
    cmap = LinearSegmentedColormap.from_list("speed_cmap", colors)

    # Dataset configuration
    datasets = [
        {
            'data': df_original,
            'x_col': 'x_position',
            'label': 'ActualData',
            'cmap': cmap
        },
        {
            'data': df_idm,
            'x_col': 'PositionX',
            'label': 'IDM',
            'cmap': cmap
        },
        {
            'data': df_dynamic_idm,
            'x_col': 'PositionX',
            'label': 'Dynamic IDM',
            'cmap': cmap
        },
        {
            'data': df_ld_idm,
            'x_col': 'PositionX',
            'label': 'LD-IDM',
            'cmap': cmap
        },
        {
            'data': df_dynamic_ld_idm,
            'x_col': 'PositionX',
            'label': 'Dynamic LD-IDM',
            'cmap': cmap
        }
    ]

    # Create position-speed heatmap for each vehicle
    for vehicle_id in vehicle_ids:
        # Create 1x5 subplot layout with square aspect ratio
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # Width sufficient for 5 square subplots
        fig.suptitle(f'FV: car{vehicle_id} position-speed density', fontsize=12, y=1.15)

        # Collect ranges from all data for unified axis ranges
        all_x = []
        all_y = []
        all_heatmaps = []

        # First pass: collect all data to determine unified ranges
        for dataset in datasets:
            try:
                vehicle_data = dataset['data'][dataset['data']['id'] == vehicle_id]
                if vehicle_data.empty:
                    continue

                x = vehicle_data[dataset['x_col']].values
                y = vehicle_data['speed'].values

                all_x.extend(x)
                all_y.extend(y)

                heatmap, _, _ = np.histogram2d(x, y, bins=20)
                all_heatmaps.append(heatmap)

            except Exception as e:
                continue

        # Determine unified axis ranges and color mapping ranges
        if all_x and all_y:
            x_min, x_max = np.percentile(all_x, 1), np.percentile(all_x, 99)  # Exclude extreme values
            y_min, y_max = np.percentile(all_y, 1), np.percentile(all_y, 99)
        else:
            x_min, x_max, y_min, y_max = 0, 100, 0, 20  # Default ranges

        if all_heatmaps:
            vmin = min(map(np.min, all_heatmaps))
            vmax = max(map(np.max, all_heatmaps))
        else:
            vmin, vmax = 0, 1

        # Second pass: draw heatmaps with unified settings
        for i, (dataset, ax) in enumerate(zip(datasets, axes)):
            try:
                # Filter vehicle data
                vehicle_data = dataset['data'][dataset['data']['id'] == vehicle_id]

                # Check if data is empty
                if vehicle_data.empty:
                    print(f"Vehicle {vehicle_id} {dataset['label']} data is empty, skipping")
                    ax.axis('off')
                    continue

                # Extract position and speed data
                x = vehicle_data[dataset['x_col']].values
                y = vehicle_data['speed'].values

                # Create grid and calculate density
                heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)

                # Draw heatmap with unified ranges
                im = ax.imshow(heatmap.T, origin='lower',
                               extent=[x_min, x_max, y_min, y_max],
                               cmap=dataset['cmap'], alpha=0.9,
                               vmin=vmin, vmax=vmax)

                # Set title and labels
                ax.set_title(f'{dataset["label"]}', fontsize=12, pad=5)

                # Unified axis labels
                ax.set_xlabel('longitudinal position (m)', fontsize=12)
                if i == 0:  # Show y-axis label only for first subplot
                    ax.set_ylabel('speed (m/s)', fontsize=12)

                # Unified tick settings
                ax.tick_params(axis='both', which='major', labelsize=7)

                # Ensure consistent aspect ratio for square appearance
                ax.set_aspect((x_max - x_min) / (y_max - y_min))

                # Set same tick intervals
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Maximum 5 ticks
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))

            except KeyError as e:
                print(f"Error processing vehicle {vehicle_id} {dataset['label']} data: missing column {e}")
                ax.axis('off')
                continue
            except Exception as e:
                print(f"Error processing vehicle {vehicle_id} {dataset['label']} data: {e}")
                ax.axis('off')
                continue

        # Add shared colorbar
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Right-side independent colorbar position
        # Adjust colorbar width parameter (change 0.02 to smaller value, e.g., 0.015)
        # Adjust colorbar position (first parameter controls horizontal position, smaller value moves left)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.005, 0.7])  # Third parameter controls colorbar width, first parameter changed from 0.92 to 0.90, moving colorbar left
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('density', fontsize=12)
        cbar.ax.tick_params(labelsize=7)

        # Adjust layout for uniform distribution
        plt.tight_layout()
        # plt.subplots_adjust(top=0.85, wspace=0.4, right=0.9)  # Leave space for title and colorbar
        # Reduce existing wspace value (e.g., from 0.4 to 0.2)
        # Adjust overall layout right boundary (smaller right value moves right boundary left)
        plt.subplots_adjust(top=0.85, wspace=0.10, right=0.86)  # Reduce wspace to decrease subplot spacing, right changed from 0.9 to 0.88, compressing right boundary

        # Save chart
        try:
            plt.savefig(os.path.join(output_dir, f'vehicle_{vehicle_id}_position_speed_heatmap.png'),
                        dpi=300, bbox_inches='tight')
            print(f"Saved position-speed heatmap for vehicle {vehicle_id}")
        except Exception as e:
            print(f"Error saving chart for vehicle {vehicle_id}: {e}")

        plt.close()

    print(f"All position-speed heatmaps saved to {output_dir} directory")


if __name__ == "__main__":
    # Define all data file paths
    original_file = 'selected_tracks.csv'  # Original data
    idm_file = 'output/IDM_data.xlsx'  # IDM
    dynamic_idm_file = 'output/Dynamic_idm_data.xlsx'  # Dynamic_IDM
    ld_idm_file = 'output/LD-IDM_data.xlsx'  # LD-IDM
    dynamic_ld_idm_file = 'output/Dynamic_LD_IDM_data.xlsx'  # Dynamic_LD-IDM

    # Execute comparison
    compare_position_speed(original_file, idm_file, dynamic_idm_file,
                           ld_idm_file, dynamic_ld_idm_file)