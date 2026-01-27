import pandas as pd
import numpy as np
import os
from openpyxl import Workbook

# Set font to ensure proper display
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False  # Solve negative sign display issue
plt.rcParams['font.family'] = 'Times New Roman'

# Define directory for saving results
output_dir = 'gap_distance_error_results'

# Define file paths
original_file = 'selected_tracks.csv'  # Original data
idm_file = 'output/fcd_IDM/1184.xlsx'  # IDM data
dynamic_idm_file = 'output/fcd_Dynamic_IDM/1184.xlsx'  # Dynamic_IDM data
lda_idm_file = 'output/fcd_LD-IDM/1184.xlsx'  # LD-IDM data
dynamic_lda_idm_file = 'output/fcd_Dynamic_LD-IDM/1184.xlsx'  # Dynamic_LD-IDM data

# Configuration parameters
FRONT_VEHICLE_ID = 1181  # Front vehicle ID
BACK_VEHICLE_ID = 1184  # Rear vehicle ID
VEHICLE_LENGTH = 4.65  # Front vehicle length (meters)


def extract_data(file_path):
    """Extract frame, vehicle ID and x coordinate data from file"""
    # Choose appropriate reading method based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Check if frame column exists
    frame_cols = ['Frame', 'frame', 'FrameID', 'frame_id']
    frame_col = next((col for col in frame_cols if col in df.columns), None)
    if frame_col is None:
        raise ValueError(f"Frame column not found in file {file_path}")

    # Check if vehicle ID column exists
    id_cols = ['VehicleID', 'vehicle_id', 'ID', 'id', 'CarID']
    id_col = next((col for col in id_cols if col in df.columns), None)
    if id_col is None:
        raise ValueError(f"Vehicle ID column not found in file {file_path}")

    # Determine x coordinate column
    x_cols = ['position_x', 'x_position', 'PositionX']
    x_col = next((col for col in x_cols if col in df.columns), None)
    if x_col is None:
        raise ValueError(f"X coordinate column not found in file {file_path}")

    # Rename columns and return required data
    result = df[[id_col, frame_col, x_col]].copy()
    result = result.rename(columns={
        id_col: 'vehicle_id',
        frame_col: 'frame',
        x_col: 'x'
    })

    # Remove duplicate (vehicle ID, frame) combinations
    if result.duplicated(subset=['vehicle_id', 'frame']).any():
        print(f"Warning: Duplicate (vehicle ID, frame) combinations found in file {file_path}, automatically deduplicated")
        result = result.drop_duplicates(subset=['vehicle_id', 'frame'], keep='first')

    return result


# Custom error calculation functions
def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    # Avoid division by zero
    mask = y_true != 0
    if len(mask) == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_distance(front_x, back_x, vehicle_length=VEHICLE_LENGTH):
    """Calculate actual gap distance: front vehicle x position - rear vehicle x position - front vehicle length"""
    return front_x - back_x - vehicle_length


def get_common_frames(*datasets):
    """Get common frames across all datasets"""
    frame_sets = [set(data['frame'].unique()) for data in datasets]
    return sorted(list(set.intersection(*frame_sets)))


def main():
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to directory: {os.path.abspath(output_dir)}")

    # Extract all data
    print("Extracting data...")
    try:
        original_data = extract_data(original_file)
        idm_data = extract_data(idm_file)
        dynamic_idm_data = extract_data(dynamic_idm_file)
        lda_idm_data = extract_data(lda_idm_file)
        dynamic_lda_idm_data = extract_data(dynamic_lda_idm_file)
    except Exception as e:
        print(f"Error extracting data: {e}")
        return

    # Model data dictionary
    model_data = {
        'IDM': idm_data,
        'LD-IDM': lda_idm_data,
        'Dynamic IDM': dynamic_idm_data,
        'Dynamic LD-IDM': dynamic_lda_idm_data
    }

    # Check if specified front and rear vehicles are present in data
    for data_name, data in [("Original data", original_data)] + list(model_data.items()):
        vehicles = set(data['vehicle_id'].unique())
        if FRONT_VEHICLE_ID not in vehicles:
            print(f"Warning: Front vehicle ID {FRONT_VEHICLE_ID} not found in {data_name}")
            return
        if BACK_VEHICLE_ID not in vehicles:
            print(f"Warning: Rear vehicle ID {BACK_VEHICLE_ID} not found in {data_name}")
            return

    # Get common frames across all datasets
    print("Finding common frames...")
    all_datasets = [original_data] + list(model_data.values())
    common_frames = get_common_frames(*all_datasets)

    print(f"Found {len(common_frames)} common frames across all datasets")
    if len(common_frames) < 2:
        print("Warning: Insufficient common frames, cannot perform calculations!")
        return

    # Extract front and rear vehicle information from original data and calculate actual gap distance
    original_front = original_data[
        (original_data['vehicle_id'] == FRONT_VEHICLE_ID) &
        (original_data['frame'].isin(common_frames))
        ].set_index('frame')['x'].reindex(common_frames)

    original_back = original_data[
        (original_data['vehicle_id'] == BACK_VEHICLE_ID) &
        (original_data['frame'].isin(common_frames))
        ].set_index('frame')['x'].reindex(common_frames)

    # Calculate actual gap distance (ground truth)
    actual_distance = calculate_distance(original_front, original_back)

    # Store error results for all models
    model_errors = {}

    # Calculate gap distance errors for each model
    for model_name, data in model_data.items():
        print(f"\nCalculating gap distance errors for {model_name} model...")

        # Extract front and rear vehicle information from model data
        model_front = data[
            (data['vehicle_id'] == FRONT_VEHICLE_ID) &
            (data['frame'].isin(common_frames))
            ].set_index('frame')['x'].reindex(common_frames)

        model_back = data[
            (data['vehicle_id'] == BACK_VEHICLE_ID) &
            (data['frame'].isin(common_frames))
            ].set_index('frame')['x'].reindex(common_frames)

        # Calculate model-predicted gap distance
        predicted_distance = calculate_distance(model_front, model_back)

        # Calculate error metrics
        mae = calculate_mae(actual_distance, predicted_distance)
        mape = calculate_mape(actual_distance, predicted_distance)
        rmse = calculate_rmse(actual_distance, predicted_distance)

        print(f"{model_name} gap distance errors - MAE: {mae:.2f}, MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")

        # Store error details for each frame
        frame_errors = pd.DataFrame({
            'frame': common_frames,
            'actual_distance': actual_distance.values,
            'predicted_distance': predicted_distance.values,
            'absolute_error': np.abs(actual_distance - predicted_distance),
            'percentage_error': np.abs(
                (actual_distance - predicted_distance) / actual_distance.replace(0, np.nan)) * 100
        })

        model_errors[model_name] = {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'frame_errors': frame_errors
        }

    # Calculate performance improvements
    improvement = {}

    # 1. Calculate performance improvement relative to IDM
    if 'IDM' in model_errors:
        idm_mae = model_errors['IDM']['mae']
        idm_mape = model_errors['IDM']['mape']
        idm_rmse = model_errors['IDM']['rmse']

        for model_name, errors in model_errors.items():
            if model_name == 'IDM':
                continue

            # Performance improvement = (IDM error - model error) / IDM error * 100%
            with np.errstate(divide='ignore', invalid='ignore'):
                mae_improve = (idm_mae - errors['mae']) / idm_mae * 100 if idm_mae != 0 else 0
                mape_improve = (idm_mape - errors['mape']) / idm_mape * 100 if idm_mape != 0 else 0
                rmse_improve = (idm_rmse - errors['rmse']) / idm_rmse * 100 if idm_rmse != 0 else 0

            if model_name not in improvement:
                improvement[model_name] = {}

            improvement[model_name]['vs_IDM_mae'] = mae_improve
            improvement[model_name]['vs_IDM_mape'] = mape_improve
            improvement[model_name]['vs_IDM_rmse'] = rmse_improve

    # 2. Calculate performance improvement of Dynamic LD-IDM compared to Dynamic IDM
    if 'Dynamic IDM' in model_errors and 'Dynamic LD-IDM' in model_errors:
        dyn_idm_mae = model_errors['Dynamic IDM']['mae']
        dyn_idm_mape = model_errors['Dynamic IDM']['mape']
        dyn_idm_rmse = model_errors['Dynamic IDM']['rmse']

        dyn_lda_errors = model_errors['Dynamic LD-IDM']

        with np.errstate(divide='ignore', invalid='ignore'):
            mae_improve = (dyn_idm_mae - dyn_lda_errors['mae']) / dyn_idm_mae * 100 if dyn_idm_mae != 0 else 0
            mape_improve = (dyn_idm_mape - dyn_lda_errors['mape']) / dyn_idm_mape * 100 if dyn_idm_mape != 0 else 0
            rmse_improve = (dyn_idm_rmse - dyn_lda_errors['rmse']) / dyn_idm_rmse * 100 if dyn_idm_rmse != 0 else 0

        if 'Dynamic LD-IDM' not in improvement:
            improvement['Dynamic LD-IDM'] = {}

        improvement['Dynamic LD-IDM']['vs_Dynamic_IDM_mae'] = mae_improve
        improvement['Dynamic LD-IDM']['vs_Dynamic_IDM_mape'] = mape_improve
        improvement['Dynamic LD-IDM']['vs_Dynamic_IDM_rmse'] = rmse_improve

    # Save results to Excel
    print("\nSaving error results to Excel...")
    excel_path = os.path.join(output_dir, f'Gap_Distance_Error_Analysis_FrontVehicle{FRONT_VEHICLE_ID}_RearVehicle{BACK_VEHICLE_ID}.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Summary error statistics
        summary_data = []
        for model_name, errors in model_errors.items():
            row = {
                'Model Name': model_name,
                'MAE (m)': errors['mae'],
                'MAPE (%)': errors['mape'],
                'RMSE (m)': errors['rmse'],
                'MAE Improvement (vs IDM) (%)': improvement[model_name][
                    'vs_IDM_mae'] if model_name in improvement and 'vs_IDM_mae' in improvement[model_name] else 0,
                'MAPE Improvement (vs IDM) (%)': improvement[model_name][
                    'vs_IDM_mape'] if model_name in improvement and 'vs_IDM_mape' in improvement[model_name] else 0,
                'RMSE Improvement (vs IDM) (%)': improvement[model_name][
                    'vs_IDM_rmse'] if model_name in improvement and 'vs_IDM_rmse' in improvement[model_name] else 0
            }

            # Add Dynamic LD-IDM improvement compared to Dynamic IDM (only for Dynamic LD-IDM)
            if model_name == 'Dynamic LD-IDM' and 'vs_Dynamic_IDM_mae' in improvement.get(model_name, {}):
                row['MAE Improvement (vs Dynamic IDM) (%)'] = improvement[model_name]['vs_Dynamic_IDM_mae']
                row['MAPE Improvement (vs Dynamic IDM) (%)'] = improvement[model_name]['vs_Dynamic_IDM_mape']
                row['RMSE Improvement (vs Dynamic IDM) (%)'] = improvement[model_name]['vs_Dynamic_IDM_rmse']
            else:
                row['MAE Improvement (vs Dynamic IDM) (%)'] = None
                row['MAPE Improvement (vs Dynamic IDM) (%)'] = None
                row['RMSE Improvement (vs Dynamic IDM) (%)'] = None

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Error Summary', index=False, float_format="%.2f")

        # 2. Frame-level error details for each model
        for model_name, errors in model_errors.items():
            errors['frame_errors'].to_excel(writer, sheet_name=f'{model_name} Frame Errors', index=False, float_format="%.2f")

        # Set percentage format
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            # Process error summary sheet
            if sheet_name == 'Error Summary':
                # Set format for all percentage columns (columns 5-10)
                for col in ['E', 'F', 'G', 'H', 'I', 'J']:
                    for row in range(2, worksheet.max_row + 1):
                        cell = worksheet[col + str(row)]
                        cell.number_format = '0.00%'
            # Process frame error sheets
            elif 'Frame Errors' in sheet_name:
                # Set format for percentage error column
                for row in range(2, worksheet.max_row + 1):
                    cell = worksheet['E' + str(row)]  # Column E is percentage error
                    cell.number_format = '0.00%'

    print(f"Gap distance error analysis results saved to: {excel_path}")
    print(f"Analysis based on front vehicle ID: {FRONT_VEHICLE_ID}, rear vehicle ID: {BACK_VEHICLE_ID}, front vehicle length: {VEHICLE_LENGTH}m")


if __name__ == "__main__":
    main()