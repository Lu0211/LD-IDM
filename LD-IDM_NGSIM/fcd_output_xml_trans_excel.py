import math
import xml.etree.ElementTree as ET
import pandas as pd
import os


def extract_vehicle_data(xml_file_path, output_file_path):
    # Parse XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Create list to store data
    data = []
    # Initialize counter and data list
    # frame = 175  # 328 (176-1)
    # frame = 151  # 323 (176-1)
    # frame = 5  # 294 (176-1)
    # frame = 33  # 305
    # frame = 150  # 316  +580
    # frame = 168  # 321   +590
    # frame = 364  # 406
    # frame = 1508  # 838
    # frame = 1945  # 943
    frame = 491  # 459
    # frame = 4213  # 1488
    # frame = 396  # 445
    # frame = 922  # 628
    # frame = 1295  # 751  +830
    # frame = 491  # 478
    # frame = 511  # 489
    # frame = 982  # 637
    # frame = 1009  # 645
    # frame = 1374  # 767
    # frame = 1398  # 775
    # frame = 2984  # 1185
    # frame = 1286  # 768
    # frame = 447  # 466  +712
    data = []
    target_time_interval = 0.1  # Target time interval (seconds)
    next_record_time = 0  # Next recording time point
    tolerance = 1e-6  # Floating point comparison tolerance

    # Iterate through each timestep
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))

        # Check if reached or exceeded next recording time point
        if time >= next_record_time - tolerance and frame < 491+730:  #630+?/925+?/850+?
            frame += 1
            print(f'time: {time}, frame: {frame}')

            # Find information for all vehicles
            for vehicle in timestep.findall('vehicle'):
                vehicle_id = vehicle.get('id')
                vehicle_type = vehicle.get('type')
                speed = float(vehicle.get('speed'))
                position_x = float(vehicle.get('x'))
                position_y = float(vehicle.get('y'))
                angle = float(vehicle.get('angle'))

                # Only process vehicles whose type is not DEFAULT_VEHTYPE
                if vehicle_type != 'DEFAULT_VEHTYPE':
                    data.append({
                        'frame': frame,
                        'id': vehicle_id,
                        'VehicleType': vehicle_type,
                        'speed': speed,
                        'PositionX': position_x,
                        'PositionY': position_y,
                        'Angle': angle
                    })

            # Update next recording time point (accumulate interval)
            next_record_time += target_time_interval

    # Create DataFrame
    df = pd.DataFrame(data)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results to Excel file
    df.to_excel(output_file_path, index=False)
    print(f"Data has been successfully saved to {output_file_path}")


if __name__ == "__main__":
    xml_file = 'output.xml'  # Replace with actual XML file path
    # car-car
    # output_file = 'output/IDM_data.xlsx'  # IDM
    # output_file = 'output/LD-IDM_data.xlsx'  # LD-IDM
    # output_file = 'output/Dynamic_idm_data.xlsx'  # Dynamic IDM
    # output_file = 'output/Centroid-guided_IDM_data.xlsx' # centroid-guided-IDM

    # car-truck
    # output_file = 'output/car-truck/IDM_data.xlsx'  # IDM
    # output_file = 'output/car-truck/LD-IDM_data.xlsx'  # LD-IDM
    # output_file = 'output/car-truck/Dynamic_idm_data.xlsx'  # Dynamic IDM
    # output_file = 'output/car-truck/Centroid-guided_IDM_data.xlsx' # centroid-guided-IDM

    # truck-car
    # output_file = 'output/truck-car/IDM_data.xlsx'  # IDM
    output_file = 'output/truck-car/LD-IDM_data.xlsx'  # LD-IDM
    # output_file = 'output/truck-car/Dynamic_idm_data.xlsx'  # Dynamic IDM
    # output_file = 'output/truck-car/Centroid-guided_IDM_data.xlsx' # centroid-guided-IDM
    extract_vehicle_data(xml_file, output_file)