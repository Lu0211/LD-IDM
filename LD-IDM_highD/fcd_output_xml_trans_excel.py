import math
import xml.etree.ElementTree as ET
import pandas as pd
import os


def extract_vehicle_data(xml_file_path, output_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    data = []

    # morning:data01
    # frame = 7717  # 377 +460
    # frame = 19887  #935  +585
    # frame = 6014  # 293
    # frame = 19149  # 908
    # frame = 5936  #286
    # frame = 4163  #191
    # frame = 0  # 3,6
    # frame = 6549 #329
    # frame = 6568 #331
    # frame = 6597  # 332
    # frame = 637 #35  +540
    # frame = 13677  # 658 +540
    # frame = 13750 #660
    # frame = 1835  # 98  +585
    # frame = 2521  # 121
    # frame = 3228  # 144  +510
    # frame = 4210  # 196
    # frame = 5136  # 249
    # frame = 2651  # 124  +570
    # frame = 5353  # 259
    # frame = 8271  # 411
    # frame = 9099  # 456
    # frame = 9606  # 476

    # noon:data06
    # frame = 400  #26
    # frame = 2341  # 114
    # frame = 11477  # 547  +580
    frame = 25655  # 1181  +475
    # frame = 28339  # 1303
    # frame = 8512  # 423
    # frame = 490  # 28

    # night:13
    # frame = 13  # 46
    # frame = 59  # 52
    # frame = 71  # 54
    # frame = 1621  # 235
    # frame = 1682  # 242
    # frame = 1714  # 245
    # frame = 1812  # 260
    # frame = 1838  # 264
    # frame = 1870  # 269  +500
    # frame = 1960  # 280  +500

    # Friday：23
    # frame = 3913 #217
    # frame = 8419  # 462
    # frame = 10734  # 574
    # frame = 455  # 45
    # frame = 827  # 62
    # frame = 1845  # 99

    data = []
    target_time_interval = 0.04  # Target time interval (seconds)
    next_record_time = 0  # Next recording time point
    tolerance = 1e-6  # Floating point comparison tolerance

    # Iterate through each timestep
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))

        # Check if reached or exceeded next recording time point
        if time >= next_record_time - tolerance and frame < 475+25655:
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

    df = pd.DataFrame(data)

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_excel(output_file_path, index=False)
    print(f"Data has been successfully saved to {output_file_path}")


if __name__ == "__main__":
    xml_file = 'output.xml'  # Replace with actual XML file path
    # output_file = 'output/IDM_data.xlsx'  # IDM
    # output_file = 'output/LD-IDM_data.xlsx'  # LD-IDM
    # output_file = 'output/Dynamic_idm_data.xlsx'  # Dynamic IDM
    output_file = 'output/Dynamic_LD_IDM_data.xlsx'  # Dynamic LD-IDM

    extract_vehicle_data(xml_file, output_file)