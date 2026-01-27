from NGSIM2HighD import NGSIM2HighD
import HighD_Columns as HC 
import NGSIM_Columns as NC 
ngsim_dataset_dir =  "data/"
ngsim_dataset_files = ['NGSIM_data-0500-0515.csv']

converter = NGSIM2HighD(ngsim_dataset_dir, ngsim_dataset_files)
#converter.infer_lane_marking()
converter.convert_tracks_info()
converter.convert_meta_info()
converter.convert_static_info()