import sys
sys.path.append('/Users/marialauradegrazia/Desktop/tesi_files_elide_robin/better_tesi-main')

import numpy as np 
from scipy.io import loadmat
import os
import pandas as pd
from plots import plot_fc, plot_bold_timeseries


BOLD_DIR = '../dataset fMRI/Data_to_share/BOLD_TS'
MOUSE_ID = 'ag171031a'
REGIONS_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'

data_dir = os.path.join(BOLD_DIR,MOUSE_ID+'_WT_tsc2_smoothed_n172.mat')
data_mat = loadmat(data_dir)
bold_data = data_mat[MOUSE_ID+'_WT_tsc2_smoothed']
regions_labels = list(pd.read_excel(io=REGIONS_LABELS_FILE)['NAME'])
regions_labels_tot = ['Right ' + reg for reg in regions_labels] + ['Left ' + reg for reg in regions_labels]


output_folder_ts = 'Timeseries'
if not os.path.exists(output_folder_ts):
    os.makedirs(output_folder_ts)

bold_data = pd.DataFrame(bold_data)
bold_data.columns = regions_labels_tot

plot_bold_timeseries(bold_ts=bold_data, mouse_id=MOUSE_ID, show=False).write_html(output_folder_ts+'/BOLD_timeseries_'+ MOUSE_ID + '.html')



output_folder_fc = 'Empirical_FC'
if not os.path.exists(output_folder_fc):
    os.makedirs(output_folder_fc)

fc_emp =  np.corrcoef(bold_data, rowvar=False)
plot_fc(FC_mat=fc_emp,regions_labels=bold_data.columns,mouse_id=MOUSE_ID,show=False).write_html(output_folder_fc+'/fc_'+ MOUSE_ID + '.html')

