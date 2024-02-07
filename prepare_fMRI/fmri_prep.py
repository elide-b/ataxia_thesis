import sys
sys.path.append('/Users/marialauradegrazia/Desktop/my_TVB/TVMB_ongoing')

import numpy as np 
from scipy.io import loadmat
import os
import pandas as pd
from plots import plot_fc, plot_bold_timeseries, plot_fcd, plot_fcd_hist
import matplotlib.pyplot as plt


BOLD_DIR = '../dataset fMRI/Data_to_share/BOLD_TS'
REGIONS_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'
SINGLE_MOUSE = True


if SINGLE_MOUSE:
    
    MOUSE_ID = 'ag171031a'
    data_dir = os.path.join(BOLD_DIR,MOUSE_ID+'_WT_tsc2_smoothed_n172.mat')
    data_mat = loadmat(data_dir)
    bold_data = data_mat[MOUSE_ID+'_WT_tsc2_smoothed']
    regions_labels = list(pd.read_excel(io=REGIONS_LABELS_FILE)['NAME'])
    regions_labels_tot = ['Right ' + reg for reg in regions_labels] + ['Left ' + reg for reg in regions_labels]
    
    print('The total number of regions in Gozzi is: ', len(regions_labels_tot))

    output_folder_ts = 'Bold_timeseries'
    if not os.path.exists(output_folder_ts):
        os.makedirs(output_folder_ts)

    bold_data = pd.DataFrame(bold_data)
    bold_data.columns = regions_labels_tot

    plot_bold_timeseries(bold_ts=bold_data, mouse_id=MOUSE_ID, show=True).write_html(output_folder_ts+'/BOLD_timeseries_'+ MOUSE_ID + '.html')


    output_folder_fc = 'Empirical_FC'
    if not os.path.exists(output_folder_fc):
        os.makedirs(output_folder_fc)

    fc_emp =  np.corrcoef(bold_data, rowvar=False)
    plot_fc(FC_mat=fc_emp,regions_labels=bold_data.columns,mouse_id=MOUSE_ID,show=True).write_html(output_folder_fc+'/fc_'+ MOUSE_ID + '.html')



    # windowed dynamical FC

    # define window size
    tau = 60 # [s] window size
    w_step = 20 # [s] window step

    #t_range = np.arange(0,bold_data.shape[0],1)
    FC_windows = []
    triu_FC_windows = []

    for t in range(0,bold_data.shape[0],w_step):
        bold_window = bold_data.iloc[t:t+tau]
        fc = np.corrcoef(bold_window,rowvar = False)
        FC_windows.append(fc)
        triu_FC_windows.append(np.triu(fc))

    n_windows = len((triu_FC_windows))
    fcd = np.zeros(shape = (n_windows,n_windows))


    for i in range(n_windows):
        for j in range(n_windows):
            fcd[i][j]=np.corrcoef(triu_FC_windows[i].ravel(),triu_FC_windows[j].ravel())[0,1]



    output_folder_fcd = 'Empirical_FCD'
    if not os.path.exists(output_folder_fcd):
        os.makedirs(output_folder_fcd)


    plot_fcd(fcd, MOUSE_ID, show = True).write_html(output_folder_fcd+'/fcd_'+ MOUSE_ID + '.html')

    plot_fcd_hist(fcd, MOUSE_ID, show = True).write_html(output_folder_fcd+'/fcd_hist_'+ MOUSE_ID + '.html')

else:
    # compute the average between subjects for all the metrics

    mice_ids = ['ag170913b','ag170919d','ag170925a','ag171004d','ag171006e','ag171013a','ag171013b','ag171013e','ag171018a','ag171018b','ag171018c','ag171019c','ag171019d','ag171023a','ag171023b','ag171030a','ag171030b','ag171030c','ag171030d','ag171031a']
    
    #for mouse_id in mice_ids:
        






