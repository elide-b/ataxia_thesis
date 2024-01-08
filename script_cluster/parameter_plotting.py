import numpy as np
import matplotlib.pyplot as plt
import scipy

from connectivity import load_mousebrain
from utils import compute_FCD, compute_FCD_exp
import os

data_dir = os.path.dirname(__file__) + "\\.."

"""3D parameter plotting e optimization di FCD con cost function come da Deco et al. (2021): Kolmogorov-Smirnov distance"""

with np.load("opt_parameters_healthy.npz") as rates:
    t = rates['time']
    data = rates['tavg']
    bold = rates['bold']
    a_range = rates['a']
    Ji_range = rates['J_i']
    wp_range = rates['w_p']
    wi_range = rates['W_i']

for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        for iwp, wp in enumerate(wp_range):
            for iwi, wi in enumerate(wi_range):
                S_e = data[ia, iJi, iwp, iwi, 0]  # a, J_i, w_p, W_i, state variable
                S_i = data[ia, iJi, iwp, iwi, 1]
                R_e = data[ia, iJi, iwp, iwi, 2]
                R_i = data[ia, iJi, iwp, iwi, 3]
                plt.figure()
                for regS_e in S_e:
                    plt.plot(t, regS_e, color='r', label='S_E')
                for regS_i in S_i:
                    plt.plot(t, regS_i, color='b', label='S_I')
                plt.title('Synaptic gating variables for a=' + str(a) + ' J_i=' + str(Ji))
                plt.legend()

                plt.figure()
                for regR_e in R_e:
                    plt.plot(t, regR_e, color='r', label='r_E')
                plt.title('Firing rates E for a=' + str(a) + ' J_i=' + str(Ji))
                plt.legend()

                plt.figure()
                for regR_i in R_i:
                    plt.plot(t, regR_i, color='b', label='r_I')
                plt.title('Firing rates I for a=' + str(a) + ' J_i=' + str(Ji))
                plt.legend()
                plt.show()

# compute FCD for mouse empirical data
BOLD_DIR = '../dataset fMRI/Data_to_share/BOLD_TS'
MOUSE_ID = 'ag171031a'
REGIONS_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'
fcd = compute_FCD(BOLD_DIR, MOUSE_ID, REGIONS_LABELS_FILE, plot_FCD=True)

# compute FCD for mouse experimental data
brain = load_mousebrain(data_dir, "Connectivity_596.h5", norm="log", scale="region")
fcd_exp = []
for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        for iwp, wp in enumerate(wp_range):
            for iwi, wi in enumerate(wi_range):
                fcd_exp.append(compute_FCD_exp(brain, a, Ji, wp, wi, plot_FCD=True))

# Kolmogorov-Smirnov distance used as a cost function
for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        for iwp, wp in enumerate(wp_range):
            for iwi, wi in enumerate(wi_range):
                scipy.stats.kstest(fcd, fcd_exp[ia, iJi, iwp, iwi])
