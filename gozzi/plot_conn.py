import sys

sys.path.append('/home/marialaura/Scrivania/TVMB_ongoing')

import numpy as np
from connectivity import load_mousebrain
import pandas as pd
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt

GOZZI_LABELS_FILE = 'atlas_gozzi_mod.xlsx'
acro_gozzi = list(pd.read_excel(io=GOZZI_LABELS_FILE)['ACRO'])
macro_gozzi = list(pd.read_excel(io=GOZZI_LABELS_FILE)['MACRO'])

conn_oh = load_mousebrain("Connectivity_02531729cb0d470d9a62dcff9158a952.h5", norm=False, scale=False)
np.fill_diagonal(conn_oh.tract_lengths, 0.)
to_plot = ['Right Primary motor area', 'Right Secondary motor area', 'Right Primary somatosensory area, nose',
           'Right Primary somatosensory area, barrel field', 'Right Primary somatosensory area, lower limb',
           'Right Primary somatosensory area, mouth', 'Right Primary somatosensory area, upper limb',
           'Right Primary somatosensory area, trunk',
           'Right Primary somatosensory area, unassigned', 'Right Supplemental somatosensory area',
           'Left Primary motor area', 'Left Secondary motor area', 'Left Primary somatosensory area, nose',
           'Left Primary somatosensory area, barrel field', 'Left Primary somatosensory area, lower limb',
           'Left Primary somatosensory area, mouth', 'Left Primary somatosensory area, upper limb',
           'Left Primary somatosensory area, trunk',
           'Left Primary somatosensory area, unassigned', 'Left Supplemental somatosensory area']
inds_to_plot = [i for i, reg in enumerate(conn_oh.region_labels) if reg in to_plot]

# remove Posterior parietal association areas (not in OH atlas)
# to_remove = ['PTLp']
# i_to_remove = [i for i,acro in enumerate(acro_gozzi) if acro in to_remove]
# acro_gozzi.pop(i_to_remove[0])
# macro_gozzi.pop(i_to_remove[0])


bilateral_acro = []
bilateral_macro = []

# bilateral acronyms
for a in acro_gozzi:
    right_acro = 'R-' + a
    bilateral_acro.append(right_acro)

for a in acro_gozzi:
    left_acro = 'L-' + a
    bilateral_acro.append(left_acro)

# bilateral macroPTLp
for a in macro_gozzi:
    right_macro = 'R-' + a
    bilateral_macro.append(right_macro)

for a in macro_gozzi:
    left_macro = 'L-' + a
    bilateral_macro.append(left_macro)

labels_path = 'results/only_gozzi/labels_gozzi.txt'
with open(labels_path, 'r') as file:
    labels_gozzi = [line.strip() for line in file]

weight_matrix = np.loadtxt('results/only_gozzi/weights_gozzi.txt')
# norm_weight = weight_matrix/np.max(weight_matrix)
w = weight_matrix.copy()
w[np.isnan(w)] = 0.0

w[w <= 0] = 0
w /= np.max(w)
w += 1
w = np.log(w)

"""
w0 = w <= 0
w_positive = w > 0
w /= w[w_positive].min()
w *= np.exp(1)

w[w_positive] = np.log(w[w_positive])
w[w0] = 0.0
"""
weight_matrix = w
# norm_weights = weight_matrix/np.max(weight_matrix)

sub1 = 'R-Isocortex'
sub2 = 'L-Isocortex'
inds_sub = [i for i, macro in enumerate(bilateral_macro) if macro == sub1 or macro == sub2]

bilateral_acro = np.array(bilateral_acro)
labels = np.array(labels_gozzi)

plot_connectivity_circle(con=weight_matrix[np.ix_(inds_sub, inds_sub)],
                         node_names=labels[inds_sub], n_lines=400, colormap='rainbow')

plt.show()
