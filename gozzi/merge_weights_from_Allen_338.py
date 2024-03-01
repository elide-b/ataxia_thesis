import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from voxcell import RegionMap

from connectivity import load_mousebrain

region_map = RegionMap.load_json("1.json")
voxel = json.loads(open('voxels.json', 'r').read())
conn_oh = load_mousebrain("Connectivity_02531729cb0d470d9a62dcff9158a952.h5", norm=False, scale=False)

GOZZI_LABELS_FILE = 'atlas_gozzi_mod.xlsx'
reg_labels_gozzi = list(pd.read_excel(io=GOZZI_LABELS_FILE)['NAME'])
regions_labels_gozzi = ['Left ' + reg for reg in reg_labels_gozzi] + ['Right ' + reg for reg in reg_labels_gozzi]

matrix = np.zeros((338, 338))
conn = np.copy(conn_oh.weights)

labels = np.copy(conn_oh.region_labels)
for i, src_name in enumerate(reg_labels_gozzi):
    ids_reg = region_map.find(src_name, "name", with_descendants=True)
    names = [region_map.get(id_reg, "name") for id_reg in ids_reg]
    l_filt = np.isin(labels, ["Left " + name for name in names])
    r_filt = np.isin(labels, ["Right " + name for name in names])
    l_labels = labels[l_filt]
    r_labels = labels[r_filt]
    l_weights = np.array([voxel[name] for name in l_labels])[..., np.newaxis]
    r_weights = np.array([voxel[name] for name in r_labels])[..., np.newaxis]

    if np.sum(l_weights) == 0 or np.sum(r_weights) == 0:
        print(f"Skip  {src_name}")
        continue
    for j, tgt_name in enumerate(reg_labels_gozzi):
        ids_reg = region_map.find(tgt_name, "name", with_descendants=True)
        if len(ids_reg) == 0:
            raise Exception(f"Label does not correspond to Allen {src_name}")
        names = [region_map.get(id_reg, "name") for id_reg in ids_reg]
        l_tgt_names = ["Left " + name for name in names]
        r_tgt_names = ["Right " + name for name in names]

        matrix[i, j] = np.sum(conn[l_filt][:, np.isin(labels, l_tgt_names)] * l_weights) / np.sum(l_weights)
        matrix[len(reg_labels_gozzi) + i, j] = np.sum(
            conn[r_filt][:, np.isin(labels, l_tgt_names)] * r_weights) / np.sum(r_weights)
        matrix[i, len(reg_labels_gozzi) + j] = np.sum(
            conn[l_filt][:, np.isin(labels, r_tgt_names)] * l_weights) / np.sum(l_weights)
        matrix[len(reg_labels_gozzi) + i, len(reg_labels_gozzi) + j] = np.sum(
            conn[r_filt][:, np.isin(labels, r_tgt_names)] * r_weights) / np.sum(r_weights)

plt.title('Connectivity Gozzi regions')
plt.imshow(matrix, cmap='viridis', interpolation="nearest")
# plt.xticks(np.arange(len(reg_gozzi)), reg_gozzi, rotation=45, ha="right")
# plt.yticks(np.arange(len(reg_gozzi)), reg_gozzi)
plt.colorbar()
plt.tight_layout()
plt.savefig('conn_gozzi.png', dpi=600)
plt.show()
np.save("matrix.npy", matrix)
