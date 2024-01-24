import sys

sys.path.append('/Users/marialauradegrazia/Desktop/my_TVB/TVMB_ongoing')

import os
from collections import OrderedDict
import time
import numpy as np
from connectivity import load_mousebrain
from plots import plot_weights, plot_matrix
import pandas as pd
import warnings

conn_oh = load_mousebrain("Connectivity_02531729cb0d470d9a62dcff9158a952.h5", norm=False, scale=False)

GOZZI_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'
reg_labels_gozzi = list(pd.read_excel(io=GOZZI_LABELS_FILE)['NAME'])
regions_labels_gozzi = ['Right ' + reg for reg in reg_labels_gozzi] + ['Left ' + reg for reg in reg_labels_gozzi]

print(len(reg_labels_gozzi))
exit()

OH_LABELS_FILE = '../gozzi/oh_table1.xls'
SHEET_NAME = 'Voxel Count_295 Structures'
reg_labels_oh = pd.read_excel(OH_LABELS_FILE, sheet_name=SHEET_NAME)['Name']
regions_labels_oh = ['Right ' + reg for reg in reg_labels_oh] + ['Left ' + reg for reg in reg_labels_oh]

voxel_count_oh = pd.read_excel(OH_LABELS_FILE, sheet_name=SHEET_NAME)['Voxel Count']
voxel_count_oh[np.isnan(voxel_count_oh)] = 0.0
voxel_count_oh_lr = pd.concat([voxel_count_oh, voxel_count_oh], ignore_index=True)
print(voxel_count_oh_lr[0], voxel_count_oh_lr[324])

df_voxel = pd.DataFrame({'Regions': regions_labels_oh, 'Voxel Count': voxel_count_oh_lr})

# print('Numero regioni nella connectivity: ', len(conn_oh.region_labels))
# print('Numero regioni nel file Oh: ', len(regions_labels_oh))
# print('Numero regioni nel file Gozzi: ', len(regions_labels_gozzi))


set1_oh = set(conn_oh.region_labels)
set2_oh = set(regions_labels_oh)
excluded_regs = set1_oh.symmetric_difference(set2_oh)

df_voxel = df_voxel[~df_voxel['Regions'].isin(excluded_regs)]
print(df_voxel)
df_voxel.to_csv('df_voxel', index=True)

key_i_k = 'Posterior parietal association areas'

print(df_voxel.loc[df_voxel['Regions'] == key_i_k, 'Voxel Count'].values)

###################CONN_OH HAS THE SAME NUMBER OF LEFT AND RIGHT REGIONS#####################
count_left = 0
count_right = 0

for reg in conn_oh.region_labels:
    count_left += reg.count('Left')
    count_right += reg.count('Right')

if count_left == count_right:
    print('n° reg left = n° reg right')
else:
    print('n° reg left =! n° reg right')

#############################################################################################


print('Difference between Oh table and Conn Oh, n° reg: ', len(excluded_regs))
# np.savetxt('diff_oh.txt', list(not_common_regs_oh), fmt='%s')
# print('Not common regions between Oh table and Conn Oh: ', not_common_regs_oh)

dict_areas = {
    "Ammon's horn": ['Field CA1', 'Field CA2', 'Field CA3'],
    "Hypothalamic lateral zone": ['Lateral hypothalamic area', 'Lateral preoptic area', 'Subthalamic nucleus',
                                  'Preparasubthalamic nucleus', 'Zona incerta', 'Parasubthalamic nucleus',
                                  'Tuberal nucleus', 'Perifornical nucleus', 'Retrochiasmatic area'],
    "Pallidum, medial region": ['Medial septal nucleus', 'Diagonal band nucleus', 'Triangular nucleus of septum'],
    "Entorhinal area": ['Entorhinal area, lateral part', 'Entorhinal area, medial part, dorsal zone',
                        'Entorhinal area, medial part, ventral zone'],
    "Pallidum, caudal region": ['Bed nucleus of the anterior commissure', 'Bed nuclei of the stria terminalis'],
    "Striatum ventral region": ['Nucleus accumbens', 'Fundus of striatum', 'Olfactory tubercle'],
    "Midbrain, behavioral state related": ['Substantia nigra, compact part', 'Pedunculopontine nucleus',
                                           'Interfascicular nucleus raphe', 'Interpeduncular nucleus',
                                           'Rostral linear nucleus raphe', 'Central linear nucleus raphe',
                                           'Dorsal nucleus raphe'],
    "Endopiriform nucleus": ['Endopiriform nucleus, dorsal part', 'Endopiriform nucleus, ventral part'],
    "Midbrain, motor related": ['Substantia nigra, reticular part', 'Ventral tegmental area', 'Paranigral nucleus',
                                'Midbrain reticular nucleus, retrorubral area', 'Midbrain reticular nucleus',
                                'Superior colliculus, motor related', 'Periaqueductal gray', 'Cuneiform nucleus',
                                'Oculomotor nucleus', 'Red nucleus', 'Medial accesory oculomotor nucleus',
                                'Edinger-Westphal nucleus', 'Trochlear nucleus', 'Paratrochlear nucleus',
                                'Ventral tegmental nucleus', 'Anterior tegmental nucleus',
                                'Lateral terminal nucleus of the accessory optic tract',
                                'Dorsal terminal nucleus of the accessory optic tract'],
    "Periventricular zone": ['Supraoptic nucleus', 'Accessory supraoptic group', 'Paraventricular hypothalamic nucleus',
                             'Periventricular hypothalamic nucleus, anterior part',
                             'Periventricular hypothalamic nucleus, intermediate part', 'Arcuate hypothalamic nucleus'],
    "Pallidum, dorsal region": ['Globus pallidus, external segment', 'Globus pallidus, internal segment'],
    "Cortical amygdalar area": ['Cortical amygdalar area, anterior part', 'Cortical amygdalar area, posterior part'],
    "Periventricular region": ['Anterodorsal preoptic nucleus', 'Anteroventral preoptic nucleus',
                               'Anteroventral periventricular nucleus', 'Dorsomedial nucleus of the hypothalamus',
                               'Median preoptic nucleus', 'Medial preoptic area',
                               'Vascular organ of the lamina terminalis', 'Posterodorsal preoptic nucleus',
                               'Parastrial nucleus', 'Suprachiasmatic nucleus',
                               'Periventricular hypothalamic nucleus, posterior part',
                               'Periventricular hypothalamic nucleus, preoptic part', 'Subparaventricular zone',
                               'Subfornical organ', 'Ventromedial preoptic nucleus', 'Ventrolateral preoptic nucleus'],
    "Orbital area": ['Orbital area, lateral part', 'Orbital area, medial part', 'Orbital area, ventrolateral part'],
    "Thalamus, polymodal association cortex related": ['Reticular nucleus of the thalamus',
                                                       'Intergeniculate leaflet of the lateral geniculate complex',
                                                       'Intermediate geniculate nucleus',
                                                       'Ventral part of the lateral geniculate complex',
                                                       'Subgeniculate nucleus', 'Rhomboid nucleus',
                                                       'Central medial nucleus of the thalamus', 'Paracentral nucleus',
                                                       'Central lateral nucleus of the thalamus',
                                                       'Parafascicular nucleus',
                                                       'Posterior intralaminar thalamic nucleus',
                                                       'Paraventricular nucleus of the thalamus', 'Parataenial nucleus',
                                                       'Nucleus of reuniens', 'Xiphoid thalamic nucleus',
                                                       'Intermediodorsal nucleus of the thalamus',
                                                       'Mediodorsal nucleus of thalamus',
                                                       'Submedial nucleus of the thalamus', 'Perireunensis nucleus',
                                                       'Anteroventral nucleus of thalamus',
                                                       'Anteromedial nucleus, dorsal part',
                                                       'Anteromedial nucleus, ventral part', 'Anterodorsal nucleus',
                                                       'Interanteromedial nucleus of the thalamus',
                                                       'Interanterodorsal nucleus of the thalamus',
                                                       'Lateral dorsal nucleus of thalamus',
                                                       'Lateral posterior nucleus of the thalamus',
                                                       'Posterior complex of the thalamus',
                                                       'Posterior limiting nucleus of the thalamus',
                                                       'Suprageniculate nucleus', 'Medial habenula',
                                                       'Lateral habenula'],
    "Thalamus, sensory-motor cortex related": ['Ventral anterior-lateral complex of the thalamus',
                                               'Ventral medial nucleus of the thalamus',
                                               'Ventral posterolateral nucleus of the thalamus',
                                               'Ventral posterolateral nucleus of the thalamus, parvicellular part',
                                               'Ventral posteromedial nucleus of the thalamus',
                                               'Ventral posteromedial nucleus of the thalamus, parvicellular part',
                                               'Posterior triangular thalamic nucleus',
                                               'Subparafascicular nucleus, magnocellular part',
                                               'Subparafascicular nucleus, parvicellular part',
                                               'Subparafascicular area', 'Peripeduncular nucleus',
                                               'Medial geniculate complex, dorsal part',
                                               'Medial geniculate complex, ventral part',
                                               'Medial geniculate complex, medial part',
                                               'Dorsal part of the lateral geniculate complex'],
    "Lateral septal complex": ['Lateral septal nucleus, caudal (caudodorsal) part',
                               'Lateral septal nucleus, rostral (rostroventral) part',
                               'Lateral septal nucleus, ventral part', 'Septofimbrial nucleus',
                               'Septohippocampal nucleus'],
    "Hypothalamic medial zone": ['Anterior hypothalamic nucleus', 'Supramammillary nucleus',
                                 'Medial mammillary nucleus', 'Lateral mammillary nucleus',
                                 'Tuberomammillary nucleus, dorsal part', 'Tuberomammillary nucleus, ventral part',
                                 'Medial preoptic nucleus', 'Dorsal premammillary nucleus',
                                 'Ventral premammillary nucleus',
                                 'Paraventricular hypothalamic nucleus, descending division',
                                 'Ventromedial hypothalamic nucleus', 'Posterior hypothalamic nucleus'],
    "Midbrain, sensory related": ['Inferior colliculus, central nucleus', 'Inferior colliculus, dorsal nucleus',
                                  'Inferior colliculus, external nucleus', 'Superior colliculus, sensory related',
                                  'Nucleus of the brachium of the inferior colliculus', 'Nucleus sagulum',
                                  'Parabigeminal nucleus', 'Midbrain trigeminal nucleus', 'Subcommissural organ'],
    "Striatum-like amygdalar nuclei": ['Central amygdalar nucleus', 'Anterior amygdalar area',
                                       'Bed nucleus of the accessory olfactory tract', 'Intercalated amygdalar nucleus',
                                       'Medial amygdalar nucleus'],
    "Striatum dorsal region": ['Caudoputamen'],
    "Pallidum, ventral region": ['Substantia innominata', 'Magnocellular nucleus'],
    "Posterior parietal association areas": ['Rostrolateral visual area', 'Anterior area']
}

final_dict_areas = {}
for key, list in dict_areas.items():
    new_list = [f'Left {reg}' for reg in list] + [f'Right {reg}' for reg in list]
    final_dict_areas[key] = new_list

labels_to_merge = [reg for list in final_dict_areas.values() for reg in list if reg not in excluded_regs]

print('Len labels to merge: ', len(labels_to_merge))

set1 = set(conn_oh.region_labels)
set2 = set(labels_to_merge)
not_merge_regs = set1.symmetric_difference(set2)

print('Not merged regions: ', len(not_merge_regs))

if len(conn_oh.region_labels) == (len(not_merge_regs) + len(labels_to_merge)):
    print('The total number of regions is OK!')
else:
    print('ERROR!')

gozzi_merging_labels = ['Right ' + reg for reg in final_dict_areas.keys()] + ['Left ' + reg for reg in
                                                                              final_dict_areas.keys()]
gozzi_remain = [elem for elem in regions_labels_gozzi if elem not in gozzi_merging_labels]
print('Gozzi regions found in merging: ', len(gozzi_merging_labels))
print('Gozzi regions not found in merging: ', len(gozzi_remain))

not_common_elem = set(gozzi_remain) - set(conn_oh.region_labels)
# print('In Gozzi but not in Oh: ', not_common_elem)

# print(conn_oh.region_labels)

# example search specific regions in a list
# search = [string for string in conn_oh.region_labels if 'Rostrolateral' in string]
# print('Search in conn oh:', search)


# MERGING
not_merge_regs = [elem for elem in not_merge_regs]

final_conn_labels = gozzi_merging_labels + not_merge_regs

def find_key(map, val):
    for k, list in map.items():
        if val in list:
            return k
    return None


with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=pd.errors.PerformanceWarning)

    df = pd.DataFrame(index=final_conn_labels, columns=final_conn_labels)
    df.to_csv('final_conn.csv', index=True)
    dict_areas_inds = {}

    for reg_gozzi, regions in final_dict_areas.items():
        regs = [r for r in regions]
        dict_areas_inds[reg_gozzi] = [i for i, labels in enumerate(conn_oh.region_labels) if labels in regs]

    # list of all the indices involved in merging
    inds_to_merge = [ind for list in dict_areas_inds.values() for ind in list]

    for i in range(len(conn_oh.region_labels)):
        # print(i)
        key_i = find_key(dict_areas_inds, i)
        if key_i == None:
            key_i = conn_oh.region_labels[i]
            key_i_k = key_i
        else:
            if 'Right' in conn_oh.region_labels[i]:
                key_i_k = 'Right ' + key_i
            else:
                key_i_k = 'Left ' + key_i

        print('Key i ', key_i)

        for j in range(len(conn_oh.region_labels)):
            # print(df)
            key_j = find_key(dict_areas_inds, j)
            if key_j == None:
                key_j = conn_oh.region_labels[j]
                key_j_k = key_j
            else:
                if 'Right' in conn_oh.region_labels[j]:
                    key_j_k = 'Right ' + key_j
                else:
                    key_j_k = 'Left ' + key_j

            voxel_count = [df_voxel.loc[df_voxel['Regions'] == key_i_k, 'Voxel Count'].values,
                           df_voxel.loc[df_voxel['Regions'] == key_j_k, 'Voxel Count'].values]

            # np.sum delle connessioni per voxel/connessioni
            flat_weights = [item for sublist in voxel_count for item in sublist]
            print(flat_weights)

            if j in inds_to_merge and i in inds_to_merge:
                # print('ENTRATO 1')
                df.at[key_i_k, key_j_k] = np.average(
                    conn_oh.weights[np.ix_(dict_areas_inds[key_i], dict_areas_inds[key_j])], axis=0,
                    weights=flat_weights)
                df.at[key_j_k, key_i_k] = np.average(
                    conn_oh.weights[np.ix_(dict_areas_inds[key_j], dict_areas_inds[key_i])], axis=0,
                    weights=flat_weights)

            if i in inds_to_merge and j not in inds_to_merge:
                # print('ENTRATO 2')
                df.at[key_i_k, conn_oh.region_labels[j]] = np.average(conn_oh.weights[dict_areas_inds[key_i], j],
                                                                      axis=0, weights=flat_weights)
                df.at[conn_oh.region_labels[j], key_i_k] = np.average(conn_oh.weights[j, dict_areas_inds[key_i]],
                                                                      axis=0, weights=flat_weights)

            if j in inds_to_merge and i not in inds_to_merge:
                # print('ENTRATO 3')
                print('Key j', key_j)  # trova le aree da mergiare -> trovare i voxel corrispondenti
                # print('Conn oh: region labels[j]', conn_oh.region_labels[j])
                print(conn_oh.weights[i, dict_areas_inds[key_j]].shape)
                df.at[conn_oh.region_labels[i], key_j_k] = np.average(conn_oh.weights[i, dict_areas_inds[key_j]],
                                                                      axis=0, weights=flat_weights)
                df.at[key_j_k, conn_oh.region_labels[i]] = np.average(conn_oh.weights[dict_areas_inds[key_j], i],
                                                                      axis=0, weights=flat_weights)

            if i not in inds_to_merge and j not in inds_to_merge:
                # print('ENTRATO 4')
                # print('Key i', key_i)
                # print('Conn oh: region labels[j]', conn_oh.region_labels[j])
                df.at[key_i_k, conn_oh.region_labels[j]] = conn_oh.weights[i, j]
                df.at[conn_oh.region_labels[j], key_i_k] = conn_oh.weights[j, i]

    print(df)
    df.to_csv('final_conn.csv', index=True)
