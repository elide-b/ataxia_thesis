import numpy as np
from matplotlib import pyplot as plt

from connectivity import load_mousebrain
import pandas as pd

conn_oh = load_mousebrain("Connectivity_02531729cb0d470d9a62dcff9158a952.h5", norm=False, scale=False)
np.fill_diagonal(conn_oh.tract_lengths, 0.)

GOZZI_LABELS_FILE = '../dataset fMRI/Data_to_share/macro_atlas_n_172_rois_excel_final.xlsx'
reg_labels_gozzi = list(pd.read_excel(io=GOZZI_LABELS_FILE)['NAME'])
reg_labels_gozzi.append('Caudoputamen')
regions_labels_gozzi = ['Right ' + reg for reg in reg_labels_gozzi] + ['Left ' + reg for reg in reg_labels_gozzi]

df_voxel = pd.read_csv('voxel.csv')

# Correct for Posterior parietal association areas voxel count = 0
list_to_correct = ['Right Rostrolateral visual area', 'Right Anterior area', 'Left Rostrolateral visual area',
                   'Left Anterior area']
for reg in list_to_correct:
    df_voxel.loc[df_voxel['Regions'] == reg, 'Voxel Count'] = 1749.0

dict_areas = {
    "Ammon's horn": ['Field CA1', 'Field CA2', 'Field CA3'],
    "Entorhinal area": ['Entorhinal area, lateral part', 'Entorhinal area, medial part, dorsal zone'],
    "Hypothalamic lateral zone": ['Lateral hypothalamic area', 'Lateral preoptic area', 'Subthalamic nucleus',
                                  'Preparasubthalamic nucleus', 'Zona incerta', 'Parasubthalamic nucleus',
                                  'Tuberal nucleus', 'Perifornical nucleus', 'Retrochiasmatic area'],
    "Pallidum, medial region": ['Medial septal nucleus', 'Diagonal band nucleus', 'Triangular nucleus of septum'],
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
    "Periventricular zone": ['Accessory supraoptic group', 'Paraventricular hypothalamic nucleus',
                             # supraoptic nucleus non è in conn_oh
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
                               # subfornical organ is not in conn_oh
                               'Ventromedial preoptic nucleus', 'Ventrolateral preoptic nucleus'],
    "Orbital area": ['Orbital area, lateral part', 'Orbital area, medial part', 'Orbital area, ventrolateral part'],
    "Thalamus, polymodal association cortex related": ['Reticular nucleus of the thalamus',
                                                       'Intergeniculate leaflet of the lateral geniculate complex',
                                                       'Intermediate geniculate nucleus',
                                                       'Ventral part of the lateral geniculate complex',
                                                       'Rhomboid nucleus',  # subgeniculate nucleus non è in conn_oh
                                                       'Central medial nucleus of the thalamus', 'Paracentral nucleus',
                                                       'Central lateral nucleus of the thalamus',
                                                       'Parafascicular nucleus',
                                                       'Posterior intralaminar thalamic nucleus',
                                                       'Paraventricular nucleus of the thalamus', 'Parataenial nucleus',
                                                       'Nucleus of reuniens', 'Xiphoid thalamic nucleus',
                                                       'Intermediodorsal nucleus of the thalamus',
                                                       'Mediodorsal nucleus of thalamus',
                                                       'Submedial nucleus of the thalamus', 'Perireunensis nucleus',
                                                       'Anteroventral nucleus of thalamus', 'Anteromedial nucleus',
                                                       # non diviso in ventral e dorsal
                                                       'Anterodorsal nucleus',
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
                                               'Medial geniculate complex',
                                               # non diviso in dorsal, medial e ventral in conn_oh
                                               'Dorsal part of the lateral geniculate complex'],
    "Lateral septal complex": ['Lateral septal nucleus, caudal (caudodorsal) part',
                               'Lateral septal nucleus, rostral (rostroventral) part',
                               'Lateral septal nucleus, ventral part', 'Septofimbrial nucleus'],
    # anche septohippocampal nucleus non è in conn_oh.region_labels
    "Hypothalamic medial zone": ['Anterior hypothalamic nucleus', 'Supramammillary nucleus',
                                 'Medial mammillary nucleus', 'Lateral mammillary nucleus',
                                 'Tuberomammillary nucleus, dorsal part', 'Tuberomammillary nucleus, ventral part',
                                 'Medial preoptic nucleus', 'Dorsal premammillary nucleus',
                                 'Ventral premammillary nucleus',
                                 'Paraventricular hypothalamic nucleus, descending division',
                                 'Ventromedial hypothalamic nucleus', 'Posterior hypothalamic nucleus'],
    "Midbrain, sensory related": ['Inferior colliculus', 'Superior colliculus, sensory related',
                                  # inferior colliculus non è diviso in conn_oh
                                  'Nucleus of the brachium of the inferior colliculus', 'Nucleus sagulum',
                                  'Parabigeminal nucleus', 'Midbrain trigeminal nucleus', 'Subcommissural organ'],
    "Striatum-like amygdalar nuclei": ['Central amygdalar nucleus', 'Anterior amygdalar area',
                                       'Bed nucleus of the accessory olfactory tract', 'Intercalated amygdalar nucleus',
                                       'Medial amygdalar nucleus'],
    "Pallidum, ventral region": ['Substantia innominata', 'Magnocellular nucleus'],
    "Posterior parietal association areas": ['Rostrolateral visual area', 'Anterior area']
}

final_dict_areas = {}
for key, list in dict_areas.items():
    new_list_left = [f'Left {reg}' for reg in list]
    new_list_right = [f'Right {reg}' for reg in list]
    key_right = 'Right ' + key
    key_left = 'Left ' + key
    final_dict_areas[key_right] = new_list_right
    final_dict_areas[key_left] = new_list_left

labels = conn_oh.region_labels
centres = conn_oh.centres

for key, list in final_dict_areas.items():
    regs_to_merge = [reg for reg in list]
    inds_to_merge = [np.where(labels == reg)[0] for reg in list]
    inds_to_merge_oh = [np.where(conn_oh.region_labels == reg)[0] for reg in list]

    # merge the labels
    labels = np.delete(labels, inds_to_merge, axis=0)
    labels = np.insert(labels, np.min(inds_to_merge), key, axis=0)

    # delete the axes
    centres = np.delete(centres, inds_to_merge, axis=0)

    # build new axis of merged regions
    new0 = np.squeeze(conn_oh.centres[inds_to_merge_oh])
    weights = [df_voxel.loc[df_voxel['Regions'] == reg, 'Voxel Count'].values for reg in regs_to_merge]
    new_0 = np.nansum(new0 * weights, axis=0) / np.nansum(weights, axis=0)

    # insert the axis
    centres = np.insert(centres, np.min(inds_to_merge), new_0, axis=0)

print('Shape centres', centres.shape)
print('Len labels', len(labels))
np.savetxt('centres_oh_merged.txt', centres)
np.savetxt('region_labels_oh_merged.txt', labels, fmt='%s')


# Build new tracts from the new centres
n_regs = len(labels)
tract_lengths = np.zeros((n_regs, n_regs))


def distance(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)


for i in range(n_regs):
    for j in range(i + 1, n_regs):
        lt = distance(centres[i], centres[j])
        tract_lengths[i, j] = lt
        tract_lengths[j, i] = lt

plt.imshow(tract_lengths)
plt.colorbar()
plt.show()


# for only gozzi regions
print('For Gozzi')

inds_gozzi = [i for i, reg in enumerate(labels) if reg in regions_labels_gozzi]
reg_gozzi = [reg for i,reg in enumerate(labels) if reg in regions_labels_gozzi]
labels_gozzi = labels[inds_gozzi]
np.savetxt('labels_gozzi.txt', labels_gozzi, fmt='%s')
centres_gozzi = centres[inds_gozzi, :]
np.savetxt('centres_gozzi.txt', centres_gozzi)
tract_gozzi = tract_lengths[np.ix_(inds_gozzi, inds_gozzi)]
np.savetxt('tract_lengths_gozzi.txt', tract_gozzi)
