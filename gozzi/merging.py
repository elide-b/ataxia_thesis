import sys

sys.path.append('/home/neuro_sim2/tvb/better_tesi')

import os
from collections import OrderedDict
import time
import numpy as np
from connectivity import load_mousebrain

# conn_oh_path='/home/neuro_sim2/tvb/better_tesi/mouse_brains/dataset/h5'
conn_oh = load_mousebrain("Connectivity_596.h5", norm=False, scale=False)

# manca Hippocampo-amygdalar transition area
dict_areas = {
    "Ammon's horn": ['Field CA1', 'Field CA2', 'Field CA3'],
    "Lateral hypothalamic zone": ['Lateral hypothalamic area', 'Lateral preoptic area', 'Subthalamic nucleus',
                                  'Preparasubthalamic nucleus',
                                  'Zona Incerta', 'Parasubthalamic nucleus', 'Tuberal nucleus', 'Perifornical nucleus',
                                  'Retrochiasmatic area'],
    "Pallidum, median region": ['Medial septal nucleus', 'Diagonal band nucleus', 'Triangular nucleus of septum'],
    "Entorhinal area": ['Entorhinal area, lateral part', 'Entorhinal area, medial part, dorsal zone ',
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
                                'Dorsal terminal nucleus of the accessory optic tract',
                                'Ventral terminal nucleus of the accessory optic tract'],
    "Periventricular zone": ['Supraoptic nucleus', 'Accessory supraoptic group', 'Paraventricular hypothalamic nucleus',
                             'Periventricular hypothalamic nucleus, anterior part',
                             'Periventricular hypothalamic nucleus, intermediate part', 'Arcuate hypothalamic nucleus'],
    "Pallidum, dorsal region": ['Globus pallidus, external segment', 'Globus pallidus, internal segment'],
    "Cortical amygdalar area": ['Cortical amygdalar area, anterior part', 'Cortical amygdalar area, posterior part'],
    "Periventricular region": ['Anterodorsal preoptic nucleus', 'Anterior hypothalamic area',
                               'Anteroventral preoptic nucleus', 'Anteroventral periventricular nucleus',
                               'Dorsomedial nucleus of the hypothalamus', 'Median preoptic nucleus',
                               'Medial preoptic area', 'Vascular organ of the lamina terminalis',
                               'Posterodorsal preoptic nucleus', 'Parastrial nucleus', 'Suprachiasmatic nucleus',
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
    "Lateral septal complex": ['Lateral septal nucleus, caudal (caudodorsal) part', 'Lateral septal nucleus',
                               'rostral (rostroventral) part', 'Lateral septal nucleus, ventral part',
                               'Septofimbrial nucleus', 'Septohippocampal nucleus'],
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
    "Pallidum, ventral region": ['Substantia innominata', 'Magnocellular nucleus']

}

dict_areas_inds_l = {}
dict_areas_inds_r = {}

for reg_gozzi, regions in dict_areas.items():
    regs_l = ['Left ' + r for r in regions]
    regs_r = ['Right ' + r for r in regions]
    dict_areas_inds_l[reg_gozzi] = [i for i, labels in enumerate(conn_oh.region_labels) if labels in regs_l]
    dict_areas_inds_r[reg_gozzi] = [i for i, labels in enumerate(conn_oh.region_labels) if labels in regs_r]

print(f'Left inds to be merged: {dict_areas_inds_l}')
# print(f'Right inds to be merged: {dict_areas_inds_r}')

