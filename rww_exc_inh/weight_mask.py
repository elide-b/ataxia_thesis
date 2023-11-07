import numpy as np
import os
from ataxia.connectivity import load_mousebrain
from ataxia.plots import plot_weights, plot_matrix

# Loading the mouse brain scaled by region and with a logarithmic normalization
data_dir = os.path.dirname(__file__) + "\\.."
brain = load_mousebrain(data_dir, "Connectivity_596.h5", norm="log", scale="region")
m_brain = load_mousebrain(data_dir, "Connectivity_596.h5", norm="log", scale="region")

# Finding the indexes of the cerebellar cortex and cerebellar nuclei
cc_labels = [
    "Lingula (I)",
    "Central lobule",
    "Culmen",
    "Declive (VI)",
    "Folium-tuber vermis (VII)",
    "Pyramus (VIII)",
    "Uvula (IX)",
    "Nodulus (X)",
    "Simple lobule",
    "Ansiform lobule",
    "Paramedian lobule",
    "Copula pyramidis",
    "Paraflocculus",
    "Flocculus"
]

cn_labels = ["Interposed nucleus",
             "Dentate nucleus",
             "Fastigial nucleus",
             "Vestibulocerebellar nucleus"
]

nreg = len(m_brain.weights)
cc_ids = [i for i in range(nreg) if any(b in m_brain.region_labels[i] for b in cc_labels)]
assert len(cc_ids) == len(cc_labels) * 2,\
    f"Expected {len(cc_labels) * 2} cereb cortex labels, but found {len(cc_ids)}"

cn_ids = [i for i in range(nreg) if any(b in m_brain.region_labels[i] for b in cn_labels)]
assert len(cn_ids) == len(cn_labels) * 2,\
    f"Expected {len(cn_labels) * 2} cereb cortex labels, but found {len(cn_ids)}"

# setting the weights of the input weights to the cerebellar nuclei to negative
for id in cc_ids:
    m_brain.weights[cn_ids, id] *= -1


# plotting the normal weights, masked weights, and the difference between them
plot_weights(brain).write_html("normal_weights.html")
plot_weights(m_brain).write_html("masked_weights.html")
diff = m_brain.weights - brain.weights
plot_matrix(diff).write_html("diff.html")

np.savez("masked_brain.npz", connectivity=m_brain, weights=m_brain.weights, centers=m_brain.centres, tract_lengths=m_brain.tract_lengths, region_labels=m_brain.region_labels)