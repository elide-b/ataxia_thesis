import itertools

import numpy as np
import os
from connectivity import load_mousebrain
from plots import plot_weights, plot_matrix
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Loading the mouse brain scaled by region and with a logarithmic normalization
brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")
m_brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")

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
assert len(cc_ids) == len(cc_labels) * 2, \
    f"Expected {len(cc_labels) * 2} cereb cortex labels, but found {len(cc_ids)}"

cn_ids = [i for i in range(nreg) if any(b in m_brain.region_labels[i] for b in cn_labels)]
assert len(cn_ids) == len(cn_labels) * 2, \
    f"Expected {len(cn_labels) * 2} cereb cortex labels, but found {len(cn_ids)}"

debug_weights = np.full(m_brain.weights.shape, np.nan)
debug_labels = np.full(m_brain.weights.shape, None, dtype=object)
# setting the weights of the input weights to the cerebellar nuclei to negative
for id in cc_ids:
    debug_weights[cn_ids, id] = -m_brain.weights[cn_ids, id]
    m_brain.weights[cn_ids, id] *= -1
    debug_labels[cn_ids, id] = [
        f"({cn}, {id} = {m_brain.weights[cn, id]}): {brain.region_labels[cn]} ← {brain.region_labels[id]}" for cn in
        cn_ids]

tc_ids = cc_ids + cn_ids

heatmap_mask = plot_matrix(debug_weights, zmin=0.00135, zmax=0.002, show=False).data[0]
# plotting the normal weights, masked weights, and the difference between them
fig = plot_weights(brain, zmin=0.00135, zmax=0.002, show=False)
fig.update_traces(opacity=0.3)
heatmap = fig.data[0]


fig = go.Figure([heatmap, heatmap_mask])
fig.update_traces(showlegend=True)
fig.update_xaxes(title="From")
fig.update_yaxes(title="To")
fig.show()
# plot_weights(m_brain, zmin=0.00135, zmax=0.002).write_html("masked_weights.html")
# diff = m_brain.weights - brain.weights
# plot_matrix(diff).write_html("diff.html")
# sub = m_brain.weights[tc_ids, :]
# plot_matrix(sub, zmin=0.00135, zmax=0.002).write_html("submatrix.html")


np.savez("masked_brain.npz", connectivity=m_brain, weights=m_brain.weights, centers=m_brain.centres,
         tract_lengths=m_brain.tract_lengths, region_labels=m_brain.region_labels, cc_ids=cc_ids, cn_ids=cn_ids)
