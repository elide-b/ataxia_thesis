import numpy as np
from ataxia.plots import plot_matrix
from tvb.datatypes.connectivity import Connectivity

# Load mouse brain
with np.load("masked_brain.npz") as brain:
    weights = brain['weights']
    centers = brain['centers']
    tract_lengths = brain['tract_lengths']
    region_labels = brain['region_labels']
    cc_labels = brain['cc_labels']
    cn_labels = brain['cn_labels']

cn_labels = int(cn_labels)
cc_labels = int(cc_labels)
connectivity = Connectivity(weights=weights, centres=centers, tract_lengths=tract_lengths, region_labels=region_labels)
nreg = len(connectivity.region_labels)
regions = connectivity.weights[cn_labels, cc_labels]

plot_matrix(regions).write_html("brain_weights_masked.html")