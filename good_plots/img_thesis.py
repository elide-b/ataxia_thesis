from connectivity import load_mousebrain
import numpy as np
from plots import plot_weights, plot_matrix
import matplotlib.pyplot as plt

"""Script to get all of the images I need for my thesis"""

brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")
nreg = len(brain.weights)
np.fill_diagonal(brain.weights, 0.)
brain.weights /= np.max(brain.weights)
#plot_weights(brain).write_html("SC_Allen.html")

