import numpy as np

from ataxia.connectivity import load_mousebrain
from ataxia.simulation import simulate

# Mouse brain
brain = load_mousebrain("Connectivity_596.h5", norm=False, scale=False)
nreg = len(brain.weights)

# Time variables
dt = 0.1
T = 100
N = int(T / dt)
# Sweep variables
nk, ng = 2, 2
k_range = np.linspace(0.0, 1.0, nk)
g_range = np.linspace(0.0, 1.0, ng)
# Output matrices, created later
data_tavg = None
data_bold = None

for ik, k in enumerate(k_range):
    for ig, g in enumerate(g_range):
        print(f"Simulating (k={k},g={g})")
        (t1, _tavg), (t2, _bold) = simulate(brain, T, dt=dt, k=k, gamma=g, G=0.02, wholebrain=True)
        if data_tavg is None:
            # Make data matrices for the returned data, we don't know before sim how long they will be.
            data_tavg = np.empty((nk, ng, *_tavg.shape), dtype=float)
            data_bold = np.empty((nk, ng, *_bold.shape), dtype=float)
        data_tavg[ik, ig] = _tavg
        data_bold[ik, ig] = _bold

np.savez("global_ataxia.npz", tavg=data_tavg, bold=data_bold)

import ataxia_596