import numpy as np

from connectivity import load_mousebrain
from simulation import simulate

# Mouse brain
brain = load_mousebrain("Connectivity_596.h5", norm=False, scale=False)
nreg = len(brain.weights)

# Time variables
dt = 0.1
T = 100
N = int(T / dt)
# Tavg is sampled per 10 time instants, BOLD per 10000
N_tavg = int(N / 10)
N_bold = int(N / 10)
# Sweep variables
nk, ng = 2, 2
k_range = np.linspace(0.0, 1.0, nk)
g_range = np.linspace(0.0, 1.0, ng)
# Output matrices
data_tavg = np.empty((nk, ng, nreg, N_tavg), dtype=float)
data_bold = np.empty((nk, ng, nreg, N_bold), dtype=float)

for ik, k in enumerate(k_range):
    for ig, g in enumerate(g_range):
        print(f"Simulating (k={k},g={g})")
        (t1, _tavg), (t2, _bold) = simulate(brain, T, dt=dt, k=k, gamma=g, G=0.02, wholebrain=True)
        data_tavg[ik, ig] = _tavg
        data_bold[ik, ig] = _bold

np.savez("global_ataxia.npz", tavg=data_tavg, bold=data_bold)
