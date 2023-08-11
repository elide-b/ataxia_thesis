import numpy as np

from ataxia.connectivity import load_mousebrain
from ataxia.simulation import simulate

# Mouse brain
mb_allen84 = load_mousebrain("allen84.zip")
nreg = len(mb_allen84.weights)

# Time variables
dt = 0.1
T = 12000
N = int(T / dt)
# Tavg is sampled per 10 time instants, BOLD per 10000
N_tavg = int(N / 10)
N_bold = int(N / 10000)
# Sweep variables
nk, ng = 3, 3
k_range = np.linspace(0.0, 1.0, nk)
g_range = np.linspace(0.0, 1.0, ng)
# Output matrices
data_tavg = np.empty((nk, ng, nreg, N_tavg), dtype=float)
data_bold = np.empty((nk, ng, nreg, N_bold), dtype=float)

for ik, k in enumerate(k_range):
    for ig, g in enumerate(g_range):
        print(f"Simulating (k={k},g={g})")
        (t1, _tavg), (t2, _bold) = simulate(mb_allen84, T, dt=dt, k=k, gamma=g, wholebrain=True)
        data_tavg[ik, ig] = _tavg
        data_bold[ik, ig] = _bold

np.savez("global_ataxia.npz", tavg=data_tavg, bold=data_bold)
