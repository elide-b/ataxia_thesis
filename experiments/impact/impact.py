import numpy as np

from ataxia.experiment import ConstantParameter, Experiment, LinspaceParameter, Result
from ataxia.connectivity import load_mousebrain
from ataxia.simulation import simulate

# Mouse brain
brain = load_mousebrain("allen84.zip", norm="pct")
nreg = len(brain.weights)

# Time variables
dt = 0.1
T = 1000
N = int(T / dt)
# Monitor resolution
tavg_res = 10
bold_res = 10000
N_tavg = int(N / tavg_res)
N_bold = int(N / bold_res)

print("tavg?", N_tavg)

experiment = Experiment(
    "impact",
    k=ConstantParameter(1.0),
    g=LinspaceParameter(1.0, 5.0, 5),
    tavg=Result((nreg, N_tavg)),
    bold=Result((nreg, N_bold)),
)

for ik, k in experiment.k:
    for ig, g in experiment.g:
        print(f"Simulating (k={k},g={g})")
        (t1, _tavg), (t2, _bold) = simulate(
            brain, T, dt=dt, k=k, gamma=g, bold_res=bold_res, tavg_res=tavg_res
        )
        experiment.tavg = _tavg
        experiment.bold = _bold

experiment.save(brain=brain, T=T, dt=dt, tavg_res=tavg_res, bold_res=bold_res)
