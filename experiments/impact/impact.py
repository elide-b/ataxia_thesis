from ataxia.connectivity import load_mousebrain
from ataxia.experiment import (
    BoolParameter,
    ConstantParameter,
    Experiment,
    LinspaceParameter,
    Result,
)
from ataxia.simulation import simulate

# Mouse brain
brain = load_mousebrain("allen84.zip")
nreg = len(brain.weights)

# Time variables
dt = 0.1
T = 120000
N = int(T / dt)
# Monitor resolution
tavg_res = 10
bold_res = 10
N_tavg = int(N / tavg_res)
N_bold = int(N / bold_res)

experiment = Experiment(
    "impact",
    from_to=BoolParameter(),
    k=ConstantParameter(1.0),
    g=LinspaceParameter(1.0, 50.0, 11),
    tavg=Result((nreg, N_tavg)),
    bold=Result((nreg, N_bold)),
)

for from_to in experiment.from_to:
    for ik, k in experiment.k:
        for ig, g in experiment.g:
            print(f"Simulating (k={k},g={g})")
            (t1, _tavg), (t2, _bold) = simulate(
                brain,
                T,
                dt=dt,
                k=k,
                gamma=g,
                bold_res=bold_res,
                tavg_res=tavg_res,
                from_to=from_to,
            )
            experiment.tavg = _tavg
            experiment.bold = _bold

experiment.save(brain=brain, T=T, dt=dt, tavg_res=tavg_res, bold_res=bold_res)
