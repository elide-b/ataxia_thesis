import numpy as np
from tvb.simulator.lab import *

from connectivity import load_mousebrain
from plots import plot_weights
from rww_model_mary import *

# Loading mouse brain
brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")
nreg = len(brain.weights)
plot_weights(brain).write_html("norm_brain_weights.html")

# Time variables
dt = 0.1
T = 500
N = int(T / dt)
# Tavg is sampled per 10 time instants, BOLD per 10000
N_tavg = int(N / 10)

# Sweeping parameters
nJi = 5
na = 5
Ji_range = np.linspace(1e-5, 5, nJi)
a_range = np.linspace(0.01, 0.5, na)
data_tavg = np.empty((na, nJi, 4, nreg, N_tavg), dtype=float)


def simulate(a, J_i):
    rww = ReducedWongWangExcIOInhI(J_i=np.array(J_i), w_p=np.array(1), W_i=np.array(0.5))
    sim = simulator.Simulator(
        model=rww,
        connectivity=brain,
        coupling=coupling.Scaling(a=np.array(a)),
        monitors=(monitors.TemporalAverage(period=1.0),),
        integrator=integrators.HeunStochastic(
            noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
            dt=0.1
        ),
    )
    sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, nreg, 1))
    sim.configure()
    (t1, tavg), = sim.run(simulation_length=T)

    return (t1, tavg),


for ia, a in enumerate(a_range):
    for iJi, Ji in enumerate(Ji_range):
        print(f"Simulating (a = {a},J_i = {Ji})")
        (t, _tavg), = simulate(a=a, J_i=Ji)
        data_tavg[ia, iJi] = np.swapaxes(_tavg, 0, 3)

np.savez("firing_rate_normal.npz", tavg=data_tavg, time=t, a_range=a_range, Ji_range=Ji_range)
