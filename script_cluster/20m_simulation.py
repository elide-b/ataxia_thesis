import numpy as np
from tvb.simulator.lab import *
from ataxia.connectivity import load_mousebrain
from rww_exc_inh.rww_model_mary import *
from ataxia.plots import plot_weights

"""Codice per la simulazione da mandare sul cluster da 20 minuti per avere i BOLD"""

# Loading mouse brain
data_dir = os.path.dirname(__file__) + "\\.."
brain = load_mousebrain(data_dir, "Connectivity_596.h5", norm="log", scale="region")
nreg = len(brain.weights)
# plot_weights(brain).write_html("norm_brain_weights.html")

# Time variables
dt = 0.1
T = 500
N = int(T / dt)
N_tavg = int(N / 10)

# Optimized parameters found
a =
J_i =
w_p =
W_i =


def simulate(a, J_i, w_p, W_i):
    rww = ReducedWongWangExcIOInhI(J_i=np.array(J_i), w_p=np.array(w_p), W_i=np.array(W_i))
    sim = simulator.Simulator(
        model=rww,
        connectivity=brain,
        coupling=coupling.Scaling(a=np.array(a)),
        monitors=(monitors.TemporalAverage(period=1.0),
                  monitors.Bold(period=  )), # TODO: define a period for the BOLD
        integrator=integrators.HeunStochastic(
            noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
            dt=0.1
        ),
    )
    sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, nreg, 1))
    sim.configure()
    (t1, tavg), (t2, bold) = sim.run(simulation_length=T)

    return (t1, tavg), (t2, bold)

(t1, tavg), (t2, bold) = simulate(a=a, J_i=J_i, w_p=w_p, W_i=W_i)