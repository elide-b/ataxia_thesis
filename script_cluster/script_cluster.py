import numpy as np
from tvb.simulator.lab import *
from ataxia.connectivity import load_mousebrain
from rww_exc_inh.rww_model_mary import *
from ataxia.plots import plot_weights

"""Script di sweep delle variabili a, J_i, w_p, e W_i per ottimizzazione nel caso sano"""

# Loading mouse brain
data_dir = os.path.dirname(__file__) + "\\.."
# Log normalization and scaled by region
brain = load_mousebrain(data_dir, "Connectivity_596.h5", norm="log", scale="region")
nreg = len(brain.weights)
plot_weights(brain).write_html("brain_weights.html")

# Time variables
dt = 0.1
T = 500
N = int(T / dt)
N_tavg = int(N / 10)

# Sweeping parameters
nJi = 5
na = 5
nwp = 5
nwi = 5
Ji_range = np.linspace(0.08, 0.12, nJi)
a_range = np.linspace(1e-7, 1e-3, na)
wp_range = np.linspace(0.8, 1.2, nwp)
wi_range = np.linspace(0.3, 0.7, nwi)
data_tavg = np.empty((nJi, na, nwp, nwi, 4, nreg, N_tavg), dtype=float)


def simulate(a, J_i, w_p, W_i):
    rww = ReducedWongWangExcIOInhI(J_i=np.array(J_i), w_p=np.array(w_p), W_i=np.array(W_i))
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
        for iwp, wp in enumerate(wp_range):
            for iwi, wi in enumerate(wi_range):
                print(f"Simulating for values a={a}, J_i={Ji}, w_p={wp}, W_i={wi}")
                (t, _tavg), = simulate(a=a, J_i=Ji, w_p=wp, W_i=wi)
                data_tavg[ia, iJi, iwp, iwi] = np.swapaxes(_tavg, 0, 3)

np.savez("opt_parameters_healthy.npz", tavg=_tavg, time=t, a=a_range, J_i=Ji_range, w_p=wp_range, W_i=wi_range)
