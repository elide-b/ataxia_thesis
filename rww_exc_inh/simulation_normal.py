import numpy as np
from tvb.simulator.lab import *
from connectivity import load_mousebrain
from rww_model_mary import *
from plots import plot_weights

# Loading mouse brain
brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")
nreg = len(brain.weights)
# plot_weights(brain).write_html("norm_brain_weights.html")

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
data_tavg = np.empty((nJi, na, 4, nreg, N_tavg), dtype=float)


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
        print(f"Simulating (a= {a},J_i= {Ji})")
        (t, _tavg), = simulate(a=a, J_i=Ji)
        data_tavg[ia, iJi] = np.swapaxes(_tavg, 0, 3)


np.savez("firing_rate_normal.npz", tavg=_tavg, time=t, a=a_range, J_i=Ji_range)

# TODO: plot 3D del variare della media dei rate al variare dei parametri con colori diversi per diverse combinazioni
#  di parametri sul cluster -> simulazioni a 20 minuti per BOLD letteratura per capire come variare i parametri,
#  resting state networks literature + firing rates di TVB!!! plot nodi cerebellari sia healthy che masked (pesi
#  prima log e poi moltiplico per -1)

# TODO: PER DOMANI
# grafici connettività + nuove equazioni e cosa abbiamo cambiato
# plot cc e cn e cambiare barra per colori
# grafici firing rate con colori diversi
# network Roberta per far vedere cosa modificare del mean field (diminuire # PC o modificare K e Q) -> chiedere
# RSN trovate + per mouse
# articolo computational ataxia (usare valori caso sano)
