import numpy as np
from tvb.simulator.lab import *
from ataxia.connectivity import load_mousebrain
from rww_model_mary import *
from ataxia.plots import plot_weights
from tvb.datatypes.connectivity import Connectivity

# Mouse brain
with np.load("masked_brain.npz") as brain:
    weights = brain['weights']
    centers = brain['centers']
    tract_lengths = brain['tract_lengths']
    region_labels = brain['region_labels']

connectivity = Connectivity(weights=weights, centres=centers, tract_lengths=tract_lengths, region_labels=region_labels)
nreg = len(connectivity.region_labels)

# plot_weights(connectivity).write_html("brain_weights.html")

rww = ReducedWongWangExcIOInhI(J_i=np.array(10))
sim = simulator.Simulator(
    model=rww,
    connectivity=connectivity,
    coupling=coupling.Scaling(a=np.array(0.5)),  # il G è a qui
    monitors=(monitors.TemporalAverage(period=1.0),),
    integrator=integrators.HeunStochastic(
        noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
        dt=0.1
    ),
)

sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, nreg, 1))
sim.configure()

(t, data), = sim.run(simulation_length=1000)

np.savez("firing_rate.npz", tavg=data, time=t)

# simulazione anche per il brain normale
# effetto sulle aree più connesse ai DCN -> se non c'è impatto aumentiamo G su aree cerebellari (ma evitiamo)
# resting state networks literature + firing rates di TVB
# selezione subnetwork e cambio G per ottimizzare su computational model (finchè non abbiamo i dati)
# TESI: TVMB -> Allen normalizzazione e maschera, simulazioni e ottimizzazione parametri per avoid saturation in model (G, J_i), resting state networks
# ottimizzazione tridimensionale G, J_i, rate
# fare log transformation perchè alcuni nodi sono troppo firing e verificare quali sono
# cambiare G con J_i normale e prova dei firing del brain no mask
