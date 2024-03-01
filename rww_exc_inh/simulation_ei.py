import numpy as np
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.lab import *

from plots import plot_weights
from rww_model_mary import *

# Mouse brain loading
with np.load("masked_brain.npz") as brain:
    weights = brain['weights']
    centers = brain['centers']
    tract_lengths = brain['tract_lengths']
    region_labels = brain['region_labels']

connectivity = Connectivity(weights=weights, centres=centers, tract_lengths=tract_lengths, region_labels=region_labels)
nreg = len(connectivity.region_labels)
plot_weights(connectivity).write_html("brain_weights_masked.html")

rww = ReducedWongWangExcIOInhI(J_i=np.array(10))
sim = simulator.Simulator(
    model=rww,
    connectivity=connectivity,
    coupling=coupling.Scaling(a=np.array(0.5)),  # the synaptic coupling 'G' is called 'a' in this function
    monitors=(monitors.TemporalAverage(period=1.0),),
    integrator=integrators.HeunStochastic(
        noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
        dt=0.1
    ),
)

sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, nreg, 1))
sim.configure()

(t, data), = sim.run(simulation_length=1000)

np.savez("firing_rate_masked.npz", tavg=data, time=t)
