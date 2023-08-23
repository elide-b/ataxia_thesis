import numpy as np
from tvb.simulator import noise
from tvb.simulator.coupling import Linear, Scaling
from tvb.simulator.integrators import EulerDeterministic, HeunStochastic
from tvb.simulator.models.wong_wang import ReducedWongWang
from tvb.simulator.monitors import Bold, TemporalAverage
from tvb.simulator.simulator import Simulator

from ataxia.disease import ataxic_w, ataxic_weights


def simulate(
    brain,
    simlen,
    dt=0.1,
    I_o=0.3,
    w=1.0,
    k=1.0,
    gamma=1.0,
    stoc=True,
    sigma=0.015,
    coupling="scaling",
    G=0.11,
    wholebrain=False,
    bold_res=10000,
    tavg_res=10
):
    simulator = Simulator()
    simulator.connectivity = brain
    with ataxic_weights(brain, gamma, wholebrain=wholebrain):
        w = ataxic_w(brain, w, k, wholebrain=wholebrain)
        simulator.model = ReducedWongWang(w=w, I_o=np.array(I_o))
        simulator.model.dt = dt
        if coupling == "scaling":
            simulator.coupling = Scaling(a=np.array(G))
        elif coupling == "linear":
            simulator.coupling = Linear()
        else:
            raise ValueError(f"Unknown coupling mode '{coupling}'")
        simulator.initial_conditions = (0.001) * np.ones((1, 1, len(brain.weights), 1))
        if stoc:
            simulator.integrator = HeunStochastic(dt=dt)
            simulator.integrator.noise = noise.Additive(nsig=np.r_[(sigma**2) / 2])
        else:
            simulator.integrator = EulerDeterministic(dt=dt)
        mon_tavg = TemporalAverage(period=dt * tavg_res)
        mon_bold = Bold(period=dt * bold_res)
        simulator.monitors = (mon_tavg, mon_bold)
        simulator.configure()
        (t1, tavg), (t2, bold) = simulator.run(simulation_length=simlen)

    return (t1, tavg[:, 0, :, 0].T), (t2, bold[:, 0, :, 0].T)
