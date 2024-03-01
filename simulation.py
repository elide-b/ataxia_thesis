import numpy as np
from tvb.simulator import noise
from tvb.simulator.coupling import Linear, Scaling
from tvb.simulator.integrators import EulerDeterministic, HeunStochastic
from tvb.simulator.models.wong_wang import ReducedWongWang
from tvb.simulator.monitors import Bold, TemporalAverage
from tvb.simulator.simulator import Simulator

from rww_exc_inh.rww_model_mary import ReducedWongWangExcIOInhI


def simulate(
        brain,
        simlen,
        dt=0.1,
        I_o=0.3,
        w=1.0,
        stoc=True,
        sigma=0.015,
        coupling="scaling",
        G=0.11,
        bold_res=10000,
        tavg_res=10,
):
    simulator = Simulator()
    simulator.connectivity = brain
    simulator.model = ReducedWongWang(w=np.array(w), I_o=np.array(I_o))
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
        simulator.integrator.noise = noise.Additive(
            nsig=np.array([(sigma ** 2) / 2]), noise_seed=1
        )
    else:
        simulator.integrator = EulerDeterministic(dt=dt)
    mon_tavg = TemporalAverage(period=dt * tavg_res)
    mon_bold = Bold(period=dt * bold_res)
    simulator.monitors = (mon_tavg, mon_bold)
    simulator.configure()
    (t1, tavg), (t2, bold) = simulator.run(simulation_length=simlen)

    return (t1, tavg[:, 0, :, 0].T), (t2, bold[:, 0, :, 0].T)


def simulate_modified(brain, a, J_i, w_p, T):
    rww = ReducedWongWangExcIOInhI(J_i=np.array(J_i), w_p=np.array(w_p))
    sim = Simulator(
        model=rww,
        connectivity=brain,
        coupling=Scaling(a=np.array(a)),
        monitors=(TemporalAverage(period=1.0),
                  Bold(period=T)),
        integrator=HeunStochastic(
            noise=noise.Additive(nsig=np.array([0.015 ** 2 / 2])),
            dt=0.1
        ),
    )
    sim.initial_conditions = np.random.uniform(0.0, 0.2, size=(2, 4, len(brain.weights), 1))
    sim.configure()
    (t1, tavg), (t2, bold) = sim.run(simulation_length=T)

    return (t1, tavg), (t2, bold)
