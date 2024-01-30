import numpy as np

from connectivity import load_mousebrain
from simulation import simulate_modified

from mpipool import MPIExecutor

from super_awesome_deap.optimization import optimize

brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")

def evaluate(individual):
    (t1, tavg), (t2, bold) = simulate_modified(
        brain, individual.G, individual.Ji, individual.w, T=2
    )
    print("EVAL", float(np.average(tavg)))
    return (float(np.average(tavg)),)

optimize(evaluate)
