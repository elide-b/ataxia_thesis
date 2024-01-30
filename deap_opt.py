import numpy as np

from connectivity import load_mousebrain
from simulation import simulate_modified

from super_awesome_deap.optimization import optimize
from super_awesome_deap.tools import Individual, Param

brain = load_mousebrain("Connectivity_596.h5", norm="log", scale="region")


class Mousebrain(Individual):
    w = Param(0, 2)
    Ji = Param(0.001, 2)
    G = Param(0, 10)


def evaluate(mousebrain):
    (t1, tavg), (t2, bold) = simulate_modified(
        brain, mousebrain.G, mousebrain.Ji, mousebrain.w, T=2
    )
    return (float(np.average(tavg)),)


optimize(Mousebrain, evaluate)
