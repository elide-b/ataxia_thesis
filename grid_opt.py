from one_sim import one_simulation
from super_awesome_grid_search import grid_search


def evaluate(G):
    pp, kd = one_simulation(G=G, simlen=3000, dt=1)
    return kd


grid_search(evaluate, 0, 10, 10)
