import numpy as np

from super_awesome_grid_search import grid_search


def evaluate(x):
    return np.interp(x, *f)


grid_search(evaluate, 0, 10, 10)
