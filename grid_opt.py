import numpy as np
import plotly.graph_objs as go

from super_awesome_grid_search import grid_search

f = (np.arange(10), [0, 0, 0, 0, -1, 0, 0, 0, 0, 0])

go.Figure(go.Scatter(x=f[0], y=f[1], name="Objective")).show()


def evaluate(x):
    return np.interp(x, *f)


grid_search(evaluate, 0, 10, 3)
