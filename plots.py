import math
from typing import Any, Union

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from tvb.datatypes.connectivity import Connectivity
from plotly.subplots import make_subplots

pio.templates.default = pio.templates["simple_white"]


def _add_matrix_panel(fig, matrix, row=None, col=None):
    fig.add_trace(go.Heatmap(z=matrix), row=row, col=col)
    fig.update_yaxes(scaleanchor="x", row=row, col=col)


def _add_weights_panel(fig, conn, row=None, col=None):
    _add_matrix_panel(fig, conn.weights, row=row, col=col)


def _add_act_panel(fig, activity, row, col):
    fig.add_trace(go.Heatmap(z=activity, zmin=0, zmax=1), row=row, col=col)


def plot_matrix(matrix, show=True):
    fig = go.Figure()
    _add_matrix_panel(fig, matrix)
    if show:
        fig.show()
    return fig


def plot_activity(activity, show=True):
    fig = go.Figure()
    _add_act_panel(fig, activity)
    if show:
        fig.show()
    return fig


def _plot_dict_grid(data: dict[str, Any], grid_panel):
    titles = [*data.keys()]
    rows = math.floor(math.sqrt(len(titles)))
    columns = math.ceil(len(titles) // rows)
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for i, c in enumerate(data.values()):
        row = i // columns + 1
        col = i % columns + 1
        grid_panel(fig, c, row, col)
    return fig


def plot_weights(conn: Union[Connectivity, dict[str, Connectivity]], show=True):
    if isinstance(conn, dict):
        fig = _plot_dict_grid(conn, _add_weights_panel)
    else:
        fig = plot_matrix(conn.weights, show=False)
    if show:
        fig.show()
    return fig


def plot_activity(act: Union[np.ndarray, dict[str, np.ndarray]], show=True):
    if isinstance(act, dict):
        fig = _plot_dict_grid(act, _add_act_panel)
    else:
        fig = plot_activity(act, show=False)
    if show:
        fig.show()
    return fig
