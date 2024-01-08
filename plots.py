import math
from typing import Any, Union
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from tvb.datatypes.connectivity import Connectivity
from plotly.subplots import make_subplots
import plotly.express as px

pio.templates.default = pio.templates["simple_white"]


def _add_matrix_panel(fig, matrix, row=None, col=None, mask_from=None, mask_to=None, zmin=None, zmax=None):
    if mask_from is not None:
        sx, sy = get_submask(mask_from, mask_to)
        matrix = apply_submask(matrix, sx, sy)
        text = np.array([f"({x}, {y})" for x, y in zip(sx.flatten(), sy.flatten())]).reshape(matrix.shape)
    else:
        text = None
    fig.add_trace(go.Heatmap(z=matrix, zmin=zmin, zmax=zmax, text=text), row=row, col=col)
    fig.update_yaxes(scaleanchor="x", row=row, col=col)


def _add_weights_panel(fig, conn, row=None, col=None):
    _add_matrix_panel(fig, conn.weights, row=row, col=col)


def _add_act_panel(fig, activity, row, col):
    fig.add_trace(go.Heatmap(z=activity), row=row, col=col)


def plot_matrix(matrix, show=True, mask_from=None, mask_to=None, zmin=None, zmax=None):
    fig = go.Figure()
    _add_matrix_panel(fig, matrix, mask_from=mask_from, mask_to=mask_to, zmin=zmin, zmax=zmax)
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


def plot_weights(conn: Union[Connectivity, dict[str, Connectivity]], show=True, mask_from=None, mask_to=None, zmin=None,
                 zmax=None):
    if isinstance(conn, dict):
        fig = _plot_dict_grid(conn, _add_weights_panel)
    else:
        fig = plot_matrix(conn.weights, show=False, mask_from=mask_from, mask_to=mask_to, zmin=zmin, zmax=zmax)
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


def get_submask(x, y=None):
    if y is None:
        y = x
    return np.meshgrid(x, y)


def apply_submask(m, x, y):
    return m[x, y].T


def plot_fc(FC_mat, regions_labels, mouse_id=None, show=True):
    fig = go.Figure(data=go.Heatmap(
        z=FC_mat,
        x=regions_labels,
        y=regions_labels,
        colorscale='Viridis'
    ))

    if mouse_id is not None:
        fig.update_layout(
            title='FC for ' + mouse_id,
            xaxis_title='Regions',
            yaxis_title='Regions',
            width=1000, height=1000
        )
    else:
        fig.update_layout(
            title='experimental FC',
            xaxis_title='Regions',
            yaxis_title='Regions',
            width=1000, height=1000
        )
    if show:
        fig.show()

    return fig


def plot_bold_timeseries(bold_ts, mouse_id=None, show=True):
    if mouse_id is not None:
        fig = px.line(bold_ts, x=bold_ts.index, y=bold_ts.columns, title='Bold timeseries for ' + mouse_id)
    else:
        fig = px.line(bold_ts, x=bold_ts.index, y=bold_ts.columns, title='Bold timeseries')
    fig.update_xaxes(title_text='Time [s]')
    fig.update_yaxes(title_text='BOLD')

    if show:
        fig.show()

    return fig
