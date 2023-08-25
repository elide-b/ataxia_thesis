import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ataxia.connectivity import get_cereb_region_mask
from ataxia.experiment import Experiment

experiment, meta = Experiment.load("impact")
means = np.mean(experiment.tavg[0], axis=2)
meandiff = means[-1] - means[0]
remaindiff = (means[-1] - means[0]) / (1 - means[0])
wlabel = "from, to" if meta.get("from_to", True) else "to, from"
go.Figure(
    [
        go.Scatter(
            name=label,
            legendgroup="Cerebellum" if is_cereb else "Cerebrum",
            legendgrouptitle_text="Cerebellum" if is_cereb else "Cerebrum",
            x=experiment.g.values,
            y=node_means,
        )
        for label, is_cereb, node_means in zip(
            meta["brain"].region_labels[np.argsort(meandiff)[::-1]],
            get_cereb_region_mask(meta["brain"])[np.argsort(meandiff)[::-1]],
            (means - means[0]).T[np.argsort(meandiff)[::-1]],
        )
    ],
    layout=dict(
        title_text=f"Response to change in cerebellar weights (assuming weights[{wlabel}])",
        xaxis=dict(
            title="Weight multiplication of cerebellar nodes",
        ),
        yaxis=dict(title="Change in activity (mu - mu0)"),
    ),
).write_html("weight_response.html")

go.Figure(
    [
        go.Scatter(
            name=label,
            legendgroup="Cerebellum" if is_cereb else "Cerebrum",
            legendgrouptitle_text="Cerebellum" if is_cereb else "Cerebrum",
            x=experiment.g.values,
            y=node_means,
        )
        for label, is_cereb, node_means in zip(
            meta["brain"].region_labels[np.argsort(remaindiff)[::-1]],
            get_cereb_region_mask(meta["brain"])[np.argsort(remaindiff)[::-1]],
            ((means - means[0]) / (1 - means[0])).T[np.argsort(remaindiff)[::-1]],
        )
    ],
    layout=dict(
        title_text=f"Response to change in cerebellar weights (assuming weights[{wlabel}])",
        xaxis=dict(
            title="Weight multiplication of cerebellar nodes",
        ),
        yaxis=dict(title="Change in activity (mu - m0) / (1 - m0)"),
    ),
).write_html("weight_response_altnorm.html")

fig = make_subplots(
    rows=1, cols=3, subplot_titles=("Regular", "Cerebellum weights x50", "Diff")
)
fig.update_layout(
    title_text=f"Activity difference for adjusted weights (assuming weights[{wlabel}])"
)
shape = experiment.tavg[0, 0].shape
fig.add_trace(go.Heatmap(z=experiment.tavg[0, 0], zmin=0, zmax=1), row=1, col=1)
fig.add_trace(go.Heatmap(z=experiment.tavg[0, -1], zmin=0, zmax=1), row=1, col=2)
fig.add_trace(
    go.Heatmap(
        z=experiment.tavg[0, -1] - experiment.tavg[0, 0],
        zmin=-1,
        zmax=1,
        zmid=0,
        text=np.tile(
            np.array(meta["brain"].region_labels), (experiment.tavg[0, 0].shape[1], 1)
        ).T,
        colorscale="RdBu",
    ),
    row=1,
    col=3,
)
fig.write_html("weight_response_tavg_diff.html")
