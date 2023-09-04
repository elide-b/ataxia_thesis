import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from ataxia.connectivity import get_cereb_region_mask
from ataxia.experiment import Experiment
from ataxia.plots import plot_matrix

experiment, meta = Experiment.load("impact")


def figmod(i):
    go.Figure(
        data=[
            go.Heatmap(
                z=np.corrcoef(experiment.bold[0, 0, i]),
                text=[meta["brain"].region_labels] * len(meta["brain"].region_labels),
            )
        ]
    )


wlabel = "from, to" if experiment.from_to else "to, from"
for i, g in experiment.g:
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=np.corrcoef(experiment.bold[0, 0, i]),
                text=[
                    [f"{f} -> {t}" for t in meta["brain"].region_labels]
                    for f in meta["brain"].region_labels
                ],
            )
        ],
        layout=dict(title_text=f"Bold crosscorrelation"),
    )
    fig.update_yaxes(scaleanchor="x")
    fig.update_layout(title_text=f"Bold crosscorrelation (cerebellar weights x{g:.2f})")
    fig.write_html(f"impact_bold_{g:.2f}.html")

# shape = experiment.tavg[0, 0].shape
# fig.add_trace(go.Heatmap(z=experiment.tavg[0, 0], zmin=0, zmax=1), row=1, col=1)
# fig.add_trace(go.Heatmap(z=experiment.tavg[0, -1], zmin=0, zmax=1), row=1, col=2)
# fig.add_trace(
#     go.Heatmap(
#         z=experiment.tavg[0, -1] - experiment.tavg[0, 0],
#         zmin=-1,
#         zmax=1,
#         zmid=0,
#         text=np.tile(
#             np.array(meta["brain"].region_labels), (experiment.tavg[0, 0].shape[1], 1)
#         ).T,
#         colorscale="RdBu",
#     ),
#     row=1,
#     col=3,
# )
# fig.write_html("weight_response_tavg_diff.html")
