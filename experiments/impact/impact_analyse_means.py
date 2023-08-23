import numpy as np
import plotly.graph_objs as go

from ataxia.connectivity import get_cereb_region_mask
from ataxia.experiment import Experiment

experiment, meta = Experiment.load("impact")
means = np.mean(experiment.tavg[0], axis=2)
meandiff = means[-1] - means[0]
print(meandiff)
remaindiff = (means[-1] - means[0]) / (1 - means[0])
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
        title_text="Response to change in cerebellar weights",
        xaxis=dict(
            title="Output of cerebellar nodes",
        ),
        yaxis=dict(title="Change in activity (mu - mu0)"),
    ),
).show()

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
            get_cereb_region_mask(meta["brain"])[np.argsort(meandiff)[::-1]],
            ((means - means[0]) / (1 - means[0])).T[np.argsort(remaindiff)[::-1]],
        )
    ],
    layout=dict(
        title_text="Response to change in cerebellar weights",
        xaxis=dict(
            title="Output of cerebellar nodes",
        ),
        yaxis=dict(title="Change in activity (mu - m0) / (1 - m0)"),
    ),
).show()

go.Figure(go.Heatmap(z=experiment.tavg[0, 0])).show()
