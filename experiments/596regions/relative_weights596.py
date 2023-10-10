import dataclasses
import itertools

import numpy as np
from plotly.subplots import make_subplots

from connectivity import get_cereb_region_mask, load_mousebrain
import plotly.graph_objs as go

brain = load_mousebrain("Connectivity_596.h5", norm=False, scale=False)
is_cereb = get_cereb_region_mask(brain)

c_w = []
nc_w = []
strong_c = []
strong_nc = []
perm_c = [[] for _ in range(10)]
perm_nc = [[] for _ in range(10)]
perm_test = np.random.randint(0, len(brain.weights), (10, sum(is_cereb)))


@dataclasses.dataclass
class Node:
    name: str
    is_cereb: bool
    sub: set[str] = dataclasses.field(default_factory=set)

    def __hash__(self):
        return hash(self.name)

    def add(self, node: str):
        if node.split(",")[0] != node:
            self.sub.add(node.split(",")[1][1:].replace("_", " "))

    def dot(self):
        if self.sub:
            subpart = '<BR /><FONT POINT-SIZE="10">' + ", ".join(self.sub) + "</FONT>"
        else:
            subpart = ""
        return f"   {self.name}[label=<{self.name}{subpart}>{', style=filled' if self.is_cereb else ''}];\n"


# <BR />
#                <FONT POINT-SIZE="10"></FONT>


@dataclasses.dataclass
class Edge:
    from_node: Node
    to_node: Node
    cereb: bool
    w: list[float] = dataclasses.field(default_factory=list)
    row_w: list[np.ndarray] = dataclasses.field(default_factory=list)
    col_w: list[np.ndarray] = dataclasses.field(default_factory=list)

    def add(self, weight, row_weights, col_weights):
        self.w.append(weight)
        self.row_w.append(row_weights)
        self.col_w.append(col_weights)

    @property
    def weight(self):
        return sum(self.w) / len(self.w)

    def dot(self):
        props = (
            f'label="{round(self.weight * 100) / 100} ({round(self.weight / np.sum(self.col_w, axis=1).mean() * 100)}%)"',
            f"weight={self.weight * self.cereb}",
        )
        edge = f"{self.from_node.name} -> {self.to_node.name}"
        return f"   {edge} [{', '.join(props)}];\n"


def get_node_name(name):
    return (
        name.split("Right ")[-1]
        .split("Left ")[-1]
        .split(",")[0]
        .replace("(", "")
        .replace(")", "")
        .replace("'", "")
    )


nodes = {
    get_node_name(l): Node(get_node_name(l), c)
    for l, c in zip(brain.region_labels, is_cereb)
}
edges = {}
used_nodes = set()

for i, l, c, row in zip(
    itertools.count(), brain.region_labels, is_cereb, brain.weights
):
    for pi, perm in enumerate(perm_test):
        if i in perm:
            perm_c[pi].extend(row)
        else:
            perm_nc[pi].extend(row)
    node = nodes[get_node_name(l)]
    if c:
        c_w.extend(row)
        for to_i, to_l, to_c, edge_w in zip(
            itertools.count(), brain.region_labels, is_cereb, row
        ):
            if edge_w > 0.05:
                to_node = nodes[get_node_name(to_l)]
                used_nodes.add(node)
                used_nodes.add(to_node)
                node.add(l)
                to_node.add(to_l)
                edge = edges.setdefault((node, to_node), Edge(node, to_node, to_c))
                edge.add(edge_w, row, brain.weights[:, to_i])
            strong_c.extend(brain.region_labels[(row > 0.05) & is_cereb])
        strong_nc.extend(brain.region_labels[(row > 0.05) & ~is_cereb])
    else:
        nc_w.extend(row)

go.Figure(
    [
        go.Histogram(
            x=c_w,
            name=f"Cerebellar weights (n={sum(is_cereb)})",
            histnorm="probability",
            nbinsx=20,
        ),
        go.Histogram(
            x=nc_w,
            name=f"Non-cerebellar weights (n={sum(~is_cereb)})",
            histnorm="probability",
            nbinsx=20,
        ),
    ],
    layout=dict(
        xaxis_title="Weight",
        yaxis_title="Probability",
        title_text="Connectivity distribution of the cerebellum",
    ),
).write_html("weight_distribution_histogram.html")

fig = make_subplots(rows=1, cols=5)
fig.update_layout(
    title_text="Connectivity distribution of random node groups permutations"
)
for i in range(5):
    fig.add_traces(
        [
            go.Histogram(
                x=c_w,
                name=f"Cerebellar weights (n={sum(is_cereb)})",
                legendgroup="c",
                showlegend=i == 0,
                histnorm="probability",
                nbinsx=20,
            ),
            go.Histogram(
                x=nc_w,
                name=f"Non-cerebellar weights (n={sum(~is_cereb)})",
                legendgroup="nc",
                showlegend=i == 0,
                histnorm="probability",
                nbinsx=20,
            ),
            go.Histogram(
                x=perm_c[i],
                name=f"Random weights (n={sum(is_cereb)})",
                legendgroup="r",
                showlegend=i == 0,
                histnorm="probability",
                nbinsx=20,
            ),
        ],
        rows=1,
        cols=i + 1,
    )
fig.write_html("permutation_histogram.html")

go.Figure(
    [
        go.Box(y=c_w, name="Cerebellar weights"),
        go.Box(y=nc_w, name="Non-cerebellar weights"),
    ],
    layout_title_text="Connectivity distribution of the cerebellum",
).write_html("box_weights.html")
n = "\n"
source = f"""
digraph g {{
    layout = "neato";
    graph [pad="0.5", sep="0.4", start="8994"];
    overlap = false;
    splines = true;
{"".join(node.dot() for node in used_nodes)}
{"".join(edge.dot() for edge in edges.values())}
}}
"""
try:
    import graphviz
except:
    print(source)
else:
    src = graphviz.Source(source)
    src.format = "svg"
    src.render("conn_diagram.gv", cleanup=False)
print()
