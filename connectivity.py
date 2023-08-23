import os
import typing
from typing import Literal

import numpy as np
import tvb.datatypes.connectivity


_cereb_labels = [
    "Left_Central_lobule",
    "Left_Culmen",
    "Left_Simple_lobule",
    "Left_Ansiform_lobule",
    "Left_Paramedian_lobule",
    "Left_Copula_pyramidis",
    "Left_Paraflocculus",
    "Right_Central_lobule",
    "Right_Culmen",
    "Right_Simple_lobule",
    "Right_Ansiform_lobule",
    "Right_Paramedian_lobule",
    "Right_Copula_pyramidis",
    "Right_Paraflocculus",
]


def _load_mousebrain(subject):
    try:
        datadir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../mouse_brains/dataset/conn/")
        )
        return tvb.datatypes.connectivity.Connectivity.from_file(
            os.path.join(datadir, f"{subject}.zip")
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mouse brain {subject} not found. Choose: "
            + ", ".join(k[:-4] for k in os.listdir(datadir) if k.endswith(".zip"))
        ) from None


def load_mousebrain(
    subject: str,
    rem_diag: bool = True,
    scale: typing.Union[Literal[False], Literal["tract"], Literal["region"]] = "region",
    norm: typing.Union[Literal[False], Literal["max"], Literal["pct"]] = "max",
) -> tvb.datatypes.connectivity.Connectivity:
    """
    Load a mouse brain from the `mouse_brains` data folder. The mouse brain is processed
    with the following steps:

    * Remove the diagonal weights; Connections to self are modeled internally in the
      node's model, not as node-connection inputs.
    * Scale the weights: Either scales the sum of the inputs, or the sum of the outputs of
      each node to one.
    * Normalize the matrix: All values will be divided by either the maximum value or the
      99th percentile.

    Each step can be toggled with its keyword argument

    :param subject: The name of the mouse brain to load.
    :param rem_diag: Toggles the "remove diagonal" step.
    :param scale: "tract" scales the output weights, "region" scales the input weights.
    :param norm: "max" normalizes by the maximum value, "pct" by the 99th percentile.
    :return: The processed mouse brain
    """
    brain = _load_mousebrain(subject)
    if rem_diag:
        np.fill_diagonal(brain.weights, 0)
    if scale:
        brain.weights = brain.scaled_weights(scale)
    if norm == "max":
        brain.weights /= np.max(brain.weights)
    elif norm == "pct":
        brain.weights /= np.percentile(brain.weights, 99)
    return brain


def get_cereb_region_mask(brain):
    return np.isin(brain.region_labels, _cereb_labels)
