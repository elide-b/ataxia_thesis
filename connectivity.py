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
        datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mouse_brains/dataset/conn/"))
        return tvb.datatypes.connectivity.Connectivity.from_file(os.path.join(datadir, f"{subject}.zip"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mouse brain {subject} not found. Choose: "
            + ", ".join(k[:-4] for k in os.listdir(datadir) if k.endswith(".zip"))
        ) from None


def load_mousebrain(
    subject: str,
    rem_diag: bool = True,
    scale: typing.Union[bool, Literal["tract"], Literal["region"]] = "region",
    norm: bool = True,
) -> tvb.datatypes.connectivity.Connectivity:
    """
    Load a mouse brain from the `mouse_brains` data folder. The mouse brain is processed
    with the following steps:

    * Remove the diagonal weights; Connections to self are modeled internally in the
      node's model, not as node-connection inputs.
    * Scale the weights: The sum of either the inputs on, or outputs of each node will be
      scaled to one
    * Normalize the matrix: All values will be multiplied so the max value is 1.

    Each step can be toggled with its keyword argument

    :param rem_diag: Toggles the "remove diagonal" step.
    :param scale:
    :param norm: Toggles the "normalize" step.
    :param subject:
    :return: The processed mouse brain
    """
    brain = _load_mousebrain(subject)
    if rem_diag:
        np.fill_diagonal(brain.weights, 0)
    if scale:
        brain.weights = brain.scaled_weights(scale)
    if norm:
        brain.weights /= np.max(brain.weights)
    return brain


def get_cereb_region_mask(brain):
    return np.isin(brain.region_labels, _cereb_labels)
