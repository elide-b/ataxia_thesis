import os
import typing
from typing import Literal

import h5py as h5py
import numpy as np
from tvb.datatypes.connectivity import Connectivity
from pathlib import Path


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


def _load_mousebrain_zip(subject):
    datadir = Path(__file__).parent / "../mouse_brains/dataset/conn/"
    try:
        return Connectivity.from_file(str(datadir / subject))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mouse brain {subject} not found. Choose: "
            + ", ".join(k[:-4] for k in os.listdir(str(datadir)) if k.endswith(".zip"))
        ) from None


def _load_mousebrain_h5(subject):
    datadir = Path(__file__).parent / "../mouse_brains/dataset/h5/"
    try:
        with h5py.File(str(datadir / subject)) as f:
            centres = np.array(f["centres"][()])
            region_labels = np.array(f["region_labels"][()]).astype("<U128")
            weights = np.array(f["weights"][()])
            tract_lengths = np.array(f["tract_lengths"][()])
            brain = Connectivity(
                centres=centres,
                region_labels=region_labels,
                weights=weights,
                tract_lengths=tract_lengths,
            )
        return brain

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mouse brain {subject} not found. Choose: "
            + ", ".join(k[:-3] for k in os.listdir(str(datadir)) if k.endswith(".h5"))
        ) from None


def load_mousebrain(
    subject: str,
    rem_diag: bool = True,
    scale: typing.Union[bool, Literal["tract"], Literal["region"]] = "region",
    norm: bool = True,
) -> Connectivity:
    """
    Load a mouse brain from the `mouse_brains` data folder. The mouse brain is processed
    with the following steps:

    * Replace any nan by 0.
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
    if subject.endswith(".zip"):
        brain = _load_mousebrain_zip(subject)
    elif subject.endswith(".h5"):
        brain = _load_mousebrain_h5(subject)
    else:
        raise ValueError("Only .zip and .h5 are supported.")
    brain.weights[np.isnan(brain.weights)] = 0.0
    if rem_diag:
        np.fill_diagonal(brain.weights, 0)
    if scale:
        brain.weights = brain.scaled_weights(scale)
    if norm:
        brain.weights /= np.max(brain.weights)
    return brain


def get_cereb_region_mask(brain):
    return np.isin(brain.region_labels, _cereb_labels)
