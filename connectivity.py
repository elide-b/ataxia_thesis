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
    scale: typing.Union[Literal[False], Literal["tract"], Literal["region"]] = "region",
    norm: typing.Union[Literal[False], Literal["max"], Literal["pct"]] = "max",
    dt: float = 0.1,
) -> Connectivity:
    """
    Load a mouse brain from the `mouse_brains` data folder. The mouse brain is processed
    with the following steps:

    * Replace any nan by 0.
    * Remove the diagonal weights; Connections to self are modeled internally in the
      node's model, not as node-connection inputs.
    * Scale the weights: Either scales the sum of the inputs, or the sum of the outputs of
      each node to one.
    * Normalize the matrix: All values will be divided by either the maximum value or the
      99th percentile.
    * Guarantee minimum tract length: The tract length will be made minimally dt * speed long

    Each step can be toggled with its keyword argument

    :param subject: The name of the mouse brain to load, with extension.
    :param dt: Minimum timestep for propagation of the signal along tracts.
    :param rem_diag: Toggles the "remove diagonal" step.
    :param scale: "tract" scales the output weights, "region" scales the input weights.
    :param norm: "max" normalizes by the maximum value, "pct" by the 99th percentile.
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
    if norm == "max":
        brain.weights /= np.max(brain.weights)
    elif norm == "pct":
        brain.weights /= np.percentile(brain.weights, 99)
    brain.tract_lengths = np.maximum(brain.speed * dt, brain.tract_lengths)
    brain.configure()
    return brain


def get_cereb_region_mask(brain):
    return np.isin(brain.region_labels, _cereb_labels)
