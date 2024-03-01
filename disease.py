import contextlib
import logging
from typing import Generator

import numpy as np
import tvb.datatypes.connectivity

from connectivity import get_cereb_region_mask


@contextlib.contextmanager
def ataxic_weights(
        brain: tvb.datatypes.connectivity.Connectivity,
        factor: float,
        wholebrain: bool = False,
        from_to: bool = True,
) -> Generator[None, None, None]:
    """
    A context manager to induce ataxia by editing the connectivity of the nodes.

    Usage:

    with ataxic_weights(brain, 0.9):
      simulator.connectivity = brain

    Inside of the `with` block, the weights are altered, outside of it they are cleaned up
    to their initial value again!

    :param brain:
    :param factor:
    :param wholebrain:
    :param from_to:
    :return:
    """
    is_cereb = get_cereb_region_mask(brain)
    old_weights = brain.weights.copy()
    if wholebrain:
        # Whole brain -> Change all weights
        brain.weights[:, :] *= factor
    elif from_to:
        # Cerebellum-only: change all cereb to non-cereb weights
        # Assume conn matrix is `weights[from, to]`
        cereb = brain.weights[is_cereb]
        cereb[:, ~is_cereb] *= factor
        brain.weights[is_cereb] = cereb
    else:
        # Assume conn matrix is `weights[to, from]`
        cereb = brain.weights[:, is_cereb]
        cereb[~is_cereb] *= factor
        brain.weights[:, is_cereb] = cereb
    # In contextlib.contextmanager, yield is where we pause waiting for the context exit.
    yield
    # After exiting the context, reset the weights to their initial values, as not to mutate
    # the connectivity that was passed in.
    brain.weights = old_weights


def ataxic_w(brain, w, k, wholebrain=False):
    is_cereb = get_cereb_region_mask(brain)
    mod_w = np.ones(brain.weights.shape[0:1]) * w
    index = slice(None) if wholebrain else is_cereb
    mod_w[index] *= k
    logging.debug(
        f"Ataxic w ({'whole brain' if wholebrain else 'cereb'}) factor = {round(mod_w[index].mean() / np.mean(w), 2)}"
    )
    return mod_w
