import contextlib
import logging
from typing import Generator

import numpy as np
import tvb.datatypes.connectivity

from ataxia.connectivity import get_cereb_region_mask


@contextlib.contextmanager
def ataxic_weights(
    brain: tvb.datatypes.connectivity.Connectivity,
    factor: float,
    from_to: bool = True,
    wholebrain: bool = False,
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
    :param from_to:
    :param wholebrain:
    :return:
    """
    is_cereb = get_cereb_region_mask(brain)
    # Construct the index of weights to set
    # Cerebellum-only: weights[is_cereb, :] *= factor
    index = [is_cereb, slice(None)]
    if wholebrain:
        # Whole brain -> weights[:, :] *= factor
        index[0] = slice(None)
    if not from_to:
        # Not sure if the weights matrix is node[from, to] or node[to, from]?
        # If from_to is false we revert the index to weights[to, from]
        index.reverse()
    index = tuple(index)
    # Use the index to alter the correct weights in the matrix
    pre = brain.weights[index].mean()
    old_weights = np.copy(brain.weights)
    brain.weights[index] *= factor
    post = brain.weights[index].mean()
    # Calculate a message to display when debugging
    stri = ",".join(":" if isinstance(i, slice) or wholebrain else "cereb_nodes" for i in index)
    logging.debug(f"Ataxia weights[{stri}] factor = {round(post / pre, 2)}")
    # This is a context manager, yield is the breakpoint where we reenter when the context
    # exits
    yield
    # When exiting the context, reset the weights to their initial values, as not to mutate
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
