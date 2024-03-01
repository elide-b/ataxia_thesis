import numpy as np
from skimage.measure import block_reduce

from plots import plot_activity

data = np.load("global_ataxia.npz")


def reduce(d):
    return block_reduce(d, block_size=(1, 1), func=np.mean)


plot_activity(
    {
        "Healthy": reduce(data["tavg"][1, 1]),
        "Global ataxia (All)": reduce(data["tavg"][0, 0]),
        "Global ataxia (W)": reduce(data["tavg"][0, 1]),
        "Global ataxia (weights)": reduce(data["tavg"][1, 0]),
    }
).write_html("global_ataxia.html")
