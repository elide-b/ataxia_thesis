import numpy as np

from ataxia.plots import plot_activity

data = np.load("global_ataxia.npz")

plot_activity(
    {
        "Healthy": data["tavg"][2, 2],
        "Global ataxia (All)": data["tavg"][0, 0],
        "Global ataxia (W)": data["tavg"][0, 2],
        "Global ataxia (weights)": data["tavg"][2, 0],
    }
).write_html("global_ataxia.html")
