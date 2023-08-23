import numpy as np

from ataxia.plots import plot_activity

data = np.load("local_ataxia.npz")

plot_activity(
    {
        "Healthy": data["tavg"][2, 2],
        "Cerebellar ataxia (All)": data["tavg"][0, 0],
        "Cerebellar ataxia (W)": data["tavg"][0, 2],
        "Cerebellar ataxia (weights)": data["tavg"][2, 0],
    }
).write_html("local_ataxia.html")
