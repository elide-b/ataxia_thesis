import builtins
from pathlib import Path

import numpy as np
import plotly.graph_objs as go

try:
    import mpi4py.MPI
    from mpipool import MPIExecutor

    pool_size = mpi4py.MPI.COMM_WORLD.Get_size()
except ImportError:
    pool_size = 1


def grid_search(evaluate, rmin, rmax, n_samples=10, max_res=0.001):
    n_samples = max(5, n_samples)
    if pool_size > 1:
        pool = MPIExecutor()
        if not pool.is_master():
            return
        map_ = pool.map
        n_samples = max(n_samples, (pool_size - 1)) // (pool_size - 1) * (pool_size - 1)
    else:
        map_ = builtins.map
    delta = float("+inf")
    bingens = 0
    results, sample_points = None, None
    unit = 0
    log = []
    best = None
    while delta > max_res:
        new_sample_points = np.linspace(rmin, rmax, n_samples, endpoint=True)
        print("Sampling", n_samples, rmin, rmax)
        new_results = np.fromiter(map_(evaluate, new_sample_points), dtype=float)
        if bingens:
            # Interleave new finer binary search results
            results = interleave(results, new_results)
            sample_points = interleave(sample_points, new_sample_points)
        else:
            results = new_results
            sample_points = new_sample_points

        delta = (rmax - rmin) / len(results)
        print("Delta is", delta)
        best = np.min(results)
        hats = [
            i
            for i in range(1, len(results) - 1)
            if results[i - 1] > results[i] < results[i + 1]
        ]
        log.append((sample_points, results, hats))
        plot_progress(log)
        print("Found", len(hats), "hats")
        if hats:
            if bingens != 0:
                n_samples = int(round(n_samples / 2 ** (bingens - 1) + 1))
                bingens = 0
            minhat = hats[np.argmin(results[hats])]
            rmin = sample_points[minhat - 1]
            rmax = sample_points[minhat + 1]
        else:
            print("Binary search iteration", bingens)
            if bingens == 0:
                unit = (rmax - rmin) / (n_samples - 1)
                rmin += unit
                rmax -= unit
                n_samples -= 1
                n_samples /= 2
            unit /= 2
            rmin -= unit
            rmax += unit
            n_samples = int(n_samples * 2)
            bingens += 1
    return best


def interleave(a, b):
    r = np.empty((a.size + b.size,), dtype=a.dtype)
    r[0::2] = a
    r[1::2] = b
    return r


def plot_progress(log):
    path = str(Path("grid_opt.html").resolve())
    print(f"Writing progress to: {path}")
    go.Figure(
        [
            go.Scatter(x=sample_points, y=results, name=f"Iteration {i}")
            for i, (sample_points, results, hats) in enumerate(log)
        ]
    ).write_html(path)
