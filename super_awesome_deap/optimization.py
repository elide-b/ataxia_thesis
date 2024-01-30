import random
import time
from pathlib import Path

import plotly.graph_objs as go

try:
    import mpi4py.MPI
    from mpipool import MPIExecutor

    pool_size = mpi4py.MPI.COMM_WORLD.Get_size()
except ImportError:
    pool_size = 1

from .tools import create_toolbox


def optimize(
    cls,
    evaluate,
    n_pop=50,
    p_cx=0.5,
    p_mut=0.20,
):
    toolbox = create_toolbox(cls, evaluate)
    if pool_size > 1:
        pool = MPIExecutor()
        if not pool.is_master():
            return
        toolbox.register("map", pool.map)
        n_pop = min(1, n_pop // (pool_size - 1)) * (pool_size - 1)
    else:
        toolbox.register("map", map)

    fitness_log = []
    pop = toolbox.population(n=n_pop)

    # Evaluate the entire population
    fitnesses = [*toolbox.map(toolbox.evaluate, pop)]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    log(fitness_log, pop)
    mfit = float("-inf")
    done = False
    while True:
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [*map(toolbox.clone, offspring)]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < p_cx:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < p_mut:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [*toolbox.map(toolbox.evaluate, invalid_ind)]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            tfit = sum(ind.fitness.wvalues)
            if tfit > mfit:
                mfit = tfit
                Path("bestfit.txt").write_text(str(ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        log(fitness_log, pop)
        plot_progress(fitness_log, done)
        if not done and should_stop(fitness_log):
            done = True


def should_stop(log):
    avg_fitness = [item[1] for item in log]
    if len(avg_fitness) < 50:
        return False
    return (avg_fitness[-1] - avg_fitness[-20]) / max(avg_fitness) < 0.01


last_plot = None


def plot_progress(log, done):
    global last_plot

    if not last_plot or (time.time() - last_plot > 10):
        last_plot = time.time()
        go.Figure(
            [
                go.Scatter(
                    y=[l[1] for l in log],
                    hovertext=[hovertext(l) for l in log],
                    name="avg",
                ),
                go.Scatter(y=[l[2] for l in log], name="max"),
            ],
            layout=dict(
                title_text="Optimization ONGOING"
                if not done
                else "Optimization PROBABLY DONE"
            ),
        ).write_html("optimization.html")


def popinfo(pop):
    params = pop[0].params
    return {
        p.name: (sum(i[p.index] for i in pop) / len(pop), max(i[p.index] for i in pop))
        for p in params
    }


def log(log, pop):
    log.append(
        (
            popinfo(pop),
            sum(ind.fitness.values[0] for ind in pop) / len(pop),
            max(ind.fitness.values[0] for ind in pop),
        )
    )


def hovertext(item):
    info, avg, max = item
    msg = "Params:<br>-------<br>"
    for k, v in info.items():
        msg += f"{k}: best={v[1]};avg={v[0]}<br>"
    return msg
