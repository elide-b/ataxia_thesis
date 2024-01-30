import random

from deap import creator, base, tools


class Param:
    def __init__(self, min, max):
        self.name = None
        self.min = min
        self.max = max

    def __set_name__(self, owner, name):
        self.name = name
        params = getattr(owner, "params", [])
        params.append(self)
        setattr(owner, "params", params)
        self.index = params.index(self)

    def __get__(self, instance, owner):
        return instance[self.index]

    def __set__(self, instance, value):
        instance[self.index] = value

    def unit(self, value):
        return (value - self.min) / (self.max - self.min)

    def abs(self, value):
        return max(self.min, min(self.max, value * (self.max - self.min) + self.min))


class Individual(list):
    params: list[Param]

    w = Param(0, 2)
    Ji = Param(0.001, 2)
    G = Param(0, 10)

    def __init__(self, args):
        super().__init__(args)

    def __str__(self):
        return f"<Individual " + ", ".join(f"{k}={v}" for k, v in self.items()) + ">"

    def keys(self):
        yield from (p.name for p in self.params)

    def values(self):
        yield from self

    def items(self):
        yield from zip(self.params, self.values())

    def mutate(self):
        mut = tools.mutGaussian(
                [p.unit(v) for p, v in self.items()],
                mu=0,
                sigma=0.1,
                indpb=1 / len(self.params),
            )[0]
        self[:] = [p.abs(v) for p, v in zip(self.params, mut)]


def create_toolbox():
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", Individual, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attribute,
        n=len(Individual.params),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", Individual.mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox
