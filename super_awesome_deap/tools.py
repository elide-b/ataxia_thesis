import functools
import random

from deap import creator, base, tools


class Param:
    def __init__(self, min, max):
        self.name = None
        self.min = min
        self.max = max

    def __set_name__(self, owner, name):
        self.name = name
        self.index = owner.params.index(self)

    def __get__(self, instance, owner):
        return instance[self.index]

    def __set__(self, instance, value):
        instance[self.index] = value

    def unit(self, value):
        return (value - self.min) / (self.max - self.min)

    def abs(self, value):
        return max(self.min, min(self.max, value * (self.max - self.min) + self.min))


class Individual(list):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    @property
    @functools.lru_cache()
    def params(cls):
        return [
            *set(
                attr
                for base in reversed(cls.__mro__)
                for attr in base.__dict__.values()
                if isinstance(attr, Param)
            )
        ]

    def __str__(self):
        return (
            f"<{type(self).__name__} "
            + ", ".join(f"{p.name}={v}" for p, v in self.items())
            + ">"
        )

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


def create_toolbox(cls, evaluate):
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create(cls.__name__, cls, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    toolbox.register("attribute", random.random)
    toolbox.register(
        "individual",
        tools.initRepeat,
        getattr(creator, cls.__name__),
        toolbox.attribute,
        n=len(cls.params),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", Individual.mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox
