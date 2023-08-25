import itertools
import sys
from typing import Mapping, Union

import numpy as np


class _Component:
    def __init__(self):
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value


class Parameter(_Component):
    def __init__(self, values):
        super().__init__()
        self._current_index = None
        self._current_value = None
        self._values = values

    def __iter__(self):
        self._current_index = -1
        return self

    def __getitem__(self, item):
        return self._values[item]

    def __next__(self):
        self._current_index += 1
        try:
            self._current_value = self._values[self._current_index]
        except:
            raise StopIteration()
        return (self._current_index, self._current_value)

    def __len__(self):
        return len(self._values)

    @property
    def values(self):
        return self._values

    @property
    def current_index(self):
        return self._current_index

    @property
    def current_value(self):
        return self._current_value

    @property
    def min(self):
        return min(self._values)

    @property
    def max(self):
        return max(self._values)

    @property
    def num(self):
        return len(self._values)

    def sweep(self):
        itr = iter(self._values)
        ctr = itertools.count()

        class SweepIter:
            def __iter__(self):
                return self

            def __next__(_):
                value = next(itr)
                count = next(ctr)
                self._current_index = count
                self._current_value = value
                return (self._current_index, self._current_value)

        return SweepIter()

    def is_usable(self):
        return self._current_value is not None and self._current_index is not None


class LinspaceParameter(Parameter):
    def __init__(self, start, stop, num):
        if num == 1:
            values = [stop]
        else:
            values = np.linspace(start, stop, num, endpoint=True)
        super().__init__(values)


class ConstantParameter(Parameter):
    def __init__(self, value):
        super().__init__([value])
        self._current_index = 0
        self._current_value = value


class Result(_Component):
    def __init__(self, shape, dtype=float):
        super().__init__()
        self._shape = shape
        self._data = None
        self._dtype = dtype

    def __array__(self):
        return self._data

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item, value):
        self._data[item] = value

    @property
    def shape(self):
        return self._shape

    def is_usable(self):
        return self._data is not None

    def init_data(self, parameter_shape):
        self._data = np.empty((*parameter_shape, *self._shape), dtype=self._dtype)

    def store(self, index, value):
        self._data[index] = value

    @property
    def data(self):
        return self._data


class Experiment:
    _name: str
    _parameters: dict[str, Parameter] = {}
    _results: dict[str, Result] = {}
    _components: dict[str, _Component] = {}

    def __init__(self, name: str, **kwargs: _Component):
        self.__dict__["_name"] = name
        self.__dict__["_parameters"] = {}
        self.__dict__["_results"] = {}
        self.__dict__["_components"] = {}
        for k, v in kwargs.items():
            v.name = k
            self._components[k] = v
            if isinstance(v, Parameter):
                self._parameters[k] = v
            elif isinstance(v, Result):
                self._results[k] = v
        for result in self._results.values():
            result.init_data(self.get_parameter_shape())

    def __getattr__(self, name) -> _Component:
        try:
            return self._components[name]
        except KeyError:
            super().__getattribute__(name)

    def __str__(self):
        params = ", ".join(
            f"{p.name} from {p.min} to {p.max} in {p.num} steps"
            for p in self._parameters.values()
        )
        results = ", ".join(f"{r.name} {r.shape}" for r in self._results.values())
        return f"<Experiment with {params}. Measures {results}>"

    def __setattr__(self, key, value):
        if key in self._results:
            self._store(self._results[key], value)
        else:
            raise AttributeError(f"Can't store unknown result '{key}'.")

    def get_parameter_shape(self):
        return tuple(len(p) for p in self._parameters.values())

    def get_parameter_indices(self):
        return tuple(p.current_index for p in self._parameters.values())

    def _store(self, result: Result, value):
        unset = [v for v in self._parameters.values() if not v.is_usable()]
        if unset:
            raise RuntimeError(
                f"Can't store results: Parameter(s) {', '.join(str(p.name) for p in unset)} have no value(s)."
            )
        result.store(self.get_parameter_indices(), value)

    def save(self, **meta):
        experiment = np.array(self._parameters, dtype=object)
        np.savez(
            f"{self._name}.npz",
            __experiment=experiment,
            __meta=meta,
            **{r.name: r.data for r in self._results.values()},
        )

    @classmethod
    def load(cls, name: str):
        data = np.load(f"{name}.npz", allow_pickle=True)
        experiment = cls(name)
        experiment.__dict__["_name"] = name
        experiment.__dict__["_parameters"] = data["__experiment"][()]
        results = {}
        for k, v in data.items():
            if k == "__experiment":
                continue
            results[k] = Result(v.shape[len(experiment._parameters) :], v.dtype)
            results[k].name = k
            setattr(results[k], "_data", v)
        experiment.__dict__["_results"] = results
        components = {}
        components.update(**experiment._results, **experiment._parameters)
        experiment.__dict__["_components"] = components
        return experiment, data["__meta"][()]
