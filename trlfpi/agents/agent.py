import abc
import numpy as np
from argparse import ArgumentParser


class Agent(abc.ABC):

    registered_agents = {}

    @classmethod
    def setup(cls, name, args):
        return cls.registerAgent[name](*args)

    @classmethod
    def register(cls, agent, name):
        if issubclass(agent, cls):
            cls.registerAgent[name] = cls

    @abc.abstractclassmethod
    def options(self, parser: ArgumentParser):
        pass

    @abc.abstractmethod
    def act(self, state: np.ndarray, ref: np.ndarray, eval=False) -> np.ndarray:
        pass

    @abc.abstractmethod
    def update(self) -> np.ndarray:
        pass
