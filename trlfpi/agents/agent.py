import abc
import numpy as np
from typing import List, Tuple


class Agent(abc.ABC):

    registered_agents: dict = {}

    default_config: dict = {}

    @classmethod
    def create(cls, name: str, config: dict):
        return cls.registered_agents[name](config)

    @classmethod
    def register(cls, name):
        def register_inner(newAgent):
            if issubclass(newAgent, cls):
                cls.registered_agents[name] = newAgent
        return register_inner

    def __init__(self, config: dict):
        self.config = {key: config.get(key, self.default_config[key]) for key in self.default_config}
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'eval'

    @abc.abstractmethod
    def setup(self, checkpoint: dict = None, device: str = 'cpu'):
        pass

    @abc.abstractmethod
    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def update(self, stepData: List[np.ndarray] = None) -> dict:
        """
        Applies gradient descent on policy and/or state value function.

        Parameters:
        stepData (list): List with the data from last step in this order.
                         [state, ref, action, log_prob, reward, done, next_state]

        Returns:
        dict: Dictionary with any losses during training.
        """
        pass

    @abc.abstractmethod
    def toDict(self) -> dict:
        return self.config
