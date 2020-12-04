import numpy as np
from .agent import Agent
from typing import List, Tuple


@Agent.register('random')
class RandomAgent(Agent):

    default_config = {
        'mean': 0.0,
        'std': 0.1,
        'outputDim': 1,
        'seed': None
    }

    def setup(self, checkpoint: dict = None, device: str = 'cpu'):
        np.random.seed(self.config['seed'])

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        action = np.random.normal(self.config['mean'], self.config['std'], self.config['outputDim'])
        action = action.reshape([1, self.config['outputDim']])
        return action, 0.0

    def update(self, stepData: List[np.ndarray] = None) -> dict:
        return {}

    def toDict(self) -> dict:
        return {'config': self.config}
