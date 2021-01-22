from .gpu import GPu
from .kernel import Kernel
from ..memory import Memory

import torch

from typing import List


class GPModel():

    default_config = {
        # Memory
        'memorySize': 1000,
        'episilon': 1e-3,  # If new sample can be predicted within that margin of error, ignore

        # Kernel
        'length': [1.0],
        'theta': [1.0],

        # GP
        'inDim': 3,
        'outDim': 1,
        'sigma': 0.1
    }

    def __init__(self, config: dict):
        self.config = {key: config.get(key, self.default_config[key]) for key in self.default_config}

    def setup(self, checkpoint: dict = None, device: str = None):

        self.device = device
        if checkpoint:
            self.fromDict(checkpoint)
        else:
            self.memory = Memory(self.config['memorySize'])
            self.epsilon = self.config['episilon']
            self.brute = True

        assert(len(self.config['length']) == self.config['inDim'])

        self.gpArray = []
        for _ in range(self.config['outDim']):
            kernel = Kernel.RBF(self.config['theta'], self.config['length'], device=device)
            gpu = GPu(kernel, sigma_n=self.config['sigma'], device=device)
            self.gpArray.append(gpu)

    def addData(self, X, Y):

        # Filter new data
        pY = self.forward(X)
        mask = torch.pow(pY - Y, 2).sum(axis=1) > self.epsilon
        for sample in zip(X[mask], Y[mask]):
            self.memory.add(sample)

    def fit(self, forceBrute=False):
        X, Y = self.memory.get()
        for index, gp in enumerate(self.gpArray):
            gp.fit(X, Y[:, [index]], fineTune=True, brute=forceBrute or self.brute)

        if self.brute:
            self.brute = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([gp(x)[0] for gp in self.gpArray], axis=1)

    def toDict(self) -> dict:
        return {
            'config': self.config,
            'gpArray': [gp.toDict() for gp in self.gpArray],
            'memory': self.memory,
            'epsilon': self.epsilon,
            'brute': self.brute
        }

    def fromDict(self, checkpoint: dict):
        self.config = checkpoint['config']
        self.gpArray = [GPu().fromDict(gpCheckpoint) for gpCheckpoint in checkpoint['gpArray']]
        self.memory = checkpoint['memory']
        self.epsilon = checkpoint['epsilon']
        self.brute = checkpoint['brute']

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
