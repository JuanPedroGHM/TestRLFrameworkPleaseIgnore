import numpy as np

from scipy.optimize import minimize

from typing import Any

from .gp import GP
from .kernel import Kernel
from trlfpi.memory import GPMemory


class Critic():

    def __init__(self,
                 inputSpace: int,
                 mean: Any = None,
                 memory: int = 1000,
                 optim_freq: int = 100,
                 bruteFactor: int = 5,
                 bGridSize: int = 5,
                 bPoolSize: int = 8):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        kernel = Kernel.RBF(1.0, np.array([1.0 for i in range(inputSpace)]))
        self.model = GP(kernel,
                        meanF=mean,
                        sigma_n=0.1,
                        bGridSize=bGridSize,
                        bPoolSize=bPoolSize)
        self.optim_freq = optim_freq
        self.updates = 0
        self.bruteFactor = bruteFactor
        self.bGridSize = bGridSize
        self.bPoolSize = bPoolSize

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        self.memory.add(x, y)
        self.updates += 1
        if self.updates % self.optim_freq == 0:
            X, Y = self.memory.data

            if self.updates % (self.optim_freq * self.bruteFactor) or self.updates == self.optim_freq:
                self.model.fit(X, Y, fineTune=True, brute=True)

            else:
                self.model.fit(X, Y, fineTune=True)

    def predict(self, x: np.ndarray):
        # x contains action
        return self.model(x)

    def getAction(self, x):
        # x without action
        # return action, value at action
        aRange = np.arange(-2, 2, 0.25).reshape(-1, 1)
        grid = np.hstack((aRange, np.repeat(x, aRange.shape[0], axis=0)))
        qs, sigmas = self.model(grid)

        bestA = grid[np.argmax(qs), 0]
        bounds = [(bestA - 0.25, bestA + 0.25)]

        def f(a):
            q_input = np.hstack((a.reshape(1, 1), x))
            q = self.model.mean(q_input).item()
            return -q

        res = minimize(f, np.array([bestA]), bounds=bounds)
        bestA = np.array(res.x).reshape(1, 1)
        return bestA, res.fun, (aRange, qs)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)
