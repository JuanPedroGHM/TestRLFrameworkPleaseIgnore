import numpy as np

from trlfpi.gp import GP, Kernel
from trlfpi.memory import GPMemory

class Critic():

    def __init__(self,
                 inputSpace: int,
                 memory: int = 500):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        self.model = GP(Kernel.RBF() + Kernel.Noise())

    def update(self, x: np.ndarray, y: np.ndarray):
        self.memory.add(x, y)
        if self.memory.size <= 10:
            X, Y = self.memory.data
            self.model.fit(X, Y)

    def predict(self, x: np.ndarray):
        if self.memory.size != 0:
            return self.model(x)
        else:
            return (0, 0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)



def updateCritic(action, deltas, reward, next_action, next_deltas):
    # Calculate TD(0) error and update critic GP
    X = np.hstack((action, deltas))
    nX = np.hstack((next_action, next_deltas))
    Y = reward + discount * critic(nX)[0]
    critic.update(X, Y)
