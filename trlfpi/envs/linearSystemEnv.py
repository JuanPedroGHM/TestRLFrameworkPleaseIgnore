import gym
from gym.spaces import Box
import numpy as np

from .simple_system_reference import LinearSystem, Reference, configSystem, configReference


class LinearSystemEnv(gym.Env):

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        self.system = LinearSystem(configSystem)
        self.reference = Reference(configReference)
        self.h = configReference['h']
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        self.reference_space = Box(low=-np.inf, high=np.inf, shape=(configReference['h'] + 1,))
        self.alpha = 0.0
        self.lastAction = None
        self.setup()

    def setup(self):
        self.system.x = np.random.normal(configSystem["x0"], 0.5)

        self.reference.generateReference()
        self.reference.counter = 1

        self.done = False
        ref = self.reference.getNext()
        self.lastAction = None
        return self.system.x.T, ref.T

    def step(self, action):
        ref = self.reference.getNext()
        systemOut = self.system.apply(action)

        if not self.lastAction:
            self.lastAction = action
            actionCost = 0
        else:
            actionCost = np.power(action[0, 0] - self.lastAction[0, 0], 2)
            self.lastAction = action

        stateCost = np.power(ref[0, 0] - systemOut[0, 0], 2)

        reward = - (stateCost + self.alpha * actionCost)

        self.done = self.reference.counter == 1

        return systemOut.T, reward, self.done, ref.T

    def predict(self, x, a, gpu=False):
        return self.system.predict(x, a, gpu=gpu)

    def reset(self):
        return self.setup()
