import gym
from gym.spaces import Box
import numpy as np

from .simple_system_reference import LinearSystem, Reference, configSystem, configReference, Logger


class LinearSystemEnv(gym.Env):

    metadata = {'render.modes': []}

    def __init__(self):
        super().__init__()
        self.system = LinearSystem(configSystem)
        self.reference = Reference(configReference)
        self.h = configReference['h']
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        self.setup()

    def setup(self):
        self.system.x = np.random.normal(configSystem["x0"], 0.5)

        self.reference.generateReference()
        self.reference.counter = 1

        self.done = False
        ref = self.reference.getNext()
        return np.hstack((self.system.x.T, ref[1:, :].T))

    def step(self, action):
        ref = self.reference.getNext()
        systemOut = self.system.apply(action)

        reward = -1 * (ref[0, 0] - systemOut[0, 0])**2
        observation = np.hstack((self.system.x.T, ref[1:, :].T))

        self.done = self.reference.counter == 1

        return observation, reward, self.done, None

    def predict(self, x, a):
        return self.system.predict(x, a)

    def reset(self):
        return self.setup()
