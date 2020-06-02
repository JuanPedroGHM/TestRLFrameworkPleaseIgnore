import gym
import numpy as np

from .simple_system_reference import LinearSystem, Reference, configSystem, configReference

class LinearSystemEnv(gym.Env):

    metadata={'render.modes': []}
    def __init__(self):
        super().__init__(self)
        self.system = LinearSystem(configSystem)
        self.reference = Reference(configReference)
        self.setup()
    
    def setup(self):
        self.system.x = np.random.normal(configSystem["x0"], 0.5)

        self.reference.generateReference()
        self.reference.counter = 1

        self.done = False


    def step(self, action):
        lastRef = self.reference.r[self.reference.counter]
        systemOut = self.system.apply(action)

        reward = (lastRef - systemOut)**2
        observation = np.hstack(systemOut,self.reference.getNext())

        self.done = (self.reference.counter == self.reference.N - self.reference.h)

        return observation, reward, self.done, None 


    def reset(self):
        self.setup()
