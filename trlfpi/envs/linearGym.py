import gym
import torch
from gym.spaces import Box
import numpy as np

from .referenceGenerator import ReferenceGenerator


class LinearEnv(gym.Env):

    metadata = {'render.modes': []}

    # Env settings
    A = np.array([[0.9590, 0.03697], [-0.5915, 0.0718]])
    At = torch.tensor(A)

    B = np.array([[0.1638], [2.3660]])
    Bt = torch.tensor(B)
    x0 = [[1.0, 0.0]]

    # Ref settings
    N = 1000
    timeStep = 0.1
    numrandref = 50
    noise_variance_ref = 0.0

    def __init__(self, horizon: int = 1, deltaActionCost: float = 0.001):
        super().__init__()
        self.alpha = deltaActionCost
        self.h = horizon
        self.refGen = ReferenceGenerator(LinearEnv.N,
                                         LinearEnv.timeStep,
                                         LinearEnv.numrandref,
                                         LinearEnv.noise_variance_ref)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        self.reference_space = Box(low=-np.inf, high=np.inf, shape=(self.h,))

    def setup(self):
        self.ref = self.refGen.generate()
        self.k = 0
        self.done = False
        self.state = np.random.normal(LinearEnv.x0, 0.5).reshape((1, 2))

        self.lastAction = None
        return self.state, self.ref[:, self.k: self.k + 1 + self.h]

    def step(self, action):

        if self.done:
            print("Trying to take a step in a finished env, please reset")
            raise

        next_state = self.system(self.state, action)
        self.state = next_state

        self.k += 1

        stateCost = np.power(self.ref[0, self.k] - next_state[0, 0], 2)
        if not self.lastAction:
            self.lastAction = action
            actionCost = 0
        else:
            actionCost = np.power(action[0, 0] - self.lastAction[0, 0], 2)
            self.lastAction = action

        reward = - (stateCost + self.alpha * actionCost)
        self.done = self.k == LinearEnv.N - 1

        lastIndex = self.k + self.h + 1 if self.k + self.h + 1 < LinearEnv.N else None
        r = self.ref[:, self.k:lastIndex] if not self.done else None

        # Pad the reference in case there are not enough points
        if r is not None and r.shape[1] < self.h + 1:
            r = np.pad(r, ((0, 0), (0, self.h + 1 - r.shape[1])), mode='edge')

        return self.state, reward, self.done, r

    def system(self, state, action, gpu=False):
        # Inputs
        # state: [[x0, x1]]
        # action : [[a]]

        if gpu:
            next_state = state @ LinearEnv.At.T + action @ LinearEnv.Bt.T
        else:
            next_state = state @ LinearEnv.A.T + action @ LinearEnv.B.T
        return next_state

    def reset(self):
        return self.setup()
