import gym
from gym.spaces import Box
import numpy as np
import torch

from .referenceGenerator import ReferenceGenerator


class ClutchEnv(gym.Env):

    metadata = {'render.modes': []}

    # Env settings
    timeStep = 0.01  # 10 ms

    # Clutch settings
    jIn = 0.209     # kgm^2
    jOut = 86.6033  # kgm^2
    gamma = 10.02   # 6.45 different for different clutches
    torqueInbase = 20
    torqueOutFactor = 2
    startState = np.array([[209.0, 10.0]])

    # Refs settings
    referenceValue = 209  # in Hz
    N = 100
    numrandref = 8
    noise_variance_ref = 0.0

    def __init__(self, horizon: int = 1, deltaActionCost: float = 0.001, rewardScaling: float = 1.0):
        super().__init__()
        self.h = horizon
        self.deltaActionCost = deltaActionCost
        self.rewardScaling = rewardScaling
        self.refGen = ReferenceGenerator(ClutchEnv.N,
                                         ClutchEnv.timeStep,
                                         ClutchEnv.numrandref,
                                         variance=ClutchEnv.noise_variance_ref,
                                         offset=ClutchEnv.referenceValue)

        self.observation_space = Box(low=0.0, high=5000, shape=(2,))
        self.action_space = Box(low=-np.inf, high=np.inf, shape=(1,))
        self.reference_space = Box(low=0.0, high=500, shape=(self.h + 1,))

    def setup(self):
        self.ref = self.refGen.generate()
        self.k = 0
        self.done = False
        self.state = ClutchEnv.startState
        self.lastAction = None

        return self.state, self.ref[:, self.k:self.k + 1 + self.h]

    def step(self, action):

        if self.done:
            print("Trying to take a step in a finished env, please reset")
            raise

        next_state, done = self.system(self.state, action)
        self.state = next_state

        self.k += 1
        stateCost = - self.rewardScaling * np.power(self.ref[0, self.k] - next_state[0, 0], 2)

        if not self.lastAction:
            actionCost = 0.0
            self.lastAction = action
        else:
            actionCost = - np.power(action[0, 0] - self.lastAction[0, 0], 2)
            self.lastAction = action

        reward = stateCost + self.deltaActionCost * actionCost

        self.done = np.any(done) or self.k == ClutchEnv.N - 1

        lastIndex = self.k + 1 + self.h if self.k + 1 + self.h < ClutchEnv.N else None
        r = self.ref[:, self.k:lastIndex] if not done else None

        # Pad the reference in case there are not enough points
        if r is not None and r.shape[1] < self.h + 1:
            r = np.pad(r, ((0, 0), (0, self.h + 1 - r.shape[1])), mode='edge')

        return next_state, reward, self.done, r

    def reset(self):
        return self.setup()

    def system(self, state, action, gpu=False):

        # Inputs
        # state : [[omegaIn, omegaOut]]
        # action: [[capacityTorque]]

        omegaIn = state[:, [0]]
        omegaOut = state[:, [1]]

        torqueIn = ClutchEnv.torqueInbase
        torqueOut = ClutchEnv.torqueOutFactor * omegaOut

        # System matrix
        if gpu:
            clutchTorque = action * torch.sign(omegaIn - ClutchEnv.gamma * omegaOut)
            a1 = (torqueIn - clutchTorque) / ClutchEnv.jIn
            a2 = (ClutchEnv.gamma * clutchTorque - torqueOut) / ClutchEnv.jOut
            A = torch.cat([a1, a2], dim=1)
        else:
            clutchTorque = action * np.sign(omegaIn - ClutchEnv.gamma * omegaOut)
            a1 = (torqueIn - clutchTorque) / ClutchEnv.jIn
            a2 = (ClutchEnv.gamma * clutchTorque - torqueOut) / ClutchEnv.jOut
            A = np.hstack([a1, a2])

        # s_k+1 = A * dT + x_k
        next_state = A * ClutchEnv.timeStep + state

        # Check if omegas crossed
        if gpu:
            return next_state
        else:
            done = ((omegaIn < ClutchEnv.gamma * omegaOut and
                     next_state[:, 0] > ClutchEnv.gamma * next_state[:, 1]) or
                    (omegaIn > ClutchEnv.gamma * omegaOut and
                     next_state[:, 0] < ClutchEnv.gamma * next_state[:, 1]))

            return next_state, done
