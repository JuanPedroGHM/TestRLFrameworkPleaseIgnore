from typing import Callable

import gym
import matplotlib.pyplot as plt
import numpy as np


def generatePlot(name, y, x=None, folder="./experiments/vpg/plots"):
    if x:
        plt.plot(x, y)
    else:
        plt.plot(y)
    plt.savefig(f'{folder}/{name}.png')
    plt.close()
    


def normFunc(inputSpace: gym.spaces.Box, low: int = -1, high: int = 1) -> Callable[[np.ndarray],np.ndarray]:

    iHigh = inputSpace.high
    iLow = inputSpace.low
    alpha = inputSpace.high - inputSpace.low
    beta = high - low
    return lambda x: ((x - iLow)/alpha)*beta + low
