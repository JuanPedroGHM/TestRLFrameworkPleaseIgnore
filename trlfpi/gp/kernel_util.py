import numpy as np
from scipy.spatial.distance import cdist


def rbf_f(x0, x1, params):
    theta = params[0]
    length = params[1:]

    if length.shape[0] != 1:
        if x0.shape[1] != x1.shape[1] != length.shape[0]:
            raise ValueError("Lenght is invalid")
    if np.any(length == 0):
        return np.zeros((x0.shape[0], x1.shape[0]))

    dist = cdist(x0 / length, x1 / length, 'sqeuclidean')
    K = theta**2 * np.exp(-.5 * dist)
    return K
