import numpy as np
import torch
from typing import Union
from scipy.spatial.distance import cdist

Tensor = Union[np.ndarray, torch.Tensor]


def rbf_f(x0: Tensor, x1: Tensor, params):
    theta = params[0]
    length = params[1:]

    if length.shape[0] != 1:
        if x0.shape[1] != x1.shape[1] != length.shape[0]:
            raise ValueError("Length is invalid")
    if np.any(length == 0):
        return np.zeros((x0.shape[0], x1.shape[0]))

    dist = cdist(x0 / length, x1 / length, 'sqeuclidean')
    K = theta**2 * np.exp(-.5 * dist)
    return K


def rbf_f_t(x0: Tensor, x1: Tensor, params):

    theta = params[0]
    length = params[1:]

    if length.shape[0] != 1:
        if x0.shape[1] != x1.shape[1] != length.shape[0]:
            raise ValueError("Length is invalid")
    if torch.any(length == 0):
        return torch.zeros((x0.shape[0], x1.shape[0]))

    a = x0 / length
    b = x1 / length
    dist = torch.cdist(a, b, p=2)
    K = theta**2 * torch.exp(-.5 * dist)
    return K
