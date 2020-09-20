import numpy as np
from scipy.spatial.distance import cdist
from typing import Callable, List


class Kernel():

    def __init__(self, f: Callable = None, params: np.ndarray = None, bounds: List[List] = None):
        self.kernelFunctions: List[Callable] = []
        self.paramsArray: List[np.ndarray] = []
        self.bounds: List[List] = []

        if f is not None and params is not None and bounds is not None:
            self.kernelFunctions.append(f)
            self.paramsArray.append(params)
            self.bounds.extend(bounds)

    def __call__(self, x1: np.ndarray, x2: np.ndarray = None, customParams: np.ndarray = None) -> np.ndarray:

        if x2 is None:
            x2 = x1

        result = np.zeros((x1.shape[0], x2.shape[0]))

        if customParams is not None:

            paramIndex = 0
            for f, params in zip(self.kernelFunctions, self.paramsArray):

                tmpParams = customParams[paramIndex:paramIndex + len(params)]
                paramIndex += len(params)

                result += f(x1, x2, tmpParams)
            return result

        else:
            for f, params in zip(self.kernelFunctions, self.paramsArray):
                result += f(x1, x2, params)

            return result

    def __add__(self, kernel2):
        self.kernelFunctions.extend(kernel2.kernelFunctions)
        self.paramsArray.extend(kernel2.paramsArray)
        self.bounds.extend(kernel2.bounds)
        return self

    @property
    def params(self) -> np.ndarray:
        return np.hstack(self.paramsArray)

    @params.setter
    def params(self, newParams: np.ndarray):
        newParamsIndex = 0
        for index, params in enumerate(self.paramsArray):
            self.paramsArray[index] = newParams[newParamsIndex: newParamsIndex + len(params)]
            newParamsIndex += len(params)

    @classmethod
    def RBF(cls, theta=1.0, lengths=1.0, bounds=None):

        def f(x0, x1, params):
            theta = params[0]
            length = params[1:]

            if length.shape[0] != 1:
                if x0.shape[1] != x1.shape[1] != length.shape[0]:
                    raise ValueError("Lenght is invalid")

            l2 = np.power(length, 2)

            dist = cdist(x0 / l2, x1 / l2, 'sqeuclidean')
            K = theta**2 * np.exp(-.5 * dist)
            return K

        params = np.hstack((theta, lengths))
        if bounds is None:
            bounds = [[1e-6, np.inf] for i in range(params.shape[0])]
        return cls(f, params, bounds)
