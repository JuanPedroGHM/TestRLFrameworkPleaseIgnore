import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from typing import Callable, List, Tuple


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

                tmpParams = customParams[paramIndex :paramIndex+len(params)]
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
    def RBF(cls, alpha = 1.0, length=1.0):
        f = lambda x0, x1, params: params[0] * rbf_kernel(x0, Y=x1, gamma=params[1])
        return cls(f, np.array([alpha, length]), [[1e-5, np.inf], [1e-5, np.inf]])

    @classmethod
    def Noise(cls, noise=1.0):
        def f(x1, x2, params):
            if x1.shape[0] == x2.shape[0]:
                return params[0] * np.identity(x1.shape[0], dtype=float)
            else:
                return np.zeros((x1.shape[0], x2.shape[0]))

        return cls(f, np.array([noise]), [[1e-5, np.inf]])
