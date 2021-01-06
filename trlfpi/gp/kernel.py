import torch
import numpy as np
from torch import Tensor
from typing import Callable, List, Union

from trlfpi.gp.kernel_util import rbf_f


class Kernel():

    def __init__(self, f: Callable[[Tensor, Tensor, Tensor], Tensor] = None, params: Tensor = None, bounds: List[List] = None):
        self.kernelFunctions: List[Callable[[Tensor, Tensor, Tensor], Tensor]] = []
        self.paramsArray: List[Tensor] = []
        self.bounds: List[List] = []

        if f is not None and params is not None and bounds is not None:
            self.kernelFunctions.append(f)
            self.paramsArray.append(params)
            self.bounds.extend(bounds)

    def __call__(self, x1: Tensor, x2: Tensor = None, customParams: Tensor = None) -> Tensor:

        if x2 is None:
            x2 = x1

        result = torch.zeros((x1.shape[0], x2.shape[0])).to(x1.device)

        if customParams is not None:

            paramIndex = 0
            for f, params in zip(self.kernelFunctions, self.paramsArray):

                tmpParams = customParams[paramIndex:paramIndex + len(params)]
                paramIndex += len(params)

                result = result + f(x1, x2, tmpParams)
            return result

        else:
            for f, params in zip(self.kernelFunctions, self.paramsArray):
                result = result + f(x1, x2, params)

            return result

    def __add__(self, kernel2):
        self.kernelFunctions.extend(kernel2.kernelFunctions)
        self.paramsArray.extend(kernel2.paramsArray)
        self.bounds.extend(kernel2.bounds)
        return self

    @property
    def params(self) -> Tensor:
        return torch.cat(self.paramsArray, axis=0)

    @params.setter
    def params(self, newParams: Tensor):
        newParamsIndex = 0
        for index, params in enumerate(self.paramsArray):
            self.paramsArray[index] = newParams[newParamsIndex: newParamsIndex + len(params)]
            newParamsIndex += len(params)

    @classmethod
    def RBF(cls, theta: float, lengths: List, bounds=None):

        if bounds is None:
            bounds = [[0, np.inf] for i in range(1 + len(lengths))]

        params = np.hstack((theta, lengths))
        return cls(rbf_f, params, bounds)
