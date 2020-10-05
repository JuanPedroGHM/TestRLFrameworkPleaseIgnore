import numpy as np
import torch
from typing import Callable, List, Union

from trlfpi.gp.kernel_util import rbf_f, rbf_f_t

Tensor = Union[np.ndarray, torch.Tensor]


class Kernel():

    def __init__(self, f: Callable[[Tensor], Tensor] = None, params: Tensor = None, bounds: List[List] = None):
        self.kernelFunctions: List[Callable[[Tensor], Tensor]] = []
        self.paramsArray: List[Tensor] = []
        self.bounds: List[List] = []

        if f is not None and params is not None and bounds is not None:
            self.kernelFunctions.append(f)
            self.paramsArray.append(params)
            self.bounds.extend(bounds)

    def __call__(self, x1: Tensor, x2: Tensor = None, customParams: Tensor = None) -> Tensor:

        if x2 is None:
            x2 = x1

        if type(x1) == torch.Tensor:
            result = torch.zeros((x1.shape[0], x2.shape[0])).to(x1.device)

        else:
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
    def params(self) -> Tensor:
        return np.hstack(self.paramsArray)

    @params.setter
    def params(self, newParams: Tensor):
        newParamsIndex = 0
        for index, params in enumerate(self.paramsArray):
            self.paramsArray[index] = newParams[newParamsIndex: newParamsIndex + len(params)]
            newParamsIndex += len(params)

    @classmethod
    def RBF(cls, theta: float, lengths: List, bounds=None, gpu=False):

        if bounds is None:
            bounds = [[0, np.inf] for i in range(1 + len(lengths))]

        if gpu:
            params = torch.tensor([theta] + lengths)
            return cls(rbf_f_t, params, bounds)
        else:
            params = np.hstack((theta, lengths))
            return cls(rbf_f, params, bounds)

