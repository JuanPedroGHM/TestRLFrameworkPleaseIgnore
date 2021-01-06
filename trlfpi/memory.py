import numpy as np
import torch
import random
from typing import Tuple, List


class GPMemory():

    def __init__(self, inputDims: int, maxSize=500, device=None):

        self.maxSize = maxSize
        if device:

            self.inputs = torch.zeros((maxSize, inputDims), device=device)
            self.outputs = torch.zeros((maxSize, 1), device=device)
        else:
            self.inputs = np.zeros((maxSize, inputDims))
            self.outputs = np.zeros((maxSize, 1))

        self.ptr = 0
        self.looped = False

    @property
    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, x, y):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        assert(x.shape[1] == self.inputs.shape[1])

        self.inputs[self.ptr, :] = x
        self.outputs[self.ptr] = y
        self.ptr += 1

    @property
    def data(self):
        if self.size == self.maxSize:
            return self.inputs, self.outputs
        else:
            return self.inputs[:self.ptr], self.outputs[:self.ptr]


class GymMemory():

    def __init__(self, size):
        self.maxSize = size
        self.data: List[Tuple] = []

    @property
    def size(self):
        return len(self.data)

    def add(self, sample: Tuple):
        self.data.append(sample)
        while self.size > self.maxSize:
            self.data.pop(0)

    def get(self, batchSize=None):
        if batchSize:
            batch = random.choices(self.data, k=batchSize)
            return map(lambda x: torch.stack(x).reshape(batchSize, -1), zip(*batch))
        else:
            return map(lambda x: torch.stack(x).reshape(self.maxSize, -1), zip(*self.data))
