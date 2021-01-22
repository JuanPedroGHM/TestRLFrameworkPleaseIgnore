import torch
import random
from typing import Tuple, List


class Memory():

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
            return map(lambda x: torch.stack(x).reshape(self.size, -1), zip(*self.data))

    def reset(self):
        self.data: List[Tuple] = []
