import torch
from torch import nn
import torch.nn.functional as F


class BiasedLiner(nn.Module):

    def __init__(self, offset=20.0):
        super(BiasedLiner, self).__init__()
        self.bias = nn.Parameter(torch.tensor(offset), requires_grad=True)

    def forward(self, x):
        return x + self.bias


class InvertedRELU(nn.Module):

    def __init__(self):
        super(InvertedRELU, self).__init__()

    def forward(self, x):
        return - F.relu(x)
