import torch
from torch import nn

from typing import List

from .util import mlp


class NNCritic(nn.Module):

    def __init__(self, input_space: int,
                 hidden: List[int],
                 activation: nn.Module = nn.ReLU,
                 outputActivation: nn.Module = nn.Identity):

        super(NNCritic, self).__init__()

        layers = [input_space] + hidden + [1]
        self.layers = mlp(layers,
                          activation=activation,
                          outputActivation=outputActivation,
                          batchNorm=True,
                          dropout=True)

    def forward(self, x: torch.Tensor):
        return self.layers(x)
