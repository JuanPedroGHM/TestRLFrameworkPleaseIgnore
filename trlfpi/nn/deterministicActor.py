import torch
from torch import nn

from typing import List
from .util import mlp


class DeterministicActor(nn.Module):

    def __init__(self, layerSizes: List[int], layerActivations: List[str], layerOptions: List[dict] = None):

        super(DeterministicActor, self).__init__()
        self.mu = mlp(layerSizes,
                      layerActivations,
                      layerOptions,
                      batchNorm=True,
                      dropout=True)

    def forward(self, obs):
        mu = self.mu(obs)
        return mu

    def act(self, obs):
        action = self.forward(obs)
        return action
