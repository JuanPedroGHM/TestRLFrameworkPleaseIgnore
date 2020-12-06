import torch
from torch.distributions import Normal
from torch import nn

from typing import List

from .util import mlp
from .stochasticActor import StochasticActor


class QFunc(nn.Module):

    def __init__(self, layerSizes: List[int], layerActivations: List[str], layerOptions: List[dict] = None):
        super(QFunc, self).__init__()
        self.layers = mlp(layerSizes, layerActivations, layerOptions, batchNorm=True, dropout=True)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def value(self, x: torch.Tensor, actor: StochasticActor, nSamples: int = 200):
        values = []
        mu, std = actor(x)
        actions = Normal(mu, std).sample([nSamples]).reshape([-1, 1])
        qs = self.forward(torch.cat([actions, x.repeat_interleave(nSamples, dim=0)], axis=1))
        values = qs.reshape([x.shape[0], nSamples]).mean(1, keepdim=True)
        return values
