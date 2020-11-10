import torch
from torch.distributions import Normal
from torch import nn

from typing import List

from .util import mlp
from .actor import NNActor


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

    def value(self, x: torch.Tensor, actor: NNActor, nSamples: int = 200):
        values = []
        mu, std = actor(x)
        actions = Normal(mu, std).sample([nSamples]).reshape([-1, 1])
        qs = self.forward(torch.cat([actions, x.repeat_interleave(nSamples, dim=0)], axis=1))
        values = qs.reshape([x.shape[0], nSamples]).mean(1, keepdim=True)
        return values
