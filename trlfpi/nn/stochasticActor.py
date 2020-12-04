import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from typing import List
from .util import mlp


class StochasticActor(nn.Module):

    def __init__(self, layerSizes: List[int], layerActivations: List[str]):

        super(StochasticActor, self).__init__()
        self.encoder = mlp(layerSizes[:-1],
                           layerActivations[:-1],
                           batchNorm=True,
                           dropout=True)
        self.mu = mlp(layerSizes[-2:],
                      layerActivations[-1:])
        self.sigma = mlp(layerSizes[-2:],
                         ['relu'])

    def forward(self, obs):
        z = self.encoder(obs)
        mu = self.mu(z)
        sigma = self.sigma(z) + 1e-6
        return mu, sigma

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -0.5 * torch.pow((actions - mu) / (std), 2)
        return alpha - torch.log((np.sqrt(2 * np.pi) * std))

    def act(self, obs, sample: bool = True, prevActions=None):
        mu, std = self.forward(obs)

        if sample:
            dist = Normal(mu, std)
            action = dist.rsample()
        else:
            action = mu

        if prevActions is not None:
            log_probs = self.log_prob(mu, std, prevActions)
        else:
            log_probs = self.log_prob(mu, std, action)

        return action, log_probs
