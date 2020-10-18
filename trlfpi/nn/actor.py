import numpy as np

import torch
from torch import nn
from torch.distributions import Normal

from typing import List

from .util import mlp


class NNActor(nn.Module):

    def __init__(self, inputSpace: int,
                 actionSpace: int,
                 hidden: List[int],
                 activation: nn.Module = nn.ReLU,
                 outputActivation: nn.Module = nn.Identity,
                 exploration_sigma: float = 0.25):

        super(NNActor, self).__init__()

        layers = [inputSpace] + hidden + [actionSpace]

        self.mu = mlp(layers,
                      activation=activation,
                      outputActivation=outputActivation,
                      batchNorm=True,
                      dropout=True)
        self.sigma = nn.Parameter(exploration_sigma * torch.ones(actionSpace))

    def forward(self, obs):
        mu = self.mu(obs)
        return mu, self.sigma

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -0.5 * torch.pow((actions - mu) / (std), 2)
        return torch.log((1 / (np.sqrt(2 * np.pi) * std)) * torch.exp(alpha))

    def act(self, obs, sample: bool = False, numpy=False, prevActions=None):
        mu, std = self.forward(obs)
        if sample:
            dist = Normal(mu, std)
            action = dist.sample()
        else:
            action = mu

        if prevActions is not None:
            log_probs = self.log_prob(mu, std, prevActions)
        else:
            log_probs = self.log_prob(mu, std, action)

        if numpy:
            return action.cpu().data.numpy(), log_probs.cpu().data.numpy()
        else:
            return action, log_probs
