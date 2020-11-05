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
                 outputActivation: nn.Module = nn.Identity):

        super(NNActor, self).__init__()

        encoderLayers = [inputSpace] + hidden
        actionLayers = [hidden[-1], actionSpace]
        sigmaLayers = [hidden[-1], actionSpace]

        self.encoder = mlp(encoderLayers,
                           activation=activation,
                           outputActivation=activation,
                           batchNorm=True,
                           dropout=True)
        self.mu = mlp(actionLayers,
                      activation=activation,
                      outputActivation=outputActivation)
        self.sigma = mlp(sigmaLayers,
                         activation=activation,
                         outputActivation=nn.ReLU)

    def forward(self, obs):
        z = self.encoder(obs)
        mu = self.mu(z)
        sigma = self.sigma(z) + 1e-6
        return mu, sigma

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -0.5 * torch.pow((actions - mu) / (std), 2)
        return alpha - torch.log((np.sqrt(2 * np.pi) * std))

    def act(self, obs, sample: bool = False, numpy=False, prevActions=None):
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

        if numpy:
            return action.cpu().data.numpy(), log_probs.cpu().data.numpy()
        else:
            return action, log_probs
