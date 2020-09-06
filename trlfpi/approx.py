import torch
from torch import nn
from torch.distributions import Normal

from typing import List

import numpy as np


def mlp(layers: List[int], activation: nn.Module = nn.ReLU, dropout=False, batchNorm=False, outputActivation: nn.Module = torch.nn.Identity) -> nn.Module:
    model = []
    lastOutputSize = layers[0]
    for index, layerSize in enumerate(layers[1:]):
        if index < len(layers) - 2:
            model.append(nn.Linear(lastOutputSize, layerSize))
            if batchNorm:
                model.append(nn.LayerNorm(layerSize))
            model.append(activation())
            if dropout:
                model.append(nn.Dropout())
            lastOutputSize = layerSize
        else:
            model.append(nn.Linear(lastOutputSize, layerSize))
            model.append(outputActivation())
    model = nn.Sequential(*model)
    return model


class NNActor(nn.Module):

    def __init__(self, inputSpace: int,
                 actionSpace: int,
                 hidden: List[int],
                 activation: nn.Module = nn.ReLU,
                 outputActivation: nn.Module = nn.Identity):

        super(NNActor, self).__init__()

        layers = [inputSpace] + hidden + [actionSpace]

        self.mu = mlp(layers, activation=activation, outputActivation=outputActivation)
        self.sigma = nn.Parameter(0.0 * torch.ones(actionSpace), requires_grad=False)

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.sigma)

        return mu, std

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -torch.pow((actions - mu) / (2 * std), 2)
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
