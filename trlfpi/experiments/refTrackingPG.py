import argparse
from typing import Callable, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from trlfpi.approx import mlp
from trlfpi.memory import Memory
from trlfpi.util import generatePlot, normFunc

class Actor(nn.Module):

    def __init__(self, observation_space:gym.spaces.Box, action_space: gym.spaces.Box, hidden: List[int]):
        super(Actor, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [action_space.shape[0]]

        self.mu = mlp(layers, activation=nn.ReLU, outputActivation=nn.Tanh)
        self.sigma = nn.Parameter(0.0 * torch.ones(action_space.shape[0]), requires_grad=False)

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.sigma)

        return mu, std

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -torch.pow((actions - mu)/(2*std), 2)
        return torch.log((1/(np.sqrt(2*np.pi)*std))* torch.exp(alpha))

    def act(self, obs, sample:bool=False, numpy=False, prevActions=None):
        mu, std = self.forward(obs)
        if sample:
            dist = Normal(mu, std)
            action = dist.sample()
        else:
            action = mu

        if prevActions != None:
            log_probs = self.log_prob(mu, std, prevActions)
        else:
            log_probs = self.log_prob(mu, std, action)

        if numpy:
            return action.cpu().data.numpy(), log_probs.cpu().data.numpy()
        else:
            return action, log_probs

class Critic(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, hidden: List[int]):
        super(Critic, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [1]
        self.v_net = mlp(layers)

    def forward(self, obs) :
        return self.v_net(obs)

    def value(self, obs):
        return self.forward(obs).numpy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--a_lr", type=float, default=5e-5)
    parser.add_argument("--c_lr", type=float, default=1e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--update_freq", type=int, default=50)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    a_lr = args.a_lr
    c_lr = args.c_lr
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    batch_size = args.batch_size
    update_freq = args.update_freq
    epsilonDecay = args.epsilonDecay

    writer = SummaryWriter(f'results/vpg/{args.env}')

    # Setup
    env = gym.make(args.env)

