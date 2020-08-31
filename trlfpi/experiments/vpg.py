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

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, hidden: List[int]):
        super(Actor, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [action_space.shape[0]]

        self.mu = mlp(layers, activation=nn.ReLU, outputActivation=nn.Tanh)
        self.sigma = nn.Parameter(0.0 * torch.ones(action_space.shape[0]), requires_grad=False)

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.sigma)

        return mu, std

    def log_prob(self, mu: torch.Tensor, std: torch.Tensor, actions: torch.Tensor):
        alpha = -torch.pow((actions - mu) / (2 * std), 2)
        return torch.log((1 / (np.sqrt(2 * np.pi) * std)) * torch.exp(alpha))

    def act(self, obs, sample=False, numpy=False, prevActions=None):
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


class Critic(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, hidden: List[int]):
        super(Critic, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [1]
        self.v_net = mlp(layers)

    def forward(self, obs):
        return self.v_net(obs)

    def value(self, obs):
        return self.forward(obs).numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
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
    norm = normFunc(env.observation_space)

    actor = Actor(env.observation_space, env.action_space, [128, 32]).to(device)
    critic = Critic(env.observation_space, [128, 32]).to(device)

    print(sum(p.numel() for p in actor.parameters()))
    print(sum(p.numel() for p in critic.parameters()))

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)
    criticOptim = torch.optim.Adam(critic.parameters(), lr=c_lr)

    memory = Memory(env.observation_space, env.action_space)

    def update():

        criticLossLog = 0
        actorLossLog = 0

        if memory.size() >= batch_size:
            for _ in range(10):
                states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), memory.get(batchSize=batch_size)))

                criticOptim.zero_grad()
                predicted_value = critic(states)
                expected_value = rewards + discount * critic(next_states) * (1 - dones)
                critic_loss = nn.functional.mse_loss(predicted_value, expected_value)
                critic_loss.backward()
                criticOptim.step()

                # Gather stats
                criticLossLog += critic_loss.item()

                actorOptim.zero_grad()
                predicted_value = critic(states)
                pred_actions, log_probs = actor.act(states, sample=False, prevActions=actions)
                actor_loss = (-log_probs * (predicted_value - rewards.mean())).mean()
                actor_loss.backward()
                actorOptim.step()

                # Gather stats
                actorLossLog += actor_loss.item()

        return criticLossLog / 10, actorLossLog / 10, actor.sigma.item()

    criticLoss = []
    actorLoss = []
    actorSigma = []
    episodeRewards = []
    stepsTaken = []

    updates = 0
    for episode in range(1, episodes + 1):
        state = norm(env.reset())
        total_reward = 0
        epsilon = np.exp(-episode * epsilonDecay / episodes)

        for step in range(max_episode_len):

            sample = (np.random.random() < epsilon)
            action, _ = actor.act(torch.as_tensor(state, dtype=torch.float32).to(device), sample=sample, numpy=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = norm(next_state)

            memory.add(state, action, reward, next_state, done)
            state = next_state
            updates += 1
            if updates % update_freq == 0:
                cl, al, sigma = update()
                writer.add_scalar('Critic Loss', cl, episode)
                writer.add_scalar('Actor Loss', al, episode)
                writer.add_scalar('Sigma', sigma, episode)

            if done:
                break

        print(f"Episode {episode}: # Steps = {step}, Reward = {total_reward}, Epsilon: {epsilon:.2f}")
        writer.add_scalar('Epsiode Rewards', total_reward, episode)
        writer.add_scalar('Epsiode length', step, episode)
        writer.add_scalar('Epsilon', epsilon, episode)
