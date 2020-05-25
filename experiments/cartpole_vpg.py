import argparse
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

env: gym.Env = gym.make("MountainCarContinuous-v0")

def mlp(layers: List[int], activation: nn.Module= nn.ReLU, dropout=False, batchNorm=False, outputActivation : nn.Module = torch.nn.Identity) -> nn.Module:

    model = []
    lastOutputSize = layers[0]
    for index, layerSize in enumerate(layers[1:]):
        model.append(nn.Linear(lastOutputSize, layerSize))
        if index < len(layers) - 1:
            if batchNorm:
                model.append(nn.BatchNorm1d(layerSize))
            model.append(activation())
            if dropout:
                model.append(nn.Dropout())
            lastOutputSize = layerSize
        else:
            model.append(nn.Linear(lastOutputSize, layerSize))
            model.append(outputActivation())

    model = nn.Sequential(*model)
    return model


class Actor(nn.Module):

    def __init__(self, observation_space:gym.spaces.Box, action_space: gym.spaces.Box, hidden: List[int]):
        super(Actor, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [action_space.shape[0]]

        self.mu = mlp(layers, activation=nn.ReLU)
        std = -0.5 * np.ones(action_space.shape[0])
        self.sigma = nn.Parameter(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.sigma)

        dist = Normal(mu, std)
        return dist

    def log_probs(self, states, actions):
        dist = self.forward(states)
        return dist.log_prob(actions)



    def act(self, obs):
        with torch.no_grad():
            dist = self.forward(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            return action.numpy(), log_prob.numpy()


class Critic(nn.Module):

    def __init__(self, observation_space: gym.spaces.Box, hidden: List[int]):
        super(Critic, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [1]
        self.v_net = mlp(layers)

    def forward(self, obs) :
        return self.v_net(obs)

    def value(self, obs):
        return self.forward(obs).numpy()


class Memory():

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, maxSize=1000):
        self.maxSize = maxSize
        self.state = np.zeros((maxSize, observation_space.shape[0]))
        self.action = np.zeros((maxSize, action_space.shape[0]))
        self.reward = np.zeros(( maxSize, 1 ))
        self.next_state = np.zeros((maxSize, observation_space.shape[0]))
        self.next_action = np.zeros((maxSize, observation_space.shape[0]))
        self.ptr = 0
        self.looped = False

    def add(self, state, act, reward, next_state, next_action):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        self.state[self.ptr,:] = state
        self.action[self.ptr,:] = act
        self.reward[self.ptr,:] = reward
        self.next_state[self.ptr,:] = next_state
        self.next_action[self.ptr,:] = next_action

        self.ptr += 1

    def get(self, batchSize=None):
        if batchSize:
            if self.looped:
                idx = np.random.choice(self.maxSize, size=batchSize, replace=False)
            else:
                idx = np.random.choice(self.ptr, size=batchSize, replace=False)

            return self.state[idx,:], self.action[idx,:], self.reward[idx,:], self.next_state[idx,:], self.next_action[idx,:]
        else:
            if self.looped:
                return self.state, self.action, self.reward, self.next_state, self.next_action

            else:
                return self.state[:self.ptr,:], self.action[:self.ptr,:], self.reward[:self.ptr,:], self.next_state[:self.ptr,:], self.next_action[:self.ptr,:]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--a_lr", type=float, default=0.0001)
    parser.add_argument("--c_lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max_episode_len", type=int, default=1000)

    args = parser.parse_args()
    
    gamma = args.gamma
    a_lr = args.a_lr
    c_lr = args.c_lr
    episodes = args.episodes
    max_episode_len = args.max_episode_len

    # Setup

    env = gym.make(args.env)

    actor = Actor(env.observation_space, env.action_space, [256, 64])
    critic = Critic(env.observation_space, [256, 64])

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)
    criticOptim = torch.optim.Adam(actor.parameters(), lr=c_lr)

    memory = Memory(env.observation_space, env.action_space)

    def update():

        states, actions, rewards, next_states, _ = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32), memory.get()))
        
        actorOptim.zero_grad()
        predicted_value = critic(states)
        expected_value = rewards + gamma * critic(next_states)
        tempDiff: torch.Tensor = expected_value - predicted_value
        log_probs = actor.log_probs(states, actions)
        actor_loss = -(log_probs * tempDiff).mean()
        actor_loss.backward()
        actorOptim.step()

        criticOptim.zero_grad()
        predicted_value = critic(states)
        expected_value = rewards + gamma * critic(next_states)
        tempDiff: torch.Tensor = expected_value - predicted_value
        critic_loss = (tempDiff**2).mean()
        critic_loss.backward()
        criticOptim.step()


    episodeRewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        env.render()
        for step in range(max_episode_len):
            action, _ = actor.act(torch.as_tensor(state, dtype=torch.float32))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            memory.add(state, action, reward, next_state, action)
            state = next_state
            env.render()

            if done:
                break

        update()
        print(f"Episode {episode}: # Steps = {step}, Reward = {total_reward}")
        episodeRewards.append(total_reward)

    plt.plot(episodeRewards)
    plt.show()
