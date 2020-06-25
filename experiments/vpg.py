import argparse
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

def mlp(layers: List[int], activation: nn.Module= nn.ReLU, dropout=False, batchNorm=False, outputActivation : nn.Module = torch.nn.Identity) -> nn.Module:

    model = []
    lastOutputSize = layers[0]
    for index, layerSize in enumerate(layers[1:]):
        model.append(nn.Linear(lastOutputSize, layerSize))
        if index < len(layers) - 1:
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


class Actor(nn.Module):

    def __init__(self, observation_space:gym.spaces.Box, action_space: gym.spaces.Box, hidden: List[int]):
        super(Actor, self).__init__()

        layers = [observation_space.shape[0]] + hidden + [action_space.shape[0]]

        self.mu = mlp(layers, activation=nn.ReLU, outputActivation=nn.Tanh)
        std = -0.5 * np.ones(action_space.shape[0])
        self.sigma = nn.Parameter(torch.as_tensor(std, dtype=torch.float32))

    def forward(self, obs):
        mu = self.mu(obs)
        std = torch.exp(self.sigma)

        return mu, std

    def act(self, obs, sample:bool=False):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if sample:
            action = dist.sample()
            return action.cpu().data.numpy(), dist.log_prob(action)
        else:
            return mu, dist.log_prob(mu)


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

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, maxSize=10000):
        self.maxSize = maxSize
        self.state = np.zeros((maxSize, observation_space.shape[0]))
        self.action = np.zeros((maxSize, action_space.shape[0]))
        self.reward = np.zeros(( maxSize, 1 ))
        self.next_state = np.zeros((maxSize, observation_space.shape[0]))
        self.done = np.zeros((maxSize, 1))
        self.ptr = 0
        self.looped = False

    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, state, act, reward, next_state, done):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        self.state[self.ptr,:] = state
        self.action[self.ptr,:] = act
        self.reward[self.ptr,:] = reward
        self.next_state[self.ptr,:] = next_state
        self.done[self.ptr,:] = done

        self.ptr += 1

    def get(self, batchSize=None):
        if batchSize:
            if self.looped:
                idx = np.random.choice(self.maxSize, size=batchSize, replace=False)
            else:
                idx = np.random.choice(self.ptr, size=batchSize, replace=False)

            return self.state[idx,:], self.action[idx,:], self.reward[idx,:], self.next_state[idx,:], self.done[idx,:]
        else:
            if self.looped:
                return self.state, self.action, self.reward, self.next_state, self.done

            else:
                return self.state[:self.ptr,:], self.action[:self.ptr,:], self.reward[:self.ptr,:], self.next_state[:self.ptr,:], self.done[:self.ptr,:]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--a_lr", type=float, default=1e-2)
    parser.add_argument("--c_lr", type=float, default=5e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_episode_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_freq", type=int, default=50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    args = parser.parse_args()
    
    gamma = args.gamma
    a_lr = args.a_lr
    c_lr = args.c_lr
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    batch_size = args.batch_size
    update_freq = args.update_freq

    # Setup

    env = gym.make(args.env)

    actor = Actor(env.observation_space, env.action_space, [256, 64]).to(device)
    actor_target = Actor(env.observation_space, env.action_space, [256, 64]).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(env.observation_space, [256, 64]).to(device)
    critic_target = Critic(env.observation_space, [256, 64]).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actorOptim = torch.optim.Adam(actor_target.parameters(), lr=a_lr)
    criticOptim = torch.optim.Adam(critic_target.parameters(), lr=c_lr)

    print(sum(p.numel() for p in actor.parameters()))
    print(sum(p.numel() for p in critic.parameters()))

    memory = Memory(env.observation_space, env.action_space)

    def update(update_count):

        if memory.size() >= batch_size: 
            states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), memory.get(batchSize=batch_size)))

            criticOptim.zero_grad()
            predicted_value = critic_target(states)
            expected_value = rewards + gamma * critic(next_states) * (1 - dones)
            critic_loss = nn.functional.mse_loss(predicted_value, expected_value)
            critic_loss.backward()
            criticOptim.step()

            actorOptim.zero_grad()
            predicted_value = critic_target(states)
            pred_actions, log_probs = actor_target.act(states)
            actor_loss = (-log_probs * predicted_value).mean()
            actor_loss.backward()
            actorOptim.step()

            if update_count % update_freq == 0:

                actor.load_state_dict(actor_target.state_dict())
                critic.load_state_dict(critic_target.state_dict())


    episodeRewards = []
    update_count = 0

    for episode in range(1, episodes+1):
        state = env.reset()
        total_reward = 0
        # env.render()
        for step in range(max_episode_len):
            action, _ = actor.act(torch.as_tensor(state, dtype=torch.float32).to(device), sample=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            memory.add(state, action, reward, next_state, done)
            state = next_state
            update(update_count)
            update_count += 1
            # env.render()

            if done:
                break

        print(f"Episode {episode}: # Steps = {step}, Reward = {total_reward}")
        episodeRewards.append(total_reward)

