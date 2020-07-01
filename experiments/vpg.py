import argparse
from typing import Dict, List, Tuple, Callable

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

    def act(self, obs, sample:bool=False, prevActions=None):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if sample:
            action = dist.sample()
            return action.cpu().data.numpy(), dist.log_prob(action)
        else:
            if prevActions != None:
                return mu, dist.log_prob(prevActions)
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


def normFunc(inputSpace: gym.spaces.Box, low: int = -1, high: int = 1) -> Callable[[np.ndarray],np.ndarray]:

    iHigh = inputSpace.high
    iLow = inputSpace.low
    alpha = inputSpace.high - inputSpace.low
    beta = high - low
    return lambda x: ((x - iLow)/alpha)*beta + low


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--a_lr", type=float, default=1e-3)
    parser.add_argument("--c_lr", type=float, default=5e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max_episode_len", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_freq", type=int, default=100)

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

    # Setup

    env = gym.make(args.env)
    norm = normFunc(env.observation_space)

    actor = Actor(env.observation_space, env.action_space, [50, 16]).to(device)
    critic = Critic(env.observation_space, [50, 16]).to(device)
    
    print(sum(p.numel() for p in actor.parameters()))
    print(sum(p.numel() for p in critic.parameters()))

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)
    criticOptim = torch.optim.Adam(critic.parameters(), lr=c_lr)

    memory = Memory(env.observation_space, env.action_space)


    
    def update():

        mean_loss = 0;
        mean_actorloss = 0;

        if memory.size() >= batch_size: 
            for _ in range(10):
                states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), memory.get(batchSize=batch_size)))

                criticOptim.zero_grad()
                predicted_value = critic(states)
                with torch.no_grad():
                    expected_value = rewards + discount * critic(next_states) * (1 - dones)
                critic_loss = nn.functional.mse_loss(predicted_value, expected_value)
                critic_loss.backward()
                criticOptim.step()

                ## Gather stats
                mean_loss += critic_loss.item()

            states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), memory.get()))
            actorOptim.zero_grad()
            import pdb; pdb.set_trace()
            predicted_value = critic(states)
            pred_actions, log_probs = actor.act(states, sample=False, prevActions=actions)
            actor_loss = (-log_probs * (predicted_value - rewards.mean())).mean()
            actor_loss.backward()
            actorOptim.step()

            #Gather stats

            print(f'Critic loss: {mean_loss/10}, Actor loss: {mean_actorloss/10}')
                


    episodeRewards = []
    steps = 0

    for episode in range(1, episodes+1):
        state = norm(env.reset())
        total_reward = 0
        # env.render()
        for step in range(max_episode_len):
            action, _ = actor.act(torch.as_tensor(state, dtype=torch.float32).to(device), sample=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = norm(next_state)

            memory.add(state, action, reward, next_state, done)
            state = next_state
            steps += 1
            if steps % update_freq == 0:
                update()

            if done:
                break

        print(f"Episode {episode}: # Steps = {step}, Reward = {total_reward}")
        episodeRewards.append(total_reward)

