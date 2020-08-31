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
from trlfpi.memory import GymMemory, GPMemory
from trlfpi.gp import GP, Kernel


class Actor(nn.Module):

    def __init__(self, inputSpace: int,
                 actionSpace: int,
                 hidden: List[int]):

        super(Actor, self).__init__()

        layers = [inputSpace] + hidden + [actionSpace]

        self.mu = mlp(layers, activation=nn.ReLU, outputActivation=nn.Identity)
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


class Critic():

    def __init__(self,
                 inputSpace: int,
                 memory: int = 500):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        self.model = GP(Kernel.RBF() + Kernel.Noise())

    def update(self, x: np.ndarray, y: np.ndarray):
        self.memory.add(x, y)
        X, Y = self.memory.data
        self.model.fit(X, Y)

    def predict(self, x: np.ndarray):
        if self.memory.size != 0:
            return self.model(x)
        else:
            return (0, 0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--nRefs", type=int, default=3)
    parser.add_argument("--a_lr", type=float, default=5e-5)
    parser.add_argument("--c_lr", type=float, default=1e-5)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--update_freq", type=int, default=10)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--plots", action='store_true')

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
    systemPlots = args.plots

    # writer = SummaryWriter(f'results/vpg/{args.env}')

    # Setup
    env = gym.make(args.env)
    nRefs = args.nRefs

    # Init policy network
    actor = Actor(2 + nRefs, 1, [128, 32]).to(device)
    actorTarget = Actor(2 + nRefs, 1, [128, 32]).to(device)
    actorTarget.load_state_dict(actor.state_dict())
    actorTarget.eval()

    actorOptim = torch.optim.Adam(actor.parameters(), lr=a_lr)

    replayBuff = GymMemory(env.observation_space, env.action_space)

    # Init critic GP
    critic = Critic(1 + nRefs, 200)

    def actWithRef(actorNet, x: np.ndarray, sample=False):
        # x : B x X
        currentState = x[:, :2]
        refs = x[:, 2:]

        # B x T x D
        state_p = np.zeros((nRefs, 2))
        deltas = np.zeros((nRefs, 1))
        actions = np.zeros((nRefs, 1))
        for i in range(nRefs):
            actorInput = torch.as_tensor(
                np.hstack((currentState, refs[:, i:nRefs + i])),
                dtype=torch.float32
            ).to(device)
            actions[i, :], _ = actorNet.act(actorInput, numpy=True, sample=sample)
            currentState = env.predict(currentState.T, actions[[i], :]).T
            state_p[i, :] = currentState

        deltas = state_p[:, [0]].T - refs[:, :nRefs]
        return actions[[0], :], deltas

    def updateCritic(action, deltas, reward, next_action, next_deltas):
        # Calculate TD(0) error and update critic GP
        X = np.hstack((action, deltas))
        nX = np.hstack((next_action, next_deltas))
        Y = reward + discount * critic(nX)[0]
        critic.update(X, Y)

    def updateActor():

        actor_loss = 0
        if replayBuff.size >= batch_size:

            states, actions, rewards, next_states, dones = tuple(map(lambda x: torch.as_tensor(x, dtype=torch.float32).to(device), replayBuff.get(batchSize=batch_size)))

            # Optimize actor
            actorOptim.zero_grad()

            # B x T x D
            currentStates = states[:, :2]
            refs = states[:, 2:]
            state_p = torch.zeros((batch_size, nRefs, 2))
            deltas = torch.zeros((batch_size, nRefs))
            next_actions = torch.zeros((batch_size, nRefs))
            log_probs = torch.zeros((batch_size, nRefs))
            
            for i in range(nRefs):
                actorInput = torch.cat((currentStates, refs[:,i:nRefs + i]), 1)

                tmpAct, tmpLP = actor.act(actorInput, sample=False, prevActions=actions)
                next_actions[:, [i]] = tmpAct
                log_probs[:, [i]] = tmpLP

                currentState = torch.as_tensor(env.predict(currentStates.T.data.numpy(), actions[:, [0]].T.data.numpy()))
                currentState = currentState.T
                state_p[:, i, :] = currentState

            deltas = state_p[:,:,0] - refs[:, :nRefs]

            criticInput = torch.cat((actions[:, [0]], deltas), 1).cpu().data.numpy()
            q_values, _ = critic(criticInput)

            actor_loss = (-log_probs[:, [0]] * (torch.as_tensor(q_values) - rewards.mean())).mean()
            actor_loss.backward()
            actorOptim.step()

        return actor_loss

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        epsilon = np.exp(-episode * epsilonDecay / episodes)

        states = [state[0]]

        for step in range(max_episode_len):
            sample = (np.random.random() < epsilon)
            action, deltas = actWithRef(actor, state, sample=sample)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update actor and critic
            next_action, next_deltas = actWithRef(actorTarget, next_state, sample=False)
            updateCritic(action, deltas, reward, next_action, next_deltas)
            actor_loss = updateActor()
            # if actor_loss != 0:
            #     writer.add_scalar('Actor loss', actor_loss)

            replayBuff.add(state, action, reward, next_state, done)
            states.append(next_state[0])

            if done:
                break

            state = next_state

        if episode % update_freq == 0:
            actorTarget.load_state_dict(actor.state_dict())

            # Plot to see how it looks
            if systemPlots:
                plt.figure(1)
                plt.plot(states, label='x')
                plt.plot(env.reference.r, '--', label='r')

                plt.grid()
                plt.legend()
                plt.show()

        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon:.2f}")
        # writer.add_scalar('Episode Rewards', total_reward, episode)
        # writer.add_scalar('Epsilon', epsilon, episode)
