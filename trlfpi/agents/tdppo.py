import torch
import numpy as np
from torch.distributions import Normal
from typing import List, Tuple
from itertools import chain

from .agent import Agent
from ..memory import Memory
from ..nn.stochasticActor import StochasticActor
from ..nn.critic import VFunc


@Agent.register("tdppo")
class TDPPO(Agent):

    default_config: dict = {
        'name': 'tdppo',

        # Env
        'h': 1,
        'discount': 0.7,

        # Policy Network
        'a_layers': [2, 64, 64, 1],
        'a_activation': ['tahn', 'tahn', 'identity'],
        'a_layerOptions': None,
        'a_lr': 1e-5,

        # Critic Network
        'c_layers': [3, 64, 64, 1],
        'c_activation': ['tahn', 'tahn', 'invRelu'],
        'c_layerOptions': None,
        'c_lr': 1e-3,

        # PPO
        'clip': 0.2,
        'klCost': 0.01,
        'explorationNoise': 1.0,

        'tau': 1e-3,
        'weightDecay': 1e-3,
        'batchSize': 512,
        'update_freq': 2,

        # ReplayBuffer
        'bufferSize': 10000
    }

    def setup(self, checkpoint: dict = None, device: str = 'cpu'):
        if checkpoint:
            self.config = checkpoint['config']

        self.device = device
        self.actor = StochasticActor(self.config['a_layers'],
                                     self.config['a_activation'],
                                     self.config['a_layerOptions'],
                                     explorationNoise=self.config['explorationNoise']).to(device)

        if checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])

        self.actorOptim = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.config['a_lr'],
                                           weight_decay=self.config['weightDecay'])

        self.c1 = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
        self.c1T = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
        if checkpoint:
            self.c1.load_state_dict(checkpoint['c1'])
            self.c1T.load_state_dict(checkpoint['c1T'])
        else:
            self.c1T.load_state_dict(self.c1.state_dict())
        self.c1T.eval()

        self.c2 = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
        self.c2T = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
        if checkpoint:
            self.c2.load_state_dict(checkpoint['c2'])
            self.c2T.load_state_dict(checkpoint['c2T'])
        else:
            self.c2T.load_state_dict(self.c2.state_dict())
        self.c2T.eval()

        self.criticOptim = torch.optim.Adam(chain(self.c1.parameters(), self.c2.parameters()),
                                            lr=self.config['c_lr'],
                                            weight_decay=self.config['weightDecay'])
        self.criticLossF = torch.nn.MSELoss()

        self.replayBuff = Memory(self.config['bufferSize'])
        self.h = self.config['h']
        self.updates = 0
        self.tau = self.config['tau']
        self.discount = self.config['discount']
        self.clip = self.config['clip']
        self.klCost = self.config['klCost']

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actorInput = torch.tensor(np.hstack([ref[:, 0:self.h + 1] - state[:, [0]],
                                             state[:, [1]]]), device=self.device)

        if self.mode == 'train':
            with torch.no_grad():
                self.actor.eval()
                action, log_probs = self.actor.act(actorInput)
            return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

        else:
            with torch.no_grad():
                self.actor.eval()
                action, log_probs = self.actor.act(actorInput, sample=False)
            return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def update(self, stepData: List[np.ndarray] = None) -> dict:

        # Save new values if any
        if stepData:
            self.replayBuff.add(tuple(map(lambda x: torch.tensor(x, device=self.device), stepData)))
        if self.replayBuff.size < self.config['batchSize']:
            return {'actor_loss': 0}

        # Get training batch
        states, refs, actions, log_probs, rewards, dones, next_states = self.replayBuff.get(self.config['batchSize'])

        # Optimize critic
        self.c1.train()
        self.c2.train()
        self.criticOptim.zero_grad()
        netInput = torch.cat([refs[:, 0:self.h + 1] - states[:, [0]],
                                states[:, [1]]], axis=1)
        netNextInput = torch.cat([refs[:, 1:self.h + 2] - next_states[:, [0]],
                                next_states[:, [1]]], axis=1)

        v1 = self.c1(netInput)
        v2 = self.c2(netInput)

        next_v = (1 - dones) * torch.min(self.c1T(netNextInput), self.c2T(netNextInput)).detach()

        c1_loss = self.criticLossF(v1, rewards + self.discount * next_v)
        c2_loss = self.criticLossF(v2, rewards + self.discount * next_v)
        critic_loss = c1_loss + c2_loss
        critic_loss.backward()
        self.criticOptim.step()

        # Optimize actor
        self.actor.train()
        self.actorOptim.zero_grad()

        # Get current log_probs and entropy
        mus, sigmas = self.actor(netInput)
        dist = Normal(mus, sigmas)
        c_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Advantage r + discount * v(s_t+1) - v(s_t)
        adv = rewards + self.discount * self.c1(netNextInput) - self.c1(netInput)

        # Importance Weight
        ratios = torch.exp(c_log_probs - log_probs.detach()) + 1e-9
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
        actor_loss = - torch.min(surr1, surr2) - self.klCost * entropy
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actorOptim.step()

        self.updates += 1

        if self.updates % self.config['update_freq'] == 0:
            for targetP, oP in zip(self.c1T.parameters(), self.c1.parameters()):
                targetP = (1 - self.tau) * targetP + self.tau * oP
            for targetP, oP in zip(self.c2T.parameters(), self.c2.parameters()):
                targetP = (1 - self.tau) * targetP + self.tau * oP

        return {'actor_loss': actor_loss.item(), 'c1_loss': c1_loss.item(), 'c2_loss': c2_loss.item()}

    def toDict(self) -> dict:
        cp: dict = {
            'config': self.config,
            'actor': self.actor.state_dict(),
            'c1': self.c1.state_dict(),
            'c1T': self.c1T.state_dict(),
            'c2': self.c2.state_dict(),
            'c2T': self.c2T.state_dict()
        }
        return cp
