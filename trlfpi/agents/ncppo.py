from typing import List, Tuple
import numpy as np
import torch
from torch.distributions import Normal

from .agent import Agent
from ..envs import LinearEnv, ClutchEnv
from ..memory import Memory
from ..nn.stochasticActor import StochasticActor
from ..nn.critic import VFunc


@Agent.register("nc-ppo")
class NCPPO(Agent):

    default_config: dict = {
        'name': 'nc-ppo',

        # Env
        'h': 1,
        'discount': 0.7,

        # Policy Network
        'a_layers': [4, 64, 64, 1],
        'a_activation': ['tahn', 'tahn', 'identity'],
        'a_layerOptions': None,
        'a_inCenter': [0.0, 0.0, 0.0, 0.0],
        'a_lr': 1e-5,

        # Model and critic
        'model': 'clutch',  # Can be linear, clutch or a GP checkpoint
        'baseline': 'zero',      # Options: ['meanRewards', 'GAE']

        # Critic Network
        'c_layers': [4, 64, 64, 1],
        'c_activation': ['tahn', 'tahn', 'invRelu'],
        'c_layerOptions': None,
        'c_inCenter': [0.0, 0.0, 0.0, 0.0],
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

        if self.config['model'] == 'clutch':
            self.modelF = ClutchEnv().system
        elif self.config['model'] == 'linear':
            self.modelF = LinearEnv().system
        else:
            raise Exception('Use of GP not implemented')

        if self.config['baseline'] == 'GAE':
            self.critic = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
            self.criticTarget = VFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
            if checkpoint:
                self.critic.load_state_dict(checkpoint['critic'])
                self.criticTarget.load_state_dict(checkpoint['criticTarget'])
            else:
                self.criticTarget.load_state_dict(self.critic.state_dict())
            self.criticTarget.eval()
            self.criticOptim = torch.optim.Adam(self.critic.parameters(),
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

        self.inCenter = torch.tensor(self.config['a_inCenter'], device=self.device).reshape([1, -1])

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actorInput = torch.tensor(np.hstack([state, ref[:, 0:self.h + 1]]), device=self.device) - self.inCenter

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
        if self.config['baseline'] == 'GAE':
            self.critic.train()
            self.criticOptim.zero_grad()
            cInput = torch.cat((states, refs[:, 0:self.h + 1]), axis=1) - self.inCenter
            cNextInput = torch.cat([next_states, refs[:, 1:self.h + 2]], axis=1) - self.inCenter

            v = self.critic(cInput)
            next_v = (1 - dones) * self.criticTarget(cNextInput).detach()

            critic_loss = self.criticLossF(v, rewards + self.discount * next_v)
            critic_loss.backward()
            self.criticOptim.step()

            baseline = self.critic(cInput)
        elif self.config['baseline'] == 'zero':
            baseline = 0
        elif self.config['baseline'] == 'meanRewards':
            baseline = rewards.mean(axis=1)

        # Optimize actor
        self.actor.train()
        self.actorOptim.zero_grad()

        # Get current log_probs and entropy
        actorInput = torch.cat([states, refs[:, 0:self.h + 1]], axis=1) - self.inCenter
        mus, sigmas = self.actor(actorInput)
        dist = Normal(mus, sigmas)
        c_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Advantage
        G = rewards
        pState = next_states
        for i in range(1, self.h + 1):
            G += - self.discount**i * torch.pow(refs[:, [i]] - pState[:, [0]], 2)
            pInput = torch.cat([pState, refs[:, i:self.h + i + 1]], axis=1) - self.inCenter
            pAction, _ = self.actor(pInput)
            pState = self.modelF(pState, pAction, gpu=True)

        adv = (G - baseline).detach()

        # Importance Weight
        ratios = torch.exp(c_log_probs - log_probs.detach()) + 1e-9
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
        actor_loss = - torch.min(surr1, surr2) - self.klCost * entropy
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actorOptim.step()

        self.updates += 1

        # if self.updates % self.config['update_freq'] == 0:
        #     for targetP, oP in zip(self.criticTarget.parameters(), self.critic.parameters()):
        #         targetP = (1 - self.tau) * targetP + self.tau * oP

        return {'actor_loss': actor_loss.item()}

    def toDict(self) -> dict:
        cp: dict = {
            'config': self.config,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict() if self.config['baseline'] == 'GAE' else None,
            'criticTarget': self.criticTarget.state_dict() if self.config['baseline'] == 'GAE' else None
        }
        return cp
