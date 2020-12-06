from typing import List, Tuple
import numpy as np
import torch
from torch.distributions import Normal

from .agent import Agent
from ..memory import GymMemory
from ..nn.stochasticActor import StochasticActor
from ..nn.critic import QFunc


@Agent.register("ppo")
class PPO(Agent):

    default_config: dict = {
        # Env
        'h': 1,
        'discount': 0.7,

        # Policy Network
        'a_layers': [3, 64, 64, 1],
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

        'tau': 1e-3,
        'weightDecay': 1e-3,
        'batchSize': 512,
        'update_freq': 2,

        # ReplayBuffer
        'bufferSize': 10000
    }

    def setup(self, checkpoint: dict = None, device: str = 'cpu'):
        print(self.config)
        if checkpoint:
            self.config = checkpoint['config']

        self.device = device
        self.actor = StochasticActor(self.config['a_layers'], self.config['a_activation'], self.config['a_layerOptions']).to(device)
        self.actorTarget = StochasticActor(self.config['a_layers'], self.config['a_activation'], self.config['a_layerOptions']).to(device)
        if checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            self.actorTarget.load_state_dict(checkpoint['actorTarget'])
        else:
            self.actorTarget.load_state_dict(self.actor.state_dict())

        self.actorTarget.load_state_dict(self.actor.state_dict())
        self.actorTarget.eval()
        self.actorOptim = torch.optim.Adam(self.actor.parameters(),
                                           lr=self.config['a_lr'],
                                           weight_decay=self.config['weightDecay'])

        self.critic = QFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
        self.criticTarget = QFunc(self.config['c_layers'], self.config['c_activation'], self.config['c_layerOptions']).to(device)
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

        self.replayBuff = GymMemory(self.config['bufferSize'])
        self.h = self.config['h']
        self.updates = 0
        self.tau = self.config['tau']
        self.discount = self.config['discount']
        self.clip = self.config['clip']
        self.klCost = self.config['klCost']

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actorInput = torch.tensor(np.hstack([state, ref[:, 1:self.h + 1]]), device=self.device)

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
        self.critic.train()
        self.criticOptim.zero_grad()
        cInput = torch.cat((actions, states, refs[:, 1:self.h + 1]), axis=1)
        q = self.critic(cInput)

        with torch.no_grad():
            next_actions, _ = self.actorTarget(torch.cat([next_states, refs[:, 2:self.h + 2]], axis=1))
            next_q = (1 - dones) * self.criticTarget(torch.cat([next_actions,
                                                           next_states,
                                                           refs[:, 2:self.h + 2]], axis=1))

        critic_loss = self.criticLossF(q, rewards + self.discount * next_q)
        critic_loss.backward()
        self.criticOptim.step()

        # Optimize actor
        self.actor.train()
        self.actorOptim.zero_grad()

        # Get current log_probs and entropy
        actorInput = torch.cat([states, refs[:, 1:self.h + 1]], axis=1)
        mus, sigmas = self.actor(actorInput)
        dist = Normal(mus, sigmas)
        c_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        # Advantage Q(a, s) - V(s)
        with torch.no_grad():
            self.critic.eval()
            qs = self.critic(torch.cat([actions, actorInput], axis=1))
            vs = self.critic.value(actorInput, self.actor, nSamples=100)
            adv = (qs - vs).detach()

        # Importance Weight
        ratios = torch.exp(c_log_probs - log_probs.detach()) + 1e-9
        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv
        actor_loss = - torch.min(surr1, surr2) - 0.01 * entropy
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actorOptim.step()

        self.updates += 1

        if self.updates % self.config['update_freq'] == 0:
            for targetP, oP in zip(self.criticTarget.parameters(), self.critic.parameters()):
                targetP = (1 - self.tau) * targetP + self.tau * oP

            for targetP, oP in zip(self.actorTarget.parameters(), self.actor.parameters()):
                targetP = (1 - self.tau) * targetP + self.tau * oP

        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

    def toDict(self) -> dict:
        cp: dict = {
            'config': self.config,
            'actor': self.actor.state_dict(),
            'actorTarget': self.actorTarget.state_dict(),
            'critic': self.critic.state_dict(),
            'criticTarget': self.criticTarget.state_dict()
        }
        return cp
