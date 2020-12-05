from typing import List, Tuple
import numpy as np
import torch

from .agent import Agent
from ..memory import GymMemory
from ..nn.deterministicActor import DeterministicActor
from ..nn.critic import QFunc


@Agent.register("dpg")
class DPG(Agent):

    default_config: dict = {
        # Env
        'h': 1,
        'discount': 0.7,

        # Policy Network
        'a_layers': [3, 64, 64, 1],
        'a_activation': ['tahn', 'tahn', 'identity'],
        'a_lr': 1e-5,

        # Critic Network
        'c_layers': [3, 64, 64, 1],
        'c_activation': ['tahn', 'tahn', 'invRelu'],
        'c_lr': 1e-3,

        # Exploration policy epsilon-greedy
        'epsilonDecay': 3e-6,
        'explorationNoise': 0.05,

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
        self.actor = DeterministicActor(self.config['a_layers'], self.config['a_activation']).to(device)
        self.actorTarget = DeterministicActor(self.config['a_layers'], self.config['a_activation']).to(device)
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

        self.critic = QFunc(self.config['c_layers'], self.config['c_activation']).to(device)
        self.criticTarget = QFunc(self.config['c_layers'], self.config['c_activation']).to(device)
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
        self.tau = self.config['tau']
        self.discount = self.config['discount']
        self.epsilonDecay = self.config['epsilonDecay']
        self.explorationNoise = self.config['explorationNoise']

        # epsilon greedy setup
        self.updates = 0
        self.epsilon = np.exp(-self.updates * self.epsilonDecay)

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actorInput = torch.tensor(np.hstack([state, ref[:, 1:self.h + 1]]), device=self.device)

        if self.mode == 'train':
            with torch.no_grad():
                self.actor.eval()
                action = self.actor.act(actorInput)

                if np.random.random() < self.epsilon:
                    action = torch.distributions.Normal(action, self.explorationNoise).sample()

            return action.detach().cpu().numpy(), 1.0

        else:
            with torch.no_grad():
                self.actor.eval()
                action = self.actor.act(actorInput)
            return action.detach().cpu().numpy(), 1.0

    def update(self, stepData: List[np.ndarray] = None) -> dict:

        # Save new values if any
        if stepData:
            self.replayBuff.add(tuple(map(lambda x: torch.tensor(x, device=self.device), stepData)))
        if self.replayBuff.size < self.config['batchSize']:
            return {'actor_loss': 0}

        # Adjust exploration noise
        self.epsilon = np.exp(-self.updates * self.epsilonDecay)

        # Get training batch
        states, refs, actions, log_probs, rewards, dones, next_states = self.replayBuff.get(self.config['batchSize'])

        # Optimize critic
        self.critic.train()
        self.criticOptim.zero_grad()
        cInput = torch.cat((actions, states, refs[:, 1:self.h + 1]), axis=1)
        q = self.critic(cInput)

        with torch.no_grad():
            next_actions = self.actorTarget(torch.cat([next_states, refs[:, 2:self.h + 2]], axis=1))
            next_q = (1 - dones) * self.criticTarget(torch.cat([next_actions,
                                                           next_states,
                                                           refs[:, 2:self.h + 2]], axis=1))

        critic_loss = self.criticLossF(q, rewards + self.discount * next_q)
        critic_loss.backward()
        self.criticOptim.step()

        # Optimize actor
        self.actor.train()
        self.actorOptim.zero_grad()

        actorInput = torch.cat([states, refs[:, 1:self.h + 1]], axis=1)
        pActions = self.actor.act(actorInput)
        actor_loss = -self.critic(torch.cat([pActions, actorInput], axis=1)).mean()
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
