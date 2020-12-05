from typing import List, Tuple
import numpy as np
import torch

from .agent import Agent
from ..memory import GymMemory
from ..nn.stochasticActor import StochasticActor


@Agent.register("reinforce")
class REINFORCE(Agent):

    default_config: dict = {
        # Env
        'h': 1,
        'discount': 0.7,

        # Network
        'layers': [3, 64, 64, 1],
        'activation': ['tahn', 'tahn', 'identity'],
        'a_lr': 1e-5,
        'weightDecay': 1e-3,
        'batchSize': 512,
        'update_freq': 1000,

        # ReplayBuffer
        'bufferSize': 10000
    }

    def setup(self, checkpoint: dict = None, device: str = 'cpu'):
        if checkpoint:
            self.config = checkpoint['config']

        self.device = device
        self.actor = StochasticActor(self.config['layers'], self.config['activation']).to(device)
        self.actorTarget = StochasticActor(self.config['layers'], self.config['activation']).to(device)
        if checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])

        self.actorTarget.load_state_dict(self.actor.state_dict())
        self.actorTarget.eval()

        self.actorOptim = torch.optim.Adam(self.actor.parameters(), lr=self.config['a_lr'], weight_decay=self.config['weightDecay'])
        self.replayBuff = GymMemory(self.config['bufferSize'])
        self.h = self.config['h']
        self.updates = 0

    def act(self, state: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        actorInput = torch.tensor(np.hstack([state, ref[:, 1:self.h + 1]]), device=self.device)

        if self.mode == 'train':
            with torch.no_grad():
                action, log_probs = self.actorTarget.act(actorInput)
            return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

        else:
            with torch.no_grad():
                self.actor.eval()
                action, log_probs = self.actor.act(actorInput)
            return action.detach().cpu().numpy(), log_probs.detach().cpu().numpy()

    def update(self, stepData: List[np.ndarray] = None) -> dict:

        # Save new values if any
        if stepData:
            self.replayBuff.add(tuple(map(lambda x: torch.tensor(x, device=self.device), stepData)))
        if self.replayBuff.size < self.config['batchSize']:
            return {'actor_loss': 0}

        # Get training batch
        states, refs, actions, log_probs, rewards, dones, next_states = self.replayBuff.get(self.config['batchSize'])

        # Optimize actor
        self.actor.train()
        self.actorOptim.zero_grad()

        actorInput = torch.cat([states, refs[:, 1:self.h + 1]], axis=1)
        pActions, c_log_probs = self.actor.act(actorInput, sample=False, prevActions=actions)

        # Importance Weight
        iw = torch.exp(c_log_probs - log_probs).detach() + 1e-9
        adv = rewards - rewards.mean()

        actor_loss = (-c_log_probs * adv * iw).mean()
        actor_loss.backward()
        self.actorOptim.step()

        self.updates += 1

        if self.updates % self.config['update_freq'] == 0:
            self.actorTarget.load_state_dict(self.actor.state_dict())

        return {'actor_loss': actor_loss.item()}

    def toDict(self) -> dict:
        cp: dict = {
            'config': self.config,
            'actor': self.actor.state_dict(),
        }
        return cp
