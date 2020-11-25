from .agent import Agent
from argparse import ArgumentParser
import numpy as np


@Agent.register("REINFORCE")
class REINFORCE(Agent):

    def __init__(self, args):
        self.args = args

    def options(self, parser: ArgumentParser):
        group = parser.add_argument_group('REINFORCE')
        group.add_argument("--nRefs", type=int, default=1)

        group.add_argument("--layers", type=int, nargs='+', default=[64, 8])
        group.add_argument("--a_lr", type=float, default=1e-3)
        group.add_argument("--weightDecay", type=float, default=0.0)
        group.add_argument("--update_freq", type=int, default=1)

        group.add_argument("--buffer_size", type=int, default=5000)
        group.add_argument("--batch_size", type=int, default=128)

    def act(self, state: np.ndarray, ref: np.ndarray, eval=False) -> np.ndarray:
        pass

    def update(self):
        pass
