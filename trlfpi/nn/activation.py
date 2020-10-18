from torch import nn
import torch.nn.functional as F


class InvertedRELU(nn.Module):

    def __init__(self):
        super(InvertedRELU, self).__init__()

    def forward(self, x):
        return - F.relu(x)
