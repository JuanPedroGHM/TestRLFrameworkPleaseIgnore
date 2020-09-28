import torch
from torch import nn

from typing import List


def mlp(layers: List[int], activation: nn.Module = nn.ReLU, dropout=False, batchNorm=False, outputActivation: nn.Module = torch.nn.Identity) -> nn.Module:
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
