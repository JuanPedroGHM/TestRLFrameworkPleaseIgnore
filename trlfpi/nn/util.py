import torch
from torch import nn

from typing import List
from .activation import InvertedRELU

activationFunctions: dict = {
    'relu': nn.ReLU,
    'identity': nn.Identity,
    'tahn': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'invRelu': InvertedRELU
}


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def mlp(layers: List[int], activations: List[str], layerOptions: List[dict] = None, dropout=False, batchNorm=False) -> nn.Module:
    moduleList: List = []
    lastOutputSize = layers[0]
    for index, layerSize in enumerate(layers[1:]):
        if index < len(layers) - 2:
            layer = nn.Linear(lastOutputSize, layerSize)
            if layerOptions:
                options = layerOptions[index]
                if 'bias' in options:
                    layer.bias = nn.Parameter(torch.tensor(options['bias']), requires_grad=True)
            moduleList.append(layer)
            if batchNorm:
                moduleList.append(nn.LayerNorm(layerSize))
            moduleList.append(activationFunctions[activations[index]]())
            if dropout:
                moduleList.append(nn.Dropout())
            lastOutputSize = layerSize
        else:
            layer = nn.Linear(lastOutputSize, layerSize)
            if layerOptions:
                options = layerOptions[index]
                if 'bias' in options:
                    layer.bias = nn.Parameter(torch.tensor(options['bias']), requires_grad=True)
            moduleList.append(layer)
            moduleList.append(activationFunctions[activations[index]]())
    model = nn.Sequential(*moduleList)
    # model.apply(init_weights)

    return model
