import torch


def rbf_f(x0: torch.Tensor, x1: torch.Tensor, params: torch.Tensor):

    theta = params[0]
    length = params[1:]

    if length.shape[0] != 1:
        if x0.shape[1] != x1.shape[1] != length.shape[0]:
            raise ValueError("Length is invalid")
    if torch.any(length == 0):
        return torch.zeros((x0.shape[0], x1.shape[0])).to(x0.device)

    a = x0 / length
    b = x1 / length
    dist = torch.cdist(a, b, p=2)
    K = theta**2 * torch.exp(-.5 * dist)
    return K
