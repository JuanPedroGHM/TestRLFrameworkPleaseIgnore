import torch

from typing import Any, Callable

from .kernel import Kernel
from trlfpi.timer import Timer


class GPu():

    def __init__(self,
                 k_function: Kernel = Kernel.RBF(1.0, [1.0]),
                 meanF: Any = None,
                 sigma_n: float = 0.1,
                 bGridSize: int = 5,
                 bPoolSize: int = 8,
                 device=None):
        self.kernel = k_function
        self.sigma_n = sigma_n
        self.meanF = meanF
        self.alpha = None
        self.L = None
        self.bGridSize = bGridSize
        self.bPoolSize = bPoolSize
        self.device = device

    def forward(self, x: torch.Tensor):
        if self.alpha is not None:

            k_1 = self.kernel(self.X_TRAIN, x)
            k_2 = self.kernel(x, x)

            mu_pred = k_1.T @ self.alpha

            mu_pred += self.priorMean(x)

            v = torch.cholesky_solve(k_1, self.L)
            sigma_pred = k_2 - v.T @ v

            return mu_pred, torch.diag(sigma_pred).reshape(-1, 1)
        else:
            return self.priorMean(x), self.kernel(x, x)

    def mean(self, x):
        if self.alpha is not None:

            k_1 = self.kernel(self.X_TRAIN, x)

            mu_pred = k_1.T @ self.alpha
            mu_pred += self.priorMean(x)

            return mu_pred
        else:
            return self.priorMean(x)

    def priorMean(self, x: torch.Tensor):
        if type(self.meanF) is float:

            pMean = torch.ones((x.shape[0], 1), device=self.device) * self.meanF
        elif type(self.meanF) is Callable:
            pMean = self.meanF(x)
        else:
            pMean = torch.zeros((x.shape[0], 1), device=self.device)
        return pMean

    def L_alpha(self, K, X, Y):
        L = torch.cholesky(K + (self.sigma_n ** 2) * torch.eye(K.shape[0], device=self.device))
        alpha = torch.cholesky_solve(Y - self.priorMean(X), L)
        return L, alpha

    def logLikelihood(self, params: torch.Tensor = None) -> torch.Tensor:
        K = self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=params)
        L, alpha = self.L_alpha(K, self.X_TRAIN, self.Y_TRAIN)
        result = 0.5 * self.Y_TRAIN.T @ alpha + torch.sum(torch.log(torch.diag(L)))

        return result

    def brute(self):

        lengthBounds = (torch.max(self.X_TRAIN, dim=0)[0] - torch.min(self.X_TRAIN, dim=0)[0]) * 10
        ranges = [(0.0, torch.tensor(100.0))]
        ranges.extend([(0.0, bound) for bound in lengthBounds])
        stepSizes = torch.tensor([(r[1] - r[0]) / self.bGridSize for r in ranges], device=self.device)
        grid = [torch.arange(r[0], r[1].item(), step.item()) for r, step in zip(ranges, stepSizes)]
        grid = torch.stack(torch.meshgrid(grid)).T.reshape(-1, len(ranges)).to(self.device)
        llArray = list(map(self.logLikelihood, grid))
        bestParams = grid[torch.tensor(llArray).argmin()]
        bounds = (bestParams - stepSizes, bestParams + stepSizes)

        return bestParams, bounds

    def fit(self, X, Y, fineTune=False, brute=False):
        self.X_TRAIN = X
        self.Y_TRAIN = Y

        bounds = None
        if brute:
            params, bounds = self.brute()
            self.kernel.params = params

        if fineTune:
            tParams = [self.kernel.params.clone().detach().requires_grad_(True)]
            lbfgsOptim = torch.optim.LBFGS(tParams, max_iter=100, lr=1e-4)

            def closure():
                lbfgsOptim.zero_grad()
                ll = self.logLikelihood(tParams[0])
                ll.backward()
                return ll

            lbfgsOptim.step(closure)

            # res = (self.logLikelihood, params, bounds=bounds)
            self.kernel.params = tParams[0]

        K = self.kernel(X, X)
        self.L, self.alpha = self.L_alpha(K, X, Y)

    def toDict(self) -> dict:
        return {
            'X_TRAIN': self.X_TRAIN,
            'alpha': self.alpha,
            'L': self.L,
            'sigma_n': self.sigma_n,
            'meanF': self.meanF,
            'kernel': self.kernel
        }

    def fromDict(self, checkpoint: dict):
        self.X_TRAIN = checkpoint['X_TRAIN']
        self.alpha = checkpoint['alpha']
        self.L = checkpoint['L']
        self.sigma_n = checkpoint['sigma_n']
        self.meanF = checkpoint['meanF']
        self.kernel = checkpoint['kernel']

    def __call__(self, x):
        return self.forward(x)
