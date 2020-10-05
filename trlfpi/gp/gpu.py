import numpy as np
import torch

from typing import Any, Callable

from .kernel import Kernel
from trlfpi.timer import Timer


class GPu():

    def __init__(self,
                 k_function: Kernel = Kernel.RBF(1.0, [1.0], gpu=True),
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
        L = torch.cholesky(K + (self.sigma_n ** 2) * torch.eye(K.shape[0]))
        alpha = torch.cholesky_solve(Y - self.priorMean(X), L)
        return L, alpha

    def logLikelihood(self, params: torch.Tensor) -> torch.Tensor:
        K = self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=params)
        L, alpha = self.L_alpha(K, self.X_TRAIN, self.Y_TRAIN)
        result = 0.5 * self.Y_TRAIN.T @ alpha + torch.sum(torch.log(torch.diag(L)))

        return result

    def brute(self):

        lengthBounds = (torch.max(self.X_TRAIN, dim=0)[0] - torch.min(self.X_TRAIN, dim=0)[0]) * 10
        ranges = [(0, 250)]
        ranges.extend([(0, bound) for bound in lengthBounds])
        stepSizes = torch.tensor([(range[1] - range[0]) / self.bGridSize for range in ranges])
        grid = [torch.arange(range[0], range[1], step) for range, step in zip(ranges, stepSizes)]
        grid = torch.stack(torch.meshgrid(grid)).T.reshape(-1, len(ranges))
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
            params = [torch.tensor(self.kernel.params, requires_grad=True)]
            lbfgsOptim = torch.optim.LBFGS(params, max_iter=100, lr=1e-4)

            def closure():
                lbfgsOptim.zero_grad()
                ll = self.logLikelihood(params[0])
                ll.backward()
                return ll

            lbfgsOptim.step(closure)

            # res = (self.logLikelihood, params, bounds=bounds)
            self.kernel.params = params[0]

        K = self.kernel(X, X)
        self.L, self.alpha = self.L_alpha(K, X, Y)

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    timer = Timer()

    # GPU TEST
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    DATAPOINTS = 1000
    TEST = 50

    f = lambda x: (x[:, [1]] - x[:, [0]]) * 20 * torch.sin(2 * np.pi * x[:, [0]]) + 10

    X_DATA = 2.0 * torch.rand(DATAPOINTS * 2, device=device).reshape((DATAPOINTS, 2)) - 1.0
    Y_DATA = f(X_DATA)

    x = torch.arange(-1.0, 1.0, 2.0 / TEST).to(device)
    x1, x2 = torch.meshgrid([x, x])
    xx = torch.stack((x1, x2), dim=2).reshape((TEST * TEST, 2))
    y = f(xx)

    kernel = Kernel.RBF(1.0, [1.0, 1.0], gpu=True)
    gpModel = GPu(kernel, sigma_n=1.0, meanF=-2.0, device=device)
    timer.start()
    gpModel.fit(X_DATA, Y_DATA, fineTune=True, brute=True)
    y_pred, sigma_pred = gpModel(xx)
    print(f"2D GPU fit time : {timer.stop()}")

    xx = xx.cpu().numpy()
    y = y.cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    X_DATA = X_DATA.detach().cpu().numpy()
    Y_DATA = Y_DATA.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y.reshape((TEST * TEST,)), color="red")
    ax.scatter(X_DATA[:, [0]], X_DATA[:, [1]], Y_DATA)

    ax = fig.add_subplot(212, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y_pred.reshape((TEST * TEST,)))
    plt.show()
