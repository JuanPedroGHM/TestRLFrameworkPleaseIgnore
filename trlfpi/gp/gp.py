import numpy as np

import scipy.optimize as optim
from scipy.linalg import cholesky, cho_solve, LinAlgError
from typing import Callable, Any
from .kernel import Kernel
from trlfpi.timer import Timer
from multiprocessing import Pool


class GP():

    def __init__(self,
                 k_function: Kernel = Kernel.RBF(1.0, [1.0]),
                 meanF: Any = None,
                 sigma_n: float = 0.1,
                 bGridSize: int = 5,
                 bPoolSize: int = 8):
        self.kernel = k_function
        self.sigma_n = sigma_n
        self.meanF = meanF
        self.alpha = None
        self.L = None
        self.bGridSize = bGridSize
        self.bPoolSize = bPoolSize

    def forward(self, x: np.ndarray):
        if self.alpha is not None:

            k_1 = self.kernel(self.X_TRAIN, x)
            k_2 = self.kernel(x, x)

            mu_pred = k_1.T @ self.alpha

            mu_pred += self.priorMean(x)

            v = cho_solve((self.L, True), k_1)
            sigma_pred = k_2 - v.T @ v

            return mu_pred, np.diag(sigma_pred).reshape(-1, 1)
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

    def priorMean(self, x: np.ndarray):
        if type(self.meanF) is float:
            pMean = np.ones((x.shape[0], 1)) * self.meanF
        elif type(self.meanF) is Callable:
            pMean = self.meanF(x)
        else:
            pMean = np.zeros((x.shape[0], 1))

        return pMean

    def L_alpha(self, K, X, Y):
        try:
            L = cholesky(K + (self.sigma_n ** 2) * np.identity(K.shape[0]), lower=True)
        except (LinAlgError, ValueError):
            raise LinAlgError

        alpha = cho_solve((L, True), Y - self.priorMean(X))
        return L, alpha

    def logLikelihood(self, params: np.ndarray):
        K = self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=params)
        try:
            L, alpha = self.L_alpha(K, self.X_TRAIN, self.Y_TRAIN)
        except LinAlgError:
            return 1e10

        result = 0.5 * self.Y_TRAIN.T @ alpha + np.sum(np.log(np.diag(L)))

        return result.item()

    def brute(self):

        lengthBounds = (np.max(self.X_TRAIN, axis=0) - np.min(self.X_TRAIN, axis=0)) * 100
        ranges = [(0, 250)]
        ranges.extend([(0, bound) for bound in lengthBounds])
        stepSizes = np.array([(range[1] - range[0]) / self.bGridSize for range in ranges])
        grid = [np.arange(range[0], range[1], step) for range, step in zip(ranges, stepSizes)]
        grid = np.array(np.meshgrid(*grid)).T.reshape(-1, len(ranges))
        with Pool(self.bPoolSize) as p:
            llArray = p.map(self.logLikelihood, grid)

        bestParams = grid[np.argmin(llArray)]
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
            params = np.hstack(self.kernel.params)
            if not bounds:
                bounds = self.kernel.bounds
            else:
                bounds = [(low, high) for low, high in zip(*bounds)]
                self.kernel.bounds = bounds

            res = optim.minimize(self.logLikelihood, params, bounds=bounds)
            self.kernel.params = res.x

        print(self.kernel.params)
        K = self.kernel(X, X)
        try:
            self.L, self.alpha = self.L_alpha(K, X, Y)
        except LinAlgError:
            print("LinAlgError: Not updated")
            print(self.kernel.params)

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    timer = Timer()

    # 1D Test1
    # DATAPOINTS = 100
    # f = lambda x: (1 / x) * np.sin(5 * 2 * np.pi * x)

    # X_DATA = np.random.uniform(-1.0, 1.0, DATAPOINTS).reshape((DATAPOINTS, 1))
    # Y_DATA = f(X_DATA) + np.random.normal(0, 0.1, DATAPOINTS).reshape((DATAPOINTS, 1))

    # kernel = Kernel.RBF(1.0, np.array([1.0]))
    # gpModel = GP(kernel, sigma_n=1, meanF=-2.0)
    # timer.start()
    # gpModel.fit(X_DATA, Y_DATA, fineTune=True, brute=True)
    # print(f"1D fit time : {timer.stop()}")

    # x = np.linspace(-1, 1, 100).reshape((100, 1))
    # y_pred, sigma_pred = gpModel(x)
    # y_real = f(x)

    # # plt.fill(np.concatenate([x, x[::-1]]),
    # #          np.concatenate([y_pred - 1.96 * sigma_pred,
    # #                          (y_pred + 1.96 * sigma_pred)[::-1]]),
    # #          alpha=.1, fc='c', ec=None)
    # plt.scatter(X_DATA, Y_DATA, marker='*', c='r')
    # plt.plot(x, y_pred, 'b-', label="GP")
    # plt.plot(x, y_real, 'r--', label="f(x)")
    # plt.legend()
    # plt.show()

    # # 2D Test
    DATAPOINTS = 1000
    TEST = 50
    f = lambda x: (x[:, [1]] - x[:, [0]]) * 20 * np.sin(2 * np.pi * x[:, [0]]) + 10
    X_DATA = np.random.uniform(-1, 1, DATAPOINTS * 2).reshape((DATAPOINTS, 2))
    Y_DATA = f(X_DATA)

    x = np.linspace(-1.0, 1.0, TEST).reshape((TEST,))
    x1, x2 = np.meshgrid(x, x)
    xx = np.stack((x1, x2), axis=2).reshape((TEST * TEST, 2))
    y = f(xx)

    kernel = Kernel.RBF(1.0, [1.0, 1.0])
    gpModel = GP(kernel, sigma_n=1.0, meanF=-2.0)
    timer.start()
    gpModel.fit(X_DATA, Y_DATA, fineTune=True, brute=True)
    y_pred, sigma_pred = gpModel(xx)
    print(f"2D fit time : {timer.stop()}")

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y.reshape((TEST * TEST,)), color="red")
    ax.scatter(X_DATA[:, [0]], X_DATA[:, [1]], Y_DATA)

    ax = fig.add_subplot(212, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y_pred.reshape((TEST * TEST,)))
    plt.show()
