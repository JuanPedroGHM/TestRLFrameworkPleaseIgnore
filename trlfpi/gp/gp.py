import numpy as np
import scipy.optimize as optim
from typing import List, Callable
from .kernel import Kernel


class GP():

    def __init__(self, k_function: Kernel = Kernel.RBF(), mean: Callable[[np.ndarray], np.ndarray] = None, sigma_n: float = 0.1):
        self.kernel = k_function
        self.sigma_n = sigma_n
        self.mean = mean
        self.alpha = None
        self.L = None

    def forward(self, x):
        if self.alpha is not None:

            k_1 = self.kernel(self.X_TRAIN, x)
            k_2 = self.kernel(x, x)

            mu_pred = k_1.T @ self.alpha

            if self.mean:
                mu_pred += self.mean(x)

            v = np.linalg.solve(self.L, k_1)
            sigma_pred = k_2 - v.T @ v

            return mu_pred, sigma_pred
        else:
            if self.mean:
                return self.mean(x), self.kernel(x, x)
            else:
                return 0, self.kernel(x, x)

    def logLikelihood(self, params: List[float]):
        K = self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=params)
        L = np.linalg.cholesky(K + (self.sigma_n ** 2) * np.identity(K.shape[0]))
        if self.mean:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, (self.Y_TRAIN - self.mean(self.X_TRAIN))))
        else:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, (self.Y_TRAIN)))

        result = 0.5 * self.Y_TRAIN.T @ alpha + np.sum(np.log(np.diag(L)))

        return result.item()

    def fit(self, X, Y, optimize=True):
        self.X_TRAIN = X
        self.Y_TRAIN = Y

        if optimize:
            params = np.hstack(self.kernel.params)
            bounds = self.kernel.bounds
            res = optim.minimize(self.logLikelihood, params, bounds=bounds)

            self.kernel.params = res.x

        self.L = np.linalg.cholesky(self.kernel(X, X) + (self.sigma_n ** 2) * np.identity(X.shape[0]))
        if self.mean:
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, (self.Y_TRAIN - self.mean(self.X_TRAIN))))
        else:
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_TRAIN))

    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":

    # 1D Test
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    DATAPOINTS = 200
    f = lambda x: (1 / x) * np.sin(5 * 2 * np.pi * x)

    X_DATA = np.random.uniform(-1.0, 1.0, DATAPOINTS).reshape((DATAPOINTS, 1))
    Y_DATA = f(X_DATA) + np.random.normal(0, 2, DATAPOINTS).reshape((DATAPOINTS, 1))

    kernel = Kernel.RBF(1, 0.1)
    gpModel = GP(kernel)
    gpModel.fit(X_DATA, Y_DATA)

    x = np.linspace(-1, 1, 1000).reshape((1000, 1))
    y_pred, sigma_pred = gpModel(x)
    y_real = f(x)

    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma_pred,
                             (y_pred + 1.96 * sigma_pred)[::-1]]),
             alpha=.1, fc='c', ec=None)
    plt.scatter(X_DATA, Y_DATA, marker='*', c='r')
    plt.plot(x, y_pred, 'b-', label="GP")
    plt.plot(x, y_real, 'r--', label="f(x)")
    plt.legend()
    plt.show()

    # 2D Test
    DATAPOINTS = 200
    TEST = 50
    f = lambda x: (x[:, [1]] - x[:, [0]]) * 20 * np.sin(2 * np.pi * x[:, [0]]) + 10
    X_DATA = np.random.uniform(-1, 1, DATAPOINTS * 2).reshape((DATAPOINTS, 2))
    Y_DATA = f(X_DATA) + np.random.normal(0, 0.0001, DATAPOINTS).reshape((DATAPOINTS, 1))

    x = np.linspace(-1.0, 1.0, TEST).reshape((TEST,))
    x1, x2 = np.meshgrid(x, x)
    xx = np.stack((x1, x2), axis=2).reshape((TEST * TEST, 2))
    y = f(xx)

    kernel = Kernel.RBF(1, [1.0, 1.0])
    gpModel = GP(kernel)
    meanF = lambda x: 10
    gpModel.fit(X_DATA, Y_DATA, meanF)
    y_pred, sigma_pred = gpModel(xx)

    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y.reshape((TEST * TEST,)), color="red")
    ax.scatter(X_DATA[:, [0]], X_DATA[:, [1]], Y_DATA)

    ax = fig.add_subplot(212, projection='3d')
    ax.plot_trisurf(xx[:, 0], xx[:, 1], y_pred.reshape((TEST * TEST,)))
    plt.show()
