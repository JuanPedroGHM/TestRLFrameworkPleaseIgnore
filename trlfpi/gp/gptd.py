import numpy as np
import scipy.optimize as optim
from typing import List
from .kernel import Kernel


class GPTD():

    def __init__(self, k_function: Kernel = Kernel.RBF()):
        self.kernel = k_function
        self.sigma_n = 1.0
        self.sigma_n_bounds = [1e-9, np.inf]

    def forward(self, x):
        k_1 = self.kernel(self.X_TRAIN, x)
        k_2 = self.kernel(x, x)

        mu_pred = k_1.T @ self.alpha

        sigma_pred = k_2 - k_1.T @ self.C @ k_1

        return mu_pred, sigma_pred

    def logLikelihood(self, params: List[float]):
        sigma_n = params[0]
        kernelParams = params[1:]

        K = self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=kernelParams)
        Sigma = sigma_n * self.H * self.H_T
        L = np.linalg.cholesky(self.H @ K @ self.H.T + Sigma)
        alpha = self.H.T @ np.linalg.solve(L.T, np.linalg.solve(L, (self.Y_TRAIN)))

        result = 0.5 * self.Y_TRAIN.T @ alpha + np.sum(np.log(np.diag(L)))

        return result.item()

    def fit(self, X, Y, H, optimize=True):
        self.X_TRAIN = X
        self.Y_TRAIN = Y
        self.H = H

        if optimize:
            params = np.hstack((self.sigma_n, self.kernel.params))
            bounds = [self.sigma_n_bounds] + self.kernel.bounds
            res = optim.minimize(self.logLikelihood, params, bounds=bounds)

            self.sigma_n = res.x[0]
            self.kernel.params = res.x[1:]

        Sigma = self.sigma_n * self.H * self.H_T
        cov_prior = self.H @ self.kernel(X, X) @ self.H.T + Sigma
        cov_inv = np.linalg.inv(cov_prior)

        self.alpha = self.H.T @ cov_inv @ Y
        self.C = self.H.T @ cov_inv @ self.H

    def __call__(self, x):
        return self.forward(x)
