import numpy as np
import scipy.optimize as optim
from typing import List, Callable
from functools import reduce
from .kernel import Kernel

class GP():

    def __init__(self, X, Y, k_function: Kernel = Kernel.RBF(), optimize:bool=True):
        self.kernel = k_function
        self.fit(X,Y, optimize=optimize)

    def forward(self, x):
        k_1 = self.kernel(self.X_TRAIN, x)
        k_2 = self.kernel(x, x)

        mu_pred = np.dot(k_1.T, self.alpha)

        v = np.linalg.solve(self.L, k_1)
        sigma_pred = k_2 - np.dot(v.T, v)

        return mu_pred, sigma_pred

    def logLikelihood(self, params: List[float]):
        print(params)
        L = np.linalg.cholesky(self.kernel(self.X_TRAIN, self.X_TRAIN, customParams=params))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.Y_TRAIN)) 

        return 0.5 * np.dot(self.Y_TRAIN.T, alpha) + np.sum(np.log(np.diag(L)))

    def fit(self, X, Y, optimize:bool= True):
        self.X_TRAIN = X
        self.Y_TRAIN = Y
        if optimize:
            print(self.kernel.params)
            res = optim.minimize(self.logLikelihood, self.kernel.params, bounds=((0.01, None), (0.01, None), (0.01, None)))
            print(res.x)
            self.kernel.params = res.x

        self.L = np.linalg.cholesky(self.kernel(X, X))
        self.alpha =np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Y_TRAIN)) 



    def __call__(self, x):
        return self.forward(x)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    DATAPOINTS = 20 

    X_DATA = np.random.uniform(-10.0, 10.0, DATAPOINTS).reshape((DATAPOINTS, 1))
    Y_DATA = X_DATA * np.sin(X_DATA) + np.random.standard_normal(DATAPOINTS).reshape((DATAPOINTS,1))

    kernel = Kernel.RBF(1, 0.1) + Kernel.Noise(0.01)
    gpModel = GP(X_DATA, Y_DATA, k_function=kernel)

    x = np.linspace(-20, 20, 100).reshape((100,1))
    y_pred, sigma_pred = gpModel(x)
    y_real = x * np.sin(x)

    plt.scatter(X_DATA, Y_DATA)
    plt.plot(x, y_pred)
    plt.plot(x, y_real)
    plt.show()
