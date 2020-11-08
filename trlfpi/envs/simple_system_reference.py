import numpy as np
import torch
from numpy import concatenate as cat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


class LinearSystem():

    def __init__(self, config):
        super().__init__()
        self.x = np.array(config["x0"], dtype=float)
        self.A = np.array(config["A"], dtype=float)
        self.B = np.array(config["B"], dtype=float)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.A_t = torch.tensor(self.A, device=device)
        self.B_t = torch.tensor(self.B, device=device)

    def apply(self, u):
        self.x = self.A @ self.x + self.B @ u
        return self.x

    def predict(self, x, u, gpu=False):
        if gpu:
            return self.predict_t(x, u)
        else:
            return self.A @ x + self.B @ u

    def predict_t(self, x: torch.Tensor, u: torch.Tensor):
        return self.A_t @ x.T + self.B_t @ u.T


class Reference():

    def __init__(self, config):
        super().__init__()
        self.h = config["h"]    # length of horizon
        self.N = config["N"]    # number of reference points
        self.dt = config["dt"]  # saple time
        self.num_of_rand_points = config["num_of_rand_points"]
        self.counter = 1
        self.noise_variance = config["noise_var"]
        self.t = []
        self.t_sparse = []
        self.r = []
        self.r_sparse = []

    def generateReference(self):
        self.t = np.linspace(0, self.N * self.dt, self.N, dtype=float)
        self.t_sparse = np.linspace(0, self.N * self.dt, self.num_of_rand_points, dtype=float)
        self.r_sparse = np.random.rand(self.num_of_rand_points)
        interpolation = interp1d(self.t_sparse, self.r_sparse, kind='cubic')
        self.r = interpolation(self.t) + self.noise_variance * np.random.randn(self.t.size)

    def getNext(self):
        r = self.r[self.counter - 1:self.counter + self.h]
        self.counter += 1
        if self.counter == self.N - self.h - 1:
            self.counter = 1
        return np.array([r]).T


class Logger():
    def __init__(self):
        super().__init__()
        self.reference_list = []
        self.action_list = []
        self.state_list = []

    def log(self, x, u, r):
        self.reference_list.append(r)
        self.action_list.append(u)
        self.state_list.append(x)


def controller(x, r):
    K = np.array([[5.4126], [0.2097], [-5.3391], [-0.4110], [0.1121], [-0.0262], [0.0047]])
    X = cat((x, r[1:, :]), axis=0)
    return -K.T @ X


configSystem = {'A': [[0.9590, 0.03697], [-0.5915, 0.0718]],
                'B': [[0.1638], [2.3660]],
                'x0': [[1.0], [0.0]]
                }

configReference = {
    'N': 1000,
    'h': 6,
    'dt': 0.1,
    'num_of_rand_points': 50,
    'noise_var': 0.0
}

if __name__ == "__main__":

    ref = Reference(configReference)
    ref.generateReference()

    sys = LinearSystem(configSystem)

    logger = Logger()

    # simple simulation
    x = sys.x
    for k in range(configReference["N"] - configReference["h"]):
        r = ref.getNext()
        u = controller(x, r)
        x = sys.apply(u)

        logger.log(x, u, r)

    x = [x[0] for x in logger.state_list]
    u = [u[0] for u in logger.action_list]
    r = [r[0] for r in logger.reference_list]
    plt.figure(1)
    plt.plot(r, '--', label="r")
    plt.plot(x, label="x")
    plt.plot(u, label="u")
    plt.ylim((-1, 1))

    plt.grid()
    plt.legend()
    plt.show()
