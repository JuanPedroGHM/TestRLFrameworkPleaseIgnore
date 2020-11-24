import numpy as np
from scipy.interpolate import interp1d


class ReferenceGenerator():

    def __init__(self, N: int,
                 stepSize: float,
                 randomPoints: int,
                 variance: float = 0.1,
                 offset: float = 0.0):
        self.N = N
        self.stepSize = stepSize
        self.randomPoints = randomPoints
        self.variance = variance
        self.offset = offset

    def generate(self) -> np.ndarray:
        t = np.linspace(0, self.N * self.stepSize, self.N, dtype=float)
        t_sparse = np.linspace(0, self.N * self.stepSize, self.randomPoints, dtype=float)
        ref_sparse = np.random.rand(self.randomPoints)
        interpolation = interp1d(t_sparse, ref_sparse, kind='cubic')
        ref = self.offset + interpolation(t) + self.variance * np.random.randn(t.size)
        return ref.reshape((1, self.N))
