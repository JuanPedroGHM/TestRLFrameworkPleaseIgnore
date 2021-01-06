import torch
import numpy as np

from trlfpi.timer import Timer
from trlfpi.gp.kernel import Kernel
from trlfpi.gp.gpu import GPu

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    timer = Timer()

    # GPU TEST
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    DATAPOINTS = 1000
    TEST = 50

    f = lambda x: (x[:, [1]] - x[:, [0]]) * 5 * torch.sin(2 * np.pi * x[:, [0]]) + 10

    X_DATA = 2.0 * torch.rand(DATAPOINTS * 2, device=device).reshape((DATAPOINTS, 2)) - 1.0
    Y_DATA = f(X_DATA)

    x = torch.arange(-1.0, 1.0, 2.0 / TEST).to(device)
    x1, x2 = torch.meshgrid([x, x])
    xx = torch.stack((x1, x2), dim=2).reshape((TEST * TEST, 2))
    y = f(xx)

    kernel = Kernel.RBF(1.0, [1.0, 1.0])
    gpModel = GPu(kernel, sigma_n=1.0, meanF=-2.0, device=device)
    timer.start()
    # import pdb; pdb.set_trace()
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
