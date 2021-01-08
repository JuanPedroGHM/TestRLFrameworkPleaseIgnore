import pathlib
import json
import pickle
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict
import numpy as np
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt


class Report():

    def __init__(self, reportPath: str, tensorboard=True):

        self.tensorboard = tensorboard
        self.reportPath: pathlib.Path = pathlib.Path(reportPath)
        if not self.reportPath.exists():
            self.reportPath.mkdir()

    def listExisting(self) -> List[int]:
        return list(map(lambda x: int(x.name), self.reportPath.iterdir()))

    def new(self) -> int:
        reportId = str(reduce(lambda x, y: x + 1, self.reportPath.iterdir(), 0))
        self.path = self.reportPath / reportId
        self.path.mkdir()

        self.plotsPath = self.path / 'plots'
        self.plotsPath.mkdir()

        self.picklePath = self.path / 'pickle'
        self.picklePath.mkdir()

        if self.tensorboard:
            self.writer = SummaryWriter(f"{self.path / 'tensorboard'}")

        self.variables: dict = {}
        return int(reportId)

    def id(self, id: int):
        self.path = self.reportPath / str(id)
        self.plotsPath = self.path / 'plots'
        self.picklePath = self.path / 'pickle'
        self.variables = self.unpickle('variables')
        if self.tensorboard:
            self.writer = SummaryWriter(f"{self.path / 'tensorboard'}")

    def logArgs(self, argsDict: Dict):
        with open(self.path / 'args.json', 'w+') as f:
            json.dump(argsDict, f, indent=4)

    def log(self, name: str, value, step=None):
        if name in self.variables.keys():
            self.variables[name][0].append(value)
            if step:
                self.variables[name][1].append(step)

        else:
            if step:
                self.variables[name] = [[value], [step]]
            else:
                self.variables[name] = [[value]]

        if self.tensorboard:
            if step:
                self.writer.add_scalar(name, value, step)
            else:
                self.writer.add_scalar(name, value)

    def generateReport(self):
        with open(f"{self.picklePath}/variables.p", "w+b") as f:
            pickle.dump(self.variables, f)

        for key, values in self.variables.items():

            if len(values) == 2:
                self.savePlot(key, key, np.array([values[0]]).T, values[1])
            else:
                self.savePlot(key, key, np.array([values[0]]).T)

    def savePlot(self, plotName: str, variableNames: List[str], Y: np.ndarray, X=None):
        plt.figure()
        for i in range(Y.shape[1]):
            if X is not None:
                plt.plot(X, Y[:, i], label=variableNames[i])
            else:
                plt.plot(Y[:, i], label=variableNames[i])

        plt.xlabel('Steps')
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.plotsPath / plotName}.pdf", format="pdf")
        plt.close()

    def pickle(self, name, obj):
        with open(f"{self.picklePath}/{name}.p", 'wb') as f:
            pickle.dump(obj, f)

    def unpickle(self, name):
        with open(self.picklePath / f"{name}.p", 'rb') as f:
            return pickle.load(f)
