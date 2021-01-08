import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

import trlfpi
from trlfpi.report import Report
from trlfpi.agents import Agent
from trlfpi.gp.gpmodel import GPModel
from trlfpi.timer import Timer


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print('Starting env')
    env = gym.make('clutch-v0', horizon=5, deltaActionCost=0.0)

    config = {
        'memorySize': 1000,
        'epsilon': 1e-3,
        'length': [1.0, 1.0, 1.0],
        'theta': 1.0,
        'inDim': 3,
        'outDim': 2,
        'sigma': 0.1
    }
    model = GPModel(config)
    model.setup(device=device)

    report = Report('results/1_TDPPO_b')
    report.id(5)

    agentCP: dict = report.unpickle('agent_best')
    agent = Agent.create('tdppo', agentCP['config'])
    agent.setup(agentCP)
    agent.eval()

    timer = Timer()

    print('Starting GPu')

    episodes = 10
    for episode in range(episodes):

        done = False
        step = 0
        state, ref = env.setup()

        gpIns = []
        states = []
        predictedStates = []
        refs = []

        totalError = 0

        while not done and step < 100:
            # Get action from controller
            u, _ = agent.act(state, ref)

            # Predict next state
            gpIn = np.hstack([state, u])
            predictedState = model(torch.tensor(gpIn, device=device)).cpu().detach().numpy()

            # Step on the env
            next_state, reward, done, next_ref = env.step(u)


            # Calc abs
            totalError += np.sum((next_state - predictedState)**2)

            gpIns.append(gpIn)
            states.append(next_state)
            predictedStates.append(predictedState)
            refs.append(next_ref[:, 0])

            state = next_state
            ref = next_ref
            step += 1

        # Plot
        states = np.vstack(states)
        gpIns = np.vstack(gpIns)
        predictedStates = np.vstack(predictedStates)
        refs = np.vstack(refs)

        print(f"Total error: {totalError}")

        plt.plot(states[:, 0], '--', label='x0')
        plt.plot(predictedStates[:, 0], '-', label="x0'")
        plt.plot(refs, '*', label="r")
        plt.plot()
        plt.legend()
        plt.grid()
        plt.show()

        plt.plot(states[:, 1], '--', label='x1')
        plt.plot(predictedStates[:, 1], '-', label="x1'")
        plt.legend()
        plt.grid()
        plt.show()

        # Learn
        timer.start()
        model.addData(torch.tensor(gpIns, device=device), torch.tensor(states, device=device))
        model.fit()

        print(f"Time to learn gp: {timer.stop()}")
