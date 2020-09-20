import argparse
import gym
import numpy as np

from trlfpi.report import Report
from trlfpi.memory import GPMemory
from trlfpi.gp import GP, Kernel
from trlfpi.timer import Timer

from scipy.optimize import brute, fmin

from typing import Callable

# Report
report = Report('dgpq')
timer = Timer()


class Critic():

    def __init__(self,
                 inputSpace: int,
                 mean: Callable[[np.ndarray], np.ndarray] = None,
                 memory: int = 1000,
                 optim_freq: int = 100):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        kernel = Kernel.RBF(1, [1.0 for i in range(inputSpace)])
        self.model = GP(kernel, mean=mean, sigma_n=0.01)
        self.optim_freq = optim_freq
        self.updates = 0

    def update(self, x: np.ndarray, y: np.ndarray) -> float:
        self.memory.add(x, y)
        self.updates += 1
        if self.updates % self.optim_freq == 0:
            X, Y = self.memory.data

            timer.start()
            self.model.fit(X, Y, optimize=True)
            report.log('gpUpdate', timer.stop())

    def predict(self, x: np.ndarray):
        # x contains action
        return self.model(x)

    def getAction(self, x):
        # x without action
        # return action, value at action
        aRange = [slice(-2, 2, 0.25)]
        params = (x[0, 0], x[0, 1], x[0, 2])

        def f(a, *params):
            a = a[0]
            x0, x1, r0 = params
            q = self.predict(np.array([[a, x0, x1, r0]]))[0]
            return -q

        resbrute = brute(f, aRange, args=params, full_output=True, finish=fmin)
        return resbrute[0].reshape(1, 1), resbrute[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)

    # DGPQ params
    parser.add_argument("--gp_size", type=int, default=1000)
    parser.add_argument("--sigma_tol", type=float, default=0.005)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--update_freq", type=int, default=100)
    parser.add_argument("--optim_freq", type=int, default=1000)

    # Exploration params
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--exploration_variance", type=float, default=0.1)

    # Plot args
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=25)

    # SETUP ARGUMENTS
    args = parser.parse_args()

    episodes = args.episodes
    max_episode_len = args.max_episode_len
    nRefs = args.nRefs
    discount = args.discount

    gp_size = args.gp_size
    sigma_tol = args.sigma_tol
    delta = args.delta
    update_freq = args.update_freq
    optim_freq = args.optim_freq

    epsilonDecay = args.epsilonDecay
    exploration_variance = args.exploration_variance

    systemPlots = args.plots
    plot_freq = args.plot_freq

    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Init critit
    dCritic = Critic(1 + 2 + nRefs, memory=gp_size, optim_freq=optim_freq)
    critic = Critic(1 + 2 + nRefs, memory=gp_size, optim_freq=optim_freq)

    updates = 0
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        tmpUpdates = 0

        epsilon = np.exp(-episode * epsilonDecay / episodes)
        if epsilon <= 0.2:
            epsilon = 0

        states = [state[0, 0]]

        for step in range(max_episode_len):

            timer.start()
            action, _ = dCritic.getAction(state[:, :2 + nRefs])
            report.log('gpAction', timer.stop())
            if np.abs(action) >= 2:
                action = np.array([[0.0]])
            if np.random.random() < epsilon:
                action += np.random.normal(0, exploration_variance)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update critic
            action_next, q_next = dCritic.getAction(next_state[:, :2 + nRefs])
            qt = reward - discount * q_next
            x = np.hstack((action, state[:, :2 + nRefs]))
            _, sigma1 = critic(x)
            if sigma1 > sigma_tol:
                tmpUpdates += 1
                critic.update(x, qt)
            mean2, sigma2 = critic(x)
            meanD, sigmaV = dCritic(x)
            if sigma_tol >= sigma2 and np.abs(meanD - mean2) > delta * 2 and critic.memory.size >= 100:

                updates += 1
                dCritic.update(x, mean2)

                if updates % update_freq == 0:
                    def meanF(x):
                        value, sigma = dCritic(x)
                        return value

                    critic = Critic(2 + 1 + nRefs, memory=gp_size, mean=meanF)

            if done:
                break

            states.append(next_state[0, 0])
            state = next_state

        if episode % plot_freq == 0 and systemPlots:

            # Plot to see how it looks
            plotData = np.stack((states, env.reference.r[:len(states)]), axis=-1)
            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref'],
                            plotData)

        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon}, Updates = {updates}, TmpUpdates = {tmpUpdates}")
        report.log('rewards', total_reward, episode)
        report.log('updates', updates, episode)
        report.log('tmpUpdates', tmpUpdates, episode)

    report.generateReport()

    report.pickle('critic', {
        "data": dCritic.memory,
        "params": dCritic.model.kernel.params,
        "sigma_n": dCritic.model.sigma_n
    })
