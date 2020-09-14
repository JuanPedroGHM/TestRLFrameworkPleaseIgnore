import argparse
import gym
import numpy as np

from trlfpi.report import Report
from trlfpi.memory import GPMemory
from trlfpi.gp import GPTD, Kernel

from scipy.optimize import brute, fmin


class Critic():

    def __init__(self,
                 inputSpace: int,
                 memory: int = 500,
                 discount: float = 0.99):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        self.model = GPTD(Kernel.RBF(1, [1.0 for i in range(inputSpace)]))
        self.H = None
        self.discount = discount

    def update(self, x: np.ndarray, y: np.ndarray):
        self.memory.add(x, y)
        if self.memory.size >= 3:

            # H
            n = self.memory.size
            if self.H:
                tmp = np.zeros((n - 1, n))
                tmp[:-1, :-1] = self.H
                self.H = tmp
                self.H[:-1, :-2] = 1
                self.H[:-1, :-1] = -self.discount

            else:
                self.H = np.array([[1, -self.discount, 0],
                                   [0, 1, -self.discount]])

            X, Y = self.memory.data
            self.model.fit(X, Y, self.H)

    def predict(self, x: np.ndarray):
        # x contains action
        if self.memory.size != 0:
            return self.model(x)
        else:
            return (0, self.model.kernel(x, x))

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
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--gp_size", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=25)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    gp_size = args.gp_size
    sigma = args.sigma
    delta = args.delta
    epsilonDecay = args.epsilonDecay
    nRefs = args.nRefs
    systemPlots = args.plots
    plot_freq = args.plot_freq

    # Report
    report = Report('GPTD')
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Init critit
    critic = Critic(1 + 2 + nRefs, memory=gp_size)
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
            sample = (np.random.random() < epsilon)
            if sample:
                action = np.random.uniform(-2, 2, size=(1, 1))
            else:
                action, _ = critic.getAction(state[:, :2 + nRefs])
                if np.abs(action) >= 2:
                    action = np.random.uniform(-2, 2, size=(1, 1))
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update critic
            critic.update(state, reward)

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
        "data": critic.memory,
        "params": critic.model.kernel.params,
        "sigma_n": critic.model.sigma_n
    })
