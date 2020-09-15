import argparse
import gym
import numpy as np

from trlfpi.report import Report
from trlfpi.memory import GPMemory
from trlfpi.gp import GP, Kernel

from scipy.optimize import brute, fmin


class Critic():

    def __init__(self,
                 inputSpace: int,
                 memory: int = 500,
                 mean=None):

        self.memory = GPMemory(inputSpace, maxSize=memory)
        self.model = GP(Kernel.RBF(1, [1.0 for i in range(inputSpace)]))
        self.mean = mean
        self.updates = 0

    def update(self, x: np.ndarray, y: np.ndarray):
        self.memory.add(x, y)
        self.updates += 1
        if self.memory.size >= 2:
            X, Y = self.memory.data
            optimize = updates % 100 == 0
            if self.mean is not None:
                self.model.fit(X, Y, self.mean, optimize=optimize)
            else:
                self.model.fit(X, Y, optimize=optimize)

    def predict(self, x: np.ndarray):
        # x contains action
        if self.memory.size >= 2:
            return self.model(x)
        else:
            if self.mean:
                return (self.mean(x), self.model.kernel(x, x))
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
    parser.add_argument("--sigma_tol", type=float, default=0.005)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--update_freq", type=int, default=1000)
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=25)

    # SETUP ARGUMENTS
    args = parser.parse_args()
    discount = args.discount
    episodes = args.episodes
    max_episode_len = args.max_episode_len
    gp_size = args.gp_size
    sigma_tol = args.sigma_tol
    delta = args.delta
    epsilonDecay = args.epsilonDecay
    update_freq = args.update_freq
    nRefs = args.nRefs
    systemPlots = args.plots
    plot_freq = args.plot_freq

    # Report
    report = Report('dgpq')
    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Init critit
    dCritic = Critic(1 + 2 + nRefs, memory=gp_size)
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
                action, _ = dCritic.getAction(state[:, :2 + nRefs])
                if np.abs(action) >= 2:
                    action = np.random.uniform(-2, 2, size=(1, 1))
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
