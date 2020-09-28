import argparse
import gym
import numpy as np

from trlfpi.report import Report
from trlfpi.gp import Critic
from trlfpi.timer import Timer

# Report
report = Report('dgpq')
timer = Timer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="linear-with-ref-v0")
    parser.add_argument("--cpus", type=int, default=1)

    # Env params
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--nRefs", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.99)

    # DGPQ params
    parser.add_argument("--gp_size", type=int, default=1000)
    parser.add_argument("--sigma_tol", type=float, default=1e-3)
    parser.add_argument("--delta", type=float, default=5e-4)
    parser.add_argument("--update_freq", type=int, default=100)
    parser.add_argument("--optim_freq", type=int, default=1000)
    parser.add_argument("--b_grid_size", type=int, default=5)

    # Exploration params
    parser.add_argument("--epsilonDecay", type=float, default=3.0)
    parser.add_argument("--exploration_std", type=float, default=0.5)

    # Plot args
    parser.add_argument("--plots", action='store_true')
    parser.add_argument("--plot_freq", type=int, default=25)

    # SETUP ARGUMENTS
    args = parser.parse_args()

    cpus = args.cpus

    episodes = args.episodes
    max_episode_len = args.max_episode_len
    nRefs = args.nRefs
    discount = args.discount

    gp_size = args.gp_size
    sigma_tol = args.sigma_tol
    delta = args.delta
    update_freq = args.update_freq
    optim_freq = args.optim_freq
    b_grid_size = args.b_grid_size

    epsilonDecay = args.epsilonDecay
    exploration_std = args.exploration_std

    systemPlots = args.plots
    plot_freq = args.plot_freq

    report.logArgs(args.__dict__)

    # Setup
    env = gym.make(args.env)

    # Init critit
    dCritic = Critic(1 + 2 + nRefs,
                     memory=gp_size,
                     optim_freq=optim_freq,
                     bruteFactor=1,
                     bGridSize=b_grid_size,
                     bPoolSize=cpus)
    critic = Critic(1 + 2 + nRefs,
                    memory=gp_size,
                    optim_freq=optim_freq,
                    bGridSize=b_grid_size,
                    bPoolSize=cpus)

    updates = 0
    episodeTimer = Timer()
    for episode in range(1, episodes + 1):
        episodeTimer.start()
        state = env.reset()
        total_reward = 0
        tmpUpdates = 0

        epsilon = np.exp(-episode * epsilonDecay / episodes)
        if epsilon <= 0.2:
            epsilon = 0

        states = []
        actions = []
        qs = []

        for step in range(max_episode_len):

            timer.start()
            action, _ = dCritic.getAction(state[:, :2 + nRefs])
            report.log('gpAction', timer.stop())
            if np.abs(action) >= 2:
                action = np.array([[0.0]])
            if np.random.random() < epsilon:
                action += np.random.normal(0, exploration_std)

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update critic
            action_next, q_next = dCritic.getAction(next_state[:, :2 + nRefs])
            qt = reward + discount * q_next
            qs.append(qt)
            x = np.hstack((action, state[:, :2 + nRefs]))

            mean1, sigma1 = critic(x)
            if sigma1 >= sigma_tol or critic.memory.size < 500:
                critic.update(x, qt)
                tmpUpdates += 1

            mean1, sigma1 = critic(x)
            meanD, sigmaD = dCritic(x)

            if sigma_tol >= sigma1 and np.abs(meanD - mean1) > delta * 2 and critic.memory.size >= 500:

                updates += 1
                dCritic.update(x, mean1)

                if updates % update_freq == 0:
                    dCritic.getAction(state[:, :2 + nRefs], plotName=f"dc_{updates}")
                    critic.getAction(state[:, :2 + nRefs], plotName=f"c_{updates}")
                    critic = Critic(2 + 1 + nRefs,
                                    memory=gp_size,
                                    mean=dCritic.model.mean)

            states.append(state[0, 0])
            actions.append(action[0, 0])

            if done:
                break

            state = next_state

        if episode % plot_freq == 0 and systemPlots:

            # Plot to see how it looks
            plotData = np.stack((states,
                                 env.reference.r[:len(states)],
                                 actions), axis=-1)

            report.savePlot(f"episode_{episode}_plot",
                            ['State', 'Ref', 'Actions'],
                            plotData)
            report.savePlot(f"episode_{episode}_q_plot",
                            ['Q'], np.array(qs).reshape(-1, 1))

        print(f"Episode {episode}: Reward = {total_reward}, Epsilon = {epsilon}, Updates = {updates}, TmpUpdates = {tmpUpdates}")
        report.log('epsilon', epsilon, episode)
        report.log('rewards', total_reward, episode)
        report.log('episodeTime', episodeTimer.stop(), episode)
        report.log('updates', updates, episode)
        report.log('tmpUpdates', tmpUpdates, episode)

    report.generateReport()

    report.pickle('critic', {
        "data": dCritic.memory,
        "params": dCritic.model.kernel.params,
        "sigma_n": dCritic.model.sigma_n
    })
