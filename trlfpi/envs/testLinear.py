import gym
import numpy as np
import matplotlib.pyplot as plt

K = np.array([[5.4126], [0.2097], [-5.3391], [-0.4110], [0.1121], [-0.0262], [0.0047]])


def controller(x, r):
    Z = np.hstack((x, r[:, 1:]))
    return -Z @ K


if __name__ == '__main__':

    env = gym.make('linear-with-ref-v0', horizon=5, deltaActionCost=0.0)
    print(env.h)
    print(env.alpha)

    done = False
    step = 0
    state, ref = env.setup()

    states = []
    actions = []
    rewards = []
    refs = []

    while not done:
        u = controller(state, ref).reshape((1, 1))
        next_state, reward, done, next_ref = env.step(u)

        states.append(state)
        actions.append(u)
        rewards.append(reward)
        refs.append(ref)

        state = next_state
        ref = next_ref

    states = np.vstack(states)
    actions = np.vstack(actions)
    rewards = np.vstack(rewards)
    refs = np.vstack(refs)

    plt.plot(refs[:, 0], '--', label='reference')
    plt.plot(states[:, 0], '-', label='x0')
    plt.plot(actions, 'x', label='u')
    plt.xlabel('n')
    plt.legend()
    plt.show()
