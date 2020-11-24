import numpy as np
import gym
import matplotlib.pyplot as plt


class PID(object):
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, dt=0.01, setpoint=0.0, bounds=[]):
        super().__init__()
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.error = np.array(0.0)
        self.last_error = np.array(0.0)
        self.integral = np.array(0.0)
        self.derivative = np.array(0.0)
        self.setpoint = np.array(setpoint)
        self.bounds = bounds

    def clamp(self, u):
        u = np.abs(u)
        if self.bounds:
            if u > self.bounds[1]:
                u = np.array(self.bounds[1])
            elif u < self.bounds[0]:
                u = np.array(self.bounds[0])
            else:
                pass
        return u

    def control(self, measurement, setpoint=[]):
        if setpoint:
            self.setpoint = np.array(setpoint)
        self.error = self.setpoint - measurement
        self.integral = self.integral + self.error * self.dt
        self.derivative = (self.error - self.last_error) / self.dt
        u = self.Kp * self.error + self.Ki * self.integral + self.Kd * self.derivative
        u = self.clamp(u)
        self.last_error = self.error
        return u


if __name__ == '__main__':
    env = gym.make('clutch-v0', horizon=3, deltaActionCost=0.0)
    print(env.h)
    print(env.deltaActionCost)

    controller = PID(15, 90, 0.001, env.timeStep, 209)

    done = False
    step = 0
    state, ref = env.setup()

    states = []
    actions = []
    rewards = []
    refs = []

    while not done:
        u = controller.control(state[0, 0], ref[0, 1]).reshape((1, 1))
        u = np.abs(u)

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

    plt.plot(states[:, 0] * 60 / 2 / np.pi, '-', label='omega_in')
    plt.plot(refs[:, 1] * 60 / 2 / np.pi, '--', label='reference')
    plt.xlabel('n')
    plt.ylabel('omega_in')
    plt.legend()
    plt.show()

    plt.plot(states[:, 0] * 60 / 2 / np.pi, '+', label='omega_in')
    plt.plot(env.gamma * states[:, 1] * 60 / 2 / np.pi, 'x', label='gamma*omega_out')
    plt.plot(refs[:, 1] * 60 / 2 / np.pi, '--', label='reference')
    plt.xlabel('n')
    plt.ylabel('omega_in')
    plt.legend()
    plt.show()
