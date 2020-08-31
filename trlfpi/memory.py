import numpy as np
import gym


class GPMemory():

    def __init__(self, inputDims: int, maxSize=500):
        self.maxSize = maxSize
        self.inputs = np.zeros((maxSize, inputDims))
        self.outputs = np.zeros((maxSize, 1))
        self.ptr = 0
        self.looped = False

    @property
    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, x: np.ndarray, y: np.ndarray):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        assert(x.shape[1] == self.inputs.shape[1])

        self.inputs[self.ptr, :] = x
        self.outputs[self.ptr] = y
        self.ptr += 1

    @property
    def data(self):
        return self.inputs, self.outputs


class GymMemory():

    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box, maxSize=10000):
        self.maxSize = maxSize
        self.state = np.zeros((maxSize, observation_space.shape[0]))
        self.action = np.zeros((maxSize, action_space.shape[0]))
        self.reward = np.zeros(( maxSize, 1 ))
        self.next_state = np.zeros((maxSize, observation_space.shape[0]))
        self.done = np.zeros((maxSize, 1))
        self.ptr = 0
        self.looped = False

    @property
    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, state, act, reward, next_state, done):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        self.state[self.ptr, :] = state
        self.action[self.ptr, :] = act
        self.reward[self.ptr, :] = reward
        self.next_state[self.ptr, :] = next_state
        self.done[self.ptr, :] = done

        self.ptr += 1

    def get(self, batchSize=None):
        if batchSize:
            if self.looped:
                idx = np.random.choice(self.maxSize, size=batchSize, replace=False)
            else:
                idx = np.random.choice(self.ptr, size=batchSize, replace=False)

            return self.state[idx,:], self.action[idx,:], self.reward[idx,:], self.next_state[idx,:], self.done[idx,:]
        else:
            if self.looped:
                return self.state, self.action, self.reward, self.next_state, self.done

            else:
                return self.state[:self.ptr,:], self.action[:self.ptr,:], self.reward[:self.ptr,:], self.next_state[:self.ptr,:], self.done[:self.ptr,:]

