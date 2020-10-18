import numpy as np
import torch
import gym


class GPMemory():

    def __init__(self, inputDims: int, maxSize=500, device=None):

        self.maxSize = maxSize
        if device:

            self.inputs = torch.zeros((maxSize, inputDims), device=device)
            self.outputs = torch.zeros((maxSize, 1), device=device)
        else:
            self.inputs = np.zeros((maxSize, inputDims))
            self.outputs = np.zeros((maxSize, 1))

        self.ptr = 0
        self.looped = False

    @property
    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, x, y):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        assert(x.shape[1] == self.inputs.shape[1])

        self.inputs[self.ptr, :] = x
        self.outputs[self.ptr] = y
        self.ptr += 1

    @property
    def data(self):
        if self.size == self.maxSize:
            return self.inputs, self.outputs
        else:
            return self.inputs[:self.ptr], self.outputs[:self.ptr]


class GymMemory():

    def __init__(self, observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Box,
                 reference_space: gym.spaces.Box = None,
                 maxSize=10000,
                 device=None):
        self.maxSize = maxSize
        self.state = torch.zeros((maxSize, observation_space.shape[0]), device=device)
        self.action = torch.zeros((maxSize, action_space.shape[0]), device=device)
        self.reward = torch.zeros((maxSize, 1), device=device)
        self.next_state = torch.zeros((maxSize, observation_space.shape[0]), device=device)
        self.done = torch.zeros((maxSize, 1), device=device)
        self.ref = torch.zeros((maxSize, reference_space.shape[0]), device=device) if reference_space else None

        self.ptr = 0
        self.looped = False

    @property
    def size(self):
        return self.maxSize if self.looped else self.ptr

    def add(self, state, act, reward, next_state, done, ref=None):
        if self.ptr >= self.maxSize:
            self.ptr = 0
            self.looped = True

        self.state[self.ptr, :] = state
        self.action[self.ptr, :] = act
        self.reward[self.ptr, :] = reward
        self.next_state[self.ptr, :] = next_state
        self.done[self.ptr, :] = done
        if self.ref is not None:
            self.ref[self.ptr, :] = ref

        self.ptr += 1

    def get(self, batchSize=None):
        if batchSize:
            if self.looped:
                idx = torch.randint(self.maxSize, (batchSize,))
            else:
                idx = torch.randint(self.ptr, (batchSize,))

            if self.ref is not None:
                return self.state[idx, :], self.action[idx, :], self.reward[idx, :], self.next_state[idx, :], self.done[idx, :], self.ref[idx, :]
            else:
                return self.state[idx, :], self.action[idx, :], self.reward[idx, :], self.next_state[idx, :], self.done[idx, :]
        else:
            if self.looped:
                if self.ref is not None:

                    return self.state, self.action, self.reward, self.next_state, self.done, self.ref
                else:
                    return self.state, self.action, self.reward, self.next_state, self.done

            else:
                if self.ref is not None:
                    return self.state[:self.ptr, :], self.action[:self.ptr, :], self.reward[:self.ptr, :], self.next_state[:self.ptr, :], self.done[:self.ptr, :], self.ref[:self.ptr, :]
                else:
                    return self.state[:self.ptr, :], self.action[:self.ptr, :], self.reward[:self.ptr, :], self.next_state[:self.ptr, :], self.done[:self.ptr, :]
