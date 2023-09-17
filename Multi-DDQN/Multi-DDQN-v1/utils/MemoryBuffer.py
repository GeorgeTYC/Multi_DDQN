import random
from collections import deque
import numpy as np



class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size):
        """ Initialization
                """

        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state):
        experience = (state, action, reward, done, new_state)

        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        """当前Buffer占有量"""
        return self.count

    def sample_batch(self, batch_size):
        """Sample a batch, optionally with (PER)"""
        batch = []

        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # return a batch of experience
        state_batch = np.array([i[0] for i in batch])
        action_batch = np.array([i[1] for i in batch])
        reward_batch = np.array([i[2] for i in batch])
        done_batch = np.array([i[3] for i in batch])
        new_state_batch = np.array([i[4] for i in batch])
        return state_batch, action_batch, reward_batch, done_batch, new_state_batch, idx

    def clear(self):
        """clear buffer"""
        self.buffer = deque()
        self.count = 0
