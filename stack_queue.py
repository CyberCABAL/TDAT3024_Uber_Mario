from collections import deque
from random import sample
from numpy import zeros, stack, reshape
from sys import getsizeof

import compression


class StackQueue:
    def __init__(self, size = 1000, stack_size = 3, state_dim = (100, 100), colour = False):
        self.size = size
        self.colour = colour
        self.state_dim = state_dim
        self.stack_size = stack_size
        self.queue = deque(maxlen=size)
        self.stack = deque([zeros(state_dim) for i in range(stack_size)], maxlen=stack_size)
        self.n_stack = deque([zeros(state_dim) for i in range(stack_size)], maxlen=stack_size)

    def append(self, obj):
        self.queue.append(obj)

    def stacks(self, values, axis):
        frames = []
        for i in values:
            frame = self.queue[i]
            done_found = False
            q = None
            for j in range(0, self.stack_size):
                q_stack = self.queue[i - j] if not done_found else q
                if q_stack[4] and j > 0:
                    done_found = True
                    q_stack = self.queue[i - j + 1]
                    q = q_stack
                self.stack.append(compression.run_length_d(q_stack[0]))
                self.n_stack.append(compression.run_length_d(q_stack[3]))

            s0 = stack(self.stack, axis=axis)
            s1 = stack(self.stack, axis=axis)
            if self.colour:
                s0 = reshape(s0, (self.state_dim[0], self.state_dim[1], self.stack_size * 3))
                s1 = reshape(s1, (self.state_dim[0], self.state_dim[1], self.stack_size * 3))
            frames.append((s0, frame[1], frame[2], s1, frame[4]))
        return frames

    def random_stacks(self, axis, amount):
        return self.stacks(sample(range(self.stack_size - 1, len(self.queue)), amount), axis)

    def __len__(self):
        return len(self.queue)

    def __sizeof__(self):
        return getsizeof(self.queue[0])# + getsizeof(self.stack) + getsizeof(self.n_stack)
