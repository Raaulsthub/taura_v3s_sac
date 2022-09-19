import numpy as np


# agent's memory
class ReplayBuffer:
    # builder
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size  # about a million state transitions
        self.mem_cntr = 0  # keeps track of the first available memory address
        # input shape will be corresponded to the observation dimensionality from the environment
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        # keeps track of the states resulted by the current action
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # stores current state and action in the memory
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size  # figure where first available memory is
        # setting the memory arrays to the current info
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    # returns a sample of our buffer
    def sample_buffer(self, batch_size):
        # how many memory is stored in our buffer
        max_mem = min(self.mem_cntr, self.mem_size)

        # randomizing memory batch to return as a sample
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


