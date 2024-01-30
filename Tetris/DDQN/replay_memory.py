import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory_board = np.zeros((self.mem_size, 1, 22, 10),
                                           dtype=np.float32)

        self.new_state_memory_board = np.zeros((self.mem_size, 1, 22, 10),
                                               dtype=np.float32)

        self.state_memory_block = np.zeros((self.mem_size, input_shape[1]),
                                           dtype=np.float32)

        self.new_state_memory_block = np.zeros((self.mem_size, input_shape[1]),
                                               dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool8)

    def store_transition(self, state_board, state_block, action, reward, state_board_, state_block_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory_board[index] = state_board
        self.state_memory_block[index] = state_block
        self.new_state_memory_board[index] = state_board_
        self.new_state_memory_block[index] = state_block_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states_board = self.state_memory_board[batch]
        states_block = self.state_memory_block[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_board_ = self.new_state_memory_board[batch]
        states_block_ = self.new_state_memory_block[batch]

        terminal = self.terminal_memory[batch]

        return states_board, states_block, actions, rewards, states_board_, states_block_, terminal