import numpy as np
import torch as T
from DDQN.deep_q_network import DeepQNetwork
from DDQN.replay_memory import ReplayBuffer

class DDQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-6,
                 replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace

        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims)

        self.q_eval = DeepQNetwork(self.lr)

        self.q_next = DeepQNetwork(self.lr)
    def store_transition(self, state_board, state_block, action, reward, state_board_, state_block_,done):
        self.memory.store_transition(state_board, state_block, action, reward, state_board_, state_block_,done)

    def sample_memory(self):
        state_board, state_block, action, reward, state_board_, state_block_, done = \
                                self.memory.sample_buffer(self.batch_size)

        states_board = T.tensor(state_board).to(self.q_eval.device)
        states_block = T.tensor(state_block).to(self.q_eval.device)

        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_board_ = T.tensor(state_board_).to(self.q_eval.device)
        states_block_ = T.tensor(state_block_).to(self.q_eval.device)

        return states_board, states_block, actions, rewards, states_board_, states_block_, dones

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state_board = T.tensor(np.array([observation[0]]), dtype=T.float).to(self.q_eval.device)
            state_block = T.tensor([observation[1]],dtype=T.float).to(self.q_eval.device)

            actions = self.q_eval.forward((state_board, state_block))
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states_board, states_block, actions, rewards, states_board_, states_block_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward((states_board, states_block))[indices, actions]
        q_next = self.q_next.forward((states_board_, states_block_))
        q_eval = self.q_eval.forward((states_board_, states_block_))

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

        return loss


