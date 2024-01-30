import math
import random

import gym
from gym import spaces
import numpy as np
from gym.core import ActType

import Env.EnvConfigs as EnvConfigs


class TetrisEnv(gym.Env):
    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.n_rows = EnvConfigs.rows
        self.n_cols = EnvConfigs.cols
        self.MAX_STEP = EnvConfigs.max_timestep

        self.observation_space = spaces.Dict({
            'board': spaces.Box(low=0, high=1, shape=(self.n_rows, self.n_cols)),  # Tetris board state
            'next_piece': spaces.Discrete(7),
            'score': spaces.Box(low=-10**10, high=10*10)
        })

        self.action_space = spaces.Discrete(5)

        '''
            0 -> No_Op
            1 -> Right
            2 -> Left
            3 -> PushDown
            4 -> Rotate
        '''

        self.next_piece = None
        self.next_piece_idx = None
        self.current_piece = None
        self.current_piece_idx = None
        self.current_piece_orientation = None
        self.current_x = None
        self.current_y = None
        self.score = None

        self.board = None
        self.game_over = None
        self.level = None
        self.lines = None
        self.current_timestep = None
        self.actions_made = None
        self.height = None
        self.place = None

    def __new_stone(self):
        self.current_piece_orientation = 0
        self.current_piece_idx = self.next_piece_idx
        self.current_piece = self.next_piece
        self.next_piece_idx = random.randint(0, 6)
        self.next_piece = EnvConfigs.tetris_shapes[self.next_piece_idx]
        self.current_x, self.current_y = EnvConfigs.start_xy(self.current_piece)

        if EnvConfigs.check_collision(self.board, self.current_piece, (self.current_x, self.current_y)):
            self.game_over = True
        else:
            self.place += 1
    def init_game(self):
        self.board = EnvConfigs.new_board()
        self.next_piece_idx = random.randint(0, 6)
        self.next_piece = EnvConfigs.tetris_shapes[self.next_piece_idx]
        self.place = 0
        self.__new_stone()

        self.game_over = False
        self.level = 1
        self.score = 0
        self.lines = 1
        self.current_timestep = 0
        self.actions_made = 0
        self.height = 0

    def __move(self, delta_x):
        if not self.game_over:
            new_x = self.current_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > self.n_cols - len(self.current_piece[0]):
                new_x = self.n_cols - len(self.current_piece[0])
            if not EnvConfigs.check_collision(self.board, self.current_piece, (new_x, self.current_y)):
                self.current_x = new_x

    def __drop(self, manual):
        if not self.game_over:
            self.score += 0.5 if manual else 0
            self.current_y += 1
            if EnvConfigs.check_collision(self.board, self.current_piece, (self.current_x, self.current_y)):
                self.board = EnvConfigs.join_matrices(self.board, self.current_piece, (self.current_x, self.current_y))
                self.__new_stone()
                cleared_rows = 0

                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = EnvConfigs.remove_row(self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                self.lines += cleared_rows
            return True
        return False

    def reset(self, **kwargs):
        self.init_game()
        negated_2d_array = [[-1 * element for element in row] for row in self.current_piece]

        changed_board = EnvConfigs.join_matrices_not_inplace(self.board, negated_2d_array,
                                                             (self.current_x, self.current_y))
        block = [self.current_piece_idx, self.current_piece_orientation, self.next_piece_idx, self.__holes(), self.__aggregated_height(), self.__bumpiness()]
        return changed_board[0:-1], block, 0, False


    def __rotate_piece(self):
        if not self.game_over:
            rotated_piece = EnvConfigs.rotate_clockwise(self.current_piece)
            if not EnvConfigs.check_collision(self.board, rotated_piece, (self.current_x, self.current_y)):
                self.current_piece = rotated_piece
                self.current_piece_orientation = (self.current_piece_orientation + math.pi/4) % 2 * math.pi

    def step(self, action: ActType):
        self.current_timestep += 1

        init_ag_height = self.__aggregated_height()
        init_holes = self.__holes()
        init_bump = self.__bumpiness()

        if self.current_timestep >= self.MAX_STEP:
            negated_2d_array = [[-1 * element for element in row] for row in self.current_piece]

            changed_board = EnvConfigs.join_matrices_not_inplace(self.board, negated_2d_array,
                                                                 (self.current_x, self.current_y))
            block = [self.current_piece_idx, self.current_piece_orientation, self.next_piece_idx, self.__holes(), self.__aggregated_height(), self.__bumpiness()]
            return changed_board[0:-1], block, 300, True

        init_lines = self.lines

        if action == 0:
            self.__drop(manual=False)
            self.actions_made = 0
        elif action == 1:
            self.__move(delta_x=1)
            self.actions_made += 1
        elif action == 2:
            self.__move(delta_x=-1)
            self.actions_made += 1
        elif action == 3:
            self.__drop(manual=True)
            self.actions_made = 0
        elif action == 4:
            self.__rotate_piece()
            self.actions_made += 1

        if self.game_over:
            block = [self.current_piece_idx, self.current_piece_orientation, self.next_piece_idx, self.__holes(), self.__aggregated_height(), self.__bumpiness()]
            return self.board[0:-1], block, -10, True

        if self.actions_made == 3:
            self.__drop(manual=False)
            self.actions_made = 0

        negated_2d_array = [[-1 * element for element in row] for row in self.current_piece]


        changed_board = EnvConfigs.join_matrices_not_inplace(self.board, negated_2d_array,
                                                             (self.current_x, self.current_y))

        lines = np.count_nonzero(np.any(self.board, axis=1))

        cleared_rows = self.lines - init_lines

        self.height = lines

        holes = self.__count_empty()


        reward = (self.height * self.n_cols - 2 * holes) / self.height

        if cleared_rows == 1:
            reward += 100
        elif cleared_rows == 2:
            reward += 150
        elif cleared_rows == 3:
            reward += 300
        elif cleared_rows == 4:
            reward += 1200

        reward += -(self.__aggregated_height() - init_ag_height)  - (self.__holes()- init_holes) - (self.__bumpiness() - init_bump)
        block = [self.current_piece_idx, self.current_piece_orientation, self.next_piece_idx, self.__holes(), self.__aggregated_height(), self.__bumpiness()]

        return changed_board[0:-1], block, reward, False

    def __count_empty(self):
        count = 0
        for row in self.board[len(self.board) - self.height:]:
            for e in row:
                if e == 0:
                    count += 1
        return count

    def __aggregated_height(self):
        column_heights = [0] * self.n_cols

        for col in range(self.n_cols):
            for row in range(self.n_rows):
                if self.board[row][col] != 0:
                    column_heights[col] = self.n_rows - row
                    break

        aggregate_height = sum(column_heights)

        return aggregate_height

    def __holes(self):
        column_heights = [0] * self.n_cols

        for col in range(self.n_cols):
            for row in range(self.n_rows):
                if self.board[row][col] != 0:
                    column_heights[col] = self.n_rows - row
                    break

        total_holes = 0

        for col in range(self.n_cols):
            for row in range(self.n_rows - column_heights[col]):
                if self.board[row][col] == 0 and any(self.board[r][col] != 0 for r in range(row + 1, self.n_rows)):
                    total_holes += 1

        return total_holes

    def __bumpiness(self):
        column_heights = [self.n_rows - max(row) if any(row) else 0 for row in zip(*self.board)]
        bumpiness = sum(abs(column_heights[i] - column_heights[i + 1]) for i in range(self.n_cols - 1))

        return bumpiness
