import numpy as np

cell_size = 18
cols = 10
rows = 22
max_timestep = 10**4

tetris_shapes = [
    [[1, 1, 1],
     [0, 1, 0]],

    [[0, 2, 2],
     [2, 2, 0]],

    [[3, 3, 0],
     [0, 3, 3]],

    [[4, 0, 0],
     [4, 4, 4]],

    [[0, 0, 5],
     [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]


def start_xy(shape):
    return int(cols / 2 - len(shape[0])/2), 0


def rotate_clockwise(shape):
    return [[shape[y][x]
             for y in range(len(shape))]
            for x in range(len(shape[0]) - 1, -1, -1)]


def check_collision(board, shape, offset):
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    del board[row]
    return [[0 for _ in range(cols)]] + board


def join_matrices(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def join_matrices_not_inplace(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    result = np.copy(mat1)

    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            result[cy + off_y - 1][cx + off_x] += val

    return result
def new_board():
    board = [[0 for __ in range(cols)]
             for _ in range(rows)]
    board += [[1 for _ in range(cols)]]
    return board
