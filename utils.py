"""
Useful Miscenlaneaus functions for connectX game.
"""

import numpy as np
import torch


def view_from(board, player_id):
    """
    Returns the board as seen from the perspective of `player_id`.
    This will return a board where every square is either -1, 0, or 1,
    where `1` will be used for the pieces of `player_id` and `-1` for
    the pieces of the opponent player.
    """
    assert player_id in [1, 2]
    palette = [0, 1, 2]
    key = np.array([0, 1, -1] if player_id == 1 else [0, -1, 1])
    board = np.array(board)
    board_index = np.digitize(board.ravel(), palette, right=True)
    board = key[board_index].reshape(board.shape)
    return board 

def get_next_board(board, column, mark, config=None):
    """The board that will result from placing `mark` into `column`"""
    EMPTY = 0
    board = list(board) # Do not modify the original
    if config is not None:
        columns = config.columns
        rows = config.rows
    else:
        # Default board shape
        columns = 7
        rows = 6

    row = max([r for r in range(rows) if board[column + (r * columns)] == EMPTY])
    board[column + (row * columns)] = mark
    return board

def get_valid_moves(board, config=None):
    """The columns where it is still valid to drop a piece."""
    if config is not None:
        columns = config.columns
    else:
        columns = 7

    valid_moves = [c for c in range(columns) if board[c] == 0]
    return valid_moves


class Dataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, X, y):
        "Initialization"
        self.y = y
        self.X = X

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.y)

    def __getitem__(self, index):
        "Generates one sample of data"
        return self.X[index], self.y[index]
