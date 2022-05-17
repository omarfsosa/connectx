import enum

import numpy as np


EMPTY = " "


def _iter_diagonals(matrix, num_connect):
    num_rows, num_cols = matrix.shape
    for row in range(num_rows - num_connect + 1):
        for offset in range(num_cols - num_connect + 1):
            diag = np.diagonal(matrix[row: row + 4], offset=offset)
            yield diag


def diagonals_view(matrix, num_connect):
    return np.array(list(_iter_diagonals(matrix, num_connect)))


def horizontals_view(matrix, num_connect):
    view = np.lib.stride_tricks.sliding_window_view(
        matrix, (1, num_connect)
    )
    return view.reshape((-1, num_connect))


def has_win(views):
    non_empty = views != EMPTY
    all_equal = views[:, 0][:, None] == views
    return (all_equal & non_empty).all(axis=1).any()


def check_for_winner(matrix, num_connect):
    hviews = horizontals_view(matrix, num_connect)
    dviews = diagonals_view(matrix, num_connect)
    hwin = has_win(hviews)
    dwin = has_win(dviews)
    if hwin or dwin:
        return True

    matrix = np.rot90(matrix)
    hviews = horizontals_view(matrix, num_connect)
    dviews = diagonals_view(matrix, num_connect)
    hwin = has_win(hviews)
    dwin = has_win(dviews)
    if hwin or dwin:
        return True

    return False


class State(enum.Enum):
    WIN = enum.auto()
    TIE = enum.auto()
    RUNNING = enum.auto()


class Evaluator:
    def __init__(self, num_connect):
        self.num_connect = num_connect

    def get_state(self, board):
        has_winner = check_for_winner(board._matrix, self.num_connect)
        if has_winner:
            return State.WIN

        last_row = board._matrix[-1, :]
        is_full = (last_row != EMPTY).all()
        if is_full:
            return State.TIE

        return State.RUNNING
