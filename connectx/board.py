import itertools

import numpy as np


EMPTY = " "


class Board:
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._matrix = np.tile(EMPTY, (num_rows, num_cols))
        self._heights = np.zeros(num_cols, dtype=int)

    def place(self, piece, position):
        row = self._heights[position]
        self._matrix[row, position] = piece
        self._heights[position] += 1

    def get_free_columns(self):
        last_row = self._matrix[-1, :]
        return np.argwhere(last_row == EMPTY).flatten()

    def __str__(self):
        lines = []
        for row in reversed(self._matrix):
            line = [f"{'' if val == EMPTY else val:^3}" for val in row]
            line = "|" + "|".join(line) + "|"
            div = itertools.islice(itertools.cycle("+---"), len(line))
            lines.append("".join(div))
            lines.append(line)

        div = itertools.islice(itertools.cycle("+---"), len(line))
        lines.append("".join(div))
        return "\n".join(lines)


if __name__ == "__main__":
    num_rows = 6
    num_cols = 7
    b = Board(num_rows, num_cols)
    print(b)
