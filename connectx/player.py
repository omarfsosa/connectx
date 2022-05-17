import abc
import random


class Player(abc.ABC):
    def __init__(self, id_, piece):
        self.id = id_
        self.piece = piece

    @abc.abstractmethod
    def choose_action(self, board, context):
        """
        Select a column to play on.
        """


class RandomPlayer(Player):
    def choose_action(self, board, context):
        options = board.get_free_columns()
        return random.choice(options)


class HumanPlayer(Player):
    def choose_action(self, board, context):
        options = list(board.get_free_columns())
        message = f"Choose a column{options}: "
        column = input(message)
        return int(column)


if __name__ == "__main__":
    p = RandomPlayer(1, "X")
    print(p.id, p.piece)
