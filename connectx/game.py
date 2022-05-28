import itertools

from connectx.evaluator import State


class Game:
    def __init__(self, board, players, evaluator):
        self.board = board
        self.players = players
        self.evaluator = evaluator
        self.setup()

    def setup(self):
        self.turns = itertools.cycle(self.players)
        self._state = None
        self.context = None

    def run(self):
        self._state = State.RUNNING
        winner = None
        while self._state == State.RUNNING:
            player = next(self.turns)
            position = player.choose_action(self.board, self.context)
            self.board.place(player.piece, position)
            self._state = self.evaluator.get_state(self.board)
            print(self.board)
        if self._state == State.WIN:
            winner = player
            # give reward / penalties

        return self._state, winner


if __name__ == "__main__":
    from connectx.board import Board
    from connectx.player import RandomPlayer
    from connectx.evaluator import Evaluator

    bd = Board(6, 7)
    p1 = RandomPlayer(1, "X")
    p2 = RandomPlayer(2, "O")
    ev = Evaluator(4)
    game = Game(bd, [p1, p2], ev)
    state, winner = game.run()
    print(state, winner)
    if winner is not None:
        print(winner.piece)
