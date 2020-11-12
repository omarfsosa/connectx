"""
A class to store the moves made in a game.
"""
import reprlib
from typing import NamedTuple

from utils import view_from

class Transition(NamedTuple):
    """(Board, Action, Next Board, Reward)"""

    board: list
    action: int
    next_board: list
    reward: int

    def __repr__(self):
        return (
            "Transition("
            f"board={reprlib.repr(self.board)}, "
            f"action={self.action!r}, "
            f"next_board={reprlib.repr(self.next_board)}, "
            f"reward={self.reward!r})"
        )

    @classmethod
    def parse_steps(cls, steps, gamma=0.99):
        """
        From the content of `steps` (provided by kaggle's connectx environment),
        recover the transitions for players 1 and 2.
        Note: This function assumes that the steps are for a single game only.
        """
        boards = [s[0]["observation"]["board"] for s in steps]
        actions1 = [s[0]["action"] for s in steps][1::2]
        actions2 = [s[1]["action"] for s in steps][2::2]
        goal1 = steps[-1][0]["reward"]
        goal2 = steps[-1][1]["reward"]
        rewards1 = [goal1 * (gamma) ** i for i in range(len(actions1))][::-1]
        rewards2 = [goal2 * (gamma) ** i for i in range(len(actions2))][::-1]

        assert rewards1[-1] in [0, 1, -1]
        assert rewards2[-1] in [0, 1, -1]
        assert rewards2[-1] == -rewards1[-1]

        player1 = zip(
            boards[0::2],
            actions1,
            boards[1::2],
            rewards1,
        )

        player2 = zip(
            boards[1::2],
            actions2,
            boards[2::2],
            rewards2,
        )

        output = (
            [cls(*x) for x in player1],
            [cls(*x) for x in player2],
        )

        return output

    def pov(self, player_id):
        """
        Transition from the point of view of player `player_id`.
        """
        board = view_from(self.board, player_id)
        next_board = view_from(self.next_board, player_id)
        return self.__class__(board, self.action, next_board, self.reward)