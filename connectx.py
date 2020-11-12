"""
Out of date!
"""

import random
import reprlib
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
        palette = [0, 1, 2]
        key = np.array([0, 1, -1] if player_id == 1 else [0, -1, 1])
        board = np.array(self.board)
        next_board = np.array(self.next_board)
        board_index = np.digitize(board.ravel(), palette, right=True)
        next_board_index = np.digitize(next_board.ravel(), palette, right=True)
        board = key[board_index].reshape(board.shape)
        next_board = key[next_board_index].reshape(next_board.shape)
        return self.__class__(board, self.action, next_board, self.reward)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


class Strategy:
    def __init__(self, model, lr=3e-5, wd=0.001):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def eval_position(self, board):
        x = self._board_to_input(board)
        prediction = self.model(x)
        return prediction

    def update(self, boards, rewards):
        self.optimizer.zero_grad()
        y_pred = self.eval_position(boards)
        y_true = torch.tensor(rewards, device=self.device, dtype=torch.float).view(
            *y_pred.shape
        )
        error = self.loss(y_true, y_pred)
        error.backward()
        self.optimizer.step()

    def _board_to_input(self, board):
        """
        Takes a board (or a list of boards) and puts
        it into a tensor shaped input
        """
        x = torch.tensor(board, dtype=torch.float, device=self.device)
        x = x.view(-1, 1, 6, 7)
        return x

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Saved state_dict at: {path}.")

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


class Agent:
    def __init__(self, strategy, greedy=False):
        self.strategy = strategy
        self.greedy = greedy
        self.experience = 0  # number of games already trained on
        self.tau = 1000

    def __call__(self, obs, config):
        return self.move(obs, config)

    def get_exploration_factor(self):
        """For epsilon-greedy policy"""
        raise NotImplementedError

    def move(self, obs, config):
        # Config is required in the signature of this function
        # just for Kaggle submissions
        board = obs["board"]
        player_id = obs["mark"]
        action = self.select_action(board, player_id)
        return action

    def select_action(self, board, player_id, debug=False):
        sample = random.random()
        epsilon = self.get_exploration_factor()

        if sample > epsilon or self.greedy:
            if debug:
                print("Choosing best.")
            action = self.select_best_action(board, player_id)
        else:
            if debug:
                print("Choosing random")
            valid_moves = self.get_valid_moves(board)
            action = random.choice(valid_moves)

        return action

    def select_best_action(self, board, player_id):
        scores = self.get_all_action_values(board, player_id)
        best_move = max(scores.keys(), key=scores.get)
        return best_move

    def get_all_action_values(self, board, player_id):
        valid_moves = self.get_valid_moves(board)
        scores = {
            move: self.evaluate_action(board, move, player_id) for move in valid_moves
        }
        return scores

    def evaluate_action(self, board, move, player_id):
        next_board = self.get_next_board(board, move, piece=player_id)
        tran = Transition(
            board, move, next_board, None
        )  # Using a transition just for the .pov method
        tran = tran.pov(player_id)  # Turn the opponent pieces into -1
        score = self.strategy.eval_position(tran.next_board).item()
        return score

    def get_next_board(self, board, move, piece):
        grid = np.reshape(board, (6, 7))
        zeros_idx = np.argwhere((grid.T[move] == 0)).flatten()
        row = max(zeros_idx)
        if row is None:
            # Invalid move was choseng
            return board

        grid[row, move] = piece
        next_board = grid.flatten().tolist()
        return next_board

    def get_valid_moves(self, board):
        valid_moves = [c for c in range(7) if board[c] == 0]
        return valid_moves

    def train(self):
        self.greedy = False
        self.strategy.model.train()

    def eval(self):
        self.strategy.model.eval()
        self.greedy = True

    def learn(self, transition, player_id, batch_size=64, epochs=5):
        transition = transition.pov(player_id)
        ds = Dataset(transition.next_board, transition.reward)
        ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for idx, (X, y) in enumerate(ld):
                self.strategy.update(X, y)
