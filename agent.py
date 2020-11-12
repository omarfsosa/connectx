"""
A connectX player suitable for Kaggle's environment.
"""
import random

from torch.utils.data import DataLoader
from utils import get_next_board, get_valid_moves, view_from, Dataset


class Agent:
    """A connectX player"""
    def __init__(self, strategy, greedy=False):
        self.strategy = strategy
        self.greedy = greedy

    def __call__(self, obs, config):
        return self.act(obs, config)

    def get_exploration_factor(self):
        """For epsilon-greedy policy"""
        raise NotImplementedError

    def act(self, obs, config=None):
        """
        Select an action for the given board setup.
        The signature of this function is chosen for compatibility with
        Kaggle submission requirements.
        """
        board = obs["board"]
        player_id = obs["mark"]
        action = self.select_action(board, player_id, config)
        return action

    def select_action(self, board, player_id, config=None):
        """
        Chooses an action for the given board configuration.
        If agent is not greedy, and epsilon-greedy policy will be used
        -- requires `self.get_exploration_factor` to be implemented.
        """
        if self.greedy:
            action = self.select_best_action(board, player_id, config)
            return action

        sample = random.random()
        epsilon = self.get_exploration_factor()
        if sample > epsilon:
            action = self.select_best_action(board, player_id, config)
        else:
            valid_moves = get_valid_moves(board, config)
            action = random.choice(valid_moves)

        return action

    def select_best_action(self, board, player_id, config=None):
        """Choose the action with maximum reward"""
        scores = self.evaluate_all_valid_moves(board, player_id, config)
        best_move = max(scores.keys(), key=scores.get)
        return best_move

    def evaluate_all_valid_moves(self, board, player_id, config=None):
        """Estimate of the reward for each possible move"""
        scores = {
            move: self.evaluate_action(board, move, player_id, config)
            for move in get_valid_moves(board, config)
        }
        return scores

    def evaluate_action(self, board, move, mark, config=None):
        """Estimate of the reward for a single action"""
        player_id = mark
        next_board = get_next_board(board, move, mark, config)
        next_board = view_from(next_board, player_id)
        score = self.strategy.eval_position(next_board, config).item()
        return score

    def train(self):
        """Set greedy to False and put the strategy in train mode"""
        self.greedy = False
        self.strategy.model.train()

    def eval(self):
        """Put the strategy in eval mode and set greedy to True"""
        self.strategy.model.eval()
        self.greedy = True

    def learn(self, transition, player_id, batch_size=64, epochs=5):
        """Update the strategy with the experience recorded in `transition`"""
        transition = transition.pov(player_id)
        ds = Dataset(transition.next_board, transition.reward)
        ld = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for idx, (X, y) in enumerate(ld):
                self.strategy.update(X, y)
