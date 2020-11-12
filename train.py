"""
Train a self-playing agent.
"""
from pathlib import Path
import itertools
import json
import torch
from kaggle_environments import make, evaluate

# ---- local
from agent import Agent
from network import CNN
from strategy import Strategy
from transition import Transition

GAMES_PER_EPISODE = 2000
EXPLORATION = 0.95
DECAY_FACTOR = 0.75
PATH_RESULTS = Path("/project/kaggle/connectx/results/")


def play_once(agents):
    env = make("connectx")
    env.reset()
    env.run(agents)
    return Transition.parse_steps(env.steps)


def play_many(agents, num_games):
    transitions_player1 = []
    transitions_player2 = []
    for _ in range(num_games):
        transitions_player_1, transitions_player_2 = play_once(agents)
        transitions_player1.extend(transitions_player_1)
        transitions_player2.extend(transitions_player_2)

    return transitions_player1, transitions_player2


def constant_function(x):
    def fn():
        return x

    return fn


if __name__ == "__main__":
    cnn = CNN()
    stg = Strategy(cnn)
    megaman = Agent(stg)  # player 1
    megaman.get_exploration_factor = constant_function(EXPLORATION)
    megaman.train()

    performance = []
    print("Training ... ")
    for t in itertools.count():
        megaman.train()
        # p1, p2 = play_many([megaman, "negamax"], num_games=GAMES_PER_EPISODE)
        p1, p2 = play_many(["negamax", megaman], num_games=GAMES_PER_EPISODE)
        transitions = Transition(*zip(*p2))
        num_wins = transitions.reward.count(1)
        win_ratio = num_wins / GAMES_PER_EPISODE
        print(f"WR: {win_ratio:>6.2%}")
        performance.append(win_ratio)
        megaman.learn(transitions, player_id=2)
        megaman.get_exploration_factor = constant_function(megaman.get_exploration_factor() * DECAY_FACTOR)
    
        torch.save(megaman.strategy.model.state_dict(), PATH_RESULTS / "megaman2.pt")
        with open(PATH_RESULTS / "megaman2_vs_negamax.json", "w") as f:
            json.dump(performance, f)

        if (win_ratio >= 0.95):
            break
    
