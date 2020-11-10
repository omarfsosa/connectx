"""
Train a self-playing agent.
"""
from pathlib import Path
import itertools
import json
import torch
from kaggle_environments import make, evaluate
from connectx import Agent, CNN, Strategy, Transition

GAMES_PER_EPISODE = 1000
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


def play_agent_vs_agent(player1, player2, player_id=1, limit=20):

    if player_id == 1:
        player1.train()
        player1.get_exploration_factor = constant_function(EXPLORATION)
    elif player_id == 2:
        player2.train()
        player2.get_exploration_factor = constant_function(EXPLORATION)

    for _ in range(limit):
        if player_id == 1:
            player2.eval()
        else:
            player1.eval()

        transitions_player_1, transitions_player_2 = play_many(
            [player1, player2], num_games=GAMES_PER_EPISODE
        )
        transitions = (
            Transition(*zip(*transitions_player_1))
            if player_id == 1
            else Transition(*zip(*transitions_player_2))
        )
        num_wins = transitions.reward.count(1)
        win_ratio = num_wins / GAMES_PER_EPISODE

        if player_id == 1:
            player1.learn(transitions, player_id=player_id)
            new_factor = player1.get_exploration_factor() * DECAY_FACTOR
            player1.get_exploration_factor = constant_function(new_factor)
        else:
            player2.learn(transitions, player_id=player_id)
            new_factor = player2.get_exploration_factor() * DECAY_FACTOR
            player2.get_exploration_factor = constant_function(new_factor)

        if win_ratio > 0.9:
            break

    else:  # For loop has reached limit
        _error_msg = f"Target win ratio of 90% not reached. Stopped at {win_ratio:5.2%} (Player {player_id})"
        raise ValueError(_error_msg)

    return player1, player2


if __name__ == "___main___":
    cnn = CNN()
    cnn.load_state_dict(torch.load(PATH_RESULTS / "mario_20201109.pt"))
    stg = Strategy(cnn)
    mario = Agent(stg)  # player 1

    cnn2 = CNN()
    cnn2.load_state_dict(torch.load(PATH_RESULTS / "luigi_20201109.pt"))
    stg2 = Strategy(cnn2)
    luigi = Agent(stg2)  # player 2

    mario.get_exploration_factor = constant_function(EXPLORATION)
    luigi.get_exploration_factor = constant_function(EXPLORATION)
    mario.train()
    luigi.train()

    performance = []

    for t in itertools.count():
        print(f"{t}: Training Mario ... ")
        mario, luigi = play_agent_vs_agent(mario, luigi, 1)
        print(f"{t}: Training Luigi ... ")
        mario, luigi = play_agent_vs_agent(mario, luigi, 2)

        if t % 10 == 0:
            mario.eval()
            luigi.eval()
            mario_outcomes = evaluate("connectx", [mario, "negamax"], num_episodes=100)
            luigi_outcomes = evaluate("connectx", ["negamax", luigi], num_episodes=100)
            wins = mario_outcomes.count([1, -1]) + luigi_outcomes.count([-1, 1])
            performance.append(wins / 200)
            print(f"{t:>5}: Win ratio vs negamax: {wins/200:6.2%}")
            with open(PATH_RESULTS / "performance.json", "w") as f:
                json.dump(performance, f)

            mario.strategy.save(PATH_RESULTS / "mario_20201110.pt")
            luigi.strategy.save(PATH_RESULTS / "luigi_20201110.pt")
