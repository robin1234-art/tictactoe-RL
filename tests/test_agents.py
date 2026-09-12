import random

from tictactoe.agents import MinimaxAgent, RandomAgent
from tictactoe.evaluation import evaluate
from tictactoe.game import Board, X
from tictactoe.training import train


def test_random_agent_only_plays_valid_moves():
    random.seed(0)
    agent = RandomAgent()
    board = Board((X, 0, 0, 0, 0, 0, 0, 0, 0))
    for _ in range(50):
        move = agent.act(board, X)
        assert move in board.valid_moves()


def test_q_learning_agent_rarely_loses_to_minimax_after_self_play():
    random.seed(0)
    agent = train(episodes=20000, opponent=None)
    agent.epsilon = 0.0  # evaluate greedily, no exploration

    stats_as_x = evaluate(agent, MinimaxAgent(), n_games=200)
    stats_as_o = evaluate(MinimaxAgent(), agent, n_games=200)

    # Symmetry-aware Q-table converges to near-optimal play well within
    # 20k self-play episodes; allow a small margin to avoid test flakiness.
    assert stats_as_x["o_wins"] <= 0.05
    assert stats_as_o["x_wins"] <= 0.05
