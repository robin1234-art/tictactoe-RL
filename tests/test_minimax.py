import random

from tictactoe.agents import MinimaxAgent, RandomAgent
from tictactoe.agents.minimax_agent import minimax_value
from tictactoe.evaluation import evaluate
from tictactoe.game import Board, X


def test_empty_board_value_is_a_draw():
    # Tic-Tac-Toe is a solved game: perfect play from both sides is a draw.
    assert minimax_value(Board(), X) == 0


def test_minimax_never_loses_to_random_as_x_or_o():
    random.seed(0)
    stats_x = evaluate(MinimaxAgent(), RandomAgent(), n_games=200)
    assert stats_x["o_wins"] == 0

    stats_o = evaluate(RandomAgent(), MinimaxAgent(), n_games=200)
    assert stats_o["x_wins"] == 0


def test_two_minimax_agents_always_draw():
    random.seed(0)
    stats = evaluate(MinimaxAgent(), MinimaxAgent(), n_games=20)
    assert stats["draws"] == 1.0
