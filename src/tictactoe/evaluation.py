"""Head-to-head evaluation between two agents, used to benchmark RL agents
against the minimax oracle (or against each other)."""
from __future__ import annotations

from tictactoe.agents.base import Agent
from tictactoe.game import Board, O, X


def play_game(agent_x: Agent, agent_o: Agent) -> int:
    """Play one game and return the result from X's perspective (1/-1/0)."""
    board = Board()
    player = X
    agents = {X: agent_x, O: agent_o}
    while not board.is_terminal():
        action = agents[player].act(board, player)
        board = board.play(action, player)
        player = -player
    result = board.result()
    assert result is not None
    return result


def evaluate(agent_x: Agent, agent_o: Agent, n_games: int = 1000) -> dict[str, float]:
    """Play `n_games` and return win/draw/loss rates from X's perspective."""
    counts = {"x_wins": 0, "o_wins": 0, "draws": 0}
    for _ in range(n_games):
        result = play_game(agent_x, agent_o)
        if result == X:
            counts["x_wins"] += 1
        elif result == O:
            counts["o_wins"] += 1
        else:
            counts["draws"] += 1
    return {k: v / n_games for k, v in counts.items()}
