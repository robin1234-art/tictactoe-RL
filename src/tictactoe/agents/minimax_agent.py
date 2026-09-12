"""Game-theoretically optimal player, via memoized minimax.

Tic-Tac-Toe is a solved game: with perfect play by both sides the outcome is
always a draw. This agent is the "ground truth" the RL agents are benchmarked
against — how close does a learned policy get to never losing?

The state space is tiny (~5478 reachable boards, ~765 up to symmetry), so
plain memoized minimax already runs instantly — no alpha-beta pruning needed.
Note: pruning combined with memoization would require care, since a
branch cut short by pruning only yields a *bound* on the true value, not the
exact value, and caching it as exact silently corrupts later lookups.
"""
from __future__ import annotations

import random

from tictactoe.agents.base import Agent
from tictactoe.game import Board, O, State, X, canonical

# Cache keyed on (canonical board, player to move) -> exact minimax value.
_value_cache: dict[tuple[State, int], int] = {}


def minimax_value(board: Board, player: int) -> int:
    """Exact game value from X's perspective (1 = X wins, -1 = O wins, 0 = draw)."""
    result = board.result()
    if result is not None:
        return result

    key = (canonical(board.cells), player)
    if key in _value_cache:
        return _value_cache[key]

    child_values = [minimax_value(board.play(move, player), -player) for move in board.valid_moves()]
    value = max(child_values) if player == X else min(child_values)

    _value_cache[key] = value
    return value


class MinimaxAgent(Agent):
    """Always plays an optimal move; ties are broken at random for variety."""

    def act(self, board: Board, player: int) -> int:
        moves = board.valid_moves()
        scored = [
            (move, minimax_value(board.play(move, player), -player))
            for move in moves
        ]
        best_value = max(v for _, v in scored) if player == X else min(v for _, v in scored)
        best_moves = [m for m, v in scored if v == best_value]
        return random.choice(best_moves)
