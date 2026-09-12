"""Tabular Q-learning agent.

The Q-table is keyed on two normalizations of the board:

1. Perspective: own pieces = +1, opponent = -1, empty = 0, rather than raw
   X/O state — so a single table can play either side and can be trained
   via self-play.
2. Symmetry: the board is reduced to its canonical form under the square's
   8-fold symmetry group (see `tictactoe.game.canonical_transform`), and the
   chosen action is remapped into that canonical board's index space. This
   mirrors the reduction the minimax oracle gets "for free" from perfect
   play, and lets experience gathered in one board orientation generalize
   to the other 7 equivalent orientations — a large sample-efficiency win
   over a raw state table.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path

from tictactoe.agents.base import Agent
from tictactoe.game import Board, State, canonical_transform

QKey = tuple[State, int]


def perspective(board: Board, player: int) -> State:
    return tuple(c * player for c in board.cells)


def canonical_key(board: Board, player: int, action: int | None = None) -> tuple[State, int | None]:
    """Canonical (state, action) key for the Q-table.

    Pass `action=None` to just canonicalize the state (e.g. for bootstrapping
    over all valid next-moves, which are canonicalized individually instead).
    """
    persp = perspective(board, player)
    canon_state, mapping = canonical_transform(persp)
    canon_action = mapping.index(action) if action is not None else None
    return canon_state, canon_action


class QLearningAgent(Agent):
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: dict[QKey, float] = {}

    def act(self, board: Board, player: int, epsilon: float | None = None) -> int:
        moves = board.valid_moves()
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.choice(moves)
        return self._greedy_move(board, player, moves)

    def _greedy_move(self, board: Board, player: int, moves: list[int]) -> int:
        values = [self.q.get(canonical_key(board, player, m), 0.0) for m in moves]
        best = max(values)
        best_moves = [m for m, v in zip(moves, values) if v == best]
        return random.choice(best_moves)

    def update(
        self,
        board: Board,
        player: int,
        action: int,
        reward: float,
        next_board: Board,
        done: bool,
    ) -> None:
        key = canonical_key(board, player, action)
        current = self.q.get(key, 0.0)

        if done:
            target = reward
        else:
            next_moves = next_board.valid_moves()
            best_next = max(
                (self.q.get(canonical_key(next_board, player, m), 0.0) for m in next_moves),
                default=0.0,
            )
            target = reward + self.gamma * best_next

        self.q[key] = current + self.alpha * (target - current)

    def save(self, path: str | Path) -> None:
        """Persist the Q-table (a plain dict of tuples/ints/floats) to `path`."""
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "QLearningAgent":
        agent = cls(**kwargs)
        with open(path, "rb") as f:
            agent.q = pickle.load(f)
        return agent
