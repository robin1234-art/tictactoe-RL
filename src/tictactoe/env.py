"""Gymnasium-compatible Tic-Tac-Toe environment.

The learning agent always plays X and moves first; the opponent (O) is a
pluggable policy so the same env can be used for self-play, training against
a random baseline, or evaluating against the minimax oracle.
"""
from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tictactoe.game import Board, O, X

OpponentPolicy = Callable[[Board], int]


def random_opponent(board: Board) -> int:
    moves = board.valid_moves()
    return int(np.random.choice(moves))


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent: OpponentPolicy = random_opponent):
        super().__init__()
        self.opponent = opponent
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.board = Board()

    def _obs(self) -> np.ndarray:
        return np.array(self.board.cells, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.board = Board()
        return self._obs(), {}

    def step(self, action: int):
        if action not in self.board.valid_moves():
            # Illegal move: end the episode with a penalty rather than crashing,
            # so a randomly-initialised policy can still learn the rules.
            return self._obs(), -10.0, True, False, {"illegal_move": True}

        self.board = self.board.play(action, X)
        if self.board.is_terminal():
            return self._obs(), self._terminal_reward(), True, False, {}

        opponent_action = self.opponent(self.board)
        self.board = self.board.play(opponent_action, O)
        if self.board.is_terminal():
            return self._obs(), self._terminal_reward(), True, False, {}

        return self._obs(), 0.0, False, False, {}

    def _terminal_reward(self) -> float:
        result = self.board.result()  # 1 = X wins, -1 = O wins, 0 = draw
        assert result is not None
        return float(result)

    def render(self):
        print(self.board.render())
