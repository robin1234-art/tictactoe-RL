from __future__ import annotations

import random

from tictactoe.agents.base import Agent
from tictactoe.game import Board


class RandomAgent(Agent):
    """Plays a uniformly random legal move. Baseline opponent for training/eval."""

    def act(self, board: Board, player: int) -> int:
        return random.choice(board.valid_moves())
