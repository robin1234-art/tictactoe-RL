from __future__ import annotations

from abc import ABC, abstractmethod

from tictactoe.game import Board


class Agent(ABC):
    """Common interface so any agent can be dropped into training/evaluation."""

    @abstractmethod
    def act(self, board: Board, player: int) -> int:
        """Return the index of the cell to play."""

    def reset(self) -> None:
        """Hook for stateful agents (e.g. to clear an episode trace). No-op by default."""
