"""Training loops for the tabular Q-learning agent.

Uses "afterstate" TD updates: for a given player, the Q-learning transition
goes from the board on their turn to the board on their *next* turn (i.e.
skipping over the opponent's reply), with the terminal win/draw/loss reward
attached to whichever player's move preceded the game ending. This is the
standard formulation for turn-based self-play Q-learning.
"""
from __future__ import annotations

from tictactoe.agents.base import Agent
from tictactoe.agents.q_learning_agent import QLearningAgent
from tictactoe.game import Board, O, X


def train(
    episodes: int,
    opponent: Agent | None = None,
    agent_player: int = X,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,  # kept > 0: decaying fully to 0 starves rarer branches (e.g. playing second) of updates
) -> QLearningAgent:
    """Train a QLearningAgent.

    If `opponent` is None, the agent trains via self-play, controlling both
    sides through the same (perspective-normalized) Q-table. Otherwise the
    agent always plays `agent_player` against the fixed `opponent` policy.
    """
    agent = QLearningAgent(alpha=alpha, gamma=gamma)
    sides = (X, O) if opponent is None else (agent_player,)

    for ep in range(episodes):
        progress = ep / max(episodes - 1, 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        board = Board()
        player = X
        trace: dict[int, list[tuple[Board, int]]] = {p: [] for p in sides}

        while True:
            if player in trace:
                state_board = board
                action = agent.act(board, player, epsilon=epsilon)
                board = board.play(action, player)
                trace[player].append((state_board, action))
            else:
                assert opponent is not None
                action = opponent.act(board, player)
                board = board.play(action, player)

            if board.is_terminal():
                break
            player = -player

        result = board.result()
        for p, seq in trace.items():
            for i, (s, a) in enumerate(seq):
                if i + 1 < len(seq):
                    agent.update(s, p, a, reward=0.0, next_board=seq[i + 1][0], done=False)
                else:
                    agent.update(s, p, a, reward=float(result * p), next_board=board, done=True)

    return agent
