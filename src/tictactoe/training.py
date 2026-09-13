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
    alternate_sides: bool = True,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,  # kept > 0: decaying fully to 0 starves rarer branches (e.g. playing second) of updates
    epsilon: float | None = None,
    agent: QLearningAgent | None = None,
) -> QLearningAgent:
    """Train a QLearningAgent.

    If `opponent` is None, the agent trains via self-play, controlling both
    sides through the same (perspective-normalized) Q-table.

    Otherwise the agent plays against the fixed `opponent` policy. With
    `alternate_sides=True` (the default) it plays X on even episodes and O on
    odd ones, so it learns a complete two-sided policy against that opponent
    rather than only ever moving first (or only ever moving second) — the two
    seats see very different state distributions. Set `alternate_sides=False`
    to pin the agent to `agent_player` for every episode instead.

    Pass `epsilon` for a constant exploration rate (e.g. when resuming
    training in checkpoints and annealing epsilon across the outer loop
    yourself); otherwise it linearly decays from `epsilon_start` to
    `epsilon_end` over this call's `episodes`.

    Pass an existing `agent` to continue training it instead of starting a
    fresh Q-table (`alpha`/`gamma` are then ignored — the agent's own values
    apply).
    """
    if agent is None:
        agent = QLearningAgent(alpha=alpha, gamma=gamma)

    for ep in range(episodes):
        if epsilon is not None:
            eps = epsilon
        else:
            progress = ep / max(episodes - 1, 1)
            eps = epsilon_start + (epsilon_end - epsilon_start) * progress

        if opponent is None:
            sides = (X, O)
        elif alternate_sides:
            sides = (agent_player if ep % 2 == 0 else -agent_player,)
        else:
            sides = (agent_player,)

        board = Board()
        player = X
        trace: dict[int, list[tuple[Board, int]]] = {p: [] for p in sides}

        while True:
            if player in trace:
                state_board = board
                action = agent.act(board, player, epsilon=eps)
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
