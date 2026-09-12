# tictactoe-RL

Tic-Tac-Toe as a small, self-contained testbed for combining **game theory**
and **reinforcement learning**: a tabular Q-learning agent trained by
self-play, benchmarked against a minimax agent that plays game-theoretically
optimally.

Tic-Tac-Toe is a *solved* game — perfect play by both sides always ends in a
draw. That gives a clean, provable yardstick for the RL agent: after
training, how close does it get to never losing?

## Project structure

```
src/tictactoe/
├── game.py              # board logic, win detection, symmetry reduction
├── env.py                # Gymnasium-compatible environment
├── agents/
│   ├── random_agent.py   # uniform random baseline
│   ├── minimax_agent.py  # game-theoretic optimal player (memoized minimax)
│   └── q_learning_agent.py  # tabular Q-learning, trained via self-play
├── training.py            # self-play / vs-fixed-opponent training loop
└── evaluation.py          # head-to-head win/draw/loss evaluation
tests/                     # pytest suite (game rules, minimax optimality, RL convergence)
notebooks/                 # game-theory analysis, training curves, agent comparison
```

## Key ideas

- **Perspective-normalized Q-table** — states are stored as "my pieces vs.
  opponent's" rather than raw X/O, so a single table trained via self-play
  can play either side.
- **Symmetry reduction** — the board has 8 equivalent orientations (the
  square's dihedral group). Both the minimax agent and the Q-learning agent
  key their tables on a canonical form, so experience learned in one
  orientation generalizes to all 7 others. This cuts the effective state
  space from ~5478 reachable boards to ~765 classes, and was the difference
  between the Q-learning agent converging in ~20k self-play episodes versus
  needing far more to reach the same performance.
- **Minimax as ground truth** — used both as an oracle (exact game value of
  any position) and as an opponent to evaluate the RL agent against.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Core game logic + minimax oracle + tabular Q-learning agent, with tests
- [ ] Notebooks: game-theory analysis, training curves, agent comparison
- [ ] DQN agent (Stable-Baselines3) as a function-approximation comparison
- [ ] Streamlit demo to play against the trained agent
- [ ] CI (GitHub Actions: tests + lint)
- [ ] Containerized deployment (Docker) to a personal server
