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
notebooks/
└── 02_training_curricula.ipynb  # self-play vs. fixed-opponent (Random/Minimax) training, compared
app/app.py                 # Streamlit demo: play against any trained agent
scripts/train_agent.py     # trains and saves the self-play Q-learning agent used by the app
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

## Does the training opponent matter?

[`notebooks/02_training_curricula.ipynb`](notebooks/02_training_curricula.ipynb)
trains two Q-learning agents with identical hyperparameters and episode
budgets — one against a random opponent, one against minimax — and
benchmarks both against the minimax oracle and against random play. The
short version: each fixed opponent teaches an incomplete policy (the
Random-trained agent is exploitable by minimax's forks; the
Minimax-trained agent is exploitable by random's off-distribution play),
while self-play avoids both blind spots from the same episode budget.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"
pytest
```

Play against the trained agents:

```bash
python scripts/train_agent.py   # trains and saves the self-play agent (once)
streamlit run app/app.py
```

## Roadmap

- [x] Core game logic + minimax oracle + tabular Q-learning agent, with tests
- [x] Streamlit demo to play against the trained agents
- [x] Notebook: training curricula (self-play vs. fixed-opponent) compared
- [ ] Notebook: game-theory analysis of the state space
- [ ] DQN agent (Stable-Baselines3) as a function-approximation comparison
- [ ] CI (GitHub Actions: tests + lint)
- [ ] Containerized deployment (Docker) to a personal server
