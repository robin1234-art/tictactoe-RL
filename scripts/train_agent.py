"""Train the tabular Q-learning agent via self-play and save it to models/q_agent.pkl.

Run once so the Streamlit app and notebooks can load a ready-trained agent
instead of retraining (~15s) on every run:

    python scripts/train_agent.py
"""
from pathlib import Path

from tictactoe.training import train

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "q_agent.pkl"


def main() -> None:
    agent = train(episodes=20000)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent.save(MODEL_PATH)
    print(f"Saved trained agent ({len(agent.q)} state-action entries) to {MODEL_PATH}")


if __name__ == "__main__":
    main()
