"""Streamlit demo: play Tic-Tac-Toe against the project's trained agents.

Run with:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tictactoe.agents import MinimaxAgent, QLearningAgent, RandomAgent  # noqa: E402
from tictactoe.game import Board, O, X  # noqa: E402
from tictactoe.training import train  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "q_agent.pkl"

st.set_page_config(page_title="Tic-Tac-Toe RL", page_icon="⭕", layout="centered")


@st.cache_resource
def load_q_agent() -> QLearningAgent:
    if MODEL_PATH.exists():
        return QLearningAgent.load(MODEL_PATH)
    with st.spinner("Training the Q-learning agent by self-play (first run only, ~15s)..."):
        agent = train(episodes=20000)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        agent.save(MODEL_PATH)
    return agent


AGENT_FACTORIES = {
    "Random": RandomAgent,
    "Minimax (optimal)": MinimaxAgent,
    "Q-learning (self-play)": load_q_agent,
}


def start_new_game(opponent_name: str, human_player: int) -> None:
    st.session_state.board = Board()
    st.session_state.opponent_name = opponent_name
    st.session_state.human_player = human_player
    st.session_state.game_over = False
    st.session_state.status = None
    if human_player == O:
        _agent_move()


def _agent_move() -> None:
    board: Board = st.session_state.board
    agent = AGENT_FACTORIES[st.session_state.opponent_name]()
    ai_player = -st.session_state.human_player
    action = agent.act(board, ai_player)
    st.session_state.board = board.play(action, ai_player)
    _check_terminal()


def _check_terminal() -> bool:
    board: Board = st.session_state.board
    if not board.is_terminal():
        return False
    result = board.result()
    if result == 0:
        st.session_state.status = "It's a draw."
    elif result == st.session_state.human_player:
        st.session_state.status = "You win! 🎉"
    else:
        st.session_state.status = "The agent wins."
    st.session_state.game_over = True
    return True


def handle_cell_click(cell: int) -> None:
    board: Board = st.session_state.board
    if st.session_state.game_over or cell not in board.valid_moves():
        return
    st.session_state.board = board.play(cell, st.session_state.human_player)
    if _check_terminal():
        return
    _agent_move()


st.title("Tic-Tac-Toe vs. RL agent")
st.caption(
    "Play against a random baseline, a game-theoretically optimal minimax agent, "
    "or a tabular Q-learning agent trained entirely by self-play."
)

with st.sidebar:
    st.header("Settings")
    opponent_name = st.selectbox("Opponent", list(AGENT_FACTORIES.keys()))
    side_label = st.radio("Play as", ["X (first)", "O (second)"])
    human_player = X if side_label.startswith("X") else O

    needs_reset = (
        "board" not in st.session_state
        or st.session_state.get("opponent_name") != opponent_name
        or st.session_state.get("human_player") != human_player
    )
    if st.button("New game") or needs_reset:
        start_new_game(opponent_name, human_player)

board: Board = st.session_state.board
symbols = {X: "X", O: "O", 0: ""}

for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        idx = r * 3 + c
        cols[c].button(
            symbols[board.cells[idx]] or " ",
            key=f"cell_{idx}",
            use_container_width=True,
            disabled=st.session_state.game_over or board.cells[idx] != 0,
            on_click=handle_cell_click,
            args=(idx,),
        )

if st.session_state.status:
    st.subheader(st.session_state.status)
