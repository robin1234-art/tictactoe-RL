"""Core Tic-Tac-Toe board logic, independent of any RL framework.

Board is a length-9 tuple of ints: 0 = empty, 1 = player X, -1 = player O.
Cells are indexed row-major:
    0 1 2
    3 4 5
    6 7 8
"""
from __future__ import annotations

from itertools import product
from typing import Iterable

EMPTY, X, O = 0, 1, -1

WIN_LINES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)

State = tuple[int, ...]


class Board:
    """Immutable-style board: every mutating operation returns a new Board."""

    def __init__(self, cells: State | None = None):
        self.cells: State = cells if cells is not None else (EMPTY,) * 9

    def valid_moves(self) -> list[int]:
        return [i for i, c in enumerate(self.cells) if c == EMPTY]

    def play(self, action: int, player: int) -> "Board":
        if self.cells[action] != EMPTY:
            raise ValueError(f"Cell {action} is already occupied")
        if player not in (X, O):
            raise ValueError(f"player must be {X} or {O}, got {player}")
        new_cells = list(self.cells)
        new_cells[action] = player
        return Board(tuple(new_cells))

    def winner(self) -> int | None:
        """Return X, O, or None (no winner yet)."""
        for a, b, c in WIN_LINES:
            s = self.cells[a] + self.cells[b] + self.cells[c]
            if s == 3:
                return X
            if s == -3:
                return O
        return None

    def is_full(self) -> bool:
        return EMPTY not in self.cells

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.is_full()

    def result(self) -> int | None:
        """Terminal outcome from X's perspective: 1 (X wins), -1 (O wins), 0 (draw).

        Returns None if the game is not over.
        """
        if not self.is_terminal():
            return None
        w = self.winner()
        return w if w is not None else 0

    def render(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        rows = [
            " ".join(symbols[c] for c in self.cells[r : r + 3])
            for r in (0, 3, 6)
        ]
        return "\n".join(rows)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.cells == other.cells

    def __hash__(self) -> int:
        return hash(self.cells)

    def __repr__(self) -> str:
        return f"Board({self.cells})"


# --- Symmetries -------------------------------------------------------
# The square board has 8 symmetries (dihedral group D4): 4 rotations x
# reflection. Exploiting them shrinks the ~5478 reachable states down to
# ~765 distinct positions up to symmetry, which is a nice game-theory
# talking point and also speeds up minimax/Q-table lookups.

_ROTATE = (6, 3, 0, 7, 4, 1, 8, 5, 2)  # 90deg rotation index mapping
_FLIP = (2, 1, 0, 5, 4, 3, 8, 7, 6)    # horizontal flip index mapping


def _apply(cells: State, mapping: Iterable[int]) -> State:
    return tuple(cells[i] for i in mapping)


def symmetries(cells: State) -> list[State]:
    """All 8 board states equivalent to `cells` under the square's symmetry group."""
    variants = []
    current = cells
    for _ in range(4):
        current = _apply(current, _ROTATE)
        variants.append(current)
        variants.append(_apply(current, _FLIP))
    return variants


def canonical(cells: State) -> State:
    """A deterministic representative of `cells`'s symmetry class (min by tuple order)."""
    return min(symmetries(cells) + [cells])


def symmetry_mappings() -> list[State]:
    """The 8 index permutations of the square's symmetry group (as used by `_apply`).

    `_apply(cells, mapping)` puts `cells[mapping[j]]` at position j, so these are
    the same transforms `symmetries()` applies to cell values, applied instead to
    position indices — which is what's needed to remap an *action* consistently
    with a board's canonicalization (see `canonical_transform`).
    """
    identity = tuple(range(9))
    mappings = []
    current = identity
    for _ in range(4):
        current = _apply(current, _ROTATE)
        mappings.append(current)
        mappings.append(_apply(current, _FLIP))
    return mappings


def canonical_transform(cells: State) -> tuple[State, State]:
    """Canonical form of `cells`, plus the index mapping used to reach it.

    Returns (canonical_cells, mapping) with canonical_cells == _apply(cells, mapping).
    To remap an action index `a` (defined on the original board) into the
    canonical board's index space, use `mapping.index(a)`.
    """
    variants = [(_apply(cells, m), m) for m in symmetry_mappings()]
    return min(variants, key=lambda variant: variant[0])


def count_reachable_states() -> int:
    """Brute-force count of reachable (legal) board states, for the README/notebook."""
    seen: set[State] = set()

    def explore(board: Board, player: int) -> None:
        if board.cells in seen:
            return
        seen.add(board.cells)
        if board.is_terminal():
            return
        for move in board.valid_moves():
            explore(board.play(move, player), -player)

    explore(Board(), X)
    return len(seen)
