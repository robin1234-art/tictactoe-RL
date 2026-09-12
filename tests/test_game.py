from tictactoe.game import Board, O, X, canonical, count_reachable_states, symmetries


def test_empty_board_has_nine_moves():
    board = Board()
    assert board.valid_moves() == list(range(9))
    assert not board.is_terminal()


def test_row_win_detected():
    board = Board((X, X, X, O, O, 0, 0, 0, 0))
    assert board.winner() == X
    assert board.is_terminal()
    assert board.result() == X


def test_column_win_detected():
    board = Board((O, X, X, O, X, 0, O, 0, 0))
    assert board.winner() == O
    assert board.result() == O


def test_diagonal_win_detected():
    board = Board((X, O, O, O, X, 0, 0, 0, X))
    assert board.winner() == X


def test_draw_detected():
    board = Board((X, O, X, X, O, O, O, X, X))
    assert board.winner() is None
    assert board.is_full()
    assert board.result() == 0


def test_play_raises_on_occupied_cell():
    board = Board((X, 0, 0, 0, 0, 0, 0, 0, 0))
    try:
        board.play(0, O)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on occupied cell")


def test_symmetries_include_original_and_have_length_eight():
    board = Board((X, 0, 0, 0, O, 0, 0, 0, 0))
    variants = symmetries(board.cells)
    assert len(variants) == 8
    assert canonical(board.cells) in variants + [board.cells]


def test_reachable_state_count_matches_known_value():
    # Well-known result for Tic-Tac-Toe: 5478 reachable legal board states.
    assert count_reachable_states() == 5478
