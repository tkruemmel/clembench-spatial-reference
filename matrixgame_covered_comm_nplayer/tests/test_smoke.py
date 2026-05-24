"""Smoke tests for matrixgame_covered_comm_nplayer."""


def test_board_module_imports():
    from utils.board import Board, DIRECTIONS
    assert DIRECTIONS  # non-empty dict/list
    board = Board(grid_size=8, walls=set())
    assert board.grid_size == 8


def test_astar_module_imports():
    from utils import astar
    assert any(callable(getattr(astar, n)) for n in dir(astar) if not n.startswith("_"))
