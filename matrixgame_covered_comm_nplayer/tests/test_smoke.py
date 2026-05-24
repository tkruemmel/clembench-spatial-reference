"""Smoke tests for matrixgame_covered_comm_nplayer."""


def test_board_module_imports():
    from utils.board import Board, DIRECTIONS
    assert DIRECTIONS  # non-empty dict/list
    board = Board(grid_size=8, walls=set())
    assert board is not None


def test_astar_module_imports():
    from utils import astar
    assert astar is not None
