"""Tests for move validation in matrixgame_covered_solo."""

from utils.board import Board
from matrixgame_covered_solo.master import SoloMatrixPlayer, _validate_move


class _StubModel:
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()


def _board_with_walls():
    b = Board(grid_size=4, walls={(1, 1)})
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    b.place_object("D", 2, 0)
    return b


def _player():
    return SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C", "D", "E", "F"})


def test_validate_accepts_valid_move():
    parsed = {"object": "A", "direction": "down", "target": "R1,C0"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is None


def test_validate_rejects_wall():
    parsed = {"object": "B", "direction": "down", "target": "R1,C1"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is not None
    assert "wall" in error.lower()


def test_validate_rejects_occupied():
    """D sits at (2,0); A trying to land there must fail."""
    b = _board_with_walls()
    b.apply_move("A", "down")
    parsed = {"object": "A", "direction": "down", "target": "R2,C0"}
    error = _validate_move(b, _player(), parsed)
    assert error is not None
    assert "occupied" in error.lower()


def test_validate_rejects_out_of_bounds():
    parsed = {"object": "A", "direction": "up", "target": "R-1,C0"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is not None
    assert "grid" in error.lower()
