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
    # A at (0,0) 0-based -> down lands on (1,0) 0-based == R2,C1 1-based.
    parsed = {"object": "A", "direction": "down", "target": "R2,C1"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is None


def test_validate_rejects_wall():
    # B at (0,1) 0-based -> down lands on (1,1) 0-based == R2,C2 1-based (a wall).
    parsed = {"object": "B", "direction": "down", "target": "R2,C2"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is not None
    code, msg = error
    assert code == "wall"
    assert "wall" in msg.lower()


def test_validate_rejects_occupied():
    """D sits at (2,0); A trying to land there must fail."""
    b = _board_with_walls()
    b.apply_move("A", "down")
    # A now at (1,0) 0-based -> down lands on (2,0) 0-based == R3,C1 1-based (occupied by D).
    parsed = {"object": "A", "direction": "down", "target": "R3,C1"}
    error = _validate_move(b, _player(), parsed)
    assert error is not None
    code, msg = error
    assert code == "occupied"
    assert "occupied" in msg.lower()


def test_validate_rejects_out_of_bounds():
    parsed = {"object": "A", "direction": "up", "target": "R-1,C0"}
    error = _validate_move(_board_with_walls(), _player(), parsed)
    assert error is not None
    code, msg = error
    assert code == "off_grid"
    assert "grid" in msg.lower()


def test_validate_move_has_no_declared_target_param():
    import inspect
    from utils.board import Board
    sig = inspect.signature(Board.validate_move)
    assert "declared_target" not in sig.parameters


def test_valid_direction_move_ok():
    from utils.board import Board
    b = Board(8, set())
    b.place_object("A", 3, 3)
    assert b.validate_move("A", "right", allowed_objects={"A"}) is None


def test_off_grid_and_wall_still_illegal():
    from utils.board import Board
    b = Board(8, {(2, 3)})
    b.place_object("A", 0, 0)
    assert b.validate_move("A", "up")[0] == "off_grid"
    b.place_object("B", 1, 3)
    assert b.validate_move("B", "down")[0] == "wall"
