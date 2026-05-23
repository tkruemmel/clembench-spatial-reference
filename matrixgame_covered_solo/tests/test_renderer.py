"""Tests for view rendering in matrixgame_covered_solo."""

from utils.board import Board
from matrixgame_covered_solo.master import SoloMatrixPlayer, _render_view


class _StubModel:
    """Minimal Model substitute for instantiating Player without a backend."""
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()


def _build_board() -> Board:
    """3 owned (A,B,C) and 3 foreign (D,E,F) on a small board, no walls."""
    b = Board(grid_size=4)
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    b.place_object("C", 0, 2)
    b.place_object("D", 3, 0)
    b.place_object("E", 3, 1)
    b.place_object("F", 3, 2)
    return b


def test_render_full_shows_all_letters():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C", "D", "E", "F"})
    rendered = _render_view(_build_board(), player, view_mode="full", compact=False)
    for letter in "ABCDEF":
        assert letter in rendered, f"{letter!r} missing from full-view render"
    assert "X" not in rendered  # nothing should be masked


def test_render_masked_shows_owned_and_x_for_foreign():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C"})
    rendered = _render_view(_build_board(), player, view_mode="masked", compact=False)
    for letter in "ABC":
        assert letter in rendered, f"owned letter {letter!r} missing from masked render"
    for letter in "DEF":
        assert letter not in rendered, f"foreign letter {letter!r} leaked into masked render"
    assert "X" in rendered  # foreigns appear as X


def test_render_compact_full_lists_all_objects():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C", "D", "E", "F"})
    rendered = _render_view(_build_board(), player, view_mode="full", compact=True)
    assert "Objects:" in rendered
    for letter in "ABCDEF":
        assert f"{letter}@" in rendered


def test_render_compact_masked_hides_foreign_identities():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C"})
    rendered = _render_view(_build_board(), player, view_mode="masked", compact=True)
    for letter in "ABC":
        assert f"{letter}@" in rendered
    for letter in "DEF":
        assert f"{letter}@" not in rendered
    # foreign cells appear with label X
    assert "X@" in rendered
