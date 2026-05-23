"""Tests for view rendering in matrixgame_covered_solo."""

from utils.board import Board
from matrixgame_covered_solo.master import _render_view


def _build_board() -> Board:
    """All 6 objects placed on a small board, no walls."""
    b = Board(grid_size=4)
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    b.place_object("C", 0, 2)
    b.place_object("D", 3, 0)
    b.place_object("E", 3, 1)
    b.place_object("F", 3, 2)
    return b


def test_render_shows_all_letters():
    rendered = _render_view(_build_board(), compact=False)
    for letter in "ABCDEF":
        assert letter in rendered, f"{letter!r} missing from render"
    assert "X" not in rendered


def test_render_compact_lists_all_objects():
    rendered = _render_view(_build_board(), compact=True)
    assert "Objects:" in rendered
    for letter in "ABCDEF":
        assert f"{letter}@" in rendered
