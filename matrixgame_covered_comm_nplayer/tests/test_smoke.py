"""Smoke tests for matrixgame_covered_comm_nplayer."""


def test_board_module_imports():
    from utils.board import Board, DIRECTIONS
    assert DIRECTIONS  # non-empty dict/list
    board = Board(grid_size=8, walls=set())
    assert board.grid_size == 8


def test_astar_module_imports():
    from utils import astar
    assert any(callable(getattr(astar, n)) for n in dir(astar) if not n.startswith("_"))


def test_common_config_shape():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "common_config.json") as f:
        cfg = json.load(f)
    assert cfg["grid_size"] == 8
    assert cfg["objects"] == ["A", "B", "C", "D", "E", "F"]
    assert cfg["directions"] == ["up", "down", "left", "right"]
    assert cfg["max_retries"] == 2
    assert cfg["strict"] is False
