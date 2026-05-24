"""Smoke tests for matrixgame_covered_comm_nplayer."""

import pytest
from pathlib import Path

PKG = Path(__file__).parent.parent


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


def test_config_has_32_variants():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    assert len(cfg) == 32, f"expected 32 variants, got {len(cfg)}"


def test_every_variant_has_required_fields():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    for key, variant in cfg.items():
        for field in ("name", "num_players", "spatial_level", "comm_protocol", "use_masking", "compact_board"):
            assert field in variant, f"variant {key} missing {field}"
        assert variant["num_players"] in (3, 6), f"{key}: num_players must be 3 or 6"
        assert variant["spatial_level"] in ("S1", "S2", "S3", "S4"), f"{key}: invalid spatial_level"
        assert variant["comm_protocol"] in ("none", "structured", "freeform", "hybrid"), f"{key}: invalid comm_protocol"
        assert variant["use_masking"] is True, f"{key}: use_masking must be true"
        assert variant["compact_board"] is True, f"{key}: compact_board must be true"


def test_variants_cover_full_matrix():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    seen = set()
    for variant in cfg.values():
        seen.add((variant["spatial_level"], variant["comm_protocol"], variant["num_players"]))
    expected = {
        (s, p, n)
        for s in ("S1", "S2", "S3", "S4")
        for p in ("none", "structured", "freeform", "hybrid")
        for n in (3, 6)
    }
    assert seen == expected, f"missing combinations: {expected - seen}"


@pytest.mark.parametrize("protocol", ["none", "structured", "freeform", "hybrid"])
def test_prompt_template_exists(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    assert path.exists(), f"missing template for {protocol}"
    text = path.read_text()
    assert "$YOUR_OBJECTS$" in text
    assert "$GRID_SIZE$" in text
    assert "$YOUR_ROLE$" in text
    assert "$N_OTHER_PLAYERS$" in text


@pytest.mark.parametrize("protocol", ["structured", "freeform", "hybrid"])
def test_message_protocols_mention_broadcast(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    text = path.read_text().lower()
    assert "broadcast" in text or "teammates" in text, \
        f"{protocol} template should mention teammates/broadcast"


def test_none_template_has_no_message_line():
    path = PKG / "resources" / "initial_prompts" / "en" / "player_prompt_none.template"
    text = path.read_text()
    assert "message:" not in text.lower(), "none template must not include 'message:'"


def test_instance_generator_module_imports():
    from instancegenerator import MatrixGameNPlayerInstanceGenerator
    gen = MatrixGameNPlayerInstanceGenerator()
    assert gen is not None
