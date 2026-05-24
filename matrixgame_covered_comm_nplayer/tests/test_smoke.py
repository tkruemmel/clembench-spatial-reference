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


import json


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("nplayer_in")
    import sys
    sys.path.insert(0, str(PKG))
    from instancegenerator import MatrixGameNPlayerInstanceGenerator

    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_32_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 32, f"expected 32 experiments, got {len(experiments)}"


def test_instance_schema(generated_instances):
    required = (
        "grid_size", "walls", "objects",
        "num_players", "player_objects",
        "start_positions", "target_positions",
        "optimal_moves", "max_turns", "max_retries", "strict",
        "use_masking", "spatial_level", "comm_protocol",
        "compact_board", "player_prompt",
    )
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in required:
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_player_partition_3p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_3p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 3
            po = inst["player_objects"]
            assert len(po) == 3, f"3p should have 3 groups, got {len(po)}"
            assert all(len(g) == 2 for g in po), f"3p: every group must own 2 objects, got {po}"
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            # Static-by-index partition
            assert po == [["A", "B"], ["C", "D"], ["E", "F"]]


def test_player_partition_6p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_6p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 6
            po = inst["player_objects"]
            assert len(po) == 6
            assert all(len(g) == 1 for g in po)
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            assert po == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]


def test_compact_board_always_true(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["compact_board"] is True


def test_experiment_naming(generated_instances):
    names = {exp["name"] for exp in generated_instances["experiments"]}
    # 32 unique names of form "S<level>_<protocol>_<n>p"
    assert len(names) == 32
    for name in names:
        parts = name.split("_")
        assert len(parts) == 3, f"unexpected experiment name shape: {name}"
        level, protocol, mode = parts
        assert level in ("S1", "S2", "S3", "S4")
        assert protocol.lower() in ("none", "structured", "freeform", "hybrid")
        assert mode in ("3p", "6p")
