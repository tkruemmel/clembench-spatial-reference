"""Tests for SoloMatrixGameInstanceGenerator."""

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("solo_in")
    import sys
    sys.path.insert(0, str(REPO / "matrixgame_covered_solo"))
    from instancegenerator import SoloMatrixGameInstanceGenerator

    gen = SoloMatrixGameInstanceGenerator()
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_16_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 16, f"expected 16 experiments, got {len(experiments)}"


def test_each_experiment_has_required_fields(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in (
                "grid_size", "walls", "objects",
                "start_positions", "target_positions",
                "optimal_moves", "max_turns", "max_retries",
                "thinking", "spatial_level", "compact_board",
                "player_prompt",
            ):
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_objects_field_is_all_six(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert set(inst["objects"]) == set("ABCDEF")


def test_origins_differ_from_targets(generated_instances):
    """Generator invariant: every object's origin != its target."""
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for obj in inst["objects"]:
                origin = tuple(inst["start_positions"][obj])
                target = tuple(inst["target_positions"][obj])
                assert origin != target, f"{exp['name']} inst {inst['game_id']}: {obj} origin==target"


def test_max_turns_matches_formula(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["max_turns"] == max(4 * inst["optimal_moves"], 20)


def test_optimal_moves_at_least_num_objects(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["optimal_moves"] >= len(inst["objects"]), (
                f"{exp['name']}: optimal {inst['optimal_moves']} < {len(inst['objects'])}"
            )


def test_thinking_and_compact_axes_covered(generated_instances):
    """Verify the 16-config matrix: 2 thinking x 4 levels x 2 compact."""
    seen = set()
    for exp in generated_instances["experiments"]:
        inst = exp["game_instances"][0]
        seen.add((inst["thinking"], inst["spatial_level"], inst["compact_board"]))
    expected = {
        (thinking, level, compact)
        for thinking in (True, False)
        for level in ("S1", "S2", "S3", "S4")
        for compact in (True, False)
    }
    assert seen == expected, f"missing combinations: {expected - seen}"
