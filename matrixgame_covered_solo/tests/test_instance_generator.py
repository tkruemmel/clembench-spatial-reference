"""Tests for SoloMatrixGameInstanceGenerator."""

import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    """Run instance generator into a temp output dir, return parsed instances.json."""
    out_dir = tmp_path_factory.mktemp("solo_in")
    import sys
    sys.path.insert(0, str(REPO / "matrixgame_covered_solo"))
    from instancegenerator import SoloMatrixGameInstanceGenerator

    gen = SoloMatrixGameInstanceGenerator()
    # Generate a small sample (2 instances per variant) for fast tests.
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_32_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 32, f"expected 32 experiments, got {len(experiments)}"


def test_each_experiment_has_required_fields(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in (
                "grid_size", "walls", "objects", "owned_objects", "foreign_objects",
                "start_positions", "target_positions",
                "optimal_moves", "max_turns", "max_retries",
                "view_mode", "thinking", "spatial_level", "compact_board",
                "player_prompt",
            ):
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_view_mode_full_owns_all_six(generated_instances):
    full_exps = [e for e in generated_instances["experiments"]
                 if e["game_instances"][0]["view_mode"] == "full"]
    assert full_exps, "no full-view experiments generated"
    for exp in full_exps:
        for inst in exp["game_instances"]:
            assert set(inst["owned_objects"]) == set("ABCDEF")
            assert set(inst["foreign_objects"]) == set()


def test_view_mode_masked_owns_three(generated_instances):
    masked_exps = [e for e in generated_instances["experiments"]
                   if e["game_instances"][0]["view_mode"] == "masked"]
    assert masked_exps
    for exp in masked_exps:
        for inst in exp["game_instances"]:
            assert len(inst["owned_objects"]) == 3
            assert len(inst["foreign_objects"]) == 3
            assert set(inst["owned_objects"]) & set(inst["foreign_objects"]) == set()
            assert set(inst["owned_objects"]) | set(inst["foreign_objects"]) == set("ABCDEF")


def test_origins_differ_from_targets(generated_instances):
    """Generator invariant: every owned object's origin != its target."""
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for obj in inst["owned_objects"]:
                origin = tuple(inst["start_positions"][obj])
                target = tuple(inst["target_positions"][obj])
                assert origin != target, f"{exp['name']} inst {inst['game_id']}: {obj} origin==target"


def test_max_turns_matches_formula(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["max_turns"] == max(4 * inst["optimal_moves"], 20)


def test_optimal_moves_positive(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["optimal_moves"] >= len(inst["owned_objects"]), (
                f"{exp['name']}: optimal {inst['optimal_moves']} < num_owned"
            )
