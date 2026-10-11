"""Prompt-arm invariants for matrixgame_covered_comm_grid_updated_prompt.

The package must differ from matrixgame_covered_comm_grid in exactly two ways:
every player_prompt is the revised one, and optimal_moves is the A* attainable
optimum A_k (with max_turns its 4x budget) rather than the generator's per-object
BFS sum. Boards and ownership are untouched, and no $PLACEHOLDER$ is gained or
lost — the master substitutes blind, so a typo ships a literal to the model.
"""

import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SOURCE = PKG.parent / "matrixgame_covered_comm_grid"
ATTAINABLE = PKG.parent / "analysis" / "efficiency_baseline"
INSTANCE_FILES = {"instances.json": "dyad"}
MUTABLE = {"player_prompt": "", "max_turns": 0, "optimal_moves": 0}
PLACEHOLDER = re.compile(r"\$[A-Z_]+\$")


def _instances(pkg: Path, filename: str):
    experiments = json.loads((pkg / "in" / filename).read_text(encoding="utf-8"))["experiments"]
    return {(e["name"], i["game_id"]): i for e in experiments for i in e["game_instances"]}


def _pair(filename: str):
    return _instances(PKG, filename), _instances(SOURCE, filename)


def _attainable(suite: str):
    rows = (ATTAINABLE / f"attainable_{suite}.jsonl").read_text(encoding="utf-8").splitlines()
    return {(d["level"], d["game_id"]): d["stuckfree"] for d in map(json.loads, rows)}


def test_only_the_prompt_and_optimum_differ_from_the_source_arm():
    for filename in INSTANCE_FILES:
        arm, source = _pair(filename)
        assert arm.keys() == source.keys(), filename
        for key, instance in arm.items():
            assert {**instance, **MUTABLE} == {**source[key], **MUTABLE}, key


def test_every_prompt_differs_from_the_source_arm():
    for filename in INSTANCE_FILES:
        arm, source = _pair(filename)
        stale = [k for k, i in arm.items() if i["player_prompt"] == source[k]["player_prompt"]]
        assert not stale, f"{filename}: {len(stale)} instances still carry the old prompt: {stale[:5]}"


def test_placeholders_survive_the_edit():
    for filename in INSTANCE_FILES:
        arm, source = _pair(filename)
        for key, instance in arm.items():
            assert PLACEHOLDER.findall(instance["player_prompt"]) == PLACEHOLDER.findall(
                source[key]["player_prompt"]
            ), key


def test_optimum_and_budget_come_from_the_attainable_solver():
    for filename, suite in INSTANCE_FILES.items():
        arm, source = _pair(filename)
        A = _attainable(suite)
        for key, instance in arm.items():
            optimum = A[(instance["spatial_level"], instance["game_id"])]
            assert instance["optimal_moves"] == optimum, key
            # The generator's invariant, restored: the budget is 4x the optimum.
            assert instance["max_turns"] == max(4 * optimum, 20), key
            # A_k is attainable, the BFS sum is a lower bound: never tighter.
            assert instance["optimal_moves"] >= source[key]["optimal_moves"], key
            assert instance["max_turns"] >= source[key]["max_turns"], key
