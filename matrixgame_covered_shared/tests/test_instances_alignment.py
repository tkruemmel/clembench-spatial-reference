"""The physical puzzle must be identical across all three games and every
variant, per (level, instance index) — thesis, comparability across variants."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
LEVELS = ["S1", "S2", "S3", "S4"]

FILES = [
    "matrixgame_covered_solo/in/instances.json",
    "matrixgame_covered_comm/in/instances.json",
    "matrixgame_covered_comm_nplayer/in/instances.json",
]


def _boards_by_level(path, level):
    data = json.loads((ROOT / path).read_text())
    out = {}
    for exp in data["experiments"]:
        name = exp["name"]
        if not (name.startswith(level + "_") or f"_{level}_" in name):
            continue
        out[name] = [
            (
                json.dumps(gi["start_positions"], sort_keys=True),
                json.dumps(gi["target_positions"], sort_keys=True),
                json.dumps(gi["walls"]),
                gi["optimal_moves"],
            )
            for gi in exp["game_instances"]
        ]
    return out


def test_boards_identical_across_all_games_and_variants():
    for level in LEVELS:
        variants = {}
        for path in FILES:
            variants.update(
                {f"{path}::{k}": v for k, v in _boards_by_level(path, level).items()}
            )
        assert variants, f"no experiments found for {level}"
        names = list(variants)
        reference = variants[names[0]]
        assert len(reference) == 8, f"{names[0]}: expected 8 instances"
        for name in names[1:]:
            assert variants[name] == reference, (
                f"{level}: boards in {name} differ from {names[0]}"
            )


def test_max_turns_formula_holds_everywhere():
    for path in FILES:
        data = json.loads((ROOT / path).read_text())
        for exp in data["experiments"]:
            for gi in exp["game_instances"]:
                assert gi["max_turns"] == max(4 * gi["optimal_moves"], 20)
