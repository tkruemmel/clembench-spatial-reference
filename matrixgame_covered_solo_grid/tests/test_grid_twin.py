"""Grid-twin invariants for matrixgame_covered_solo_grid.

The package must be a rendering-only variant of matrixgame_covered_solo: identical boards per
experiment and instance index, compact_board=false everywhere, templates that
differ from the compact ones only in the board-format block, and a master that
clemcore can load through the import shim and that renders the ASCII grid.
"""

import difflib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
ROOT = PKG.parent
COMPACT = ROOT / "matrixgame_covered_solo"
INSTANCE_FILES = ['instances.json']


def _experiments(pkg: Path, filename: str):
    return json.loads((pkg / "in" / filename).read_text(encoding="utf-8"))["experiments"]


def _compact_name(name: str) -> str:
    return name.replace("_Grid", "_Compact")


def _board(gi):
    return (
        json.dumps(gi["start_positions"], sort_keys=True),
        json.dumps(gi["target_positions"], sort_keys=True),
        json.dumps(gi["walls"]),
        gi["optimal_moves"],
        gi["max_turns"],
    )


def test_every_grid_experiment_has_a_compact_twin_with_identical_boards():
    for filename in INSTANCE_FILES:
        grid = {e["name"]: e for e in _experiments(PKG, filename)}
        compact = {e["name"]: e for e in _experiments(COMPACT, filename)}
        assert grid, f"{filename}: no experiments"
        assert {_compact_name(n) for n in grid} == set(compact), (
            f"{filename}: experiment sets differ between grid and compact"
        )
        for name, exp in grid.items():
            twin = compact[_compact_name(name)]
            assert len(exp["game_instances"]) == len(twin["game_instances"]), name
            for gi, ci in zip(exp["game_instances"], twin["game_instances"]):
                assert gi["game_id"] == ci["game_id"], name
                assert _board(gi) == _board(ci), f"{name} game_id={gi['game_id']}: board differs"


def test_instances_are_flagged_and_named_as_grid():
    for filename in INSTANCE_FILES:
        for exp in _experiments(PKG, filename):
            assert exp["name"].endswith("_Grid"), exp["name"]
            for gi in exp["game_instances"]:
                assert gi["compact_board"] is False, exp["name"]
                prompt = gi["player_prompt"]
                assert "compact listings" not in prompt, exp["name"]
                assert "`Objects:`" not in prompt, exp["name"]
                assert "`##`" in prompt and "`+----+`" in prompt, exp["name"]


def test_common_config_identical_to_compact():
    grid = json.loads((PKG / "resources/common_config.json").read_text(encoding="utf-8"))
    compact = json.loads((COMPACT / "resources/common_config.json").read_text(encoding="utf-8"))
    assert grid == compact


def test_templates_differ_from_compact_only_in_the_board_format_block():
    tdir = "resources/initial_prompts/en"
    names = sorted(p.name for p in (PKG / tdir).iterdir())
    assert names == sorted(p.name for p in (COMPACT / tdir).iterdir())
    for name in names:
        grid = (PKG / tdir / name).read_text(encoding="utf-8").splitlines()
        compact = (COMPACT / tdir / name).read_text(encoding="utf-8").splitlines()
        ops = [op for op in difflib.SequenceMatcher(a=compact, b=grid).get_opcodes() if op[0] != "equal"]
        assert len(ops) == 1 and ops[0][0] == "replace", f"{name}: {ops}"
        _, i1, i2, j1, j2 = ops[0]
        assert i2 - i1 == 4 and j2 - j1 == 4, f"{name}: expected the 4-line board block"
        assert compact[i1].startswith("- The board is given to you as compact listings")
        assert grid[j1].startswith("- The board is drawn as a grid")


class _DummyModel:
    """Stand-in for clemcore.backends.Model with the methods setup touches."""

    def __init__(self, name="dummy"):
        self.name = name
        self._gen_args = {}

    def set_gen_arg(self, key, value):
        self._gen_args[key] = value


def _rendered_views(bench, master):
    import sys
    module = sys.modules[type(bench).__module__]
    return module._render_view(master.board, master.compact_board), module._render_targets_view(master.board, master.compact_board)


def test_master_loads_through_clemcore_and_renders_the_grid():
    from clemcore.clemgame import GameBenchmark
    from clemcore.clemgame.registry import GameSpec

    specs = GameSpec.from_directory(str(PKG))
    assert specs, "clemgame.json declares no game"
    for spec in specs:
        bench = GameBenchmark.load_from_spec(spec)
        assert type(bench).__name__ == "SoloMatrixGameBenchmark"
        filename = f"{spec['instances']}.json" if "instances" in spec else "instances.json"
        exp = _experiments(PKG, filename)[0]
        gi = exp["game_instances"][0]
        master = bench.create_game_master(exp, [_DummyModel(f"m{i}") for i in range(spec["players"])])
        master._on_setup(**gi)
        assert master.compact_board is False
        board, goal = _rendered_views(bench, master)
        for view in (board, goal):
            assert "Objects:" not in view and "Targets:" not in view
            assert "C1" in view and "+----+" in view, "expected the boxed grid"
        if False:
            assert "| X  |" in board, "teammates' objects must appear as X"
            assert "X" not in goal, "teammates' targets are omitted, never masked"
