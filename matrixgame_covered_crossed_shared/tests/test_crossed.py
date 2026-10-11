"""Invariants for the crossed-device packages (S3x/S4x, grid rendering).

Each crossed package must be its reference arm with only the boards swapped:
same instance fields and prompts (dyad/3p/6p: the updated-prompt arm, with
the A_k budget; solo: the grid twin, the only solo prompt), the S3/S4 structure (longest chain 3,
4 edges, 2 chains, one target in the dead end), the device on the other
chain, the corridor of the paired released board, and ownership that keeps
chain neighbours and the device pair on different agents at 2, 3 and 6 players.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from matrixgame_covered_crossed_shared.boards import generate_crossed_board  # noqa: E402

# crossed package -> (grid twin, instance files)
PACKAGES = {
    "matrixgame_covered_solo_crossed_grid": ("matrixgame_covered_solo_grid", ["instances.json"]),
    "matrixgame_covered_comm_crossed_grid": ("matrixgame_covered_comm_grid", ["instances.json"]),
    "matrixgame_covered_comm_nplayer_crossed_grid": (
        "matrixgame_covered_comm_nplayer_grid", ["instances_3p.json", "instances_6p.json"]
    ),
}
# crossed package -> (prompt arm it follows, template source, A_k jsonl per instance file)
PROMPT_ARM = {
    "matrixgame_covered_solo_crossed_grid": (
        "matrixgame_covered_solo_grid", "matrixgame_covered_solo_grid/resources/initial_prompts/en", None
    ),
    "matrixgame_covered_comm_crossed_grid": (
        "matrixgame_covered_comm_grid_updated_prompt", "prompt_updated/dyad",
        {"instances.json": "attainable_dyad_crossed.jsonl"},
    ),
    "matrixgame_covered_comm_nplayer_crossed_grid": (
        "matrixgame_covered_comm_nplayer_grid_updated_prompt", "prompt_updated/nplayer",
        {"instances_3p.json": "attainable_3p_crossed.jsonl",
         "instances_6p.json": "attainable_6p_crossed.jsonl"},
    ),
}
DEEP, SHALLOW = set("ABCD"), set("EF")
DEVICE_CHAIN = {"S3x": SHALLOW, "S4x": DEEP}


def _instances(pkg, filename):
    data = json.loads((ROOT / pkg / "in" / filename).read_text(encoding="utf-8"))
    return [(e["name"], gi) for e in data["experiments"] for gi in e["game_instances"]]


def _crossed():
    for pkg, (_, files) in PACKAGES.items():
        for f in files:
            yield from _instances(pkg, f)


def _edges(gi):
    s, t = gi["start_positions"], gi["target_positions"]
    return {(x, y) for x in s for y in s if x != y and s[y] == t[x]}


def _longest(edges):
    succ = {}
    for x, y in edges:
        succ.setdefault(x, []).append(y)

    def depth(x):
        return max((1 + depth(y) for y in succ.get(x, [])), default=0)

    return max(depth(x) for x, _ in edges)


def _owners(gi):
    if "player_objects" in gi:
        groups = gi["player_objects"]
    elif "player_a_objects" in gi:
        groups = [gi["player_a_objects"], gi["player_b_objects"]]
    else:
        return None
    return {o: i for i, g in enumerate(groups) for o in g}


def _device_objects(gi):
    d = gi["device"]
    s, t = gi["start_positions"], gi["target_positions"]
    return {o for o in s if s[o] in (d["entrance"], d["dead_end"]) or t[o] == d["dead_end"]}


def test_every_package_has_both_levels_at_n15():
    for pkg, (_, files) in PACKAGES.items():
        for f in files:
            levels = {}
            for _, gi in _instances(pkg, f):
                levels.setdefault(gi["spatial_level"], set()).add(gi["game_id"])
            assert levels == {"S3x": set(range(1, 16)), "S4x": set(range(1, 16))}, (pkg, f)


def test_chain_structure_matches_s3_s4():
    for name, gi in _crossed():
        edges = _edges(gi)
        assert len(edges) == 4, name
        assert _longest(edges) == 3, name
        objs_in_edges = {o for e in edges for o in e}
        assert objs_in_edges == DEEP | SHALLOW, name
        # exactly one target in the dead end, and it is a dead end
        t = gi["target_positions"]
        assert sum(t[o] == gi["device"]["dead_end"] for o in t) == 1, name


def test_device_sits_on_the_other_chain():
    for name, gi in _crossed():
        dev = _device_objects(gi)
        assert len(dev) == 2 and dev <= DEVICE_CHAIN[gi["spatial_level"]], (name, dev)


def test_ownership_keeps_chain_neighbours_and_device_pair_apart():
    for name, gi in _crossed():
        owner = _owners(gi)
        if owner is None:
            continue
        for x, y in _edges(gi):
            assert owner[x] != owner[y], (name, x, y)
        a, b = sorted(_device_objects(gi))
        assert owner[a] != owner[b], name


def test_corridor_equals_paired_released_board():
    released = {
        (gi["spatial_level"], gi["game_id"]): gi["device"]["corridor"]
        for _, gi in _instances("matrixgame_covered_solo_grid", "instances.json")
        if gi["spatial_level"] in ("S3", "S4")
    }
    for name, gi in _crossed():
        paired = {"S3x": "S3", "S4x": "S4"}[gi["spatial_level"]]
        assert gi["device"]["corridor"] == released[(paired, gi["game_id"])], name


def test_instances_match_a_fresh_generation():
    for name, gi in _crossed():
        b = generate_crossed_board(gi["spatial_level"], gi["grid_size"], gi["objects"], 42, gi["game_id"] - 1)
        assert b["start_positions"] == gi["start_positions"], name
        assert b["target_positions"] == gi["target_positions"], name
        assert b["walls"] == gi["walls"], name


def test_budget_follows_the_prompt_arm():
    """Solo keeps the per-object BFS sum; dyad/3p/6p use A_k, as the updated arm."""
    for pkg, (_, files) in PACKAGES.items():
        budgets = PROMPT_ARM[pkg][2]
        for f in files:
            if budgets:
                rows = [json.loads(l) for l in open(ROOT / "analysis/efficiency_baseline" / budgets[f])]
                assert all(r["status"] == "exact" for r in rows), f
                ak = {(r["level"], r["game_id"]): r["stuckfree"] for r in rows}
            for name, gi in _instances(pkg, f):
                if budgets:
                    want = ak[(gi["spatial_level"], gi["game_id"])]
                else:
                    want = generate_crossed_board(
                        gi["spatial_level"], gi["grid_size"], gi["objects"], 42, gi["game_id"] - 1
                    )["optimal_moves"]
                assert gi["optimal_moves"] == want, (pkg, name, gi["game_id"])
                assert gi["max_turns"] == max(4 * want, 20), (pkg, name, gi["game_id"])


def test_same_fields_prompts_and_config_as_prompt_arm():
    for pkg, (twin, files) in PACKAGES.items():
        arm, tsrc, _ = PROMPT_ARM[pkg]
        for f in files:
            ref = [gi for _, gi in _instances(arm, f) if gi["spatial_level"] == "S3"]
            ref_keys = set(ref[0])
            ref_prompts = {gi["player_prompt"] for gi in ref}
            for name, gi in _instances(pkg, f):
                assert set(gi) == ref_keys, name
                assert gi["compact_board"] is False, name
                assert gi["player_prompt"] in ref_prompts, name
        tdir = "resources/initial_prompts/en"
        names = sorted(p.name for p in (ROOT / pkg / tdir).iterdir())
        assert names == sorted(p.name for p in (ROOT / tsrc).iterdir()), pkg
        for n in names:
            assert (ROOT / pkg / tdir / n).read_bytes() == (ROOT / tsrc / n).read_bytes(), (pkg, n)
        assert (ROOT / pkg / "resources/common_config.json").read_bytes() == (
            ROOT / twin / "resources/common_config.json"
        ).read_bytes(), pkg


class _DummyModel:
    """Stand-in for clemcore.backends.Model with the methods setup touches."""

    def __init__(self, name="dummy"):
        self.name = name
        self._gen_args = {}

    def set_gen_arg(self, key, value):
        self._gen_args[key] = value


def test_every_instance_sets_up_through_the_shared_master():
    from clemcore.clemgame import GameBenchmark
    from clemcore.clemgame.registry import GameSpec

    for pkg in PACKAGES:
        specs = GameSpec.from_directory(str(ROOT / pkg))
        assert specs, pkg
        for spec in specs:
            bench = GameBenchmark.load_from_spec(spec)
            filename = f"{spec['instances']}.json" if "instances" in spec else "instances.json"
            data = json.loads((ROOT / pkg / "in" / filename).read_text(encoding="utf-8"))
            for exp in data["experiments"]:
                for gi in exp["game_instances"]:
                    models = [_DummyModel(f"m{i}") for i in range(spec["players"])]
                    master = bench.create_game_master(exp, models)
                    master._on_setup(**gi)
                    assert master.compact_board is False, (spec["game_name"], exp["name"])
