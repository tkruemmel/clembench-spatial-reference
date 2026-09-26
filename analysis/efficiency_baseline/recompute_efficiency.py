"""Recompute move efficiency against a sound baseline.

``Move Efficiency`` and the clembench ``Main Score`` both divide the stored
``optimal_moves`` by the episode's move count.  That numerator is the sum of
per-object BFS distances, which ignores every interaction between objects and
is provably too small at S4 (see ``s4_device.py``).  Levels are affected
unequally, so the published level ordering is not safe to read.

This script re-derives the table against the exact joint optimum from
``joint_optimum.jsonl``.  All 60 released grid boards are solved exactly, so
every cell is a single number.

For any board whose optimum is not yet known the script falls back to a
bracket, so a partial run still yields sound conclusions:

  lower bound   max(stored optimal_moves,
                    the S4 corridor-swap correction where it applies,
                    the A* frontier recorded for that board)

  upper bound   the shortest play-through any episode actually achieved

Any real play-through is a valid solution, so its length bounds the optimum
from above; the A* frontier bounds it from below.  Where two levels' intervals
are disjoint, their ordering is settled rather than estimated.
"""

import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_optimum import load_instances  # noqa: E402
from s4_device import s4_correction  # noqa: E402

LEVELS = ("S1", "S2", "S3", "S4")

# S2 #10 is unplayable in the 6-player partition: object B has no legal move at
# move 0, cannot declare `done`, and the only player acting before it cannot
# free it, so all eight 6p episodes abort after one accepted move.  Section 7.2
# of results_three_models_grid.md proves it and shows it is the only such board
# of the sixty.  That document excludes it from every figure, in all four
# suites, because the boards are byte-identical across suites.  Matched here so
# the corrected efficiency table is comparable with the published one.
EXCLUDED = {("S2", 10)}
DEFAULT_JOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_optimum.jsonl")

# (model label, suite) -> results root.  Edit to point at other campaigns.
DEFAULT_ROOTS = {
    ("Qwen3.8", "solo"): "results_qwen38_grid_n15_solo",
    ("Qwen3.8", "dyad"): "results_qwen38_grid_n15_dyad",
    ("Qwen3.8", "3p"): "results_qwen38_grid_n15_3p",
    ("Qwen3.8", "6p"): "results_qwen38_grid_n15_6p",
    ("DeepSeek-V4.1", "solo"): "results_dsv41_grid_n15_solo",
    ("DeepSeek-V4.1", "dyad"): "results_dsv41_grid_n15_dyad",
    ("DeepSeek-V4.1", "3p"): "results_dsv41_grid_n15_3p",
    ("DeepSeek-V4.1", "6p"): "results_dsv41_grid_n15_6p",
}


def read_episodes(roots: Dict[Tuple[str, str], str]):
    """Successful episodes as (model, suite, level, game_id, move_count)."""
    episodes = []
    for (model, suite), root in roots.items():
        for path in glob.glob(f"{root}/*/*/*/instance_*/scores.json"):
            match = re.match(r"(?:Solo_)?(S\d)_", path.split("/")[-3])
            if not match:
                continue
            gid = int(path.split("/")[-2].split("_")[-1])
            scores = json.load(open(path))["episode scores"]
            if scores.get("Success") == 1:
                episodes.append((model, suite, match.group(1), gid, scores["Move Count"]))
    return episodes


def read_joint(path: str) -> Dict[Tuple[str, int], dict]:
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["level"], r["game_id"])] = r
    return out


def build_bounds(instances, joint, episodes):
    """Per board: (lower bound, upper bound, 'exact' | 'bracket')."""
    stored = {k: gi["optimal_moves"] for k, gi in instances.items()}
    s4_delta = {
        ("S4", r["game_id"]): r["corrected"] - r["stored"]
        for (lvl, _), gi in instances.items()
        if lvl == "S4"
        for r in [s4_correction(gi)]
    }
    best: Dict[Tuple[str, int], int] = {}
    for _, _, lvl, gid, mc in episodes:
        k = (lvl, gid)
        best[k] = min(best.get(k, 10 ** 9), mc)

    bounds = {}
    for k in instances:
        j = joint.get(k)
        if j and j["status"] == "exact":
            bounds[k] = (j["opt"], j["opt"], "exact")
            continue
        lo = stored[k]
        if k in s4_delta:
            lo = max(lo, stored[k] + s4_delta[k])
        if j:
            lo = max(lo, j["lb"])
        hi = best.get(k)
        bounds[k] = (lo, max(lo, hi) if hi else None, "bracket")
    return bounds, stored


def _cell(values: List[float]) -> str:
    return f"{st.mean(values):.2f}" if values else "n/a"


def _interval(lo: List[float], hi: List[float]) -> str:
    """One number once the joint optimum is exact, an interval while it is not."""
    a, b = _cell(lo), _cell(hi)
    return a if a == b else f"{a}-{b}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", default="matrixgame_covered_comm_grid/in/instances.json")
    ap.add_argument("--joint", default=DEFAULT_JOINT)
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    ap.add_argument("--keep-excluded", action="store_true",
                    help=f"keep boards normally excluded from the figures: {sorted(EXCLUDED)}")
    args = ap.parse_args()
    excluded = set() if args.keep_excluded else EXCLUDED

    instances = load_instances(args.instances)
    joint = read_joint(args.joint)
    episodes = read_episodes(DEFAULT_ROOTS)
    bounds, stored = build_bounds(instances, joint, episodes)

    n_exact = sum(1 for k in instances if bounds[k][2] == "exact")
    print(f"joint optimum known exactly for {n_exact}/{len(instances)} boards")
    if excluded:
        print(f"excluded from the aggregates: {sorted(excluded)}")
    print()

    print("Per-level baseline, mean over the boards of that level")
    head = f"{'level':>6} {'stored':>8} {'lower':>8} {'upper':>8} {'exact':>7}"
    print(head)
    for lvl in LEVELS:
        ks = [k for k in instances if k[0] == lvl and k not in excluded]
        print(
            f"{lvl:>6} {st.mean(stored[k] for k in ks):>8.1f} "
            f"{st.mean(bounds[k][0] for k in ks):>8.1f} "
            f"{st.mean(bounds[k][1] for k in ks if bounds[k][1]):>8.1f} "
            f"{sum(1 for k in ks if bounds[k][2] == 'exact'):>5}/{len(ks)}"
        )

    print("\nExact gap (joint optimum minus stored optimal_moves), boards solved exactly")
    for lvl in LEVELS:
        gaps = sorted(
            joint[k]["gap"] for k in instances
            if k[0] == lvl and k not in excluded
            and k in joint and joint[k]["status"] == "exact"
        )
        print(f"  {lvl}: {gaps}")

    published = defaultdict(list)
    lower = defaultdict(list)
    upper = defaultdict(list)
    for model, suite, lvl, gid, mc in episodes:
        if (lvl, gid) in excluded:
            continue
        key = (model, suite, lvl)
        published[key].append(stored[(lvl, gid)] / mc)
        lo, hi, _ = bounds[(lvl, gid)]
        lower[key].append(lo / mc)
        if hi:
            upper[key].append(hi / mc)

    order = sorted({(m, s) for m, s, _, _, _ in episodes},
                   key=lambda x: (x[0], ["solo", "dyad", "3p", "6p"].index(x[1])))

    if args.markdown:
        gids = sorted({k[1] for k in instances})
        print("\nPer board, `stored optimal_moves` / joint optimum (gap):\n")
        print("| board | " + " | ".join(LEVELS) + " |")
        print("|---|" + "---|" * len(LEVELS))
        for gid in gids:
            cells = []
            for lvl in LEVELS:
                j = joint.get((lvl, gid))
                s = stored[(lvl, gid)]
                if j and j["status"] == "exact":
                    cells.append(f"{s} / {j['opt']} ({j['gap']:+d})")
                elif j:
                    cells.append(f"{s} / >= {j['lb']}")
                else:
                    cells.append(f"{s} / not solved")
                if (lvl, gid) in EXCLUDED:
                    cells[-1] += " *"
            print(f"| {gid} | " + " | ".join(cells) + " |")
        if EXCLUDED:
            print("\n`*` excluded from the aggregates above; "
                  "the optimum is still a property of the board.")

        print("\n| model | suite | " + " | ".join(LEVELS) + " |")
        print("|---|---|" + "---|" * len(LEVELS))
        for model, suite in order:
            cells = [
                _interval(lower[(model, suite, l)], upper[(model, suite, l)])
                for l in LEVELS
            ]
            print(f"| {model} | {suite} | " + " | ".join(cells) + " |")
    else:
        print("\nMove efficiency, [proven lower bound, achievable upper bound]")
        print(f"{'model':<15}{'suite':<6}" + "".join(f"{l:>15}" for l in LEVELS))
        for model, suite in order:
            cells = [
                _interval(lower[(model, suite, l)], upper[(model, suite, l)]).rjust(15)
                for l in LEVELS
            ]
            print(f"{model:<15}{suite:<6}" + "".join(cells))
        print("\nAs published, stored optimal_moves / move count")
        print(f"{'model':<15}{'suite':<6}" + "".join(f"{l:>15}" for l in LEVELS))
        for model, suite in order:
            cells = [_cell(published[(model, suite, l)]).rjust(15) for l in LEVELS]
            print(f"{model:<15}{suite:<6}" + "".join(cells))


if __name__ == "__main__":
    main()
