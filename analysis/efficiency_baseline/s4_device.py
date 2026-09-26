"""Verify the S4 corridor-swap correction to ``optimal_moves``.

S4 boards are built around a forced swap inside a one-wide dead-end corridor
(``matrixgame_covered_shared/boardgen.py:_generate_s4``):

  * INNER starts *on* the dead end.
  * OUTER starts on the corridor cell adjacent to the dead end, and its target
    *is* the dead end.

The per-object BFS therefore credits OUTER with a distance of 1.  That is
impossible.  INNER occupies OUTER's target and can only leave through the
corridor, which OUTER blocks; inside a one-wide corridor INNER can never pass
OUTER, so OUTER has to stay ahead of it and leave the corridor completely.
INNER's only route out runs through the entrance cell as well, so OUTER must
clear that too.  With corridor length L the cheapest legal trajectory is

    L moves out   (to the first cell beyond the entrance)
  + L + 1 back    (to the dead end)
  = 2L + 1 moves, against the 1 the BFS counts.

Every S4 board uses L = 4, so the stored ``optimal_moves`` is understated by
exactly 8.  This script checks that on the generated boards and, if result
roots are given, checks the bound against every successful episode: no
play-through may finish an S4 board in fewer than ``optimal_moves + 8`` moves.
"""

import argparse
import glob
import json
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from joint_optimum import DIRS, load_instances

DEFAULT_INSTANCES = "matrixgame_covered_comm_grid/in/instances.json"


def bfs_distances(size: int, walls: Set[Tuple[int, int]], src: Tuple[int, int]):
    """Distance from ``src`` to every reachable free cell, ignoring objects."""
    d = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if (
                0 <= nxt[0] < size
                and 0 <= nxt[1] < size
                and nxt not in walls
                and nxt not in d
            ):
                d[nxt] = d[(r, c)] + 1
                q.append(nxt)
    return d


def s4_correction(gi: dict) -> dict:
    """Forced-detour correction for one S4 board.

    OUTER cannot use its BFS path; it must reach some cell outside the corridor
    and its entrance, then come back.  The cheapest such trajectory is
    ``min over x outside of  d(start, x) + d(x, target)``, which is the number
    reported as ``outer_true``.
    """
    size = gi["grid_size"]
    walls = {tuple(w) for w in gi["walls"]}
    device = gi["device"]
    corridor = {tuple(c) for c in device["corridor"]}
    entrance = tuple(device["entrance"])
    dead_end = tuple(device["dead_end"])
    inner_adjacent = tuple(device["inner_adjacent"])

    starts = {o: tuple(p) for o, p in gi["start_positions"].items()}
    targets = {o: tuple(p) for o, p in gi["target_positions"].items()}
    outer = next(o for o in starts if starts[o] == inner_adjacent and targets[o] == dead_end)
    inner = next(o for o in starts if starts[o] == dead_end)

    per_object = {
        o: bfs_distances(size, walls, starts[o])[targets[o]] for o in starts
    }
    blocked = corridor | {entrance}
    from_start = bfs_distances(size, walls, starts[outer])
    from_target = bfs_distances(size, walls, targets[outer])
    outer_true = min(
        from_start[x] + from_target[x]
        for x in from_start
        if x not in blocked and x in from_target
    )
    return dict(
        game_id=gi["game_id"],
        corridor_length=len(corridor),
        outer=outer,
        inner=inner,
        outer_bfs=per_object[outer],
        outer_true=outer_true,
        bfs_sum=sum(per_object.values()),
        stored=gi["optimal_moves"],
        corrected=sum(per_object.values()) - per_object[outer] + outer_true,
    )


def render(gi: dict) -> str:
    """ASCII view of a board with objects on their start cells."""
    size = gi["grid_size"]
    walls = {tuple(w) for w in gi["walls"]}
    at_start = {tuple(p): o for o, p in gi["start_positions"].items()}
    lines = ["     " + "".join(f"C{c + 1:<3}" for c in range(size))]
    for r in range(size):
        row = []
        for c in range(size):
            if (r, c) in walls:
                row.append("##  ")
            elif (r, c) in at_start:
                row.append(f"{at_start[(r, c)]}   ")
            else:
                row.append(".   ")
        lines.append(f"R{r + 1:<3} " + "".join(row))
    return "\n".join(lines)


def successful_move_counts(roots: List[str]) -> Dict[Tuple[str, int], List[int]]:
    """Move counts of every successful episode, keyed by (level, game_id)."""
    out: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    for root in roots:
        for path in glob.glob(f"{root}/*/*/*/instance_*/scores.json"):
            match = re.match(r"(?:Solo_)?(S\d)_", path.split("/")[-3])
            if not match:
                continue
            gid = int(path.split("/")[-2].split("_")[-1])
            scores = json.load(open(path))["episode scores"]
            if scores.get("Success") == 1:
                out[(match.group(1), gid)].append(scores["Move Count"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="result roots to validate the bound against, e.g. results_*_grid_n15_*",
    )
    ap.add_argument("--render", type=int, default=None, help="also print this S4 board")
    args = ap.parse_args()

    instances = load_instances(args.instances)
    rows = [s4_correction(gi) for (lvl, _), gi in sorted(instances.items()) if lvl == "S4"]

    print(f"{'board':>5} {'L':>2} {'outer':>5} {'BFS':>4} {'true':>5} {'stored':>7} {'corrected':>10} {'delta':>6}")
    for r in rows:
        print(
            f"{r['game_id']:>5} {r['corridor_length']:>2} {r['outer']:>5} {r['outer_bfs']:>4} "
            f"{r['outer_true']:>5} {r['stored']:>7} {r['corrected']:>10} "
            f"{r['corrected'] - r['stored']:>6}"
        )
    deltas = sorted({r["corrected"] - r["stored"] for r in rows})
    print(f"\ndistinct corrections across {len(rows)} S4 boards: {deltas}")
    assert all(r["bfs_sum"] == r["stored"] for r in rows), \
        "stored optimal_moves is not the per-object BFS sum"

    if args.render is not None:
        gi = instances[("S4", args.render)]
        print(f"\nS4 board #{args.render}  device={gi['device']}")
        print(render(gi))

    if args.results:
        counts = successful_move_counts(args.results)
        correction = {r["game_id"]: r["corrected"] - r["stored"] for r in rows}
        violations, tight, total = [], [], 0
        for r in rows:
            bound = r["stored"] + correction[r["game_id"]]
            for mc in counts.get(("S4", r["game_id"]), []):
                total += 1
                if mc < bound:
                    violations.append((r["game_id"], mc, bound))
                elif mc == bound:
                    tight.append(r["game_id"])
        print(f"\nvalidation over {total} successful S4 episodes")
        print(f"  episodes finishing below stored+correction : {len(violations)}")
        print(f"  boards where the best play meets it exactly: {sorted(set(tight))}")
        for v in violations[:10]:
            print("  VIOLATION", v)


if __name__ == "__main__":
    main()
