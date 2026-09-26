#!/usr/bin/env python3
"""Per-board property audit of the 60 released instances.

Measures the *structural devices* each spatial level claims to install, rather
than bulk quantities (wall count, detour ratio) that do not track them.  Every
column is computed from the instance files alone -- no model, no episode.

Columns, and why each one is here
---------------------------------
Device severity (the ladder's actual claim, quantified):

  interaction gap   opt - h0, where h0 is the per-object BFS sum (each object
                    flies to its target ignoring all others) and opt is the
                    exact joint optimum.  The difference is the number of moves
                    the board forces purely because objects must get out of
                    each other's way.  Joined from joint_optimum.jsonl.
  chain length      Longest path in the graph  X -> Y  iff  Y.start == X.target
                    ("Y must vacate before X can land").  This is exactly the
                    construction boardgen uses: S2 builds A<-B<-C (2 edges),
                    S3 and S4 build a length-3 chain (3 edges).
  swap              Whether a forced non-monotonic swap is present: an object
                    pair (X, Y) with X.target == Y.start, where Y sits in a
                    cell with exactly one free neighbour and that neighbour is
                    X.start.  Y cannot leave until X retreats; X cannot land
                    until Y leaves.  This is S4's declared device.

Irreversibility and pressure:

  dead-end cells    Free cells with exactly one free neighbour.  A trap: an
                    object driven in can only be recovered by reversing.
  targets in DE     Targets that sit in such a cell -- the trap is on the
                    critical path, not decorative.
  mobility          Legal moves available per object at the start state, mean
                    and minimum over the six objects.  A move is legal if the
                    destination is on the grid, not a wall and not occupied
                    (utils/board.py validate_move).
  forced@6p         Objects with exactly one legal move at the start.  At 6p
                    each player owns one object, so such a player has no
                    choice on turn one.  zero@6p is the unplayable case.

Descriptive only (no difficulty claim attached):

  walls, density, free cells, grid size.

Reported for reference, not as a difficulty measure:

  detour ratio      Per object, (BFS path length with walls) / (Manhattan
                    distance).  boardgen's _carve_for_ratio carves PER OBJECT
                    and accepts at ``target - 0.15``, with documented targets
                    of 1.5 at S2 and 2.0 at S3/S4 -- so the object is the unit
                    and the pass rate against that threshold, not an average,
                    says whether carving succeeded.  The distribution is
                    bounded below at 1.0 with 31-51% of objects on that floor;
                    it is summarised by its median.  Path *length* is the S2
                    device only; it is not what distinguishes S3 from S4.

Usage
-----
    python3 analysis/efficiency_baseline/instance_audit.py                  # summary
    python3 analysis/efficiency_baseline/instance_audit.py --per-board      # all 60
    python3 analysis/efficiency_baseline/instance_audit.py --csv audit.csv
"""

import argparse
import collections
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

LEVELS = ["S1", "S2", "S3", "S4"]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# S2 #10 is unplayable under the six-player partition and is excluded from every
# figure in the thesis, so the audit reports on the same 59 boards the models
# played.  The one exception is the screening step (--all-boards), which has to
# see all 60: the audit is what identifies the board in the first place.
EXCLUDED = ("S2", 10)

# The physical puzzle is generated once in matrixgame_covered_shared.boardgen
# and shared across packages; --verify-packages checks that on disk.
SOLO = "matrixgame_covered_solo/in/instances.json"
PACKAGES = [
    SOLO,
    "matrixgame_covered_solo_grid/in/instances.json",
    "matrixgame_covered_comm/in/instances.json",
    "matrixgame_covered_comm_nplayer/in/instances.json",
]


# ── loading ────────────────────────────────────────────────────────────

def board_key(inst: Dict) -> Tuple:
    """Identity of the physical puzzle, independent of protocol or rendering."""
    return (
        inst["grid_size"],
        tuple(sorted(tuple(w) for w in inst["walls"])),
        tuple(sorted((k, tuple(v)) for k, v in inst["start_positions"].items())),
        tuple(sorted((k, tuple(v)) for k, v in inst["target_positions"].items())),
    )


def load_boards(path: str) -> Dict[Tuple[str, int], Dict]:
    """One entry per (level, game_id); duplicates across variants are checked."""
    with open(os.path.join(ROOT, path)) as fh:
        data = json.load(fh)
    boards: Dict[Tuple[str, int], Dict] = {}
    for exp in data["experiments"]:
        for inst in exp["game_instances"]:
            level = inst["spatial_level"]
            key = (level, inst["game_id"])
            if key in boards:
                if board_key(boards[key]) != board_key(inst):
                    raise SystemExit(
                        f"{path}: {key} differs between variants -- the physical "
                        "puzzle is supposed to be identical across them"
                    )
                continue
            boards[key] = inst
    return boards


# ── geometry ───────────────────────────────────────────────────────────

def free_cells(n: int, walls: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(n) for c in range(n) if (r, c) not in walls]


def free_neighbours(cell, n, walls) -> List[Tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in walls:
            out.append((nr, nc))
    return out


def bfs_dist(start, goal, n, walls) -> Optional[int]:
    """Shortest path over free cells, ignoring other objects (boardgen's h0)."""
    if start == goal:
        return 0
    seen = {start}
    frontier = [start]
    d = 0
    while frontier:
        d += 1
        nxt = []
        for cell in frontier:
            for nb in free_neighbours(cell, n, walls):
                if nb in seen:
                    continue
                if nb == goal:
                    return d
                seen.add(nb)
                nxt.append(nb)
        frontier = nxt
    return None


def legal_moves(obj, starts, n, walls) -> int:
    """Moves available at the start state (walls, edges and other objects)."""
    occupied = {tuple(p) for o, p in starts.items() if o != obj}
    r, c = starts[obj]
    count = 0
    for dr, dc in DIRS:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < n):
            continue
        if (nr, nc) in walls or (nr, nc) in occupied:
            continue
        count += 1
    return count


# ── device measures ────────────────────────────────────────────────────

def chain_length(starts: Dict, targets: Dict) -> int:
    """Longest path in  X -> Y  iff  Y.start == X.target  (edges, not nodes).

    boardgen builds exactly this relation: starts[D] = targets[C] means D must
    vacate before C can land.  The graph is a forest of paths by construction,
    so a memoised depth suffices; a cycle would be a generator bug and raises.
    """
    at_start = {tuple(p): o for o, p in starts.items()}
    succ = {}
    for obj, tgt in targets.items():
        blocker = at_start.get(tuple(tgt))
        if blocker is not None and blocker != obj:
            succ[obj] = blocker

    depth: Dict[str, int] = {}

    def walk(o: str, seen: Set[str]) -> int:
        if o in depth:
            return depth[o]
        if o in seen:
            raise SystemExit(f"cyclic dependency at {o}: generator invariant broken")
        nxt = succ.get(o)
        d = 0 if nxt is None else 1 + walk(nxt, seen | {o})
        depth[o] = d
        return d

    return max((walk(o, set()) for o in starts), default=0)


def dependency_edges(starts: Dict, targets: Dict) -> Tuple[int, int]:
    """Total dependency edges, and the number of separate chains.

    The longest path alone hides real structure: S3 installs a primary chain of
    3 edges *plus* a secondary 1-edge chain (E<-F), and S4 installs a primary
    chain of 3 edges plus the swap edge.  Both therefore carry 4 edges in 2
    chains while reporting a longest path of 3.
    """
    at_start = {tuple(p): o for o, p in starts.items()}
    succ = {}
    for obj, tgt in targets.items():
        blocker = at_start.get(tuple(tgt))
        if blocker is not None and blocker != obj:
            succ[obj] = blocker
    heads = [o for o in starts if o in succ and o not in succ.values()]
    return len(succ), len(heads)


def swap_pairs(starts: Dict, targets: Dict, n: int, walls: Set) -> int:
    """Forced non-monotonic swaps: X wants Y's cell, and Y's only way out is X's.

    S4's declared device.  Y sits in a cell with one free neighbour (a dead
    end); that neighbour holds X; and X's target is Y's cell.  Neither can move
    first without the other retreating, and X must move *away* from its target
    to resolve it.
    """
    pos = {o: tuple(p) for o, p in starts.items()}
    tgt = {o: tuple(p) for o, p in targets.items()}
    count = 0
    for x in starts:
        for y in starts:
            if x == y:
                continue
            if tgt[x] != pos[y]:
                continue
            nbs = free_neighbours(pos[y], n, walls)
            if len(nbs) == 1 and nbs[0] == pos[x]:
                count += 1
    return count


def dead_end_stats(starts, targets, n, walls) -> Tuple[int, int]:
    """Free cells with exactly one free neighbour, and targets sitting in one."""
    cells = [c for c in free_cells(n, walls) if len(free_neighbours(c, n, walls)) == 1]
    de = set(cells)
    in_de = sum(1 for t in targets.values() if tuple(t) in de)
    return len(cells), in_de


def detour_ratios(starts, targets, n, walls) -> Optional[List[float]]:
    """Per-object BFS path / Manhattan distance.

    The generator carves PER OBJECT and accepts an object at
    ``ratio >= target_ratio - 0.15`` (_carve_for_ratio), so the object is the
    unit and a threshold pass-rate is the criterion.  The distribution is
    bounded below at 1.0 with 31-51% of objects sitting exactly on that floor,
    so it is summarised by its median; a mean over it is misleading.
    """
    ratios = []
    for obj, s in starts.items():
        s = tuple(s)
        t = tuple(targets[obj])
        man = abs(s[0] - t[0]) + abs(s[1] - t[1])
        if man == 0:
            continue
        d = bfs_dist(s, t, n, walls)
        if d is None:
            return None
        ratios.append(d / man)
    return ratios or None


# ── audit ──────────────────────────────────────────────────────────────

def load_solver_outputs():
    jo, sf = {}, {}
    p = os.path.join(HERE, "joint_optimum.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            jo[(d["level"], d["game_id"])] = d
    p = os.path.join(HERE, "stuckfree.jsonl")
    if os.path.exists(p):
        for line in open(p):
            d = json.loads(line)
            sf[(d["level"], d["game_id"])] = d
    return jo, sf


def audit_board(level, gid, inst, jo, sf) -> Dict:
    n = inst["grid_size"]
    walls = {tuple(w) for w in inst["walls"]}
    starts = inst["start_positions"]
    targets = inst["target_positions"]

    mob = [legal_moves(o, starts, n, walls) for o in starts]
    de_cells, de_targets = dead_end_stats(starts, targets, n, walls)

    j = jo.get((level, gid), {})
    h0, opt = j.get("h0"), j.get("opt")
    gap = None if (h0 is None or opt is None) else opt - h0
    s = sf.get((level, gid), {})

    row = {
        "level": level,
        "game_id": gid,
        # device severity
        "h0": h0,
        "opt": opt,
        "gap": gap,
        "gap_pct": None if not h0 or gap is None else 100.0 * gap / h0,
        "chain": chain_length(starts, targets),
        "edges": dependency_edges(starts, targets)[0],
        "chains": dependency_edges(starts, targets)[1],
        "swap": swap_pairs(starts, targets, n, walls),
        # irreversibility / pressure
        "dead_end_cells": de_cells,
        "targets_in_dead_end": de_targets,
        "mob_mean": st.mean(mob),
        "mob_min": min(mob),
        "forced_6p": sum(1 for m in mob if m == 1),
        "zero_6p": sum(1 for m in mob if m == 0),
        "overhead_6p": s.get("overhead_vs_joint"),
        # descriptive
        "walls": len(walls),
        "density": len(walls) / (n * n),
        "free_cells": n * n - len(walls),
        # reference only
        "detour_objs": detour_ratios(starts, targets, n, walls),
        "device_declared": bool(inst.get("device")),
    }
    ro = row.pop("detour_objs")
    row["detour_objs"] = ro
    row["detour"] = st.median(ro) if ro else None
    return row


def verify_packages() -> None:
    ref = load_boards(SOLO)
    for path in PACKAGES[1:]:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"  {path:52s} absent, skipped")
            continue
        other = load_boards(path)
        shared = sorted(set(ref) & set(other))
        bad = [k for k in shared if board_key(ref[k]) != board_key(other[k])]
        status = "identical" if not bad else f"{len(bad)} DIFFER"
        print(f"  {path:52s} {len(shared):2d} boards, {status}")


def fmt(v, spec="6.2f"):
    return "    --" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=SOLO)
    ap.add_argument("--per-board", action="store_true", help="all 60 rows")
    ap.add_argument("--csv", metavar="PATH", help="write the per-board table")
    ap.add_argument("--verify-packages", action="store_true",
                    help="check the boards are identical across packages")
    ap.add_argument("--all-boards", action="store_true",
                    help=f"include {EXCLUDED[0]} #{EXCLUDED[1]}, which is excluded by "
                         "default as unplayable at 6p; use this for the screening "
                         "step that identifies it")
    ap.add_argument("--screen", action="store_true",
                    help="list every board carrying an object with no legal move "
                         "at the start, over all 60 boards")
    args = ap.parse_args()

    if args.verify_packages:
        print("Physical puzzle identity across packages")
        verify_packages()
        print()

    boards = load_boards(args.instances)
    jo, sf = load_solver_outputs()

    if args.screen:
        print("SCREENING over all 60 boards: objects with no legal move at the start")
        print("(zero start mobility is necessary but not sufficient for an")
        print(" unplayable partition -- the stuck-free search decides)")
        print("level  id  zero-mobility objects  stuck-free solution")
        for lv, gid in sorted(boards, key=lambda k: (k[0], k[1])):
            r = audit_board(lv, gid, boards[(lv, gid)], jo, sf)
            if not r["zero_6p"]:
                continue
            st_ = sf.get((lv, gid), {}).get("status", "?")
            print(f"{lv:>4}  {gid:3d}         {r['zero_6p']}              "
                  f"{'NONE EXISTS' if st_ == 'none exists' else st_}")
        print()

    if not args.all_boards:
        boards.pop(EXCLUDED, None)
    rows = [audit_board(lv, gid, boards[(lv, gid)], jo, sf)
            for lv, gid in sorted(boards, key=lambda k: (k[0], k[1]))]
    scope = ("all 60 boards as generated" if args.all_boards else
             f"the {len(rows)}-board analysis set "
             f"({EXCLUDED[0]} #{EXCLUDED[1]} excluded, unplayable at 6p)")
    print(f"Scope: {scope}\n")

    if args.csv:
        flat = [{k: v for k, v in r.items() if k != "detour_objs"} for r in rows]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)
        print(f"wrote {args.csv} ({len(rows)} boards)\n")

    if args.per_board:
        print("level  id   h0  opt  gap  gap%  chain swap  DE  T@DE  mob  min  f@6p  detour")
        for r in rows:
            print(f"{r['level']:>4}  {r['game_id']:3d} {fmt(r['h0'],'4.0f')} "
                  f"{fmt(r['opt'],'4.0f')} {fmt(r['gap'],'4.0f')} "
                  f"{fmt(r['gap_pct'],'5.1f')} {r['chain']:5d} {r['swap']:4d} "
                  f"{r['dead_end_cells']:3d} {r['targets_in_dead_end']:5d} "
                  f"{r['mob_mean']:4.1f} {r['mob_min']:4d} {r['forced_6p']:5d} "
                  f"{fmt(r['detour'],'6.2f')}")
        print()

    by = collections.defaultdict(list)
    for r in rows:
        by[r["level"]].append(r)

    def col(lv, k):
        return [r[k] for r in by[lv] if r[k] is not None]

    print("DEVICE SEVERITY  (what the levels claim to install)")
    print("level  n  |  h0    opt   gap   gap%  | chain | edges | chains | swap | 6p ovh")
    print("           (gap% is the ratio of the means, as in the denominator section)")
    for lv in LEVELS:
        rs = by[lv]
        ch = col(lv, "chain")
        sw = [r for r in rs if r["swap"] > 0]
        ov = col(lv, "overhead_6p")
        print(f"{lv:>4} {len(rs):3d}  | {fmt(st.mean(col(lv,'h0')),'5.1f')} "
              f"{fmt(st.mean(col(lv,'opt')),'6.1f')} "
              f"{fmt(st.mean(col(lv,'gap')),'5.1f')} "
              f"{fmt(100*st.mean(col(lv,'gap'))/st.mean(col(lv,'h0')),'5.1f')} | "
              f"{st.mean(ch):4.1f}  | "
              f"{st.mean(col(lv,'edges')):4.1f}  | "
              f"{st.mean(col(lv,'chains')):5.1f}  | "
              f"{len(sw):2d}/{len(rs):<2d} | "
              f"{fmt(st.mean(ov) if ov else None,'5.3f')}")

    print("\nIRREVERSIBILITY AND PRESSURE")
    print("level  | dead-end cells | targets in DE | mobility mean | min | forced@6p | zero@6p")
    for lv in LEVELS:
        rs = by[lv]
        print(f"{lv:>4}   | {st.mean(col(lv,'dead_end_cells')):6.1f}         "
              f"| {st.mean(col(lv,'targets_in_dead_end')):5.2f}         "
              f"| {st.mean(col(lv,'mob_mean')):6.2f}        "
              f"| {min(col(lv,'mob_min'))}   "
              f"| {sum(col(lv,'forced_6p')):3d}       "
              f"| {sum(col(lv,'zero_6p')):3d}")

    print("\nDESCRIPTIVE  (no difficulty claim attached)")
    print("level  | walls | density | free cells")
    for lv in LEVELS:
        print(f"{lv:>4}   | {st.mean(col(lv,'walls')):5.1f} "
              f"| {st.mean(col(lv,'density')):7.3f} "
              f"| {st.mean(col(lv,'free_cells')):10.1f}")

    print("\nDETOUR RATIO  (the S2 device; reference at S3/S4)")
    print("The generator carves per object and accepts at target - 0.15, so the")
    print("pass rate below, not an average, is what says whether carving succeeded.")
    print("level | target | accept | pooled median (n) | objects at accept | boards all-6")
    target = {"S1": None, "S2": 1.5, "S3": 2.0, "S4": 2.0}
    for lv in LEVELS:
        t = target[lv]
        pooled = [r for row in by[lv] for r in (row["detour_objs"] or [])]
        if t is None:
            print(f"{lv:>4}  |   --   |   --   | {st.median(pooled):6.2f} ({len(pooled):3d})    "
                  f"|        --         |    --")
            continue
        acc = t - 0.15
        ok = sum(1 for x in pooled if x >= acc - 1e-9)
        allsix = sum(1 for row in by[lv]
                     if all(x >= acc - 1e-9 for x in (row["detour_objs"] or [])))
        print(f"{lv:>4}  |  {t:.1f}   |  {acc:.2f}  | {st.median(pooled):6.2f} ({len(pooled):3d})    "
              f"|  {ok:3d}/{len(pooled)} = {100*ok/len(pooled):3.0f}%   |  {allsix:2d}/{len(by[lv])}")


if __name__ == "__main__":
    main()
