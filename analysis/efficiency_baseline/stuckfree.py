"""Optimal stuck-free play-through of a board under the n-player protocol.

Covers every suite's partition (``--suite solo|dyad|3p|6p``): a player owning
several objects moves any one of them per turn and may declare ``done`` once all
of them are home.  Solo is one player owning all six, i.e. the free-order joint
optimum, which is the solver's self-check against ``joint_optimum.jsonl``.
The description below is written for the 6-player case.

The free-order joint optimum of ``joint_optimum.py`` is what a *solo* player can
achieve: any object may move on any turn.  A six-player team cannot schedule
freely.  The protocol adds four constraints, and this module searches under all
of them:

  * one object per player, strict round-robin turn order A, B, C, D, E, F;
  * on its turn a player must move its own object exactly one cell, unless it
    has declared ``done``;
  * a player may declare ``done`` only when its object is on target, and is then
    skipped forever -- its object becomes a permanent obstacle;
  * the episode ends the instant every object is on its target, so the team
    never has to "finish the round".

A player whose object is off target and has no legal move cannot produce a legal
response, which is what aborts an episode in the real game (Section 7.1 of
``results_three_models_grid.md``).  Every such line is pruned, so a solution
returned here is a play-through in which **no player is ever enclosed**.

Method
------
A* over (positions, done-set, player to move).  The heuristic is the sum of
per-object BFS distances on the walls-only grid, which is admissible: one move
reduces that sum by at most one.  ``done`` declarations consume a turn but cost
no move, so the g-value counts moves only, matching ``Move Count`` in the logs.
The heuristic is not consistent across a ``done`` edge (cost 0, h unchanged), so
the closed set is keyed on state and re-opened on a strictly better g.

Usage
-----
    python3 stuckfree.py                 # the fifteen boards of Section 7.2
    python3 stuckfree.py --level S3 --game 10 --path
    python3 stuckfree.py --all           # all sixty boards
"""

import argparse
import heapq
import json
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_optimum import load_instances  # noqa: E402

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRNAME = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUITES = {
    # solo instance files name experiments Solo_S1_..., which load_instances does
    # not parse; the boards are identical, so solo reads the dyad file and drops
    # the partition (see main).
    "solo": os.path.join(REPO, "matrixgame_covered_comm_grid", "in", "instances.json"),
    "dyad": os.path.join(REPO, "matrixgame_covered_comm_grid", "in", "instances.json"),
    "3p": os.path.join(REPO, "matrixgame_covered_comm_nplayer_grid", "in", "instances_3p.json"),
    "6p": os.path.join(REPO, "matrixgame_covered_comm_nplayer_grid", "in", "instances_6p.json"),
}

# The fifteen boards on which either strong model was aborted at 6p
# (Section 7.2 of results_three_models_grid.md).
SECTION_72 = [("S2", 2), ("S2", 6), ("S2", 10), ("S2", 12),
              ("S3", 6), ("S3", 7), ("S3", 10), ("S3", 13), ("S3", 15),
              ("S4", 1), ("S4", 5), ("S4", 8), ("S4", 10), ("S4", 11), ("S4", 13)]


def partition(gi: dict) -> List[List[str]]:
    """Objects per player, in turn order.  Solo instances are one player owning all."""
    if "player_objects" in gi:
        return [list(g) for g in gi["player_objects"]]
    if "player_a_objects" in gi:
        return [list(gi["player_a_objects"]), list(gi["player_b_objects"])]
    return [sorted(gi["objects"])]


def board(gi: dict):
    """(letters, groups, starts, targets, neighbour table, distance tables, free).

    ``letters`` is the object order of the position tuple; ``groups[p]`` lists the
    object indices player ``p`` controls, players in turn order.
    """
    size = gi["grid_size"]
    walls = {tuple(w) for w in gi["walls"]}
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in walls]
    idx = {p: i for i, p in enumerate(free)}
    nbr = [[] for _ in free]
    for p, i in idx.items():
        for d in DIRS:
            q = (p[0] + d[0], p[1] + d[1])
            if q in idx:
                nbr[i].append((idx[q], d))

    parts = partition(gi)
    letters = [o for g in parts for o in g]
    groups, k = [], 0
    for g in parts:
        groups.append(tuple(range(k, k + len(g))))
        k += len(g)
    starts = tuple(idx[tuple(gi["start_positions"][l])] for l in letters)
    targets = tuple(idx[tuple(gi["target_positions"][l])] for l in letters)

    dist = []
    for t in targets:
        d = [-1] * len(free)
        d[t] = 0
        q = deque([t])
        while q:
            u = q.popleft()
            for v, _ in nbr[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        dist.append(d)
    return letters, groups, starts, targets, nbr, dist, free


def solve(gi: dict, cap: int = 20_000_000, want_path: bool = False, start=None):
    """Optimal stuck-free play-through, or None if every line runs into a dead end.

    Returns (moves, expanded, path, dead) where path is a list of
    (player, action) with action either a direction name or "done", and dead
    counts the expanded states in which the player to move had no legal
    response -- the abort condition of Section 7.1.  ``start`` is a state
    (positions, done mask, player to move) to search from instead of the
    board's start; it is the reference hstar.c is checked against.
    """
    letters, groups, starts, targets, nbr, dist, free = board(gi)
    n = len(groups)                                   # players
    m = len(letters)                                  # objects

    def h(pos):
        return sum(dist[i][pos[i]] for i in range(m))

    def nxt(turn, done_mask):
        """Next player to act, skipping those that have declared done."""
        for k in range(1, n + 1):
            t = (turn + k) % n
            if not (done_mask >> t) & 1:
                return t
        return -1                                     # everybody is done

    start = start or (starts, 0, 0)
    best = {start: 0}
    came: Dict[tuple, tuple] = {}
    pq = [(h(start[0]), 0, start)]
    expanded = 0
    dead = 0                                          # states with no legal response

    while pq:
        f, g, st = heapq.heappop(pq)
        if g > best.get(st, 1 << 30):
            continue
        pos, done_mask, turn = st
        if pos == targets:
            path = []
            if want_path:
                cur = st
                while cur in came:
                    prev, act = came[cur]
                    path.append(act)
                    cur = prev
                path.reverse()
            return g, expanded, path, dead
        expanded += 1
        if expanded > cap:
            return "cap", expanded, [], dead

        occupied = set(pos)
        succ = []
        own = groups[turn]
        # option 1: declare done (free, consumes the turn, freezes the objects)
        if all(pos[i] == targets[i] for i in own):
            t2 = nxt(turn, done_mask | (1 << turn))
            if t2 >= 0:
                succ.append((0, (pos, done_mask | (1 << turn), t2), "done"))
        # option 2: move one of the player's own objects one cell
        for i in own:
            for v, d in nbr[pos[i]]:
                if v in occupied:
                    continue
                np = list(pos)
                np[i] = v
                np = tuple(np)
                act = f"{letters[i]} {DIRNAME[d]}"
                if np == targets:
                    succ.append((1, (np, done_mask, turn), act))
                    continue
                succ.append((1, (np, done_mask, nxt(turn, done_mask)), act))

        # no legal response: the player would be aborted here, so the line is dead
        if not succ:
            dead += 1
            continue

        for cost, ns, act in succ:
            ng = g + cost
            if ng < best.get(ns, 1 << 30):
                best[ns] = ng
                if want_path:
                    came[ns] = (st, (turn, act))
                heapq.heappush(pq, (ng + h(ns[0]), ng, ns))

    return None, expanded, [], dead


BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stuckfree_c")


def solve_c(gi: dict, cap: int = 200_000_000, timelimit: float = 3600.0):
    """Same answer as ``solve`` via stuckfree.c (cc -O3 -o stuckfree_c stuckfree.c).

    No path is returned; ``solve`` with ``want_path`` is the replay-checked
    reference, and the two are cross-checked on the 3p and 6p boards.
    """
    import subprocess
    letters, groups, starts, targets, nbr, dist, free = board(gi)
    lines = [f"{len(free)} {len(letters)} {len(groups)}"]
    lines += [" ".join(map(str, [len(ns)] + [v for v, _ in ns])) for ns in nbr]
    lines += [" ".join(map(str, starts)), " ".join(map(str, targets))]
    lines += [" ".join(map(str, [len(g)] + list(g))) for g in groups]
    out = subprocess.run([BINARY, str(cap), str(timelimit)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True).stdout.split()
    got = dict(zip(out[0::2], out[1::2]))
    moves = int(got["opt"]) if got["status"] == "exact" else \
        (None if got["status"] == "none" else "cap")
    return moves, int(got["expanded"]), [], 0


def replay(gi: dict, path) -> str:
    """Re-execute a returned path against the rules; returns '' if it is legal."""
    letters, groups, starts, targets, nbr, dist, free = board(gi)
    n = len(groups)
    pos = list(starts)
    done = [False] * n
    turn = 0
    moves = 0
    total = len([a for a in path if a[1] != "done"])
    for who, act in path:
        if who != turn:
            return f"out of turn: player {who} acted on player {turn}'s turn"
        if act == "done":
            if any(pos[i] != targets[i] for i in groups[who]):
                return f"player {who} declared done off target"
            done[who] = True
        else:
            obj, dname = act.split()
            i = letters.index(obj)
            if i not in groups[who]:
                return f"player {who} moved {obj}, which it does not control"
            d = [k for k, v in DIRNAME.items() if v == dname][0]
            tgt = [v for v, dd in nbr[pos[i]] if dd == d]
            if not tgt:
                return f"{obj} moved {dname} into a wall or off the grid"
            if tgt[0] in pos:
                return f"{obj} moved {dname} onto an occupied cell"
            pos[i] = tgt[0]
            moves += 1
        if tuple(pos) == targets:
            return "" if moves == total else f"solved early after {moves} moves"
        for k in range(1, n + 1):
            t = (turn + k) % n
            if not done[t]:
                turn = t
                break
        # the player about to act must have a legal response
        own = groups[turn]
        if not all(pos[i] == targets[i] for i in own) and \
           not any(v not in pos for i in own for v, _ in nbr[pos[i]]):
            return f"player {turn} is enclosed with no legal move"
    return "path ended without solving the board"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=SUITES, default="6p")
    ap.add_argument("--instances", help="instance file; defaults to the suite's")
    ap.add_argument("--level")
    ap.add_argument("--game", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--path", action="store_true", help="print and replay-check the solution")
    ap.add_argument("--cap", type=int, default=20_000_000)
    ap.add_argument("--c", action="store_true", help="use the C port (no paths)")
    ap.add_argument("--save", help="append one JSON record per board to this file")
    a = ap.parse_args()

    inst = load_instances(a.instances or SUITES[a.suite])
    if a.suite == "solo":
        inst = {k: {f: v for f, v in gi.items() if f not in ("player_a_objects", "player_b_objects")}
                for k, gi in inst.items()}
    jo = {}
    jp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_optimum.jsonl")
    if os.path.exists(jp):
        for line in open(jp):
            d = json.loads(line)
            jo[(d["level"], d["game_id"])] = d["opt"]

    if a.level and a.game:
        keys = [(a.level, a.game)]
    elif a.all:
        keys = sorted(inst, key=lambda k: (k[0], k[1]))
    else:
        keys = SECTION_72

    out = open(a.save, "w") if a.save else None
    print(f"{'board':<10}{'stuck-free':>12}{'joint opt':>11}{'stored':>8}"
          f"{'budget':>8}{'overhead':>10}{'expanded':>11}{'dead ends':>11}{'secs':>7}")
    for k in keys:
        gi = inst.get(k)
        if gi is None:
            print(f"{k[0]} #{k[1]:<6}  not in instance file")
            continue
        t0 = time.time()
        if a.c:
            moves, expanded, path, dead = solve_c(gi, cap=max(a.cap, 200_000_000))
        else:
            moves, expanded, path, dead = solve(gi, cap=a.cap, want_path=a.path)
        secs = time.time() - t0
        stored = gi.get("optimal_moves")
        opt = jo.get(k)
        if moves is None:
            cell, over = "none exists", "—"
        elif moves == "cap":
            cell, over = f"cap {a.cap}", "—"
        else:
            cell = str(moves)
            over = f"{moves / opt:.2f}" if opt else "—"
        budget = max(4 * stored, 20) if stored else None
        print(f"{k[0]} #{k[1]:<6}{cell:>12}{str(opt):>11}{str(stored):>8}"
              f"{str(budget):>8}{over:>10}{expanded:>11}{dead:>11}{secs:>7.1f}")
        if out:
            out.write(json.dumps({
                "suite": a.suite, "level": k[0], "game_id": k[1],
                "stuckfree": moves if isinstance(moves, int) else None,
                "status": "exact" if isinstance(moves, int) else
                          ("none exists" if moves is None else "cap"),
                "joint_optimum": opt, "stored": stored, "move_budget": budget,
                "overhead_vs_joint": round(moves / opt, 4) if isinstance(moves, int) and opt else None,
                "expanded": expanded, "dead_ends": dead, "secs": round(secs, 1)}) + "\n")
            out.flush()
        if a.path and isinstance(moves, int):
            err = replay(gi, path)
            print("   replay:", "legal, no player ever enclosed" if not err else f"INVALID: {err}")
            print("   " + "  ".join(f"{w}:{x}" for w, x in path))


    if out:
        out.close()


if __name__ == "__main__":
    main()
