"""Exact joint optimum for a matrixgame board.

The stored ``optimal_moves`` in every instance file is the sum of per-object
BFS distances (see ``matrixgame_covered_shared/boardgen.py:_verify_reachability``).
That is a lower bound on the number of moves a board really costs, because it
ignores every interaction between objects.  At S4 the bound is provably wrong
by 8 moves on every board (see ``s4_device.py``).

This module computes the real thing: the minimum number of moves to get every
object from its start to its target, where one move relocates one object by one
cell and two objects may never share a cell.  Move order is free and ownership
is ignored, so the result is

  * exactly the optimum a *solo* player can achieve (the solo game has no
    ``done`` mechanism and the player may move any object on any turn), and
  * a lower bound on what a multiplayer team can achieve (strict round-robin,
    a permanent ``done`` declaration and the missing ``pass`` action all force
    extra moves on top).

Method
------
A* over joint configurations.  A state is the tuple of all object positions;
a successor moves one object into a free neighbouring cell.  The heuristic is
the sum of per-object BFS distances, i.e. the stored ``optimal_moves`` of the
current state.  One move changes exactly one object's distance by +-1 at unit
cost, so the heuristic is consistent: A* is optimal, and the f-value of the
last expanded state is a *proven* lower bound even when the search is cut off.
"""

import heapq
import json
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def load_instances(path: str) -> Dict[Tuple[str, int], dict]:
    """Map (spatial level, game_id) -> game instance.

    Instance files repeat the same board once per protocol; the physical board
    is identical across protocols and across the solo / dyad / 3p / 6p
    packages, so the first occurrence is kept.
    """
    out: Dict[Tuple[str, int], dict] = {}
    for exp in json.load(open(path))["experiments"]:
        level = exp["name"].split("_")[0]
        for gi in exp["game_instances"]:
            out.setdefault((level, gi["game_id"]), gi)
    return out


def _grid(gi: dict):
    """Free cells, neighbour lists and a cell -> index map for one instance."""
    size = gi["grid_size"]
    walls = {tuple(w) for w in gi["walls"]}
    free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in walls]
    idx = {p: i for i, p in enumerate(free)}
    nbr = [
        tuple(
            idx[(p[0] + dr, p[1] + dc)]
            for dr, dc in DIRS
            if (p[0] + dr, p[1] + dc) in idx
        )
        for p in free
    ]
    return free, idx, nbr


def distance_table(nbr: List[Tuple[int, ...]], n: int, source: int) -> List[int]:
    """BFS distance from every free cell to ``source``, ignoring other objects."""
    INF = 10 ** 6
    d = [INF] * n
    d[source] = 0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in nbr[u]:
            if d[v] == INF:
                d[v] = d[u] + 1
                q.append(v)
    return d


def solve(gi: dict, cap: int = 3_000_000, timelimit: float = 600.0) -> dict:
    """Exact joint optimum for one instance.

    Returns a dict with:
      ``h0``      sum of per-object BFS distances, i.e. the stored bound
      ``opt``     the exact joint optimum, or None if the search was cut off
      ``lb``      a proven lower bound on the optimum (== opt when exact)
      ``status``  'exact' | 'cutoff' | 'unsolvable'

    ``cap`` bounds expanded states (and therefore memory); ``timelimit`` is in
    seconds.  On a cut-off the returned ``lb`` is still sound and is normally
    much tighter than ``h0``.
    """
    free, idx, nbr = _grid(gi)
    n = len(free)
    objs = sorted(gi["objects"])
    k = len(objs)
    start = tuple(idx[tuple(gi["start_positions"][o])] for o in objs)
    goal = tuple(idx[tuple(gi["target_positions"][o])] for o in objs)

    dist = [distance_table(nbr, n, g) for g in goal]
    h0 = sum(dist[i][start[i]] for i in range(k))
    if start == goal:
        return dict(h0=h0, opt=0, lb=0, popped=0, status="exact")

    heap = [(h0, 0, start)]
    closed = set()
    popped = 0
    last_f = h0
    t0 = time.time()

    while heap:
        f, g_cost, state = heapq.heappop(heap)
        if state in closed:
            continue
        closed.add(state)
        popped += 1
        last_f = f
        if state == goal:
            return dict(h0=h0, opt=g_cost, lb=g_cost, popped=popped, status="exact")
        if popped >= cap or (popped % 65536 == 0 and time.time() - t0 > timelimit):
            # Every state with f < last_f has been expanded, so optimum >= last_f.
            return dict(h0=h0, opt=None, lb=last_f, popped=popped, status="cutoff")
        ng = g_cost + 1
        for i in range(k):
            u = state[i]
            du = dist[i][u]
            di = dist[i]
            for v in nbr[u]:
                if v in state:
                    continue
                nxt = state[:i] + (v,) + state[i + 1:]
                if nxt in closed:
                    continue
                heapq.heappush(heap, (ng + f - g_cost - du + di[v], ng, nxt))

    return dict(h0=h0, opt=None, lb=last_f, popped=popped, status="unsolvable")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("level", help="S1 | S2 | S3 | S4")
    ap.add_argument("game_id", type=int)
    ap.add_argument(
        "--instances",
        default="matrixgame_covered_comm_grid/in/instances.json",
        help="instance file to read the board from",
    )
    ap.add_argument("--cap", type=int, default=3_000_000)
    ap.add_argument("--timelimit", type=float, default=600.0)
    args = ap.parse_args()

    gi = load_instances(args.instances)[(args.level, args.game_id)]
    t = time.time()
    r = solve(gi, cap=args.cap, timelimit=args.timelimit)
    gap = None if r["opt"] is None else r["opt"] - r["h0"]
    print(
        f"{args.level} #{args.game_id}: stored={gi['optimal_moves']} h0={r['h0']} "
        f"joint_optimum={r['opt']} lower_bound={r['lb']} gap={gap} "
        f"{r['status']} {time.time() - t:.1f}s"
    )


if __name__ == "__main__":
    _main()
