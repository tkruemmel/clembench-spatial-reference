"""Targets in the way, per board of the final arm (thesis/method_targets_in_way.md).

A pair (x, y) of objects counts when x's target lies on every shortest route of
y from its start to its own target, over the walls alone, and y does not start
on that cell. It counts as an only way when it lies on every route of y. The
count depends on the board alone, and the 60 boards are the same in every game,
so the solo instances are read. Run once, prints one row per level:

    python3 analysis/thesis/targets_in_way.py
"""

import json
import os
import statistics
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSTANCES = os.path.join(ROOT, "matrixgame", "in", "instances_solo.json")


def bfs(gi, src, blocked=None):
    """Walls-only distance and number of shortest routes from src to every cell."""
    n, walls = gi["grid_size"], {tuple(w) for w in gi["walls"]} | ({blocked} if blocked else set())
    dist, cnt, q = {src: 0}, {src: 1}, deque([src])
    while q:
        r, c = cell = q.popleft()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nb[0] < n and 0 <= nb[1] < n) or nb in walls:
                continue
            if nb not in dist:
                dist[nb], cnt[nb] = dist[cell] + 1, 0
                q.append(nb)
            if dist[nb] == dist[cell] + 1:
                cnt[nb] += cnt[cell]
    return dist, cnt


def pairs(gi):
    """[(x, y)] in the way, and the subset in which x's target is y's only way."""
    start = {o: tuple(p) for o, p in gi["start_positions"].items()}
    target = {o: tuple(p) for o, p in gi["target_positions"].items()}
    out, only = [], []
    for y in start:
        ds, ns = bfs(gi, start[y])
        dt, nt = bfs(gi, target[y])
        for x in target:
            c = target[x]
            if x == y or c == start[y] or c not in ds:
                continue
            # c is on every shortest route when the routes through it are all the routes
            if ds[c] + dt[c] == ds[target[y]] and ns[c] * nt[c] == ns[target[y]]:
                out.append((x, y))
                if target[y] not in bfs(gi, start[y], blocked=c)[0]:
                    only.append((x, y))
    return out, only


def main():
    with open(INSTANCES) as f:
        experiments = json.load(f)["experiments"]
    print("level  pairs/board  range  only-way boards  shortest solution")
    for e in experiments:
        gis = e["game_instances"]
        level = gis[0]["spatial_level"]
        res = [pairs(gi) for gi in gis]
        n = [len(p) for p, _ in res]
        only = [o for _, o in res]
        if level == "S3":  # one only way per board: B's target, the corridor entrance, in D's way
            assert all(o == [("B", "D")] for o in only), only
            assert all(tuple(gi["target_positions"]["B"]) == tuple(gi["device"]["entrance"]) for gi in gis)
        else:
            assert not any(only), (level, only)
        print(f"{level:5}  {statistics.mean(n):11.2f}  {min(n)}-{max(n):<3}  {sum(map(bool, only)):15}"
              f"  {statistics.mean(gi['optimal_moves'] for gi in gis):17.1f}")


if __name__ == "__main__":
    main()
