"""Optimal play-through of a Matrixgame board under the real rules (A*).

Searches under every rule of the game: strict round-robin turn order, one
move of one of the player's own objects per turn, a `done` only with all own
objects home (the player is then skipped and its objects never move again),
and the episode ending the moment every object is home. A player with an
object off target and no legal move cannot respond, which aborts an episode,
so every such line is pruned: a solution is a play-through in which no player
is ever enclosed. Solo is one player owning all six objects, so there the
search finds the free-order optimum.

A* over (positions, done set, player to move). The heuristic, the sum of
per-object BFS distances over the walls, is admissible: one move lowers it by
at most one. A `done` consumes a turn but costs no move, so g counts moves
only, matching `Move Count` in the logs.

The optimum A of each game's board sets its turn budget, max(4 * A, 20), and
the efficiency denominator. resources/optimal_moves.json holds them:

    cc -O3 -o matrixgame/utils/solver_c matrixgame/utils/solver.c
    python3 matrixgame/utils/solver.py --table           # recompute the table
    python3 matrixgame/utils/solver.py 6p S3 10 --path   # one board, path replay-checked
"""

import argparse
import heapq
import json
import os
import time
from collections import deque
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = os.path.dirname(HERE)
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRNAME = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
LEVELS = ("S1", "S2", "S3", "S4")


def boards(game: str) -> Dict[tuple, dict]:
    """(level, game_id) -> instance of the game's first experiment at that level."""
    out = {}
    for e in json.load(open(os.path.join(GAME, "in", f"instances_{game}.json")))["experiments"]:
        for gi in e["game_instances"]:
            out.setdefault((gi["spatial_level"], gi["game_id"]), gi)
    return out


def partition(gi: dict) -> List[List[str]]:
    """Objects per player, in turn order.  Solo instances are one player owning
    all; the played dyad instances (results/) name the two groups separately."""
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
    response, which aborts an episode.  ``start`` is a state
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


BINARY = os.path.join(HERE, "solver_c")


def solve_c(gi: dict, cap: int = 200_000_000, timelimit: float = 3600.0):
    """Same answer as ``solve`` via solver.c, much faster; no path is returned.

        cc -O3 -o matrixgame/utils/solver_c matrixgame/utils/solver.c
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game", nargs="?", choices=("solo", "dyad", "3p", "6p"))
    ap.add_argument("level", nargs="?", choices=LEVELS)
    ap.add_argument("board", nargs="?", type=int)
    ap.add_argument("--path", action="store_true", help="print and replay-check the solution")
    ap.add_argument("--table", action="store_true", help="recompute resources/optimal_moves.json with solver_c")
    a = ap.parse_args()

    if a.table:
        table = {}
        for game in ("solo", "dyad", "3p", "6p"):
            inst = boards(game)
            table[game] = {}
            for level in LEVELS:
                row = []
                for gid in range(1, 16):
                    t0 = time.time()
                    moves, expanded, _, _ = solve_c(inst[(level, gid)])
                    assert isinstance(moves, int), f"{game} {level} #{gid}: {moves}"
                    print(f"{game} {level} #{gid}: {moves} ({expanded} expanded, {time.time() - t0:.1f} s)")
                    row.append(moves)
                table[game][level] = row
        lines = ["{"] + [f'  "{g}": {{\n' + ",\n".join(f'    "{l}": {json.dumps(v)}' for l, v in lv.items())
                         + "\n  }" + ("," if g != "6p" else "") for g, lv in table.items()] + ["}"]
        with open(os.path.join(GAME, "resources", "optimal_moves.json"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return

    gi = boards(a.game)[(a.level, a.board)]
    t0 = time.time()
    moves, expanded, path, dead = solve(gi, want_path=a.path)
    print(f"{a.game} {a.level} #{a.board}: optimum {moves}, budget {gi['max_turns']}, "
          f"{expanded} expanded, {dead} dead ends, {time.time() - t0:.1f} s")
    if a.path and isinstance(moves, int):
        err = replay(gi, path)
        print("replay:", "legal, no player ever enclosed" if not err else f"INVALID: {err}")
        print("  ".join(f"{w}:{x}" for w, x in path))


if __name__ == "__main__":
    main()
