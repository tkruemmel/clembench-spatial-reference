"""The sealing move of every enclosure abort: did the sealer have a harmless
alternative? Answers the claim in S9.3 that it had one in 30 of 32 cases, and
prints the two exceptions. Also the data source for figures/aborts.py.

    python3 analysis/qualitative_analysis/seal_alt.py
"""
import collections
import json
import os
import re
import string
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STRONG = ("qwen38", "dsv41")  # the tier S9.3 reports; nemotron aborts on format
MODELS = {"qwen38": "Qwen3.8", "dsv41": "DeepSeek"}


def legal_moves(pos, objs, walls, n):
    """Every (object, direction) the given objects can legally play from `pos`."""
    occ = set(pos.values())
    out = []
    for o in objs:
        r0, c0 = pos[o]
        for d, (dr, dc) in DIRS.items():
            nr, nc = r0 + dr, c0 + dc
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in walls and (nr, nc) not in occ:
                out.append((o, d))
    return out


def step(pos, obj, direction):
    p = dict(pos)
    p[obj] = (p[obj][0] + DIRS[direction][0], p[obj][1] + DIRS[direction][1])
    return p


def _aborts():
    path = os.path.join(HERE, "aborts.json")
    if not os.path.exists(path):  # forced.py writes it from episodes.json
        subprocess.run([sys.executable, "forced.py"], cwd=HERE, check=True,
                       capture_output=True)
    return json.load(open(path))


def cases(models=STRONG):
    """One record per enclosure abort, with the board replayed up to the seal.

    The sealing move is the accepted move that took the victim from some legal
    move to none; `alts` are everything the sealing player could have played
    instead and `harmless` the subset that would have left every player a move.
    """
    out = []
    for a in _aborts():
        if a["n_legal_own"] or a["model"] not in models:
            continue
        d = os.path.join(REPO, a["path"])
        ins = json.load(open(os.path.join(d, "instance.json")))
        it = json.load(open(os.path.join(d, "interactions.json")))
        n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
        groups = ins["player_objects"]
        owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
        mine = collections.defaultdict(list)
        for o, p in owner.items():
            mine[p].append(o)
        victim = groups[string.ascii_uppercase.index(a["who"])][0]

        moves, done_at, done = [], [], set()
        for t in it["turns"]:
            for e in t:
                ac = e.get("action", {})
                if ac.get("type") == "move":
                    m = MOVE.match(ac.get("content", ""))
                    if m:
                        moves.append((m.group(2), m.group(3)))
                        done_at.append(frozenset(done))
                elif ac.get("type") == "player_done":
                    done.add(ac["content"].split()[1])

        pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
        seal = None
        for i, (o, dr) in enumerate(moves):
            had = bool(legal_moves(pos, [victim], walls, n))
            pre = pos
            pos = step(pos, o, dr)
            if had and not legal_moves(pos, [victim], walls, n):
                seal = (i, o, dr, pre)
        if seal is None:
            continue
        i, o, dr, pre = seal
        sealer = owner[o]
        alts = legal_moves(pre, mine[sealer], walls, n)
        harmless = [(o2, d2) for o2, d2 in alts
                    if not [pl for pl in mine
                            if not legal_moves(step(pre, o2, d2), mine[pl], walls, n)]]
        out.append(dict(
            meta=a, grid=n, walls=walls, groups=groups, owner=owner, mine=dict(mine),
            targets={o2: tuple(p) for o2, p in ins["target_positions"].items()},
            start={o2: tuple(p) for o2, p in ins["start_positions"].items()},
            moves=moves, victim=victim, sealer=sealer, played=(o, dr), ply=i + 1,
            pre=pre, alts=alts, harmless=harmless, done=set(done_at[i]),
        ))
    return out


def dist_map(walls, n, target):
    """Shortest-path distance to `target` over walls, other objects ignored.

    Objects move, so a teammate standing in the way does not make a direction
    the wrong way to go; walls do. Manhattan distance would call three of the
    sealing moves retreats when they are in fact the way round a wall column.
    """
    seen = {target: 0}
    queue = collections.deque([target])
    while queue:
        cell = queue.popleft()
        for dr, dc in DIRS.values():
            nb = (cell[0] + dr, cell[1] + dc)
            if (0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls
                    and nb not in seen):
                seen[nb] = seen[cell] + 1
                queue.append(nb)
    return seen


def progress(case, pos, obj, direction):
    """Does this move take `obj` nearer its own target?"""
    d = dist_map(case["walls"], case["grid"], case["targets"][obj])
    return d[step(pos, obj, direction)[obj]] < d[pos[obj]]


def greedy_rows(cases):
    """The counts of tab:seal-greedy, per model and pooled.

    Restricted to the aborts whose sealer had a harmless alternative, which is
    the set the question is about: where a choice existed, what did the sealing
    move buy the player that made it?
    """
    keys, out = [MODELS[m] for m in STRONG] + ["all"], {}
    for case in cases:
        if not case["harmless"]:
            continue  # the two forced seals: no choice to characterise
        obj, direction = case["played"]
        d = dist_map(case["walls"], case["grid"], case["targets"][obj])
        delta = d[step(case["pre"], obj, direction)[obj]] - d[case["pre"][obj]]
        made = [(o, dd) for o, dd in case["alts"]
                if progress(case, case["pre"], o, dd)]
        safe = [m for m in made if m in set(case["harmless"])]
        # the two ways of asking it coincide on all 30, which the caption says
        assert (made == [case["played"]]) == (not safe), case["meta"]
        for key in (MODELS[case["meta"]["model"]], "all"):
            row = out.setdefault(key, collections.Counter())
            row["n"] += 1
            row["nearer" if delta < 0 else "level" if delta == 0 else "further"] += 1
            row["free" if safe else "costly"] += 1
    return keys, out


def last_free_cell(case):
    """The one cell the victim could still move to, just before the seal."""
    free = legal_moves(case["pre"], [case["victim"]], case["walls"], case["grid"])
    if len(free) != 1:
        return None
    r0, c0 = case["pre"][case["victim"]]
    dr, dc = DIRS[free[0][1]]
    return (r0 + dr, c0 + dc)


def replay(case, upto):
    """Board after the first `upto` accepted moves."""
    pos = dict(case["start"])
    for o, d in case["moves"][:upto]:
        pos = step(pos, o, d)
    return pos


def main():
    rows = cases()
    print("enclosure aborts (%s): %d" % ("+".join(STRONG), len(rows)))
    print("sealing player had a harmless legal alternative: %d of %d"
          % (sum(1 for r in rows if r["harmless"]), len(rows)))
    print("victim was already down to one legal move: %d of %d"
          % (sum(1 for r in rows if len(legal_moves(r["pre"], [r["victim"]],
                                                    r["walls"], r["grid"])) == 1), len(rows)))
    print("sealer stepped into the victim's last free cell: %d of %d"
          % (sum(1 for r in rows if last_free_cell(r) == step(r["pre"], *r["played"])[r["played"][0]]),
             len(rows)))
    print()
    keys, counts = greedy_rows(rows)
    print("of the %d seals with a harmless alternative, the move played took"
          % counts["all"]["n"])
    print("   the sealer nearer its own target:      %d" % counts["all"]["nearer"])
    print("   nowhere (distance unchanged):          %d" % counts["all"]["level"])
    print("   further from its own target:           %d" % counts["all"]["further"])
    print()
    for r in (x for x in rows if not x["harmless"]):
        m, (o, d) = r["meta"], r["played"]
        print(f"{m['model']:8s} {m['level']} {m['proto']:10s} {m['inst']} | "
              f"victim {r['victim']} | sealer {r['sealer']} played {o} {d} "
              f"at move {r['ply']}/{m['moves']}")
        print(f"   sealer's legal moves there: {r['alts']}")
        print(f"   done players then: {sorted(r['done']) or 'none'}")


if __name__ == "__main__":
    main()
