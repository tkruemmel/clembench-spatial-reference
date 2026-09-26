"""The permanent cut behind every cycle loss of S9.3.2: which `done` killed the
board, and what the sealed door was made of.

Answers, from the recorded move logs alone, the question S9.3.2 leaves open in
words: the cut is never created by a move. An object stands on the fatal cell,
harmlessly, for as long as it likes; the board dies at the instant its owner
declares `done` and the cell stops being something that can step aside.

    python3 analysis/qualitative_analysis/cut.py            # the counts S9.3.2 quotes
    python3 analysis/qualitative_analysis/cut.py --list     # candidate episodes per kind

Also the data source for figures/cuts.py.
"""
import argparse
import collections
import itertools
import json
import os
import re
import string

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STRONG = ("qwen38", "dsv41")   # the tier S9.3 reports; Nemotron aborts on format
MODELS = {"qwen38": "Qwen3.8", "dsv41": "DeepSeek"}

# The shapes a cut takes. The pocket test comes first because it is a different
# mechanism rather than a different place: the object is shut out of its
# destination cell itself, not off a route to it. The corridor tests come next,
# because a victim inside the corridor is sealed by whatever closes the mouth,
# whether or not that cell is also on some other route.
POCKET = "the target's own neighbourhood sealed"
ENTRANCE = "seal on the corridor entrance, target inside the corridor"
ENTRANCE_OTHER = "seal on the corridor entrance, target elsewhere"
INSIDE = "victim locked inside the corridor"
APPROACH = "seal on the approach to the corridor"
OPEN = "open-board chokepoint, no corridor involved"
CHOKEPOINTS = (ENTRANCE, ENTRANCE_OTHER, INSIDE, APPROACH, OPEN)


def path(src, dst, n, walls, blocked):
    """Shortest route src -> dst over free cells, or None. `blocked` are walls too."""
    if src == dst:
        return [src]
    seen, queue = {src}, collections.deque([[src]])
    while queue:
        route = queue.popleft()
        for dr, dc in DIRS.values():
            nb = (route[-1][0] + dr, route[-1][1] + dc)
            if not (0 <= nb[0] < n and 0 <= nb[1] < n):
                continue
            if nb in walls or nb in blocked or nb in seen:
                continue
            if nb == dst:
                return route + [nb]
            seen.add(nb)
            queue.append(route + [nb])
    return None


def min_vertex_cut(src, dst, n, walls, limit=3):
    """Fewest free cells that must become walls to separate src from dst.

    The board generator certifies the free-cell graph 2-connected
    (`boardgen._is_2connected`), so this is the number that says whether a cut
    was within the guarantee. It is 1 only where the corridor is involved: that
    check is run with the corridor treated as blocked, which makes the entrance
    an articulation point by design rather than by accident.
    """
    free = [(r, c) for r in range(n) for c in range(n)
            if (r, c) not in walls and (r, c) not in (src, dst)]
    for k in range(1, limit + 1):
        for combo in itertools.combinations(free, k):
            if path(src, dst, n, walls, set(combo)) is None:
                return k
    return limit + 1


def _events(interactions):
    """Accepted moves and done declarations, in the order they were recorded."""
    out = []
    for turn in interactions["turns"]:
        for entry in turn:
            action = entry.get("action", {})
            if action.get("type") == "move":
                m = MOVE.match(action.get("content", ""))
                if m:
                    out.append(("move", (m.group(2), m.group(3))))
            elif action.get("type") == "player_done":
                out.append(("done", action["content"].split()[1]))
    return out


def step(pos, obj, direction):
    p = dict(pos)
    p[obj] = (p[obj][0] + DIRS[direction][0], p[obj][1] + DIRS[direction][1])
    return p


def _classify(victim_pos, victim_tgt, cells, entrance, corridor, n, walls, frozen):
    """Where the cut is, or that it is a pocket rather than a place on a route.

    A pocket is the degenerate case: every side of the target cell is a wall,
    the board edge or a frozen object, so there is no route left to pinch. It
    needs no corridor and is the only shape S1 and S2 produce.
    """
    if all((r, c) in walls or (r, c) in frozen
           or not (0 <= r < n and 0 <= c < n)
           for r, c in ((victim_tgt[0] + dr, victim_tgt[1] + dc)
                        for dr, dc in DIRS.values())):
        return POCKET
    if victim_pos in corridor:
        return INSIDE
    if entrance in cells:
        return ENTRANCE if victim_tgt in corridor else ENTRANCE_OTHER
    if victim_tgt in corridor:
        return APPROACH
    return OPEN


def cases(models=STRONG):
    """One record per (cycle loss, object that can no longer reach its target).

    Each record is replayed to the event that severed the path, so `pre` is the
    board as it stood one instant before the fatal `done` — with the sealing
    object already standing where it stands, and the victim's route still open
    through it.
    """
    out = []
    episodes = json.load(open(os.path.join(HERE, "episodes.json")))
    for rec in episodes:
        if rec["outcome"] != "lose" or rec["suite"] not in ("3p", "6p"):
            continue
        if rec["model"] not in models or "cycle" not in (rec["end_msg"] or ""):
            continue
        d = os.path.join(REPO, rec["path"])
        ins = json.load(open(os.path.join(d, "instance.json")))
        it = json.load(open(os.path.join(d, "interactions.json")))
        n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
        groups = ins["player_objects"]
        owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
        mine = collections.defaultdict(list)
        for o, p in owner.items():
            mine[p].append(o)
        start = {o: tuple(p) for o, p in ins["start_positions"].items()}
        tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
        device = ins.get("device") or {}
        corridor = {tuple(c) for c in device.get("corridor", [])}
        entrance = tuple(device["entrance"]) if device.get("entrance") else None

        # walk the episode, recording the instant each object's path is severed
        pos, done, plies = dict(start), set(), 0
        sealed = {}
        for kind, payload in _events(it):
            before, before_done = dict(pos), frozenset(done)
            if kind == "move":
                pos = step(pos, *payload)
                plies += 1
            else:
                done.add(payload)
            frozen = {pos[o] for o in owner if owner[o] in done}
            for o in tgt:
                if o in sealed or owner[o] in done or pos[o] == tgt[o]:
                    continue
                if path(pos[o], tgt[o], n, walls, frozen - {pos[o]}) is None:
                    sealed[o] = (plies, kind, payload, before, before_done)

        frozen_end = {o: pos[o] for o in owner if owner[o] in done}
        victims = [o for o in tgt
                   if owner[o] not in done and pos[o] != tgt[o]
                   and path(pos[o], tgt[o], n, walls,
                            set(frozen_end.values()) - {pos[o]}) is None]
        for v in victims:
            ply, kind, payload, pre, pre_done = sealed[v]
            assert kind == "done", (rec["path"], v, kind)
            frozen_pre = {o: pre[o] for o in owner if owner[o] in pre_done}
            frozen_post = {o: pre[o] for o in owner
                           if owner[o] in pre_done or owner[o] == payload}
            # every route that existed one instant ago ran through a cell this
            # player is about to freeze, so the sealing objects are the ones
            # standing on it; at three players a single `done` can close two.
            route = path(pre[v], tgt[v], n, walls,
                         set(frozen_pre.values()) - {pre[v]})
            seals = [o for o in mine[payload] if pre[o] in route]
            assert seals, (rec["path"], v)
            sealer = seals[0]
            # objects that, released on their own, reopen a route
            blocked = set(frozen_post.values()) - {pre[v]}
            culprits = [o for o, c in frozen_post.items() if o != v
                        and path(pre[v], tgt[v], n, walls, blocked - {c}) is not None]
            others = [o for o in culprits if o != sealer]
            out.append(dict(
                meta=rec, grid=n, walls=walls, owner=owner, mine=dict(mine),
                groups=groups, start=start, targets=tgt, device=device,
                corridor=corridor, entrance=entrance,
                victim=v, sealer=sealer, sealer_player=payload, culprits=culprits,
                others=others, ply=ply, pre=pre,
                frozen_pre=frozen_pre, frozen_post=frozen_post,
                route=route, seals=seals,
                other_route=path(pre[v], tgt[v], n, walls, {pre[sealer]}),
                kind=_classify(pre[v], tgt[v], [pre[o] for o in seals],
                               entrance, corridor, n, walls,
                               set(frozen_post.values())),
                mincut=min_vertex_cut(pre[v], tgt[v], n, walls),
                tail=rec["moves"] - ply,
            ))
    return out


def summary(rows):
    eps = {r["meta"]["path"] for r in rows}
    print("cycle losses ending with an object that can no longer reach its target:")
    print("   %d episodes, %d cut objects" % (len(eps), len(rows)))
    print()
    print("the event that severed the path: %s"
          % dict(collections.Counter("done" for _ in rows)))
    print("the sealing object stands on its own target: %d of %d"
          % (sum(r["pre"][r["sealer"]] == r["targets"][r["sealer"]] for r in rows), len(rows)))
    print("the victim's target cell is occupied by a frozen object: %d of %d"
          % (sum(r["targets"][r["victim"]] in set(r["frozen_post"].values())
                 for r in rows), len(rows)))
    print("one frozen object is the whole door (releasing it reopens a route): %d of %d"
          % (sum(bool(r["culprits"]) for r in rows), len(rows)))
    print("objects the declaring player froze on the victim's last route: %s"
          % dict(collections.Counter(len(r["seals"]) for r in rows)))
    print()
    print("shape of the cut:")
    for k, v in collections.Counter(r["kind"] for r in rows).most_common():
        print("   %-46s %3d" % (k, v))
    print()
    print("minimum vertex cut of the board between victim and target:")
    for k, v in sorted(collections.Counter((r["kind"], r["mincut"]) for r in rows).items()):
        print("   %-46s cut = %d : %3d" % (k[0], k[1], v))
    print()
    print("by level: %s" % dict(collections.Counter(r["meta"]["level"] for r in rows)))
    print("by suite: %s" % dict(collections.Counter(r["meta"]["suite"] for r in rows)))
    tails = sorted(r["tail"] for r in rows)
    print("moves played after the cut formed: median %d, max %d"
          % (tails[len(tails) // 2], tails[-1]))


def listing(rows, kind=None):
    seen = collections.Counter(r["meta"]["path"] for r in rows)
    for r in rows:
        if kind and not r["kind"].startswith(kind):
            continue
        m = r["meta"]
        print("%-8s %-3s %-4s %-10s %-14s victim %s  sealer %s (player %s)  "
              "culprits %s  victims/ep %d  cut %d  %s"
              % (m["model"], m["suite"], m["level"], m["proto"], m["inst"],
                 r["victim"], r["sealer"], r["sealer_player"],
                 "".join(sorted(r["culprits"])), seen[m["path"]], r["mincut"],
                 r["kind"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", metavar="KIND", nargs="?", const="",
                    help="per-episode candidates, optionally filtered by kind prefix")
    args = ap.parse_args()
    rows = cases()
    if args.list is None:
        summary(rows)
    else:
        listing(rows, args.list)


if __name__ == "__main__":
    main()
