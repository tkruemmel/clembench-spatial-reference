"""Upstream of the fork: order of completion at the S3 door, holder waiting, message evidence.

Read-only on the repo. Strong tier (qwen38 + dsv41), 3p + 6p.
    python3 order.py            # all tables
"""
import collections, json, os, re, string, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doors import comp, genuine, REPO, DIRS
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
PROTOS = ("none", "structured", "freeform", "hybrid")
STRONG = ("qwen38", "dsv41")


def cell_rx(c):
    r, k = c[0] + 1, c[1] + 1
    return r"(?:\(?\bR%d\s*,?\s*C%d\b\)?|\brow\s*%d\W{1,3}col(?:umn)?\s*%d\b)" % (r, k, r, k)


def load(rec):
    d = os.path.join(REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    ev = []
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {}); t = a.get("type"); c = a.get("content", "")
            if t == "move":
                m = MOVE.match(c)
                if m:
                    ev.append(("move", m.group(2), m.group(3)))
            elif t == "player_done":
                ev.append(("done", c.split()[1], None))
            elif t == "message":
                m = re.match(r"Player ([A-F]): (.*)", c, re.S)
                if m:
                    ev.append(("msg", m.group(1), m.group(2)))
    return ins, ev


class Board:
    def __init__(s, ins):
        s.n = ins["grid_size"]; s.walls = {tuple(w) for w in ins["walls"]}
        g = ins["player_objects"]
        s.owner = {o: string.ascii_uppercase[i] for i, gg in enumerate(g) for o in gg}
        s.mine = {string.ascii_uppercase[i]: list(gg) for i, gg in enumerate(g)}
        s.start = {o: tuple(p) for o, p in ins["start_positions"].items()}
        s.tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}

    def dist(s, src, dst, blocked):
        """BFS distance over free cells (walls + blocked)."""
        if src == dst:
            return 0
        seen = {src}; fr = [src]; d = 0
        while fr:
            d += 1; nx = []
            for a, b in fr:
                for dr, dc in DIRS:
                    q = (a + dr, b + dc)
                    if not (0 <= q[0] < s.n and 0 <= q[1] < s.n) or q in s.walls or q in blocked or q in seen:
                        continue
                    if q == dst:
                        return d
                    seen.add(q); nx.append(q)
            fr = nx
        return None


def replay(ins, ev):
    """Yield (event, ply, pos_before, done_before) for every event; pos is a dict."""
    B = Board(ins)
    pos, done, ply = dict(B.start), set(), 0
    for e in ev:
        yield e, ply, pos, frozenset(done)
        if e[0] == "move":
            dr, dc = STEP[e[2]]
            pos = dict(pos); pos[e[1]] = (pos[e[1]][0] + dr, pos[e[1]][1] + dc); ply += 1
        elif e[0] == "done":
            done.add(e[1])
    yield ("end", None, None), ply, pos, frozenset(done)


# ---------------------------------------------------------------- S3 door order
def door_order(rec, ins, ev, h, v):
    B = Board(ins)
    E = B.tgt[h]; lab = comp(B.n, B.walls, {E})
    passed = lambda p: lab.get(p) is not None and lab.get(p) == lab.get(B.tgt[v])
    hp = B.owner[h]
    r = dict(first_arr=None, first_pass=None, elig=None, reach=None, after=None,
             pass_before_arr=None, ply_end=None, arr_moves_h=None, dmin_h=None,
             win_end=None, v_home_end=None)
    nmoves_h = 0
    for e, ply, pos, done in replay(ins, ev):
        if r["first_pass"] is None and passed(pos[v]):
            r["first_pass"] = ply
        if r["first_arr"] is None and pos[h] == E:
            r["first_arr"] = ply; r["arr_moves_h"] = nmoves_h
            r["pass_before_arr"] = passed(pos[v])
        elig = hp not in done and all(pos[o] == B.tgt[o] for o in B.mine[hp])
        if elig and r["elig"] is None:
            r["elig"] = (ply, passed(pos[v]))
        if elig and not passed(pos[v]) and r["reach"] is None:
            r["reach"] = ply
        if r["reach"] is not None and r["after"] is None:
            # what the reaching player does next
            if e[0] == "done" and e[1] == hp:
                r["after"] = ("cut" if not passed(pos[v]) else "declared after pass", ply)
            elif e[0] == "move" and e[1] == h and pos[h] == E:
                r["after"] = ("gave way", ply)
            elif e[0] == "move" and B.owner[e[1]] == hp and pos[e[1]] == B.tgt[e[1]]:
                r["after"] = ("other own object off target", ply)
            elif e[0] == "end":
                r["after"] = ("held to end", ply)
        if e[0] == "move" and e[1] == h:
            nmoves_h += 1
        if e[0] == "end":
            r["ply_end"] = ply; r["v_home_end"] = pos[v] == B.tgt[v]
    r["dmin_h"] = B.dist(B.start[h], E, set())
    r["dmin_v"] = B.dist(B.start[v], B.tgt[v], set())
    # first cell on v's side reached: distance from v's start to the first passed cell
    r["order"] = ("h never arrived" if r["first_arr"] is None
                  else "v first" if r["pass_before_arr"] else "h first")
    ends = [x for x in (r["first_pass"], r["first_arr"]) if x is not None]
    r["win_end"] = min(ends) if ends else r["ply_end"]
    return r


# ---------------------------------------------------------------- holder waiting
def waiting(rec, ins, ev, static_pairs=None):
    """Decisions of a player whose object h sits 1-2 cells from target with a free
    progress step available. pending = h's target is a door for some pending v.

    static_pairs: {h: [v,...]} -> pending iff some v has not passed (walls-only test).
    else dynamic: t_h separates some not-home v (other owner, not done) from its
    target given walls + frozen objects.
    """
    B = Board(ins); out = []
    labs = {h: comp(B.n, B.walls, {B.tgt[h]}) for h in (static_pairs or {})}
    for e, ply, pos, done in replay(ins, ev):
        if e[0] != "move":
            continue
        p = B.owner[e[1]]
        frozen = {pos[o] for o in B.owner if B.owner[o] in done}
        occ = set(pos.values())
        for h in B.mine[p]:
            d = B.dist(pos[h], B.tgt[h], frozen)
            if d not in (1, 2):
                continue
            prog = []
            for dr, dc in DIRS:
                q = (pos[h][0] + dr, pos[h][1] + dc)
                if 0 <= q[0] < B.n and 0 <= q[1] < B.n and q not in B.walls and q not in occ:
                    dq = B.dist(q, B.tgt[h], frozen)
                    if dq is not None and dq < d:
                        prog.append(q)
            if not prog:
                continue
            if static_pairs is not None:
                if h not in static_pairs:
                    pend = None  # not a static holder
                else:
                    lab = labs[h]
                    pend = any(not (lab.get(pos[v]) is not None and lab.get(pos[v]) == lab.get(B.tgt[v]))
                               for v in static_pairs[h])
            else:
                before = comp(B.n, B.walls, frozen)
                after = comp(B.n, B.walls, frozen | {B.tgt[h]})
                pend = any(B.owner[v] != p and B.owner[v] not in done and pos[v] != B.tgt[v]
                           and pos[v] != B.tgt[h]
                           and before.get(pos[v]) == before.get(B.tgt[v]) is not None
                           and after.get(pos[v]) != after.get(B.tgt[v])
                           for v in B.tgt)
            if e[1] == h:
                dr, dc = STEP[e[2]]
                q = (pos[h][0] + dr, pos[h][1] + dc)
                dq = B.dist(q, B.tgt[h], frozen)
                kind = "progress" if dq is not None and dq < d else "h non-progress"
            else:
                kind = "other object"
            out.append(dict(h=h, d=d, pend=pend, kind=kind, ply=ply, p=p))
    return out


# ---------------------------------------------------------------- arrivals on live doors (all levels)
def arrivals(rec, ins, ev):
    """Each move that puts h on its target: does t_h (given frozen set) cut a pending v?"""
    B = Board(ins); out = []
    for e, ply, pos, done in replay(ins, ev):
        if e[0] != "move":
            continue
        h = e[1]; dr, dc = STEP[e[2]]
        q = (pos[h][0] + dr, pos[h][1] + dc)
        if q != B.tgt[h]:
            continue
        frozen = {pos[o] for o in B.owner if B.owner[o] in done}
        before = comp(B.n, B.walls, frozen); after = comp(B.n, B.walls, frozen | {q})
        p = B.owner[h]
        vic = [v for v in B.tgt if B.owner[v] != p and B.owner[v] not in done and pos[v] != B.tgt[v]
               and v != h and pos[v] != q
               and before.get(pos[v]) is not None and before.get(pos[v]) == before.get(B.tgt[v])
               and after.get(pos[v]) != after.get(B.tgt[v])]
        # would this arrival make the player done-eligible?
        np_ = dict(pos); np_[h] = q
        elig = all(np_[o] == B.tgt[o] for o in B.mine[p])
        out.append(dict(h=h, vic=vic, elig=elig, ply=ply + 1))
    return out


def episodes(levels=None, suites=("3p", "6p")):
    eps = json.load(open(os.path.join(REPO, f"analysis/qualitative_analysis/episodes.{ARM}.json")))
    return [e for e in eps if e["model"] in STRONG and e["suite"] in suites
            and e["proto"] in PROTOS and (levels is None or e["level"] in levels)]


def fisher(a, b, c, d):
    from math import comb
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def p(x):
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9))


if __name__ == "__main__":
    pass
