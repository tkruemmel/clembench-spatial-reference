"""Every accepted 6p move, with its full choice set annotated.

Replays boards with seal_averted's own helpers (legal/step/dist_map/stream) so
`pinch`, `taken`, `costly` are identical to seal_averted.scan -- asserted per
episode in check().
"""
import sys
sys.dont_write_bytecode = True
QA = "/Users/tom/uni/clembench-spatial-reference/analysis/qualitative_analysis"
sys.path.insert(0, QA)
import json, os, pickle, string, collections
import seal_averted as sa
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"plies.{ARM}.pkl")


def obj_dist(walls, n, target, others):
    """BFS to target treating walls AND the other objects' current cells as blocked.
    An occupied target is unreachable for now (empty map -> every move scores 0)."""
    if target in others:
        return {}
    seen, qq = {target: 0}, collections.deque([target])
    while qq:
        c = qq.popleft()
        for dr, dc in sa.STEP.values():
            nb = (c[0] + dr, c[1] + dc)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls and nb not in others and nb not in seen:
                seen[nb] = seen[c] + 1
                qq.append(nb)
    return seen


def plies(rec):
    d = os.path.join(sa.REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
    groups = ins["player_objects"]
    L = string.ascii_uppercase
    owner = {o: L[i] for i, g in enumerate(groups) for o in g}
    mine = {L[i]: list(g) for i, g in enumerate(groups)}
    seat = {"Player %d" % (i + 1): L[i] for i in range(len(groups))}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    done, ply, out = set(), 0, []
    said = collections.defaultdict(list)

    def counts(board):
        occ = set(board.values())
        return {q: len(sa.legal(board, mine[q], walls, n, occ)) for q in mine if q not in done}

    for kind, a, b in sa.stream(it):
        if kind == "msg":
            p = seat.get(a)
            if p:
                said[p].append((ply, b))
            continue
        if kind == "done":
            done.add(a)
            continue
        p = owner[a]
        moves = sa.legal(pos, mine[p], walls, n)
        before = counts(pos)
        mates = {q: c for q, c in before.items() if q != p}
        alts = []
        for m in moves:
            nb = sa.step(pos, *m)
            after = counts(nb)
            dm = sa.dist_map(walls, n, tgt[m[0]])
            delta = dm[nb[m[0]]] - dm[pos[m[0]]]
            red = [q for q in mates if after[q] < mates[q]]
            occ = set(nb.values())
            dest = nb[m[0]]
            # board-visible neighbourhood of the destination: every other object (done or not)
            adjX = []
            for dr, dc in sa.STEP.values():
                c = (dest[0] + dr, dest[1] + dc)
                for o2, p2 in pos.items():
                    if p2 == c and o2 != m[0]:
                        adjX.append((sum(1 for e in sa.STEP.values()
                                         if 0 <= c[0] + e[0] < n and 0 <= c[1] + e[1] < n
                                         and (c[0] + e[0], c[1] + e[1]) not in walls
                                         and (c[0] + e[0], c[1] + e[1]) not in set(pos.values())),
                                     owner[o2] in done, o2))
            dobj = obj_dist(walls, n, tgt[m[0]], frozenset(v for k2, v in pos.items() if k2 != m[0]))
            alts.append(dict(
                m=m, dest=nb[m[0]], delta=delta, gain=delta < 0,
                kill=sorted(q for q in mates if mates[q] > 0 and after[q] == 0),
                red=len(red),                                   # teammates losing an option
                to1=any(after[q] == 1 and mates[q] >= 2 for q in red),
                redk=[(mates[q], after[q]) for q in red],
                adjX=adjX, own_after=len(sa.legal(nb, [m[0]], walls, n, occ)),
                dobj=(dobj.get(dest, 999) - dobj.get(pos[m[0]], 999))))
        prev = [t for k, t in said.get(p, []) if k < ply]
        cur = [t for k, t in said.get(p, []) if k == ply]
        out.append(dict(
            model=rec["model"], proto=rec["proto"], level=rec["level"], inst=rec["inst"],
            path=rec["path"], outcome=rec["outcome"], ply=ply, player=p, obj=a,
            pos=dict(pos), target=tgt[a], alts=alts, played=(a, b),
            chosen=[x["m"] for x in alts].index((a, b)),
            immobile=any(c == 0 for c in mates.values()),     # some active teammate already stuck
            done=sorted(done), owner=owner, mine=mine, walls=walls, n=n,
            prev=prev[-1] if prev else None, cur=cur[-1] if cur else None))
        pos = sa.step(pos, a, b)
        ply += 1
    return out


def derive(r):
    """seal_averted's pinch flags, recomputed from the annotated ply."""
    seals = [x for x in r["alts"] if x["kill"]]
    if not seals:
        return None
    safe = [x for x in r["alts"] if not x["kill"]]
    return dict(taken=bool(r["alts"][r["chosen"]]["kill"]), free=bool(safe),
                costly=any(x["gain"] for x in seals) and not any(x["gain"] for x in safe))


def build():
    eps = [e for e in json.load(open(os.path.join(QA, f"episodes.{ARM}.json"))) if e["suite"] == "6p"]
    allp = []
    for rec in eps:
        ps = plies(rec)
        # definitional identity with seal_averted.scan, per episode
        mine = {(r["ply"]): derive(r) for r in ps if derive(r)}
        theirs = {r["ply"]: dict(taken=r["taken"], free=bool(r["safe"]), costly=r["costly"])
                  for r in sa.scan(rec)}
        assert mine == theirs, (rec["path"], mine, theirs)
        allp.extend(ps)
    pickle.dump(allp, open(OUT, "wb"))
    return allp


def load():
    return pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else build()


if __name__ == "__main__":
    ps = build()
    pin = [r for r in ps if derive(r)]
    strong = [r for r in pin if r["model"] in ("qwen38", "dsv41")]
    pr = [r for r in strong if r["proto"] in ("freeform", "hybrid")]
    rg = [r for r in strong if r["proto"] in ("none", "structured")]
    f = lambda rs: [derive(r) for r in rs if derive(r)["free"]]
    # strong tier, prose then none+structured: free pinches, declined, costly, costly declined
    # (= seal_averted.py --tier strong [--proto none structured]; original arm is seal_averted.md's headline)
    got = [(len(f(x)), sum(not d["taken"] for d in f(x)), sum(d["costly"] for d in f(x)),
            sum(d["costly"] and not d["taken"] for d in f(x))) for x in (pr, rg)]
    EXPECT = {"original": [(73, 43, 46, 20), (106, 62, 72, 32)],
              "updated": [(88, 58, 41, 16), (133, 77, 87, 39)]}
    assert got == EXPECT[ARM], got
    print("plies", len(ps), "pinches", len(pin), "strong check ok", got)
