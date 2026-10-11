"""Residual (obs vs M1) at free pinches, split by whether the would-be victim stands in the mover's own way.  python3 inway.py"""
import io, contextlib, json, collections
with contextlib.redirect_stdout(io.StringIO()):
    from decomp import M1, M0, ok, clogit, predict, ps, STRONG
    from analysis import poibin_p
import plies as P
L = "ABCDEF"


def inway(r):
    ins = json.load(open(P.sa.REPO + "/" + r["path"] + "/instance.json"))
    n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
    s = [x for x in r["alts"] if x["kill"]][0]
    vc = r["pos"][ins["player_objects"][L.index(s["kill"][0])][0]]
    d = P.obj_dist(walls, n, r["target"], frozenset({vc}))
    # with only the victim as an obstacle, the mover's best route gets longer or is cut: sealing it is self-defeating
    dw = P.sa.dist_map(walls, n, r["target"])
    return d.get(r["pos"][r["obj"]], 999) > dw[r["pos"][r["obj"]]]


for lab, protos in (("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    out = []
    for name, F in (("M0", M0), ("M1", M1)):
        b, _ = clogit(fit, F)
        for sub in (True, False):
            rs = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"] and inway(r) == sub]
            pr = [predict(b, F, r) for r in rs]
            o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rs)
            out.append("%s victim-in-way=%s: obs %d/%d (%.0f%%) exp %.0f%% p=%.2g" % (name, sub, o, len(rs), 100 * o / len(rs), 100 * sum(pr) / len(rs), poibin_p(o, pr)))
    print(lab); [print("   " + x) for x in out]
