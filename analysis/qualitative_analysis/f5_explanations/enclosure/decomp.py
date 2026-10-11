"""Decompose the pinch decline rate: progress-only vs + generic X-adjacency vs observed.
Models fitted on NON-pinch plies only, then used to predict the pinch choices.  python3 decomp.py"""
import collections
import numpy as np
from scipy import stats
import plies as P
from robust import ps, clogit, minK, costly, STRONG
from analysis import poibin_p

ok = lambda r: r["K"] >= 2 and not r["immobile"] and not r["s4open"]
freep = lambda r: r["pin"] and r["pin"]["free"]
M0 = {"gain(walls)": lambda r, x: x["gain"], "gain*dsv41": lambda r, x: x["gain"] and r["model"] == "dsv41",
      "gain(objects)": lambda r, x: x["dobj"] < 0, "gain(objects)*dsv41": lambda r, x: x["dobj"] < 0 and r["model"] == "dsv41"}
M1 = dict(M0, **{"adjacent to any X": lambda r, x: bool(x["adjX"]), "#X adjacent": lambda r, x: len(x["adjX"]),
                 "own moves after": lambda r, x: x["own_after"]})


def predict(b, feats, r):
    X = np.array([[f(r, x) for f in feats.values()] for x in r["alts"]], float)
    u = X @ b; p = np.exp(u - u.max()); p /= p.sum()
    return sum(pi for pi, x in zip(p, r["alts"]) if x["kill"])


print("free pinches: observed seals vs expected from choice models fitted on non-pinch plies of the same tier x protocol group")
print("(M0 = wall + object-aware progress only = 'blind'; M1 = M0 + generic aversion to stepping next to any X + own mobility)")
for tier, models in (("strong", STRONG), ("weak", ("nemotron",))):
    for lab, protos in (("none", ("none",)), ("structured", ("structured",)), ("freeform", ("freeform",)), ("hybrid", ("hybrid",)),
                        ("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
        pin = [r for r in ps if r["model"] in models and r["proto"] in protos and freep(r)]
        if not pin:
            continue
        fit = [r for r in ps if r["model"] in models and r["proto"] in protos and ok(r) and not r["pin"]]
        out = []
        for name, F in (("M0", M0), ("M1", M1)):
            F = dict(F)
            if tier == "weak":
                F = {k: v for k, v in F.items() if "dsv41" not in k}
            b, _ = clogit(fit, F)
            for sub, f in (("costly", lambda r: r["pin"]["costly"]), ("all-free", lambda r: True)):
                rs = [r for r in pin if f(r)]
                pr = [predict(b, F, r) for r in rs]
                o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rs)
                out.append((name, sub, o, len(rs), sum(pr), poibin_p(o, pr)))
        print("  %-6s %-10s " % (tier, lab) + " | ".join(
            "%s %s: obs %d/%d (%.0f%%) exp %.1f (%.0f%%) p=%.2g" % (nm, sub, o, n, 100 * o / n, e, 100 * e / n, p)
            for nm, sub, o, n, e, p in out if sub == "costly") )
        print("  %-6s %-10s " % ("", "") + " | ".join(
            "%s %s: obs %d/%d (%.0f%%) exp %.1f (%.0f%%) p=%.2g" % (nm, sub, o, n, 100 * o / n, e, 100 * e / n, p)
            for nm, sub, o, n, e, p in out if sub == "all-free"))

# protocol interaction on the lethal-specific term, strong tier, all plies
print("\nstrong tier: lethal-specific term (adjacent X with k=1, over and above generic adjacency) x prose, board-clustered")
rows = [r for r in ps if r["model"] in STRONG and ok(r)]
pr = lambda r: r["proto"] in ("freeform", "hybrid")
F = dict(M1, **{"k=1": lambda r, x: minK(x) == 1, "k=1*prose": lambda r, x: minK(x) == 1 and pr(r),
                "adj*prose": lambda r, x: bool(x["adjX"]) and pr(r), "gain*prose": lambda r, x: x["gain"] and pr(r),
                "gainobj*prose": lambda r, x: x["dobj"] < 0 and pr(r)})
b, V = clogit(rows, F)
se = np.sqrt(np.diag(V))
for k, bb, s in zip(F, b, se):
    print("   %-22s OR %.2f [%.2f,%.2f] z=%.2f" % (k, np.exp(bb), np.exp(bb - 1.96 * s), np.exp(bb + 1.96 * s), bb / s))

# how many board-visible k=1 moves are next to a done (inactive) X
c = collections.Counter()
for r in rows:
    for x in r["alts"]:
        if minK(x) == 1:
            c[("active" if any(k == 1 and not d for k, d, *_ in x["adjX"]) else "done-only", r["chosen"] == r["alts"].index(x))] += 1
print("\nk=1 alternatives (strong, analysed plies): %s" % dict(c))
