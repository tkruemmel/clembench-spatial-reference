"""Test 1 by d and the requested k1 x next / k1 x relievable model, none arm vs message arms.  python3 followup4.py"""
import collections, io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from followup import ps, STRONG, M1, ok, clogit, predict, poibin_p, F1
for lab, protos in (("none", ("none",)), ("msg arms", ("structured", "freeform", "hybrid"))):
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    b, _ = clogit(fit, M1)
    pin = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"]]
    out = []
    for name, f in (("d=1", lambda x: x["d"] == 1), ("d=2", lambda x: x["d"] == 2), ("d=3", lambda x: x["d"] == 3), ("d=4-5", lambda x: x["d"] >= 4),
                    ("relievable", lambda x: x["rel"]), ("not relievable", lambda x: not x["rel"])):
        rs = [r for r in pin if f([x for x in r["alts"] if x["kill"]][0])]
        pr = [predict(b, M1, r) for r in rs]
        o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rs)
        out.append("%s %d/%d (%.0f%%) exp %.0f%%" % (name, o, len(rs), 100 * o / len(rs), 100 * sum(pr) / len(rs)))
    print("%-9s " % lab + " | ".join(out))
    rows = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r)]
    b, V = clogit(rows, F1); se = np.sqrt(np.diag(V)); ks = list(F1)
    print("          " + " | ".join("%s OR %.2f [%.2f, %.2f]" % (k, np.exp(b[ks.index(k)]), np.exp(b[ks.index(k)] - 1.96 * se[ks.index(k)]),
                                                            np.exp(b[ks.index(k)] + 1.96 * se[ks.index(k)])) for k in ks[-4:-1]))
