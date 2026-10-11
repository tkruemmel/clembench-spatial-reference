"""Separate turn logic from board look: pocket (victim walled in) vs crowd (victim also next to other X's),
and within crowd, relievable (an adjacent X moves before the victim) vs not (same look).  python3 followup2.py"""
import collections, io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from followup import ps, STRONG, M1, ok, clogit, predict, poibin_p, sa, GROUPS, relieved
for r in ps:
    for x in r["alts"]:
        if x["kill"]:
            v = x["kill"][0]
            nb = sa.step(r["pos"], *x["m"])
            vc = nb[v]
            occ = {c: o for o, c in nb.items()}
            others = [occ[(vc[0] + a, vc[1] + b)] for a, b in sa.STEP.values()
                      if (vc[0] + a, vc[1] + b) in occ and occ[(vc[0] + a, vc[1] + b)] != x["m"][0]]
            x["shape"] = "pocket" if not others else ("crowd-relievable" if x["rel"] else "crowd-unrelievable")
SH = ("pocket", "crowd-unrelievable", "crowd-relievable")
print("free pinches, strong tier: seal taken, obs vs M1-expected; [victim actually had a move at its turn, taken seals]")
for lab, protos in (("none", ("none",)), ("structured", ("structured",)), ("prose", ("freeform", "hybrid")), ("all", GROUPS[-1][1])):
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    b, _ = clogit(fit, M1)
    pin = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"]]
    out = []
    for s in SH:
        rs = [r for r in pin if [x for x in r["alts"] if x["kill"]][0]["shape"] == s]
        if not rs:
            out.append("%s --" % s); continue
        pr = [predict(b, M1, r) for r in rs]
        tk = [r for r in rs if r["alts"][r["chosen"]]["kill"]]
        out.append("%-19s %2d/%-2d (%2.0f%%) exp %2.0f%% p=%.2g  [relieved %d/%d]" % (s, len(tk), len(rs), 100 * len(tk) / len(rs),
                   100 * sum(pr) / len(rs), poibin_p(len(tk), pr), sum(relieved(r) for r in tk), len(tk)))
    print("  %-10s " % lab + "\n             ".join(out))
print("\nconditional logit, strong tier, M1 controls, board-clustered; reference = generic adjacency (seal dummies are on top of it)")
for lab, protos in (("none", ("none",)), ("structured+prose", ("structured", "freeform", "hybrid")), ("all", GROUPS[-1][1])):
    rows = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r)]
    F = dict(M1, **{s: (lambda s: lambda r, x: bool(x["kill"]) and x["shape"] == s)(s) for s in SH})
    F["k1 done X"] = lambda r, x: any(k == 1 for k, *_ in x["adjX"]) and not x["kill"]
    b, V = clogit(rows, F); se = np.sqrt(np.diag(V)); ks = list(F)
    n = collections.Counter(x["shape"] for r in rows for x in r["alts"] if x["kill"])
    print("  %-16s " % lab + " | ".join("%s (n=%d) %.2f [%.2f,%.2f]" % (s, n[s], np.exp(b[ks.index(s)]), np.exp(b[ks.index(s)] - 1.96 * se[ks.index(s)]),
                                                                  np.exp(b[ks.index(s)] + 1.96 * se[ks.index(s)])) for s in SH))
    i, j = ks.index("crowd-unrelievable"), ks.index("crowd-relievable")
    print("  %-16s crowd-unrelievable vs crowd-relievable: ratio %.2f z=%.2f" % ("", np.exp(b[i] - b[j]), (b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])))
