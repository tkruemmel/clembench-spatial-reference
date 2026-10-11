"""Pure turn-order contrast: victim's adjacent X can move but its owner moves AFTER the victim (unrelievable by turn
order alone) vs an adjacent X whose owner moves BEFORE the victim (relievable). Same board look.  python3 followup3.py"""
import collections, io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from followup2 import ps, STRONG, M1, ok, clogit, predict, poibin_p, sa, relieved
    from followup import after


def cat(r, x):
    v = x["kill"][0]
    nb = sa.step(r["pos"], *x["m"])
    vc = nb[v]
    occ = {c: o for o, c in nb.items()}
    adj = [occ[(vc[0] + a, vc[1] + b)] for a, b in sa.STEP.values()
           if (vc[0] + a, vc[1] + b) in occ and occ[(vc[0] + a, vc[1] + b)] != x["m"][0]]
    if not adj:
        return "pocket"
    if x["rel"]:
        return "relievable"
    seq = after(r)
    late = [o for o in adj if o not in r["done"] and o in seq and seq.index(o) > seq.index(v)
            and sa.legal(nb, [o], r["walls"], r["n"])]
    return "unrelievable: turn order only" if late else "unrelievable: adjacent X done/stuck"


CATS = ("relievable", "unrelievable: turn order only", "unrelievable: adjacent X done/stuck", "pocket")
for r in ps:
    for x in r["alts"]:
        if x["kill"]:
            x["cat"] = cat(r, x)
print("free pinches, strong tier: seal taken obs vs M1-expected  [taken seals whose victim had a move at its turn]")
for lab, protos in (("none", ("none",)), ("structured", ("structured",)), ("prose", ("freeform", "hybrid")),
                    ("msg arms", ("structured", "freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    b, _ = clogit(fit, M1)
    pin = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"]]
    out = []
    for s in CATS[:3]:
        rs = [r for r in pin if [x for x in r["alts"] if x["kill"]][0]["cat"] == s]
        if not rs:
            out.append("%s --" % s); continue
        pr = [predict(b, M1, r) for r in rs]
        tk = [r for r in rs if r["alts"][r["chosen"]]["kill"]]
        out.append("%-36s %2d/%-2d (%2.0f%%) exp %2.0f%% p=%.2g [relieved %d/%d]" % (s, len(tk), len(rs), 100 * len(tk) / len(rs),
                   100 * sum(pr) / len(rs), poibin_p(len(tk), pr), sum(relieved(r) for r in tk), len(tk)))
    print("  %-10s " % lab + "\n             ".join(out))
print("\nlogit (M1 controls + k1 done X control), OR for each seal category, board-clustered")
for lab, protos in (("none", ("none",)), ("msg arms", ("structured", "freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    rows = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r)]
    F = dict(M1, **{s: (lambda s: lambda r, x: bool(x["kill"]) and x["cat"] == s)(s) for s in CATS})
    F["k1 done X"] = lambda r, x: any(k == 1 for k, *_ in x["adjX"]) and not x["kill"]
    b, V = clogit(rows, F); se = np.sqrt(np.diag(V)); ks = list(F)
    n = collections.Counter(x["cat"] for r in rows for x in r["alts"] if x["kill"])
    print("  %-9s " % lab + " | ".join("%s (n=%d) %.2f [%.2f,%.2f]" % (s, n[s], np.exp(b[ks.index(s)]), np.exp(b[ks.index(s)] - 1.96 * se[ks.index(s)]),
                                                                np.exp(b[ks.index(s)] + 1.96 * se[ks.index(s)])) for s in CATS[:3]))
    i, j = ks.index(CATS[1]), ks.index(CATS[0])
    print("  %-9s turn-order-only vs relievable: ratio %.2f [%.2f, %.2f] z=%.2f" % ("", np.exp(b[i] - b[j]),
          np.exp(b[i] - b[j] - 1.96 * np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])), np.exp(b[i] - b[j] + 1.96 * np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])),
          (b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])))
# examples: message arms, turn-order-only, declined vs taken
print()
for r in ps:
    if r["model"] in STRONG and r["proto"] != "none" and r["pin"] and r["pin"]["free"]:
        x = [x for x in r["alts"] if x["kill"]][0]
        if x["cat"] in CATS[:2]:
            print("%s ply %d %s seal %s victim %s d=%d %s %s" % (r["path"], r["ply"], r["player"], x["m"], x["kill"][0], x["d"], x["cat"],
                  "TAKEN" if r["alts"][r["chosen"]] is x else "declined"))
