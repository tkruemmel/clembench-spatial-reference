"""Placebo: last free cell of a DONE X (harmless, board looks identical) vs of an ACTIVE X.  python3 placebo.py"""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from decomp import M1, ok, clogit, minK, ps, STRONG
import numpy as np
act = lambda x: any(k == 1 and not d for k, d, *_ in x["adjX"])
don = lambda x: minK(x) == 1 and not act(x)
for lab, protos in (("none", ("none",)), ("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    rows = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r)]
    F = dict(M1, **{"k=1 ACTIVE X": lambda r, x: act(x), "k=1 DONE X only": lambda r, x: don(x)})
    b, V = clogit(rows, F); se = np.sqrt(np.diag(V)); ks = list(F)
    n_act = sum(act(x) for r in rows for x in r["alts"]); n_don = sum(don(x) for r in rows for x in r["alts"])
    print("%-6s " % lab + " | ".join("%s OR %.2f [%.2f,%.2f]" % (k, np.exp(b[i]), np.exp(b[i] - 1.96 * se[i]), np.exp(b[i] + 1.96 * se[i]))
                                   for i, k in enumerate(ks) if k.startswith("k=1") or k == "adjacent to any X"),
          " (alternatives: active %d, done %d)" % (n_act, n_don))
