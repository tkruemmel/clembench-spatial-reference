"""d x relievable split of free pinches, strong tier, none vs message arms.  python3 dsplit.py"""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from followup import ps, STRONG, M1, ok, clogit, predict, poibin_p
for lab, protos in (("none", ("none",)), ("msg arms", ("structured", "freeform", "hybrid"))):
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    b, _ = clogit(fit, M1)
    pin = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"]]
    print(lab)
    for dl, df in (("d=1", lambda d: d == 1), ("d=2", lambda d: d == 2), ("d=3", lambda d: d == 3), ("d=4-5", lambda d: d >= 4)):
        for rl in (False, True):
            rs = [r for r in pin if (lambda x: df(x["d"]) and x["rel"] == rl)([x for x in r["alts"] if x["kill"]][0])]
            if not rs:
                print("  %-6s %-14s --" % (dl, "relievable" if rl else "fatal")); continue
            pr = [predict(b, M1, r) for r in rs]
            o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rs)
            print("  %-6s %-14s %2d/%-2d (%3.0f%%) exp %3.0f%%  p=%.2g" % (dl, "relievable" if rl else "fatal", o, len(rs), 100*o/len(rs), 100*sum(pr)/len(rs), poibin_p(o, pr)))
