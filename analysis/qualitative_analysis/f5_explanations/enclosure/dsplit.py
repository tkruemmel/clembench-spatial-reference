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

# Turn horizon (exploratory; the d split was chosen after the data were seen). Message arms, fatal pinches only:
# seal played when the victim moves next or one player later (d<=2) vs two players later (d=3).
from scipy.stats import fisher_exact
seal = lambda r: [x for x in r["alts"] if x["kill"]][0]
took = lambda r: bool(r["alts"][r["chosen"]]["kill"])
fatal = [r for r in ps if r["model"] in STRONG and r["proto"] != "none" and r["pin"] and r["pin"]["free"] and not seal(r)["rel"]]
near, d3 = [r for r in fatal if seal(r)["d"] <= 2], [r for r in fatal if seal(r)["d"] == 3]
a, b = sum(map(took, near)), sum(map(took, d3))
print("turn horizon, msg arms, fatal pinches: seal played d<=2 %d/%d vs d=3 %d/%d  Fisher p=%.2g"
      % (a, len(near), b, len(d3), fisher_exact([[a, len(near) - a], [b, len(d3) - b]])[1]))
t3 = [r for r in d3 if took(r)]
print("  d=3 fatal seals played %d, episode aborted %d, episodes %d; prose %d"
      % (len(t3), sum(r["outcome"] == "abort" for r in t3), len({r["path"] for r in t3}),
         sum(r["proto"] in ("freeform", "hybrid") for r in t3)))
for r in t3:  # the mover's messages, for the hand reading of the prose cases
    if r["proto"] in ("freeform", "hybrid"):
        print("   %s %s %s %s ply %d mover %s victim %s | prev: %s | cur: %s" % (
            r["model"], r["level"], r["proto"], r["inst"], r["ply"], r["player"], seal(r)["kill"][0],
            " ".join((r["prev"] or "").split())[:400], " ".join((r["cur"] or "").split())[:400]))
