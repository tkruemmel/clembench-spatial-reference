"""Q4: prose arms, costly pinches -- what did the mover's previous message say about the sealing cell?  python3 q4.py [--show]"""
import re, sys, collections, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from robust import ps, costly, base, G, played_gain
import plies as P
from scipy import stats
cell = lambda c: P.sa.CELL % (c[0] + 1, c[1] + 1)
SENT = re.compile(r"[^.!?\n;]+")
# the sentence naming the cell says the mover will go there vs keep it clear / give it to someone
AVOID = re.compile(r"\b(?:clear|open|free|avoid|stay off|keep|leave|won'?t|not|vacat\w*|yours|for [A-F]\b|hold off|block)", re.I)


def tag(r):
    g = G(r)
    if not r["prev"]:
        return "no previous message"
    rx = re.compile(cell(g["dest"]), re.I)
    ss = [s for s in SENT.findall(r["prev"]) if rx.search(s)]
    if not ss:
        return "cell not named"
    return "named as keep-clear" if all(AVOID.search(s) for s in ss) else "named as own route"


for tier, models in (("strong", ("qwen38", "dsv41")), ("weak", ("nemotron",))):
    for kind, f in (("costly pinch", costly), ("non-pinch base", lambda r: base(r) and r["ng"] == 1)):
        t = collections.defaultdict(lambda: [0, 0])
        for r in ps:
            if r["model"] in models and r["proto"] in ("freeform", "hybrid") and f(r):
                k = tag(r); t[k][0] += played_gain(r); t[k][1] += 1
        print("%-6s %-15s " % (tier, kind) + " | ".join("%s: played %d/%d (%.0f%%)" % (k, a, n, 100 * a / n) for k, (a, n) in sorted(t.items())))
if "--show" in sys.argv:
    for r in ps:
        if r["model"] != "nemotron" and r["proto"] in ("freeform", "hybrid") and costly(r):
            g = G(r)
            print("\n%s ply %d %s seal->%s %s [%s]" % (r["path"].split("/")[-2:], r["ply"], r["player"], "R%dC%d" % (g["dest"][0] + 1, g["dest"][1] + 1),
                                                    "TAKEN" if played_gain(r) else "declined", tag(r)))
            print("   prev:", " ".join((r["prev"] or "")[:400].split()))
            print("   cur :", " ".join((r["cur"] or "")[:400].split()))
