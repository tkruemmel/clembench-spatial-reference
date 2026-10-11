"""Robustness checks for analysis.py.  python3 robust.py"""
import collections, random, re
import numpy as np
from scipy import stats
import plies as P

STRONG = ("qwen38", "dsv41")
ps = P.load()
for r in ps:
    r["pin"] = P.derive(r)
    r["K"] = len(r["alts"])
    r["ng"] = sum(x["gain"] for x in r["alts"])
    r["s4open"] = r["level"] == "S4" and r["ply"] == 0
costly = lambda r: r["pin"] and r["pin"]["free"] and r["pin"]["costly"]
base = lambda r: r["K"] >= 2 and not r["pin"] and not r["immobile"] and not r["s4open"]
G = lambda r: [x for x in r["alts"] if x["gain"]][0]
played_gain = lambda r: r["alts"][r["chosen"]]["gain"]

# ---- A. is the sealing move's "progress" real once the other objects are counted as obstacles? ----
print("A. unique-progress plies, strong tier: P(play gaining move) split by whether it also gains with objects as obstacles")
for lab, f in (("obj-aware gain", lambda r: G(r)["dobj"] < 0), ("blocked/not real", lambda r: G(r)["dobj"] >= 0)):
    c = [r for r in ps if r["model"] in STRONG and costly(r) and f(r)]
    b = [r for r in ps if r["model"] in STRONG and base(r) and r["ng"] == 1 and f(r)]
    cp, bp = sum(map(played_gain, c)), sum(map(played_gain, b))
    print("   %-16s pinch %d/%d (%.0f%%)  baseline %d/%d (%.0f%%)  Fisher p=%.2g"
          % (lab, cp, len(c), 100 * cp / max(1, len(c)), bp, len(b), 100 * bp / len(b),
             stats.fisher_exact([[cp, len(c) - cp], [bp, len(b) - bp]])[1]))

# ---- B. level- and K-matched baseline, and per model ----
print("\nB. costly pinches vs unique-progress baseline matched on model x protocol x level x K (indirect standardisation)")
num, den = collections.Counter(), collections.Counter()
for r in ps:
    if base(r) and r["ng"] == 1:
        k = (r["model"], r["proto"], r["level"], r["K"])
        den[k] += 1; num[k] += played_gain(r)
for lab, protos in (("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    for m in STRONG + (STRONG,):
        c = [r for r in ps if r["model"] in (m if isinstance(m, tuple) else (m,)) and r["proto"] in protos and costly(r)]
        e = sum(num[(r["model"], r["proto"], r["level"], r["K"])] / den[(r["model"], r["proto"], r["level"], r["K"])] for r in c)
        o = sum(map(played_gain, c))
        print("   %-6s %-14s taken %d/%d (%.0f%%)  expected %.1f (%.0f%%)" % (lab, m if isinstance(m, str) else "strong", o, len(c), 100 * o / len(c), e, 100 * e / len(c)))

# ---- C. board-cluster bootstrap of taken-rate minus expected (costly, strong) ----
print("\nC. board-clustered bootstrap (2000x) of costly pinch take-rate minus matched baseline (model x proto x K), strong tier")
num2, den2 = collections.Counter(), collections.Counter()
for r in ps:
    if base(r) and r["ng"] == 1:
        k = (r["model"], r["proto"], r["K"]); den2[k] += 1; num2[k] += played_gain(r)
random.seed(1)
for lab, protos in (("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    c = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and costly(r)]
    byb = collections.defaultdict(list)
    for r in c:
        byb[(r["level"], r["inst"])].append((played_gain(r), num2[(r["model"], r["proto"], r["K"])] / den2[(r["model"], r["proto"], r["K"])]))
    boards = list(byb)
    stat = lambda bs: (lambda xs: np.mean([a for a, _ in xs]) - np.mean([e for _, e in xs]))([x for b in bs for x in byb[b]])
    boot = sorted(stat([random.choice(boards) for _ in boards]) for _ in range(2000))
    print("   %-6s n=%d on %d boards: diff %.2f  95%% CI [%.2f, %.2f]" % (lab, len(c), len(boards), stat(boards), boot[50], boot[1949]))
# prose - rigid difference in take rate at costly pinches (board bootstrap)
cr = [r for r in ps if r["model"] in STRONG and costly(r)]
byb = collections.defaultdict(list)
for r in cr:
    byb[(r["level"], r["inst"])].append(r)
boards = list(byb)
def dd(bs):
    xs = [r for b in bs for r in byb[b]]
    a = [played_gain(r) for r in xs if r["proto"] in ("freeform", "hybrid")]
    b = [played_gain(r) for r in xs if r["proto"] not in ("freeform", "hybrid")]
    return (np.mean(a) if a else np.nan) - (np.mean(b) if b else np.nan)
boot = sorted(x for x in (dd([random.choice(boards) for _ in boards]) for _ in range(2000)) if x == x)
print("   prose - rigid take-rate difference at costly pinches: %.2f  95%% CI [%.2f, %.2f]" % (dd(boards), boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]))
print("   costly pinches per board (strong): %s" % sorted(collections.Counter({b: len(v) for b, v in byb.items()}).values(), reverse=True)[:10])

# ---- D. extended conditional logit: board-visible neighbourhood, own mobility, object-aware progress ----


def clogit(rows, feats):
    X = [np.array([[f(r, x) for f in feats.values()] for x in r["alts"]], float) for r in rows]
    y = [r["chosen"] for r in rows]
    cl = [(r["level"], r["inst"]) for r in rows]              # cluster by BOARD (conservative)
    b = np.zeros(len(feats))
    for _ in range(60):
        g, H = np.zeros_like(b), np.zeros((len(b), len(b)))
        for Xi, yi in zip(X, y):
            u = Xi @ b; p = np.exp(u - u.max()); p /= p.sum()
            xb = p @ Xi; g += Xi[yi] - xb; D = Xi - xb; H -= (D * p[:, None]).T @ D
        st = np.linalg.solve(H, g); b -= st
        if np.abs(st).max() < 1e-9:
            break
    S = collections.defaultdict(lambda: np.zeros_like(b))
    for Xi, yi, c in zip(X, y, cl):
        u = Xi @ b; p = np.exp(u - u.max()); p /= p.sum(); S[c] += Xi[yi] - p @ Xi
    Hi = np.linalg.inv(H)
    V = Hi @ sum(np.outer(s, s) for s in S.values()) @ Hi
    return b, V


minK = lambda x: min((k for k, *_ in x["adjX"]), default=0)
FE = {
    "gain(walls)": lambda r, x: x["gain"],
    "gain*dsv41": lambda r, x: x["gain"] and r["model"] == "dsv41",
    "gain(objects too)": lambda r, x: x["dobj"] < 0,
    "adj X, its last cell (k=1)": lambda r, x: minK(x) == 1,
    "adj X, k=2": lambda r, x: minK(x) == 2,
    "adj X, k>=3": lambda r, x: minK(x) >= 3,
    "#X adjacent": lambda r, x: len(x["adjX"]),
    "own moves after": lambda r, x: x["own_after"],
}
print("\nD. conditional logit, board-visible features (k = free neighbours of the adjacent X before the move; done X's included,")
print("   since the board does not mark them), controls: object-aware progress, #adjacent X, mover's own mobility after. SE clustered by board.")
for lab, protos in (("none", ("none",)), ("structured", ("structured",)), ("freeform", ("freeform",)), ("hybrid", ("hybrid",)),
                    ("rigid", ("none", "structured")), ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))):
    for tier, models in (("strong", STRONG), ("weak", ("nemotron",))):
        rows = [r for r in ps if r["model"] in models and r["proto"] in protos and r["K"] >= 2 and not r["immobile"] and not r["s4open"]]
        if not rows:
            continue
        fe = dict(FE)
        if tier == "weak":
            fe.pop("gain*dsv41")
        if not any(minK(x) == 1 for r in rows for x in r["alts"]):
            fe.pop("adj X, its last cell (k=1)")
        b, V = clogit(rows, fe)
        se = np.sqrt(np.diag(V))
        ks = list(fe)
        out = ["%s %.2f [%.2f,%.2f]" % (k, np.exp(bb), np.exp(bb - 1.96 * s), np.exp(bb + 1.96 * s)) for k, bb, s in zip(ks, b, se) if k != "gain*dsv41"]
        extra = ""
        if "adj X, its last cell (k=1)" in ks:
            i, j = ks.index("adj X, its last cell (k=1)"), ks.index("adj X, k=2")
            z = (b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
            extra = " || k=1 vs k=2: ratio %.2f z=%.2f" % (np.exp(b[i] - b[j]), z)
        print("  %-6s %-10s n=%5d  %s%s" % (tier, lab, len(rows), " | ".join(out), extra))
