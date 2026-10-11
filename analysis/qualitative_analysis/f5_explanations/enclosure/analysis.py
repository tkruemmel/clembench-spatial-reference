"""Blind or sensitive? Baselines for the enclosure decline rate.  python3 analysis.py"""
import collections, re
import numpy as np
from scipy import stats
import plies as P

TIERS = {"strong": ("qwen38", "dsv41"), "weak": ("nemotron",)}
GROUPS = [("none", ("none",)), ("structured", ("structured",)), ("freeform", ("freeform",)),
          ("hybrid", ("hybrid",)), ("rigid", ("none", "structured")),
          ("prose", ("freeform", "hybrid")), ("all", ("none", "structured", "freeform", "hybrid"))]

ps = P.load()
for r in ps:
    r["pin"] = P.derive(r)
    r["K"] = len(r["alts"])
    r["ng"] = sum(x["gain"] for x in r["alts"])
    r["s4open"] = r["level"] == "S4" and r["ply"] == 0


def base_ok(r):
    """Baseline plies: a real choice, no pinch, nobody already stuck (rescue motive), not the S4 opening."""
    return r["K"] >= 2 and not r["pin"] and not r["immobile"] and not r["s4open"]


def sel(tier, protos, f=lambda r: True):
    return [r for r in ps if r["model"] in TIERS[tier] and r["proto"] in protos and f(r)]


# ---------- per-move baseline rate q(stratum, class) from non-pinch plies ----------
q_num, q_den = collections.Counter(), collections.Counter()
for r in ps:
    if base_ok(r):
        for i, x in enumerate(r["alts"]):
            k = (r["model"], r["proto"], r["K"], r["ng"], x["gain"])
            q_den[k] += 1
            q_num[k] += i == r["chosen"]


def q(r, x):
    k = (r["model"], r["proto"], r["K"], r["ng"], x["gain"])
    return q_num[k] / q_den[k] if q_den[k] else float("nan")


def poibin_p(o, probs):
    """Exact two-sided p for observed count o under a Poisson-binomial(probs)."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1 - p, p])
    lo, hi = pmf[: o + 1].sum(), pmf[o:].sum()
    return min(1.0, 2 * min(lo, hi))


def oe(rows):
    """Observed seals played vs expected under 'blind' (seal move chosen at its class's baseline rate)."""
    probs = [sum(q(r, x) for x in r["alts"] if x["kill"]) for r in rows]
    o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rows)
    e = sum(probs)
    return o, len(rows), e, poibin_p(o, probs) if rows else float("nan")


def fmt_oe(t):
    o, n, e, p = t
    if not n:
        return "  --"
    return "%3d/%-3d %4.0f%% | exp %5.1f (%3.0f%%) O/E %.2f p=%.2g" % (o, n, 100 * o / n, e, 100 * e / n, o / e if e else 0, p)


costly = lambda r: r["pin"] and r["pin"]["free"] and r["pin"]["costly"]
free_gain = lambda r: r["pin"] and r["pin"]["free"] and not r["pin"]["costly"] and any(x["kill"] and x["gain"] for x in r["alts"])
free_lose = lambda r: r["pin"] and r["pin"]["free"] and not r["pin"]["costly"] and any(x["kill"] and not x["gain"] for x in r["alts"])
anyfree = lambda r: r["pin"] and r["pin"]["free"]

print("=" * 100)
print("Q1/Q2  seals played vs blind expectation (baseline = same model x protocol x #legal x #gaining x move class, non-pinch plies)")
for tier in TIERS:
    print("\n--", tier)
    for name, protos in GROUPS:
        print("  %-10s costly  %s" % (name, fmt_oe(oe(sel(tier, protos, costly)))))
        print("  %-10s fr-gain %s" % ("", fmt_oe(oe(sel(tier, protos, free_gain)))))
        print("  %-10s fr-lose %s" % ("", fmt_oe(oe(sel(tier, protos, free_lose)))))
        print("  %-10s ALLfree %s" % ("", fmt_oe(oe(sel(tier, protos, anyfree)))))
    # sensitivity: drop pinches where some teammate is already immobile
    print("  [excl. immobile-present pinches] all costly %s" % fmt_oe(oe(sel(tier, GROUPS[-1][1], lambda r: costly(r) and not r["immobile"]))))
    print("  [excl. immobile-present pinches] all free   %s" % fmt_oe(oe(sel(tier, GROUPS[-1][1], lambda r: anyfree(r) and not r["immobile"]))))

# ---------- Q1 transparent version: unique-progress plies, P(play the progress move), by K ----------
print("\n" + "=" * 100)
print("Q1  plies with exactly one gaining move: P(play it), pinch (it seals) vs non-pinch baseline, by #legal moves")
for tier in TIERS:
    for name, protos in GROUPS[-3:]:
        line = []
        for K in (2, 3, 4):
            b = sel(tier, protos, lambda r: base_ok(r) and r["ng"] == 1 and r["K"] == K)
            c = sel(tier, protos, lambda r: costly(r) and r["K"] == K)
            bp = sum(r["alts"][r["chosen"]]["gain"] for r in b)
            cp = sum(r["alts"][r["chosen"]]["gain"] for r in c)
            pf = stats.fisher_exact([[cp, len(c) - cp], [bp, len(b) - bp]])[1] if c else float("nan")
            line.append("K=%d pinch %2d/%-3d base %4d/%-5d (%2.0f%%) Fisher p=%.2g"
                        % (K, cp, len(c), bp, len(b), 100 * bp / max(1, len(b)), pf))
        print("  %-6s %-6s " % (tier, name) + " | ".join(line))

# ---------- conditional logit with episode-clustered SEs ----------


def clogit(rows, feats):
    X = [np.array([[f(r, x) for f in feats.values()] for x in r["alts"]], float) for r in rows]
    y = [r["chosen"] for r in rows]
    cl = [r["path"] for r in rows]
    b = np.zeros(len(feats))
    for _ in range(50):
        g, H = np.zeros_like(b), np.zeros((len(b), len(b)))
        for Xi, yi in zip(X, y):
            u = Xi @ b
            p = np.exp(u - u.max()); p /= p.sum()
            xb = p @ Xi
            g += Xi[yi] - xb
            D = Xi - xb
            H -= (D * p[:, None]).T @ D
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-9:
            break
    # sandwich, clustered by episode
    S = collections.defaultdict(lambda: np.zeros_like(b))
    for Xi, yi, c in zip(X, y, cl):
        u = Xi @ b
        p = np.exp(u - u.max()); p /= p.sum()
        S[c] += Xi[yi] - p @ Xi
    B = sum(np.outer(s, s) for s in S.values())
    Hi = np.linalg.inv(H)
    V = Hi @ B @ Hi
    return b, np.sqrt(np.diag(V)), len(rows), len(S)


FEATS = {
    "gain": lambda r, x: x["gain"],
    "gain*dsv41": lambda r, x: x["gain"] and r["model"] == "dsv41",
    "seal(->0)": lambda r, x: bool(x["kill"]),
    "squeeze(2->1)": lambda r, x: x["to1"] and not x["kill"],
    "reduce(k->k-1,k>=3)": lambda r, x: x["red"] > 0 and not x["to1"] and not x["kill"],
}


def clogit_rows(tier, protos):
    return sel(tier, protos, lambda r: r["K"] >= 2 and not r["immobile"] and not r["s4open"])


print("\n" + "=" * 100)
print("Q2/Q3  conditional logit over every choice set (K>=2, no teammate already stuck, not S4 ply 0); OR [95% CI], SE clustered by episode")
for tier in TIERS:
    for name, protos in GROUPS:
        rows = clogit_rows(tier, protos)
        if not rows:
            continue
        feats = dict(FEATS)
        if tier == "weak":
            feats.pop("gain*dsv41")
        if not any(x["kill"] for r in rows for x in r["alts"]):
            feats.pop("seal(->0)")
        b, se, n, ncl = clogit(rows, feats)
        cells = ["%s %.2f [%.2f,%.2f]" % (k, np.exp(bb), np.exp(bb - 1.96 * s), np.exp(bb + 1.96 * s))
                 for k, bb, s in zip(feats, b, se) if k != "gain*dsv41"]
        print("  %-6s %-10s n=%5d eps=%3d  %s" % (tier, name, n, ncl, " | ".join(cells)))

# prose vs rigid interaction on the seal and squeeze terms (strong tier)
rows = clogit_rows("strong", GROUPS[-1][1])
pr = lambda r: r["proto"] in ("freeform", "hybrid")
feats = dict(FEATS)
feats["seal*prose"] = lambda r, x: bool(x["kill"]) and pr(r)
feats["squeeze*prose"] = lambda r, x: x["to1"] and not x["kill"] and pr(r)
feats["reduce*prose"] = lambda r, x: x["red"] > 0 and not x["to1"] and not x["kill"] and pr(r)
feats["gain*prose"] = lambda r, x: x["gain"] and pr(r)
b, se, n, _ = clogit(rows, feats)
print("  strong interaction model: " + " | ".join("%s b=%.2f z=%.2f" % (k, bb, bb / s) for k, bb, s in zip(feats, b, se)))

# ---------- Q3 transparent version: dose-response on the unique progress move ----------
print("\n" + "=" * 100)
print("Q3  plies with exactly one gaining move whose alternatives reduce nobody: P(play the gaining move) by what it does to a teammate")


def effect(x):
    return "seal ->0" if x["kill"] else "squeeze 2->1" if x["to1"] else "k->k-1 (k>=3)" if x["red"] else "none"


for tier in TIERS:
    for name, protos in GROUPS[-3:]:
        t = collections.defaultdict(lambda: [0, 0])
        for r in sel(tier, protos, lambda r: r["K"] >= 2 and r["ng"] == 1 and not r["immobile"] and not r["s4open"]):
            g = [x for x in r["alts"] if x["gain"]][0]
            if any(x["red"] or x["kill"] for x in r["alts"] if not x["gain"]):
                continue
            for k in (r["K"], "all"):
                t[(effect(g), k)][0] += r["alts"][r["chosen"]]["gain"]
                t[(effect(g), k)][1] += 1
        print("  %-6s %-6s " % (tier, name) + " | ".join(
            "%s: %d/%d %.0f%% (K2 %s)" % (e, *t[(e, "all")], 100 * t[(e, "all")][0] / max(1, t[(e, "all")][1]),
                                         "%d/%d" % tuple(t[(e, 2)]))
            for e in ("none", "k->k-1 (k>=3)", "squeeze 2->1", "seal ->0")))

# same-class pairs: K=2, both moves same progress class, exactly one reduces a teammate (any degree, incl. seal)
print("\n  K=2, both moves same progress class, exactly one hurts a teammate: P(choose the hurting move), null 50%")
for tier in TIERS:
    for name, protos in GROUPS[-3:]:
        out = []
        for lab, f in (("non-lethal", lambda x: x["red"] > 0 and not x["kill"]), ("seal", lambda x: bool(x["kill"]))):
            rows = sel(tier, protos, lambda r: r["K"] == 2 and not r["immobile"] and not r["s4open"]
                       and r["alts"][0]["gain"] == r["alts"][1]["gain"]
                       and sum(bool(x["red"] or x["kill"]) for x in r["alts"]) == 1
                       and any(f(x) for x in r["alts"]))
            k = sum(f(r["alts"][r["chosen"]]) for r in rows)
            out.append("%s %d/%d p=%.2g" % (lab, k, len(rows), stats.binomtest(k, len(rows)).pvalue if rows else float("nan")))
        print("  %-6s %-6s %s" % (tier, name, " | ".join(out)))

# ---------- Q4 plan-following ----------
print("\n" + "=" * 100)
print("Q4  prose arms: did the mover's PREVIOUS message name the destination cell of the unique gaining move?")
CELLRX = lambda c: re.compile(P.sa.CELL % (c[0] + 1, c[1] + 1), re.I)
for tier in TIERS:
    t = collections.defaultdict(lambda: [0, 0])
    for r in sel(tier, ("freeform", "hybrid"), lambda r: r["ng"] == 1 and r["K"] >= 2 and (costly(r) or base_ok(r))):
        g = [x for x in r["alts"] if x["gain"]][0]
        named = bool(r["prev"] and CELLRX(g["dest"]).search(r["prev"]))
        kind = "costly pinch" if costly(r) else "non-pinch"
        t[(kind, named)][0] += r["alts"][r["chosen"]]["gain"]
        t[(kind, named)][1] += 1
    for kind in ("costly pinch", "non-pinch"):
        a, b_ = t[(kind, True)], t[(kind, False)]
        print("  %-6s %-12s named: played %d/%d (%.0f%%) | not named: played %d/%d (%.0f%%) | share named %.0f%%"
              % (tier, kind, *a, 100 * a[0] / max(1, a[1]), *b_, 100 * b_[0] / max(1, b_[1]), 100 * a[1] / max(1, a[1] + b_[1])))
    # destination is the mover's own target
    c = sel(tier, ("none", "structured", "freeform", "hybrid"), costly)
    onto = [r for r in c if [x for x in r["alts"] if x["gain"]][0]["dest"] == r["target"]]
    print("  %-6s costly pinches whose sealing move lands on the mover's own target: %d/%d, played %d"
          % (tier, len(onto), len(c), sum(r["alts"][r["chosen"]]["gain"] for r in onto)))
