"""Coverage audits (spec §9.2) and the cross-arm equivalence report (§9.5).

Audits count omissions only (`grammar.audits`); they are reported per 1,000
messages (exact repeats within a seat dropped) by model and protocol, on the
development arm and the calibration set. Outside the development arm a grammar
whose audit rate exceeds twice its development rate is flagged
"coverage-limited"; the measure itself is not changed.

`audit_rates` is one long table of every per-scope quantity of §9.2 and §9.5:
audit and grammar rates, frame shares, G-letter/G-seat collisions and the §5.5
opportunity counts. run.py reads the development run's table back from
out/dev/audit_rates.csv as `dev_rates`, which gives the test-arm rule its
thresholds and `cross_arm_equivalence` its development column.

    cov.run(scope, eps, A, dev_rates=<rows of out/dev/audit_rates.csv or None>,
            opportunity_counts=<rows of the §5.5 opportunity_counts table or None>)
"""
import collections

import numpy as np

from analysis.messages import corpus, grammar, stats
from analysis.messages.measures import accuracy

# which grammar each audit bounds; `X at (R,C)` is a position statement G-loc cannot pair
GRAMMAR_OF = {"cell": "G-cell", "letter_cell_fwd": "G-loc", "letter_cell_rev": "G-loc", "direction": "G-dir",
              "goal_noun": "G-target", "x_at": "G-loc"}
FACTOR = 2.0  # §9.2 test-arm rule
AUDIT, GRAM = "audit per 1,000 messages", "grammar audit per 1,000 messages"
FRAME, COLL, OPP = "frame share", "G-letter/G-seat collisions per 1,000 messages", "opportunity count"
KEY = ("block", "quantity", "model", "suite", "proto", "tier")


def coverage_limited(rate, dev_rate):
    """§9.2 test-arm rule: a rate above FACTOR × the development rate marks
    the measure coverage-limited. None when either rate is missing."""
    if rate is None or dev_rate is None:
        return None
    return rate > FACTOR * dev_rate


def bounds(rows, value, resid, key, a, b, unit=("model", "suite", "level", "board"), denom=None, seed=0):
    """§9.2 bounds of a board-paired contrast a - b (rows whose `key` is a or b)
    of a prose-detected episode measure. Lower: the `resid` residual
    candidates are non-events; upper: they are events. For a count `value`
    that is value and value + resid; for a rate (value / `denom`) the
    residuals join the denominator, value / (denom + resid) and
    (value + resid) / (denom + resid). Both scenarios get the sign-flip test;
    worst_lo / worst_hi are the mean pair differences with the two arms at
    opposite bounds."""
    def lo(r):
        if not denom:
            return r[value]
        return r[value] / (r[denom] + r[resid]) if r[denom] + r[resid] else None

    def hi(r):
        if not denom:
            return r[value] + r[resid]
        return (r[value] + r[resid]) / (r[denom] + r[resid]) if r[denom] + r[resid] else None

    rl = stats.paired_contrast([dict(r, _v=lo(r)) for r in rows], "_v", key, a, b, unit, seed)
    ru = stats.paired_contrast([dict(r, _v=hi(r)) for r in rows], "_v", key, a, b, unit, seed)
    arm = {x: {tuple(r[u] for u in unit): (lo(r), hi(r)) for r in rows if r[key] == x and lo(r) is not None}
           for x in (a, b)}
    both = sorted(set(arm[a]) & set(arm[b]))
    return dict(contrast=rl["contrast"], pairs=rl["pairs"], missing=rl["missing"],
                est_lower=rl["mean"], p_lower=rl["p"], est_upper=ru["mean"], p_upper=ru["p"],
                worst_lo=float(np.mean([arm[a][k][0] - arm[b][k][1] for k in both])) if both else np.nan,
                worst_hi=float(np.mean([arm[a][k][1] - arm[b][k][0] for k in both])) if both else np.nan)


def _num(v):
    """A number from an in-memory value or a CSV string; None if empty or not numeric."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _row(scope, block, quantity, value, n=None, model="all", suite="all", proto="all", tier="all"):
    return dict(scope=scope, block=block, quantity=quantity, model=model, suite=suite, proto=proto, tier=tier,
                n=n, value=value)


def _messages(eps):
    """Message units with exact repeats within a seat dropped (message-level rates)."""
    return [(ep, t) for ep, t in corpus.messages(eps) if not t["dup"]]


def audit_rows(scope, eps, A, P=grammar.DEFAULT):
    """§9.2 per 1,000 messages by model and protocol ("all" pools), per audit and per grammar."""
    n, tot = collections.Counter(), collections.defaultdict(collections.Counter)
    for ep, t in _messages(eps):
        m, p = ep["meta"]["model"], ep["meta"]["proto"]
        res = grammar.audits(t["msg"], A[(ep["meta"]["id"], t["ply"])], P)
        for k in ((m, p), (m, "all"), ("all", p), ("all", "all")):
            n[k] += 1
            tot[k].update(res)
            for au, c in res.items():
                tot[k][GRAMMAR_OF[au]] += c
    out = []
    for (m, p), nm in sorted(n.items()):
        tier = corpus.TIER.get(m, "all")
        for block, names in ((AUDIT, GRAMMAR_OF), (GRAM, sorted(set(GRAMMAR_OF.values())))):
            out += [_row(scope, block, q, 1000 * tot[(m, p)][q] / nm, nm, m, "all", p, tier) for q in names]
    return out


def frame_share_rows(scope, eps, A, P=grammar.DEFAULT):
    """§9.5 frame shares: G-loc (all regions) and prose G-from candidates about
    own and teammates' letters, per tier and pooled."""
    cands = accuracy.candidates(eps, A, P)
    for c in cands:
        c["whose"], c["all"] = ("own" if c["own"] else "teammate"), "all"
    out = []
    for kind, sel in (("loc", [c for c in cands if c["kind"] == "loc"]),
                      ("frm", [c for c in cands if c["kind"] == "frm" and c["region"] == "prose"])):
        for by in (("all", "whose"), ("tier", "whose")):
            out += [_row(scope, FRAME, f"G-{kind} {r['frame']} ({r['whose']})", r["share"], r["n"], tier=r[by[0]])
                    for r in accuracy.frame_rows(sel, by, kind)]
    return out


def collision_rows(scope, eps, A):
    """G-letter/G-seat collisions per 1,000 messages: letter tokens next to a
    G-seat token, and letter tokens inside a seat-like string (`player` + letter)."""
    msgs = _messages(eps)
    adj = inside = 0
    for ep, t in msgs:
        for _, toks in A[(ep["meta"]["id"], t["ply"])]["tokens"]:
            for i, x in enumerate(toks):
                if x.kind != "letter":
                    continue
                adj += (i > 0 and toks[i - 1].kind == "seat") or (i + 1 < len(toks) and toks[i + 1].kind == "seat")
                inside += i > 0 and toks[i - 1].kind == "word" and toks[i - 1].val == "player"
    nm = len(msgs)
    return [_row(scope, COLL, q, 1000 * k / nm if nm else None, nm)
            for q, k in (("letter next to a seat", adj), ("letter inside a seat-like string", inside))]


def opportunity_rows(scope, opportunity_counts):
    """§5.5 opportunity counts (the alignment module's rows), one row per count column."""
    out = []
    for r in opportunity_counts or ():
        for k, v in r.items():
            if k not in ("scope", "model", "tier", "suite", "proto", "level") and _num(v) is not None:
                out.append(_row(scope, OPP, k, _num(v), None, r.get("model", "all"), r.get("suite", "all"),
                                r.get("proto", "all"), r.get("tier", "all")))
    return out


def run(scope, eps, A, dev_rates=None, opportunity_counts=None, P=grammar.DEFAULT):
    """{table: rows}: `audit_rates` (every §9.2/§9.5 quantity of this scope;
    grammar rows carry the test-arm rule against `dev_rates`) and, with
    `dev_rates` outside the development arm, `cross_arm_equivalence`
    (development and this scope side by side)."""
    eps = [ep for ep in eps if ep["meta"]["proto"] != "none"]
    rows = (audit_rows(scope, eps, A, P) + frame_share_rows(scope, eps, A, P) + collision_rows(scope, eps, A)
            + opportunity_rows(scope, opportunity_counts))
    key = lambda r: tuple(str(r[k]) for k in KEY)
    dev = {key(r): r for r in dev_rates or ()}
    for r in rows:
        d = _num(dev[key(r)]["value"]) if key(r) in dev else None
        r["dev_value"] = d
        if r["block"] == GRAM:
            r["threshold"] = None if d is None else FACTOR * d
            r["coverage_limited"] = coverage_limited(r["value"], d) if scope != "dev" else None
    out = dict(audit_rates=rows)
    if dev_rates is not None and scope != "dev":
        here = {key(r): r for r in rows}
        out["cross_arm_equivalence"] = [
            dict(zip(KEY, k), n_dev=_num(dev[k]["n"]) if k in dev else None,
                 dev=_num(dev[k]["value"]) if k in dev else None,
                 **{f"n_{scope}": here[k]["n"] if k in here else None, scope: here[k]["value"] if k in here else None})
            for k in sorted(set(dev) | set(here))]
    return out
