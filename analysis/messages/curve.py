"""Specification curve over equivalent choices (spec §9.3), for the D2 tier contrast.

Factors: the clause comma rule, the G-loc gap (2, 4, 8), the look-back (0, 3),
strict vs lenient frames, the article-A rule, compass words and duplicate
handling. Compass words change only G-dir, so they cannot move D2; the
factor is kept and its specifications are identical by construction.
Non-equivalent choices are separate sensitivity analyses, not curve factors.

Per specification: the strong - weak difference of per-episode accuracy
and its two-sample permutation p. Summary: the median estimate, the share of
specifications with the primary's sign, the share with that sign and
p < .05, and a joint permutation test that relabels tiers at the episode
level once per draw for every specification (statistic: the median estimate).
The curve is monomethod: a misspecification shared by every variant is not
detected.
"""
import itertools

import numpy as np
from scipy import stats as sps

from . import grammar

FACTORS = dict(comma=(False, True), gap=(2, 4, 8), lookback=(0, 3), frames=("strict", "lenient"),
               article=(True, False), compass=(False, True), dedupe=(False, True))


def d2_values(eps, A, frames="strict", dedupe=False):
    """{episode id: (correct, statements)} over teammate G-loc pairs kept under
    `frames`, true at B_t-; episodes with no statement are absent."""
    out = {}
    for ep in eps:
        ok = n = 0
        for t in ep["turns"]:
            if not t["is_msg"] or (dedupe and t["dup"]):
                continue
            for p in A[(ep["meta"]["id"], t["ply"])]["loc"]:
                if grammar.kept(p["frames"], frames) and ep["owner"][p["letter"]] != t["seat"]:
                    n += 1
                    ok += t["board"][p["letter"]] == p["cell"]
        if n:
            out[ep["meta"]["id"]] = (ok, n)
    return out


def _quick(text):
    """Messages that can hold a G-loc pair (a capital A-F and a digit); the
    rest are skipped when re-analysing for the curve."""
    return any(c in text for c in "ABCDEF") and any(c.isdigit() for c in text)


def specs():
    keys = list(FACTORS)
    for vals in itertools.product(*FACTORS.values()):
        yield dict(zip(keys, vals))


def run(scope, eps, A=None, B=2000, seed=0, n_resamples=10_000):
    """One grammar variant at a time, keeping only its G-loc pairs (memory)."""
    eps = [e for e in eps if e["meta"]["proto"] != "none"]
    tier = {e["meta"]["id"]: e["meta"]["tier"] for e in eps}
    ids = sorted(tier)
    rows, mats = [], []
    by_key = {}
    for s in specs():
        by_key.setdefault((s["comma"], s["gap"], s["lookback"], s["article"]), []).append(s)
    for (comma, gap, lookback, article), group in by_key.items():
        P = dict(grammar.DEFAULT, comma=comma, gap=gap, lookback=lookback, article=article)
        locs = {(ep["meta"]["id"], t["ply"]): dict(loc=grammar.analyse(t["msg"], ep["meta"]["arm"], ep["meta"]["proto"],
                                                                        t["slots"], P)["loc"] if _quick(t["msg"]) else [])
                for ep in eps for t in ep["turns"] if t["is_msg"]}
        for s in group:  # compass cannot change G-loc, so it shares the analysis
            vals = d2_values(eps, locs, s["frames"], s["dedupe"])
            v = np.array([vals[i][0] / vals[i][1] if i in vals else np.nan for i in ids])
            mats.append(v)
            strong = v[[tier[i] == "strong" and i in vals for i in ids]]
            weak = v[[tier[i] == "weak" and i in vals for i in ids]]
            ok = len(strong) > 1 and len(weak) > 1
            est = strong.mean() - weak.mean() if ok else np.nan
            p = (sps.permutation_test((strong, weak), lambda a, b, axis: a.mean(axis=axis) - b.mean(axis=axis),
                                      permutation_type="independent", n_resamples=n_resamples, random_state=seed,
                                      vectorized=True).pvalue if ok else np.nan)
            rows.append(dict(s, estimate=est, p=p, n_strong=len(strong), n_weak=len(weak)))
        del locs
    primary = next(r for r in rows if all(r[k] == (grammar.DEFAULT[k] if k in grammar.DEFAULT else False)
                                          for k in ("comma", "gap", "lookback", "frames", "article", "compass", "dedupe")))
    sign = np.sign(primary["estimate"])
    est = np.array([r["estimate"] for r in rows])
    M = np.vstack(mats)  # specifications x episodes, NaN where no statement
    is_strong = np.array([tier[i] == "strong" for i in ids])

    def median_diff(lab):
        s_mask, w_mask = ~np.isnan(M) & lab, ~np.isnan(M) & ~lab
        d = np.nansum(np.where(s_mask, M, 0), 1) / s_mask.sum(1) - np.nansum(np.where(w_mask, M, 0), 1) / w_mask.sum(1)
        return np.median(d)

    obs = median_diff(is_strong)
    rng = np.random.default_rng(seed)
    null = np.array([median_diff(rng.permutation(is_strong)) for _ in range(B)])
    joint_p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + B)
    summary = [dict(estimand="D2 strong - weak (grammar alone)", n_specs=len(rows), primary=primary["estimate"],
                    median=float(np.median(est)), share_same_sign=float(np.mean(np.sign(est) == sign)),
                    share_same_sign_p05=float(np.mean((np.sign(est) == sign) & (np.array([r["p"] for r in rows]) < .05))),
                    joint_statistic="median estimate", joint_p=float(joint_p), B=B, seed=seed)]
    return dict(curve_d2=rows, curve_summary=summary)
