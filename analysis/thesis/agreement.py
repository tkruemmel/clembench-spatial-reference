#!/usr/bin/env python3
"""Between-model agreement of the two strong models.  Exploratory throughout.

    ARM=updated python3 analysis/thesis/agreement.py

1. Episode-level success agreement, per suite and pooled, over the cells both
   models played (same suite, level, protocol and board):
     both / neither / Qwen-only / DeepSeek-only, the discordance rate, exact
     McNemar on the discordant pairs, Cohen's kappa, the failure agreement
     p_neg = 2*neither / (2*neither + discordant) (the share of failures that
     are shared, Cicchetti & Feinstein), and the odds ratio with Yule's
     Q = (OR-1)/(OR+1).  Kappa moves with the success base rate; the odds
     ratio does not depend on either model's marginal rate, so it is the
     base-rate-free reading.  Woolf 95 % interval on log OR, with 0.5 added to
     every cell when one is zero (Haldane-Anscombe).
2. Board-level agreement: per (suite, level), Pearson r over the 15 boards of
   the two models' failure counts (out of 4 protocols; 2 conditions in solo),
   with a Fisher-z interval and a percentile bootstrap over boards (10,000
   resamples, seed 0; resamples with no variance in either model are dropped
   and counted).  r is undefined where a model has no variance.
3. Per-board success correlation over all 60 boards (14 episodes per board
   and model).

ARM picks the team suites (qual_corpus.tree); solo is always the original
campaign.
"""
import collections
import math
import os
import sys

import numpy as np
from scipy.stats import binomtest, norm, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as T                                          # noqa: E402

Q, D = T.STRONG
GAMES = ("solo",) + T.SUITES
Z = norm.ppf(0.975)


def concordance(pairs):
    """pairs: [(qwen_success, deepseek_success)] -> the agreement measures."""
    n = len(pairs)
    a = sum(1 for x, y in pairs if x and y)                 # both succeed
    b = sum(1 for x, y in pairs if x and not y)             # Qwen only
    c = sum(1 for x, y in pairs if y and not x)             # DeepSeek only
    d = n - a - b - c                                       # neither
    po = (a + d) / n
    pq, pd = (a + b) / n, (a + c) / n
    pe = pq * pd + (1 - pq) * (1 - pd)
    aa, bb, cc, dd = (v + .5 for v in (a, b, c, d)) if 0 in (a, b, c, d) else (a, b, c, d)
    lor = math.log(aa * dd / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    yule = lambda l: (math.exp(l) - 1) / (math.exp(l) + 1)
    return dict(n=n, both=a, neither=d, q_only=b, d_only=c, disc=100 * (b + c) / n,
                mcnemar=binomtest(b, b + c, .5).pvalue if b + c else float("nan"),
                kappa=(po - pe) / (1 - pe), sq=100 * pq, sd=100 * pd,
                pneg=2 * d / (2 * d + b + c) if d + b + c else float("nan"),
                OR=math.exp(lor), or_lo=math.exp(lor - Z * se), or_hi=math.exp(lor + Z * se),
                yq=yule(lor), yq_lo=yule(lor - Z * se), yq_hi=yule(lor + Z * se),
                haldane=0 in (a, b, c, d))


def board_r(x, y, seed=0, B=10_000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return None
    r = pearsonr(x, y)[0]
    n = len(x)
    z = math.atanh(min(r, 0.999999))
    fz = (math.tanh(z - Z / math.sqrt(n - 3)), math.tanh(z + Z / math.sqrt(n - 3)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    xs, ys = x[idx], y[idx]
    ok = (xs.std(axis=1) > 0) & (ys.std(axis=1) > 0)
    xs, ys = xs[ok], ys[ok]
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    rb = (xc * yc).sum(axis=1) / np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    return dict(r=r, n=n, fz=fz, boot=tuple(np.percentile(rb, [2.5, 97.5])), dropped=int((~ok).sum()))


def main():
    eps = T.all_episodes(T.load_optima())
    idx = {(e["model"], e["suite"], e["level"], e["cond"], e["game_id"]): e for e in eps}
    keys = sorted({k[1:] for k in idx if k[0] == Q} & {k[1:] for k in idx if k[0] == D})
    print(f"ARM={T.ARM}  paired cells {len(keys)}")

    print("\n1. Episode-level success agreement, Qwen vs DeepSeek")
    print(f"{'suite':6}{'n':>5}{'both':>6}{'neith':>6}{'Q-only':>7}{'D-only':>7}{'disc%':>7}"
          f"{'McNemar':>9}{'kappa':>7}{'Q%':>6}{'D%':>6}{'p_neg':>7}   OR [95%]            Yule Q [95%]")
    for su in GAMES + ("all",):
        ks = [k for k in keys if su in ("all", k[0])]
        c = concordance([(idx[(Q,) + k]["success"], idx[(D,) + k]["success"]) for k in ks])
        print(f"{su:6}{c['n']:>5}{c['both']:>6}{c['neither']:>6}{c['q_only']:>7}{c['d_only']:>7}"
              f"{c['disc']:>7.1f}{c['mcnemar']:>9.3g}{c['kappa']:>7.3f}{c['sq']:>6.1f}{c['sd']:>6.1f}"
              f"{c['pneg']:>7.2f}   {c['OR']:6.1f} [{c['or_lo']:.1f}, {c['or_hi']:.1f}]"
              f"   {c['yq']:+.2f} [{c['yq_lo']:+.2f}, {c['yq_hi']:+.2f}]{'  (H-A)' if c['haldane'] else ''}")

    print("\n2. Board-level agreement: r of per-board failure counts, 15 boards per cell")
    fails = collections.Counter()
    for (m, su, lv, cond, g), e in idx.items():
        fails[(m, su, lv, g)] += 1 - e["success"]
    for su in GAMES:
        for lv in T.ALL_LEVELS:
            gs = sorted({k[3] for k in keys if k[0] == su and k[1] == lv})
            x = [fails[(Q, su, lv, g)] for g in gs]
            y = [fails[(D, su, lv, g)] for g in gs]
            r = board_r(x, y)
            tag = (f"r={r['r']:+.2f}  Fisher-z [{r['fz'][0]:+.2f}, {r['fz'][1]:+.2f}]  "
                   f"bootstrap [{r['boot'][0]:+.2f}, {r['boot'][1]:+.2f}] (dropped {r['dropped']})"
                   if r else "r undefined (no variance)")
            print(f"  {su:5}{lv}: fails Q={sum(x):3} D={sum(y):3}  {tag}")

    print("\n3. Per-board success, all 60 boards, 14 episodes per board and model")
    sq, sd = collections.Counter(), collections.Counter()
    for k in keys:
        sq[(k[1], k[3])] += idx[(Q,) + k]["success"]
        sd[(k[1], k[3])] += idx[(D,) + k]["success"]
    bs = sorted(sq)
    r = board_r([sq[b] for b in bs], [sd[b] for b in bs])
    print(f"  {len(bs)} boards: r={r['r']:.3f}  Fisher-z [{r['fz'][0]:.2f}, {r['fz'][1]:.2f}]  "
          f"bootstrap [{r['boot'][0]:.2f}, {r['boot'][1]:.2f}]")


def _selfcheck():
    c = concordance([(1, 1)] * 40 + [(1, 0)] * 5 + [(0, 1)] * 5 + [(0, 0)] * 10)
    assert c["both"] == 40 and c["neither"] == 10 and abs(c["disc"] - 100 / 6) < 1e-9
    assert abs(c["OR"] - 16.0) < 1e-9 and abs(c["yq"] - 15 / 17) < 1e-9
    assert abs(c["pneg"] - 20 / 30) < 1e-9 and abs(c["kappa"] - 5 / 9) < 1e-9


if __name__ == "__main__":
    _selfcheck()
    main()
