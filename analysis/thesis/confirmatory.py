#!/usr/bin/env python3
r"""The nineteen confirmatory contrasts of tab:confirmatory, in one script.

    python3 analysis/thesis/confirmatory.py            > thesis/confirmatory.tex
    python3 analysis/thesis/confirmatory.py --check    # the same numbers as plain text

Stack item D-b2.  Eleven of the nineteen carried an effect size and no test
(A1, A2, A5, A6, A9, B4, C1--C4, D2); all nineteen are computed here, in one
place, because m has to be fixed in one place -- eleven ad-hoc tests is the
thing a designated confirmatory set exists to prevent.

The machinery is analysis/thesis/tables_comm.py's and is imported rather than copied:
the episode loader (all_episodes), the exact joint optimum (load_optima), the
paired sign-flip permutation test (perm_p).  Only Woolf's homogeneity test is
new, because nothing else needed it.

Three emissions, from the one run:

  1. the p column of tab:confirmatory, one line per contrast, to paste;
  2. tab:holm -- the nineteen sorted, with the Holm threshold on each row and
     the stop point marked;
  3. tab:contrasts -- Appendix A, the same nineteen rows with effect size, n,
     raw p and Holm-adjusted p, computed twice: on the 840-episode set as
     released, and on the 826 that exclude the replaced S2 #10.  The second
     column is the sensitivity analysis S6.3 promises; no conclusion in this
     thesis rests on the replacement, and these columns are how that is shown
     rather than asserted.

Three decisions the table's own family headers force, stated here because they
are what makes m equal nineteen rather than twenty-six:

  * Families B, C and D are declared per model and never pooled, and their
    claims are "in both strong models".  Each such row therefore carries one
    p per model and the row's p is the LARGER of the two -- the intersection-
    union test of the conjunction, which needs no correction of its own and is
    exactly the claim the prose makes.  Both per-model values are printed.
  * A9 spans S1--S4 although the rest of family A is S2--S4: an interaction
    contrast needs the stratum where the effect is absent.  Its 2x2 per level
    is prose (FREEFORM and HYBRID pooled) against NONE, which is the protocol
    effect S8.3 and S8.6 report.  B4's is dyad against 6p, B3's total step.
  * Haldane--Anscombe (+0.5 to all four cells) is applied to any stratum with
    a zero cell and to no other, and the corrected strata are named in the
    note.  It is not optional here: the dyad is 30/30 under all four protocols
    at S1, so A9's S1 stratum has an infinite log odds ratio without it.

P1 is outside the correction and is reported as a bound on the Qwen-DeepSeek
difference, not as a p-value: it licenses family A's pooling, and failing to
reject is not evidence of equivalence.  The interval is exact (Clopper-Pearson
on the discordant pairs, rescaled), which is the paired analogue of the exact
McNemar the rest of the success contrasts use.
"""
import argparse
import json
import math
import os
import statistics as st
import sys

import numpy as np
from scipy.stats import binomtest, chi2 as chi2_dist, permutation_test

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as T                                      # noqa: E402
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

STRONG = T.STRONG                       # ("Qwen3.8", "DeepSeek-V4.1-Flash")
WEAK = "Nemotron"
PROSE = ("freeform", "hybrid")
ALPHA = 0.05
BOARD = ("S2", 10)                      # the replaced board, for the sensitivity column


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def mcnemar(prs):
    """Exact conditional McNemar on the discordant pairs."""
    n01 = sum(1 for a, b in prs if a["success"] and not b["success"])
    n10 = sum(1 for a, b in prs if b["success"] and not a["success"])
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else float("nan")
    ra = 100.0 * sum(a["success"] for a, _ in prs) / len(prs)
    rb = 100.0 * sum(b["success"] for _, b in prs) / len(prs)
    return dict(est=ra - rb, unit="pp", n=len(prs), p=p,
                note=f"{n01}:{n10} discordant")


def paired_eff(prs):
    """Mean per-board efficiency difference over the boards solved under both,
    two-sided paired sign-flip permutation test (tables_comm.perm_p)."""
    d = [a["eff"] - b["eff"] for a, b in prs if a["success"] and b["success"]]
    return dict(est=st.mean(d) if d else float("nan"), unit="", n=len(d),
                p=T.perm_p(d), note="matched boards", floor=True)


def two_sample_p(x, y):
    """Two-sided permutation test on the difference of means, unpaired: the
    across-level contrasts compare different boards and cannot be paired."""
    if not x or not y:
        return float("nan")
    return permutation_test(
        (np.asarray(x, float), np.asarray(y, float)),
        lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        permutation_type="independent", n_resamples=100_000,
        alternative="two-sided", random_state=0, vectorized=True).pvalue


def woolf(strata):
    """Woolf's chi-square test of homogeneity of the odds ratio across strata.

    strata: [(name, a, b, c, d)] -- successes and failures under the first
    condition, then under the second.  Haldane-Anscombe on any stratum with a
    zero cell; the names of the corrected strata come back so the note can
    disclose them, which the caption of tab:confirmatory promises.
    """
    logs, ws, fixed = [], [], []
    for name, a, b, c, d in strata:
        if 0 in (a, b, c, d):
            fixed.append(name)
            a, b, c, d = a + .5, b + .5, c + .5, d + .5
        logs.append(math.log((a * d) / (b * c)))
        ws.append(1.0 / (1 / a + 1 / b + 1 / c + 1 / d))
    bar = sum(w * l for w, l in zip(ws, logs)) / sum(ws)
    chi2 = sum(w * (l - bar) ** 2 for w, l in zip(ws, logs))
    df = len(logs) - 1
    return dict(est=chi2, unit="chi2", n=df, p=chi2_dist.sf(chi2, df),
                note="H--A: " + (", ".join(fixed) if fixed else "none"),
                logs=logs)


def paired_diff_ci(prs, conf=0.95):
    """Exact two-sided interval for the paired difference in success rate.

    The discordant pairs are binomial; a Clopper-Pearson interval on
    theta = n01/(n01+n10) rescales to the difference by
    d = (n01+n10)/N * (2*theta - 1), which is exact for the same reason the
    conditional McNemar test is.
    """
    n01 = sum(1 for a, b in prs if a["success"] and not b["success"])
    n10 = sum(1 for a, b in prs if b["success"] and not a["success"])
    m, n = n01 + n10, len(prs)
    if not m:
        return 0.0, (0.0, 0.0), n01, n10, n
    lo, hi = binomtest(n01, m, 0.5).proportion_ci(confidence_level=conf,
                                                 method="exact")
    f = m / n
    return (100 * f * (2 * n01 / m - 1),
            (100 * f * (2 * lo - 1), 100 * f * (2 * hi - 1)), n01, n10, n)


# --------------------------------------------------------------------------
# the episode set
# --------------------------------------------------------------------------
def build(excluded):
    """(episodes, index) for one analysis set.  index is keyed
    (model, suite, level, cond, board)."""
    T.EXCLUDED = excluded
    eps = T.all_episodes(T.load_optima())
    idx = {(e["model"], e["suite"], e["level"], e["cond"], e["game_id"]): e
           for e in eps}
    return eps, idx


def gids(eps, level):
    return sorted({e["game_id"] for e in eps if e["level"] == level})


def pairs(idx, over, akey, bkey):
    out = []
    for t in over:
        a, b = idx.get(akey(t)), idx.get(bkey(t))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def dyad_keys(eps, levels=("S2", "S3", "S4")):
    return [(m, l, g) for m in STRONG for l in levels for g in gids(eps, l)]


def dyad_pairs(eps, idx, a, b, levels=("S2", "S3", "S4")):
    """Family A's pairing: dyad condition a against condition b on the same
    board and model, pooled over the two strong models.  A solo condition is
    written "solo_silent" and is looked up in the solo suite."""
    def key(cond):
        suite = "solo" if cond.startswith("solo_") else "dyad"
        return lambda t: (t[0], suite, t[1], cond, t[2])
    return pairs(idx, dyad_keys(eps, levels), key(a), key(b))


def suite_pairs(eps, idx, model, a, b):
    """Family B's pairing: the same (level, protocol, board) in two suites."""
    over = [(model, l, p, g) for l in T.ALL_LEVELS for p in T.PROTOCOLS
            for g in gids(eps, l)]
    return pairs(idx, over,
                 lambda t: (t[0], a, t[1], t[2], t[3]),
                 lambda t: (t[0], b, t[1], t[2], t[3]))


def whole_set_pairs(eps, idx, model_a, model_b):
    """Families D and P1: all 840 cells of one model against the other's."""
    over = [(l, s, c, g) for l in T.ALL_LEVELS
            for s, cs in (("solo", ("solo_silent", "solo_thinking")),
                          ("dyad", T.PROTOCOLS), ("3p", T.PROTOCOLS),
                          ("6p", T.PROTOCOLS))
            for c in cs for g in gids(eps, l)]
    return pairs(idx, over,
                 lambda t: (model_a, t[1], t[0], t[2], t[3]),
                 lambda t: (model_b, t[1], t[0], t[2], t[3]))


def solo_boards(eps):
    """Family C's analysis set: the boards solved under all four solo
    conditions -- two protocols in each of the two strong models."""
    ok = {}
    for e in eps:
        if e["suite"] == "solo" and e["model"] in STRONG:
            ok.setdefault((e["level"], e["game_id"]), []).append(e["success"])
    return {k: v for k, v in ok.items() if len(v) == 4 and all(v)}


def solo_eff(idx, model, level, g):
    """One number per board per model: the mean efficiency of its SILENT and
    THINKING episodes.  Both are successes by construction of solo_boards."""
    return st.mean([idx[(model, "solo", level, c, g)]["eff"]
                    for c in ("solo_silent", "solo_thinking")])


# --------------------------------------------------------------------------
# the message corpus, for D2 only.  48 MB and ~40 s, so it is loaded once.
# --------------------------------------------------------------------------
_CLAIMS = None
CORPUS_MODEL = {"qwen38": "Qwen3.8", "dsv41": "DeepSeek-V4.1-Flash",
                "nemotron": "Nemotron"}


def claims():
    """(model, suite, level, protocol, board) -> (correct, wrong) teammate
    claims, from qual_corpus's cache -- the annotation R5 verified."""
    global _CLAIMS
    if _CLAIMS is None:
        cache = os.path.join(T.ROOT, f"analysis/qualitative_analysis/corpus.{ARM}.json")
        rows = json.load(open(cache))["rows"]
        _CLAIMS = {}
        for r in rows:
            if not (r["oth_ok"] or r["oth_bad"]):
                continue
            k = (CORPUS_MODEL[r["model"]], r["suite"], r["lvl"], r["proto"],
                 r["gid"])
            ok, bad = _CLAIMS.get(k, (0, 0))
            _CLAIMS[k] = (ok + r["oth_ok"], bad + r["oth_bad"])
    return _CLAIMS


def claim_rates(excluded, models):
    """One proportion per episode: teammate claims correct / teammate claims
    made, over the episodes that made at least one."""
    return [ok / (ok + bad) for (m, _s, l, _p, g), (ok, bad) in claims().items()
            if m in models and not (excluded and (l, g) == excluded)]


# --------------------------------------------------------------------------
# the nineteen, plus P1
# --------------------------------------------------------------------------
def both(fn):
    """A per-model row: run fn for each strong model, take the larger p.

    The claims of families B, C and D are conjunctions over the two strong
    models, so the valid p for the row is the maximum -- an intersection-union
    test.  The per-model values are kept for the prose, which cites them.
    """
    def run(eps, idx):
        rs = [fn(eps, idx, m) for m in STRONG]
        out = dict(rs[max(range(2), key=lambda i: rs[i]["p"])])
        out["per_model"] = rs
        out["note"] = " / ".join(f"{T.MODEL_NAMES[m]} $p$ {T.sci(r['p'])}"
                                 for m, r in zip(STRONG, rs))
        return out
    return run


def a_success(a, b, levels=("S2", "S3", "S4")):
    return lambda eps, idx: mcnemar(dyad_pairs(eps, idx, a, b, levels))


def a_eff(a, b, levels=("S2", "S3", "S4")):
    return lambda eps, idx: paired_eff(dyad_pairs(eps, idx, a, b, levels))


def a9(eps, idx):
    """Protocol x level: prose against NONE, one 2x2 per level, dyad, both
    strong models pooled.  S1 is in deliberately -- it is the stratum where
    the effect is absent, which is what the interaction claims."""
    strata = []
    for l in T.ALL_LEVELS:
        cells = []
        for conds in (PROSE, ("none",)):
            es = [idx[(m, "dyad", l, c, g)] for m in STRONG for c in conds
                  for g in gids(eps, l) if (m, "dyad", l, c, g) in idx]
            k = sum(e["success"] for e in es)
            cells += [k, len(es) - k]
        strata.append((l, *cells))
    return woolf(strata)


def b_step(a, b):
    def one(eps, idx, m):
        return mcnemar(suite_pairs(eps, idx, m, a, b))
    return both(one)


def b4_one(eps, idx, m):
    """Team size x level: dyad against 6p, one 2x2 per level, per model."""
    strata = []
    for l in T.ALL_LEVELS:
        cells = []
        for s in ("dyad", "6p"):
            es = [idx[(m, s, l, p, g)] for p in T.PROTOCOLS
                  for g in gids(eps, l) if (m, s, l, p, g) in idx]
            k = sum(e["success"] for e in es)
            cells += [k, len(es) - k]
        strata.append((l, *cells))
    return woolf(strata)


def c_levels(hi, lo):
    """Efficiency at one level against another, solo, per model.  Different
    boards, so unpaired; the unit is the board's mean over SILENT and
    THINKING, on the boards solved under all four solo conditions."""
    def one(eps, idx, m):
        bs = solo_boards(eps)
        x = [solo_eff(idx, m, l, g) for (l, g) in bs if l == hi]
        y = [solo_eff(idx, m, l, g) for (l, g) in bs if l == lo]
        return dict(est=st.mean(x) - st.mean(y), unit="", n=f"{len(x)}/{len(y)}",
                    p=two_sample_p(x, y), floor=True,
                    note=f"{hi} {st.mean(x):.3f} vs {lo} {st.mean(y):.3f}")
    return both(one)


def c4_one(eps, idx, m):
    """SILENT -> THINKING within board, all four levels, same analysis set."""
    d = [idx[(m, "solo", l, "solo_thinking", g)]["eff"]
         - idx[(m, "solo", l, "solo_silent", g)]["eff"]
         for (l, g) in solo_boards(eps)]
    return dict(est=st.mean(d), unit="", n=len(d), p=T.perm_p(d),
                note="paired within board", floor=True)


def d1_one(eps, idx, m):
    return mcnemar(whole_set_pairs(eps, idx, m, WEAK))


def d2(eps, idx):
    """Teammate-claim accuracy, one proportion per episode, strong against
    weak.  Unpaired: an episode with no teammate claim contributes none, and
    which episodes those are differs by tier."""
    x = claim_rates(T.EXCLUDED, STRONG)
    y = claim_rates(T.EXCLUDED, (WEAK,))
    return dict(est=100 * (st.mean(x) - st.mean(y)), unit="pp", floor=True,
                n=f"{len(x)}/{len(y)}", p=two_sample_p(x, y),
                note=f"{100*st.mean(x):.1f}\\,\\% vs {100*st.mean(y):.1f}\\,\\%")


# id, the contrast as tab:confirmatory words it, outcome, and the function.
ROWS = [
    ("A1", r"solo-\textsc{silent} $\to$ dyad-\textsc{none}", "success",
     a_success("none", "solo_silent")),
    ("A2", r"solo-\textsc{silent} $\to$ dyad-\textsc{none}", "efficiency",
     a_eff("none", "solo_silent")),
    ("A3", r"\textsc{none} $\to$ \textsc{structured}", "success",
     a_success("structured", "none")),
    ("A4", r"\textsc{none} $\to$ \textsc{structured}", "efficiency",
     a_eff("structured", "none")),
    ("A5", r"\textsc{structured} $\to$ \textsc{freeform}", "success",
     a_success("freeform", "structured")),
    ("A6", r"\textsc{structured} $\to$ \textsc{freeform}", "efficiency",
     a_eff("freeform", "structured")),
    ("A7", r"\textsc{freeform} $\to$ \textsc{hybrid}", "success",
     a_success("hybrid", "freeform")),
    ("A8", r"\textsc{freeform} $\to$ \textsc{hybrid}", "efficiency",
     a_eff("hybrid", "freeform")),
    ("A9", r"protocol $\times$ level", "success", a9),
    ("B1", r"dyad $\to$ 3p", "success", b_step("3p", "dyad")),
    ("B2", r"3p $\to$ 6p", "success", b_step("6p", "3p")),
    ("B3", r"dyad $\to$ 6p (the total)", "success", b_step("6p", "dyad")),
    ("B4", r"team size $\times$ level", "success", both(b4_one)),
    ("C1", r"S1 $>$ S2", "efficiency", c_levels("S1", "S2")),
    ("C2", r"S2 $>$ S3", "efficiency", c_levels("S2", "S3")),
    ("C3", r"S3 $<$ S4 (S3 last)", "efficiency", c_levels("S4", "S3")),
    ("C4", r"\textsc{silent} $\to$ \textsc{thinking}", "efficiency",
     both(c4_one)),
    ("D1", r"strong tier $>$ weak tier", "success", both(d1_one)),
    ("D2", r"teammate claim accuracy, strong vs.\ weak", "accuracy", d2),
]
M = len(ROWS)


def holm(ps):
    """Holm-Bonferroni over the m contrasts: the threshold each rank is tested
    against, the adjusted p, and where the ladder stops."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, stop = {}, 0.0, None
    for rank, i in enumerate(order):
        thr = ALPHA / (len(ps) - rank)
        run = max(run, min(1.0, ps[i] * (len(ps) - rank)))
        rej = ps[i] <= thr and stop is None
        if not rej and stop is None:
            stop = rank
        out[i] = dict(rank=rank + 1, thr=thr, adj=run, reject=rej)
    return out, order, stop


def run(excluded):
    eps, idx = build(excluded)
    return {rid: fn(eps, idx) for rid, _c, _o, fn in ROWS}, eps, idx


def p1(eps, idx):
    d, (lo, hi), n01, n10, n = paired_diff_ci(
        whole_set_pairs(eps, idx, *STRONG))
    return d, lo, hi, n01, n10, n


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def ptex(p, floor=False):
    """Exact tests print their value however small; only the permutation
    tests are floored, at their 10^5-resample limit."""
    return "$<10^{-4}$" if floor and p < 1e-4 else f"${T.sci(p)}$"


def fl(r):
    return bool(r.get("floor"))


def nfmt(r):
    """The n column: matched pairs, the two group sizes, or Woolf's df."""
    return f"df {r['n']}" if r["unit"] == "chi2" else str(r["n"])


def est(r, outcome):
    if r["unit"] == "chi2":
        return f"$\\chi^2 = {r['est']:.2f}$"
    if r["unit"] == "pp":
        return T.signed(r["est"], 1) + "\\,pp"
    return T.signed(r["est"], 3)


def emit(res, res_x, sets, check):
    (eps, idx), (eps_x, idx_x) = sets
    ps = [res[rid]["p"] for rid, *_ in ROWS]
    h, order, stop = holm(ps)
    hx, _, _ = holm([res_x[rid]["p"] for rid, *_ in ROWS])
    d, lo, hi, n01, n10, n = p1(eps, idx)
    dx, lox, hix, _, _, _ = p1(eps_x, idx_x)
    last = order[stop - 1] if stop else None

    if check:
        print(f"{'id':<4}{'effect':>14}{'n':>10}{'p':>12}{'p Holm':>12}"
              f"{'p (826)':>12}  note")
        for i, (rid, _c, _o, _f) in enumerate(ROWS):
            r = res[rid]
            print(f"{rid:<4}{r['est']:>14.4f}{str(r['n']):>10}{r['p']:>12.3g}"
                  f"{h[i]['adj']:>12.3g}{res_x[rid]['p']:>12.3g}  {r['note']}")
        print()
        for rank, i in enumerate(order):
            rid = ROWS[i][0]
            print(f"{rank+1:>3}. {rid:<4} p={ps[i]:<12.3g} "
                  f"threshold {ALPHA/(M-rank):<10.5f} "
                  f"{'reject' if h[i]['reject'] else 'STOP' if rank == stop else 'not rejected'}")
        print(f"\nHolm stops after rank {stop} of {M}"
              if stop else f"\nHolm rejects all {M}")
        print(f"P1: {d:+.2f} pp [{lo:+.2f}, {hi:+.2f}] "
              f"({n01}:{n10} discordant of {n});  "
              f"826: {dx:+.2f} [{lox:+.2f}, {hix:+.2f}]")
        return

    # 1 -- the p column of tab:confirmatory, to paste
    out = ["% ------------------------------------------------------------",
           "% 1. The p column of tab:confirmatory.  One cell per row of the",
           "%    table, in its order; a dagger marks a contrast Holm retains",
           "%    as a null, and the caption needs the legend for it.",
           "% ------------------------------------------------------------"]
    for i, (rid, _c, _o, _f) in enumerate(ROWS):
        out.append(f"% {rid} & {ptex(res[rid]['p'], fl(res[rid]))}"
                   + ("" if h[i]["reject"] else r" $\dagger$"))
    out.append("")

    # 2 -- the ladder
    body = []
    for rank, i in enumerate(order):
        rid, contrast, outcome, _f = ROWS[i]
        mark = "" if h[i]["reject"] else r" $\dagger$"
        f = fl(res[rid])
        row = (f"{rank+1} & {rid} & {contrast}, {outcome} & {ptex(ps[i], f)} & "
               f"{ALPHA/(M-rank):.5f} & {ptex(h[i]['adj'], f)}{mark}")
        if rank == stop:
            body.append(r"\midrule")
        body.append(row)
    note = (f"The ladder stops at rank {stop+1 if stop is not None else M}: "
            + (f"{ROWS[last][0]} is the last contrast rejected, at "
               f"$\\alpha/{M - stop + 1} = {ALPHA/(M-stop+1):.5f}$, and every "
               f"contrast below the rule is retained as a null ($\\dagger$). "
               if stop else "every contrast is rejected. ")
            + "Rows below the rule are not tested further, which is what "
              "makes Holm a step-down procedure rather than nineteen separate "
              "comparisons. \\textsc{p1} is outside the ladder by "
              "construction.")
    out.append(T.table(
        "The Holm ladder over the confirmatory set, sorted by raw $p$. Each "
        f"row is tested against $\\alpha/(m - i + 1)$ with $m = {M}$ and "
        "$\\alpha = 0.05$; the procedure stops at the first row that fails "
        "and every row below it is retained. $p < 10^{-4}$ marks the "
        "resampling floor of the permutation tests.",
        "tab:holm", "llp{0.36\\textwidth}rrr",
        r"$i$ & \# & Contrast & $p$ & $\alpha/(m-i+1)$ & $p_\mathrm{Holm}$",
        body, note=note, tight=True))

    # 3 -- Appendix A, with and without the replaced board
    body = []
    for i, (rid, contrast, outcome, _f) in enumerate(ROWS):
        r, rx = res[rid], res_x[rid]
        if body and rid[0] != ROWS[i - 1][0][0]:
            body.append(r"\midrule")
        f = fl(r)
        body.append(f"{rid} & {contrast}, {outcome} & {est(r, outcome)} & "
                    f"{nfmt(r)} & {ptex(r['p'], f)} & {ptex(h[i]['adj'], f)} & "
                    f"{ptex(rx['p'], f)} & {ptex(hx[i]['adj'], f)}")
    note = (
        "\\emph{Effect} is the difference in success rate in percentage "
        "points, the mean per-board efficiency difference, or Woolf's "
        "$\\chi^2$ on 3 degrees of freedom. \\emph{$n$} is matched pairs, or "
        "the two group sizes where the contrast is unpaired. Families B, C "
        "and D are per model and never pooled: the $p$ shown is the larger of "
        "the two strong models', which is the valid test of the conjunction "
        "the prose claims; both values are in the per-model notes below. "
        "\\emph{840} is the analysis set as released; \\emph{826} excludes "
        "the replaced S2 \\#10 from every cell. No contrast changes its "
        "verdict between the two columns"
        + (". " if all(
            (res[rid]["p"] <= ALPHA) == (res_x[rid]["p"] <= ALPHA)
            for rid, *_ in ROWS) else ", except where noted. ")
        + f"\\textsc{{p1}}: Qwen3.8 minus DeepSeek-V4.1-Flash on success over "
          f"all {n} paired cells is {d:+.2f} points, exact 95\\,\\% interval "
          f"$[{lo:+.2f}, {hi:+.2f}]$ ({n01}:{n10} discordant). It is reported "
          "as a bound and not as a $p$-value: failing to reject is not "
          "evidence of equivalence, and the interval is what licenses "
          f"family~A's pooling. On the 826 set it is {dx:+.2f} points, "
          f"$[{lox:+.2f}, {hix:+.2f}]$.")
    out.append(T.table(
        "The confirmatory set: effect size, test and Holm-adjusted $p$ for "
        "each of the nineteen contrasts of Table~\\ref{tab:confirmatory}, on "
        "the 840-episode analysis set and on the 826 that exclude the "
        "replaced board of \\S6.3.",
        "tab:contrasts",
        "l>{\\raggedright\\arraybackslash}p{0.225\\textwidth}rlrrrr",
        r" & & & & \multicolumn{2}{c}{840 episodes} & "
        r"\multicolumn{2}{c}{826 (board excluded)} \\"
        "\n    \\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n"
        r"    \# & Contrast & Effect & $n$ & $p$ & $p_\mathrm{Holm}$ & "
        r"$p$ & $p_\mathrm{Holm}$",
        body, note=note, tight=True))

    # the per-model detail the prose cites
    out.append("% per-model detail (families B, C, D):")
    for rid, *_ in ROWS:
        r = res[rid]
        if "per_model" in r:
            out.append(f"%   {rid}: " + "; ".join(
                f"{T.MODEL_NAMES[m]} {d['est']:+.4f} (n={d['n']}, "
                f"p={d['p']:.3g})" for m, d in zip(STRONG, r["per_model"])))
        else:
            out.append(f"%   {rid}: {r['note']}")
    print("\n".join(out))


def _selfcheck():
    """The three pieces of arithmetic that are not imported from elsewhere.

    Woolf against a hand-computed two-stratum case, the Haldane--Anscombe
    branch, the exact paired interval, and the Holm step-down.  Runs on every
    invocation: it costs microseconds and the alternative is a silent sign
    error in a table nobody can recompute by eye.
    """
    w = woolf([("a", 10, 10, 10, 10), ("b", 20, 10, 10, 20)])
    assert abs(w["est"] - 2.7454) < 1e-3, w   # by hand: log ORs 0 and 1.386,
    assert w["n"] == 1 and w["note"].endswith("none")  # weights 2.5 and 3.33
    z = woolf([("a", 10, 0, 10, 10), ("b", 20, 10, 10, 20)])
    assert math.isfinite(z["est"]) and "a" in z["note"], z
    prs = [({"success": 1}, {"success": 0})] * 3 + [({"success": 0}, {"success": 1})] * 3
    d, (lo, hi), n01, n10, n = paired_diff_ci(prs)
    assert d == 0 and lo < 0 < hi and (n01, n10, n) == (3, 3, 6)
    h, order, stop = holm([0.001, 0.04, 0.2])
    assert order == [0, 1, 2] and stop == 1 and h[0]["reject"] and not h[1]["reject"]
    assert abs(h[0]["adj"] - 0.003) < 1e-12 and h[2]["adj"] >= h[1]["adj"]


def main():
    _selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="plain text instead of LaTeX")
    args = ap.parse_args()
    res_x, eps_x, idx_x = run(BOARD)    # 826: the board excluded
    res, eps, idx = run(None)           # 840: as released
    emit(res, res_x, ((eps, idx), (eps_x, idx_x)), args.check)


if __name__ == "__main__":
    main()
