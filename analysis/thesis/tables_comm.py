#!/usr/bin/env python3
"""Tables for the communication/dyad results chapter (items W8a-g on the stack).

    python3 analysis/thesis/tables_comm.py > thesis/tables_comm.tex

Named for its content, not its position: this is the chapter that answers the
communication question.  After Q1 swaps RQ1 and RQ2 it becomes Chapter 7; the
stack's W8a-g names are unchanged for the same reason.

Emits ten tables and Appendix B.

  tab:clembench     the framework's own scores (% Played, Quality Score,
                    clemscore) per model and game, aggregated as clemeval does,
                    beside the success rate the rest of the thesis reports.

  tab:solo          the \tab{6}: solo success, efficiency and moves wasted by
                    level, all three arms.  S7.1's table, not this chapter's; it
                    is here because every baseline below is read off it.
  tab:scratchpad    the \tab{7}: solo SILENT against THINKING by level, S7.2's
                    table, here for the same reason as tab:solo.
  tab:recovery      the \tab{8} of the table inventory: cell-level success and
                    efficiency, dyad against the pooled solo baseline.
  tab:decomposition the deliberation-matched contrasts.  This is D-b's table,
                    not W8a's; it is here because the analysis is computed from
                    the same episodes.  If Q1 moves D-b to the other chapter,
                    move the label with it.
  tab:interaction   protocol by spatial level, dyad over its per-level solo
                    baseline.  The only table here that includes S1, because its
                    point is that S1 is flat on both sides: there is no gap for
                    a channel to close until the board hides something.
  tab:dialogue      S8.5's null: message features in successful against failed
                    episodes, protocol held constant.  The only table here
                    computed from messages rather than from scores.json, and
                    the only one whose scope is all three multiplayer suites
                    rather than the dyad.
  tab:carries       the \tab{10}: what a dyad message carries, by model and
                    protocol.  Read from qual_corpus.py's cache, so its cells
                    are the ones R5 verified as Part II Table 12.
  tab:ladder        the \tab{11}: success by model, suite and protocol, all
                    three models and suites.  Asserts the ordering its caption
                    states.
  tab:teamsize      the \tab{12}: Quality, success and efficiency by team
                    size at each level, strong tier pooled, with the paired
                    per-model dyad->3p->6p Quality drops S9.1 cites.
  tab:seal-greedy   S9.3's enclosure aborts: what the sealing move was worth to
                    the object that played it, and what the sealer could have
                    played instead.  Computed by analysis/qualitative_analysis/
                    seal_alt.py, which figures/aborts.py also draws from.
  tab:cut-shape     S9.3.2's permanent cuts: where the sealing object stood, by
                    level, with the board's minimum vertex cut beside it.
                    Computed by analysis/qualitative_analysis/cut.py, which
                    figures/cuts.py also draws from.

tab:interaction is the \tab{9}.  It had an S2-S4 twin, dropped in D-e because
it repeated the same cells; its note on the two S1 solo non-successes moved
into tab:interaction's.  F3 is figures/protocol_level.py, which reads ladder().

Why two baselines rather than one.  A communication channel is also somewhere
to write before acting, and so is a scratchpad, so a protocol measured against
a no-scratchpad baseline is credited for both.  tab:recovery uses the pooled
solo baseline because S8.1's prose does; tab:decomposition holds the writing
surface fixed across each contrast instead -- SILENT against dyad-NONE, and
THINKING against the three protocols -- which is the pairing D-b specifies.

Scope, and every part of it is a decision the caption has to state:

  * S2-S4 only.  S1 is at ceiling in all eight dyad cells of both strong
    models, so "how much of the gap is recovered" has no gap to measure there.
  * The two strong models only.  Nemotron is a different tier and family A's
    pooling is licensed for Qwen and DeepSeek by P1, not for all three.
  * Solo is reported as SILENT and THINKING separately and as their pool.
    The pool is the baseline the gap is measured from, which is what S8.1's
    prose asserts; the two components are shown because they are also the
    baselines D-b decomposes against -- SILENT is deliberation-free, like
    dyad-NONE, and THINKING is the deliberation-matched ceiling the protocol
    conditions should be read against.
  * Efficiency is the suite's attainable optimum over the episode's move count
    (the exact joint optimum for solo, where the two coincide), NOT
    scores.json's `Move Efficiency`.  That key divides by the stored per-object
    BFS sum, which S5.4 defines as the budgeting bound and not the scoring one;
    it understates efficiency by 0.07-0.08 here and by 20.4% at S4.  Verified
    2026-09-20: both denominators were computed over these episodes and only
    the exact one reproduces the figures S8.1 states.
  * Efficiency is conditioned on success, so the cells rest on different board
    sets.  The note carries the matched-subset figure for the one contrast
    whose direction depends on it.

Format, as clembench papers report: the Quality Score and % Played lead every
results table, and success and move efficiency -- the Quality Score's two
factors -- follow as game-specific rows or columns.  The Quality Score is
BENCH_SCORE on the suite's attainable optimum, averaged over played episodes; where
aborts occur (6p, the weak tier) cells are "Quality / % Played", the
Plant-in-Cupboard layout.  Quality contrasts use the paired permutation test
S5.5 declares (perm_p), not Wilcoxon: the score is a mean with a point mass at
0, and the signed-rank test answers a different question about it.
tab:dialogue and tab:carries are message-level and carry no score.

EXCLUDED is the live analysis set.  R3b landed on 21 Sep, so it is None and the
tables are on 840; thesis/tables_comm.826.tex is the version they replace.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics as st
import sys

import numpy as np
from scipy.stats import (binomtest, fisher_exact, kruskal, mannwhitneyu,
                         permutation_test)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis/qualitative_analysis"))
from qual_corpus import ARM, tree                                # noqa: E402

TREES = {
    "Qwen3.8": ("results_qwen38_grid_n15_solo", tree("results_qwen38_grid_n15", "dyad")),
    "DeepSeek-V4.1-Flash": ("results_dsv41_grid_n15_solo", tree("results_dsv41_grid_n15", "dyad")),
}
# tab:ladder and F3 need all three multiplayer suites and, for tab:ladder, the
# weak tier too.  Same result trees qual_corpus.py's ARMS reads.
ARMS = {"Qwen3.8": "results_qwen38_grid_n15",
        "DeepSeek-V4.1-Flash": "results_dsv41_grid_n15",
        "Nemotron": "results_nemotron_fixthink_grid_n15"}
STRONG = ("Qwen3.8", "DeepSeek-V4.1-Flash")
SUITES = ("dyad", "3p", "6p")
LEVELS = ("S2", "S3", "S4")          # scope of tab:recovery and tab:decomposition
ALL_LEVELS = ("S1",) + LEVELS        # tab:interaction needs the flat level too
EXCLUDED = None                # was ("S2", 10); R3b landed 21 Sep
PROTOCOLS = ("none", "structured", "freeform", "hybrid")
NICE = {"none": r"\textsc{none} (no channel)", "structured": r"\textsc{structured}",
        "freeform": r"\textsc{freeform}", "hybrid": r"\textsc{hybrid}"}

# tab:dialogue.  Binary features first, then the two per-message means, which
# are in different units and so cannot share a column with them.  Display order
# is by the success-minus-failure difference, computed at run time: the ordering
# is the argument, and hard-coding it would let a rerun silently contradict the
# caption.
DIALOGUE = [("target disclosed", "tgt_disclosed"),
            ("acknowledgement", "ack"),
            ("sequencing marker", "seq"),
            ("corridor or dead end named", "corr"),
            ("request", "req"),
            ("names a teammate's object", "names_other"),
            ("reports itself blocked", "wait")]
MEANS = [("mean words", "words", 1), ("mean coordinates", "ncoord", 1)]


def corpus():
    """(qual_corpus module, every annotated message).

    The lexicon and the board replay are analysis/qualitative_analysis/qual_corpus.py's
    and are not duplicated here; this reads that script's cache -- the one R5
    verified -- and writes it if it is missing.  Refresh it with
    `python3 analysis/qualitative_analysis/qual_corpus.py --refresh` after any change to
    the annotation.
    """
    sys.path.insert(0, os.path.join(ROOT, "analysis/qualitative_analysis"))
    import qual_corpus as qc
    cache = os.path.join(ROOT, f"analysis/qualitative_analysis/corpus.{ARM}.json")
    if os.path.exists(cache):
        rows = json.load(open(cache))["rows"]
    else:
        rows, epi = qc.extract(ROOT)
        json.dump({"rows": rows, "epi": epi}, open(cache, "w"))
    return qc, rows


def messages():
    """The message corpus restricted to the scope of S8.5.

    Strong models, prose protocols, S2-S4, all three multiplayer suites -- not
    the dyad-only scope the rest of this file uses, because the claim is
    reported on the full multiplayer corpus.
    """
    qc, rows = corpus()
    return [r for r in rows if r["model"] in qc.STRONG
            and r["proto"] in qc.PROSE and r["lvl"] != "S1"]


def ladder():
    """(model, suite, level, protocol) -> list of 0/1 successes.

    Success only, over all episodes, so an abort counts as a failure -- the
    convention tab:interaction uses.  No efficiency, so no optimum lookup.
    """
    out = {}
    for model, prefix in ARMS.items():
        for suite in SUITES:
            pat = os.path.join(ROOT, tree(prefix, suite), "**", "scores.json")
            for f in sorted(glob.glob(pat, recursive=True)):
                exp = f.split(os.sep)[-3]          # S3_structured_3p_grid
                if not re.match(r"S\d_", exp):
                    continue
                level, proto = exp.split("_")[:2]
                s = json.load(open(f))["episode scores"]
                out.setdefault((model, suite, level, proto), []).append(
                    int(s.get("Success", 0)))
    return out


def load_optima():
    """Exact optima, refusing anything inexact.

    (level, game_id) -> the free-order joint optimum J (S6.1).
    (suite, level, game_id) -> the attainable optimum of that suite (S6.4):
    the cheapest play-through a flawless team of that size achieves under the
    suite's own rules.  Solo's is J by definition.  Every score in the thesis
    divides by the second; J survives as `eff_j`, the cost including the rules.
    """
    opt = {}
    path = os.path.join(ROOT, "analysis/efficiency_baseline/joint_optimum.jsonl")
    for ln in open(path):
        d = json.loads(ln)
        if d.get("status") != "exact":
            continue
        opt[(d["level"], d["game_id"])] = d["opt"]
        opt[("solo", d["level"], d["game_id"])] = d["opt"]
    for suite in SUITES:
        path = os.path.join(ROOT, f"analysis/efficiency_baseline/attainable_{suite}.jsonl")
        for ln in open(path):
            d = json.loads(ln)
            if d.get("status") == "exact":
                opt[(suite, d["level"], d["game_id"])] = d["stuckfree"]
    return opt


def episodes(opt):
    """One record per scored episode in scope."""
    out = []
    missing = set()
    for model, (solo_tree, dyad_tree) in TREES.items():
        for tree, is_solo in ((solo_tree, True), (dyad_tree, False)):
            pat = os.path.join(ROOT, tree, "**", "scores.json")
            for f in sorted(glob.glob(pat, recursive=True)):
                exp = f.split(os.sep)[-3]
                if is_solo:
                    if not exp.startswith("Solo_"):
                        continue
                    # Solo_S2_Silent_Grid -> level S2, cond solo_silent
                    level = exp.split("_")[1]
                    cond = "solo_" + exp.split("_")[2].lower()
                else:
                    if not re.match(r"S\d_", exp):
                        continue
                    level, cond = exp.split("_")[0], exp.split("_")[1]
                if level not in ALL_LEVELS:
                    continue
                gid = int(re.search(r"instance_(\d+)", f).group(1))
                if EXCLUDED and (level, gid) == EXCLUDED:
                    continue
                s = json.load(open(f))["episode scores"]
                success = int(s.get("Success", 0))
                eff = eff_j = None
                suite = "solo" if is_solo else "dyad"
                if success and s.get("Move Count"):
                    if (suite, level, gid) in opt:
                        eff = opt[(suite, level, gid)] / s["Move Count"]
                        eff_j = opt[(level, gid)] / s["Move Count"]
                    else:
                        missing.add((level, gid))
                out.append(dict(model=model, level=level, game_id=gid,
                                cond=cond, success=success, eff=eff, eff_j=eff_j,
                                ab=int(s["Aborted"]), q=quality(s, eff)))
    return out, missing


# Order carries the argument: the coordination contrast, then the three
# deliberation-matched ones, then the deliberation benefit the middle three are
# matched on.  The caption states the grouping; it does not need header rows.
CONTRASTS = [
    (r"Dyad \textsc{none} vs.\ solo \textsc{silent}", ("none", "solo_silent")),
    (r"Dyad \textsc{structured} vs.\ solo \textsc{thinking}", ("structured", "solo_thinking")),
    (r"Dyad \textsc{freeform} vs.\ solo \textsc{thinking}", ("freeform", "solo_thinking")),
    (r"Dyad \textsc{hybrid} vs.\ solo \textsc{thinking}", ("hybrid", "solo_thinking")),
    (r"Solo \textsc{thinking} vs.\ solo \textsc{silent}", ("solo_thinking", "solo_silent")),
]


def contrast(idx, keys, a, b):
    """One paired contrast.  Exact conditional McNemar on success; two-sided
    paired permutation test on efficiency over the boards solved under both.

    The permutation test is the one S5.5 and tab:confirmatory declare, and
    replaced an exact Wilcoxon here on 24 Sep so that the prose of S8.1 and the
    table agree.  It is on the mean difference, so ties carry information and
    the efficiency n is the whole matched set rather than the non-zero
    differences.
    """
    pairs = [(idx[(m, l, g, a)], idx[(m, l, g, b)]) for m, l, g in keys
             if (m, l, g, a) in idx and (m, l, g, b) in idx]
    n01 = sum(1 for x, y in pairs if x["success"] == 1 and y["success"] == 0)
    n10 = sum(1 for x, y in pairs if x["success"] == 0 and y["success"] == 1)
    pm = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else float("nan")
    d = [x["eff"] - y["eff"] for x, y in pairs
         if x["success"] and y["success"] and x["eff"] is not None and y["eff"] is not None]
    pw = perm_p(d)
    ra = 100.0 * sum(x["success"] for x, _ in pairs) / len(pairs)
    rb = 100.0 * sum(y["success"] for _, y in pairs) / len(pairs)
    dq = [x["q"] - y["q"] for x, y in pairs
          if x["q"] is not None and y["q"] is not None]
    return dict(pairs=len(pairs), d_succ=ra - rb, n01=n01, n10=n10, pm=pm,
                d_eff=st.mean(d) if d else float("nan"), n_eff=len(d), pw=pw,
                d_q=st.mean(dq) if dq else float("nan"), n_q=len(dq), pq=perm_p(dq))


def fp(x):
    """p-value, with a floor rather than a misleading 0.0000."""
    return "$<$0.0001" if x < 0.0001 else f"{x:.4f}"


def sci(x):
    """p-value in the prose's style: 0.28, or 1.2\\times10^{-8} below 0.001."""
    if x >= 0.001:
        return f"{x:.2g}" if x < 0.995 else "1.0"
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def signed(x, dp):
    return f"{x:+.{dp}f}".replace("-", "$-$").replace("+", "$+$")


def quality(s, eff):
    """BENCH_SCORE on the given efficiency: None if aborted, else 0 or
    100 x efficiency.  The clembench Quality Score is its mean over played
    episodes (tab:clembench)."""
    if s["Aborted"]:
        return None
    return min(100.0, 100.0 * eff) if s["Success"] and eff is not None else 0.0


def cell(rows):
    n = len(rows)
    k = sum(r["success"] for r in rows)
    effs = [r["eff"] for r in rows if r["success"] and r["eff"] is not None]
    effj = [r["eff_j"] for r in rows if r["success"] and r.get("eff_j") is not None]
    qs = [r["q"] for r in rows if r["q"] is not None]
    qj = [min(100.0, 100.0 * r["eff_j"]) if r["success"] else 0.0
          for r in rows if r["q"] is not None and "eff_j" in r]
    return dict(n=n, k=k, rate=100.0 * k / n if n else float("nan"),
                eff=st.mean(effs) if effs else float("nan"), eff_n=len(effs),
                eff_j=st.mean(effj) if effj else float("nan"),
                q_j=st.mean(qj) if qj else float("nan"),
                played=100.0 * len(qs) / n if n else float("nan"),
                q=st.mean(qs) if qs else float("nan"))


def perm_p(d):
    """Two-sided paired permutation (sign-flip) test on the mean of the
    per-board differences d: the test S5.5 declares for per-board contrasts.
    Exact below 17 pairs, else 100 000 resamples at a fixed seed."""
    d = np.asarray([x for x in d], dtype=float)
    if not len(d) or not d.any():
        return float("nan")
    return permutation_test(
        (d,), lambda x, axis: np.mean(x, axis=axis), permutation_type="samples",
        n_resamples=100_000, alternative="two-sided", random_state=0).pvalue


def psci(x):
    """A permutation p in sci()'s style, floored at the resampling limit."""
    return "<10^{-4}" if x < 1e-4 else f"= {sci(x)}"


def qpairs(a_rows, b_rows, key, field="q"):
    """Quality differences a - b over the keys both played."""
    ia = {key(r): r for r in a_rows if r[field] is not None}
    ib = {key(r): r for r in b_rows if r[field] is not None}
    return [ia[k][field] - ib[k][field] for k in ia if k in ib]


def table(caption, label, spec, header, body, note=None, tight=False):
    out = [r"\begin{table}[tbp]", r"  \centering", r"  \footnotesize"]
    if tight:  # wide contrast tables: narrower column gaps, not smaller type
        out.append(r"  \setlength{\tabcolsep}{3pt}")
    out += [f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule",
            "    " + header + r" \\", r"    \midrule"]
    for b in body:
        out.append(r"    \midrule" if b == r"\midrule" else "    " + b + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        out += [r"  \\[2pt]", r"  \begin{minipage}{0.9\textwidth}\footnotesize",
                "  " + note, r"  \end{minipage}"]
    out += [r"\end{table}", ""]
    return "\n".join(out)


def thousands(n):
    """Digit grouping the thesis's way: a thin space, not a comma."""
    return f"{n:,}".replace(",", "\\,")


def pct(x):
    """Percent, with a real LaTeX minus rather than a hyphen."""
    return f"{x:.1f}".replace("-", "$-$") + "\\,\\%"


def all_episodes(opt, games=("solo",) + SUITES):
    """Every episode of the three arms, one dict each: model, suite, level,
    cond (solo_silent / solo_thinking, or the protocol), game_id, success,
    eff (the suite's attainable optimum over move count, successes only),
    eff_j (the same on the free-order joint optimum), ab, q."""
    out = []
    for model, prefix in ARMS.items():
        for suite in games:
            pat = os.path.join(ROOT, tree(prefix, suite), "**", "scores.json")
            for f in sorted(glob.glob(pat, recursive=True)):
                exp = f.split(os.sep)[-3]  # Solo_S2_Silent_Grid, S3_none_3p_grid
                if suite == "solo":
                    if not exp.startswith("Solo_"):
                        continue
                    _, level, proto = exp.split("_")[:3]
                    cond = "solo_" + proto.lower()
                else:
                    if not re.match(r"S\d_", exp):
                        continue
                    level, cond = exp.split("_")[:2]
                gid = int(re.search(r"instance_(\d+)", f).group(1))
                if EXCLUDED and (level, gid) == EXCLUDED:
                    continue
                s = json.load(open(f))["episode scores"]
                ok = int(s.get("Success", 0))
                eff = opt[(suite, level, gid)] / s["Move Count"] if ok else None
                eff_j = opt[(level, gid)] / s["Move Count"] if ok else None
                out.append(dict(model=model, suite=suite, level=level, cond=cond,
                                game_id=gid, success=ok, eff=eff, eff_j=eff_j,
                                ab=int(s["Aborted"]), q=quality(s, eff),
                                q_j=quality(s, eff_j)))
    return out


def f2(x):
    return "--" if x != x else f"{x:.2f}"


def f3(x):
    return "--" if x is None or x != x else f"{x:.3f}"


def solo_table(opt, check):
    """tab:solo (the inventory's tab 6): solo play by level, clembench first.

    All three arms, both solo protocols pooled, all four levels -- S7.1 reads
    the weak tier off the same table.  % Played and the Quality Score lead, as
    in clembench papers; success, efficiency and wasted moves (move count
    minus the exact optimum) are the game-specific rows beneath them.
    Wasted moves are cited in S7.1 because the ratio scale is not comparable
    across levels.
    """
    eps = [e for e in all_episodes(opt, ("solo",))]
    cells, fisher, kw, s3s4 = {}, {}, {}, {}
    for model in ARMS:
        for level in ALL_LEVELS:
            rows = [e for e in eps if e["model"] == model and e["level"] == level]
            c = cell(rows)
            solved = [e for e in rows if e["success"]]
            c["waste"] = (st.mean(opt[(level, e["game_id"])] / e["eff"]
                                  - opt[(level, e["game_id"])] for e in solved)
                          if solved else None)
            c["qs"] = [e["q"] for e in rows if e["q"] is not None]
            cells[(model, level)] = c
        ks = [(cells[(model, l)]["k"], cells[(model, l)]["n"]) for l in ALL_LEVELS]
        fisher[model] = min(
            fisher_exact([[a, n - a], [b, m - b]]).pvalue
            for i, (a, n) in enumerate(ks) for b, m in ks[i + 1:])
        qs = [cells[(model, l)]["qs"] for l in ALL_LEVELS]
        kw[model] = kruskal(*qs).pvalue if all(len(set(q)) > 1 for q in qs) \
            and len({x for q in qs for x in q}) > 1 else float("nan")
        a, b = cells[(model, "S3")]["qs"], cells[(model, "S4")]["qs"]
        s3s4[model] = mannwhitneyu(a, b).pvalue if set(a) | set(b) != {0.0} else float("nan")

    if check:
        for (model, level), c in sorted(cells.items()):
            w = "--" if c["waste"] is None else f"{c['waste']:.1f}"
            print(f"{model:<20} {level} P {c['played']:6.2f}  Q {f2(c['q']):>6}  "
                  f"{c['k']:>2}/{c['n']}  eff {f3(c['eff'])}  waste {w}")
        for model in ARMS:
            print(f"{model:<20} Fisher(success) min p = {fisher[model]:.3f}  "
                  f"Kruskal(Q) p = {kw[model]:.3g}  S3 vs S4 (Q) p = {s3s4[model]:.3g}")
        print()
        return None

    def f1(x):
        return "--" if x is None else f"{x:.1f}"

    body = []
    for model in ARMS:
        cs = [cells[(model, l)] for l in ALL_LEVELS]
        if body:
            body.append(r"\midrule")
        body += [
            f"{MODEL_NAMES[model]} & \\% Played & " + " & ".join(f2(c["played"]) for c in cs),
            " & Quality Score & " + " & ".join(f2(c["q"]) for c in cs),
            " & Solved & " + " & ".join(f"{c['k']}/{c['n']}" for c in cs),
            " & Efficiency & " + " & ".join(f3(c["eff"]) for c in cs),
            " & Moves wasted & " + " & ".join(f1(c["waste"]) for c in cs)]
    low = {m: min(ALL_LEVELS, key=lambda l: cells[(m, l)]["q"]) for m in STRONG}
    assert set(low.values()) == {"S3"}, low
    strong = "; ".join(f"{MODEL_NAMES[m]} ${sci(kw[m])}$ and ${sci(s3s4[m])}$" for m in STRONG)
    ps = ", ".join(f"${sci(fisher[m])}$ ({MODEL_NAMES[m]})" for m in ARMS)
    caption = (
        "Solo play by level, both solo protocols pooled, so each cell is 30 "
        "episodes: 15 boards under \\textsc{silent} and \\textsc{thinking}. "
        "\\emph{\\% Played} and the \\emph{Quality Score} are the clembench scores "
        "of Table~\\ref{tab:clembench}; the rows beneath them are its two "
        "game-specific factors. In the strong tier every episode is played and "
        "the Quality Score is lowest at S3, not S4, in both models: success does "
        "not move across the levels, so the ordering is carried by efficiency. "
        "The weak tier solves nothing above S1. "
        "\\emph{Efficiency} is the exact joint optimum over the move count and "
        "\\emph{moves wasted} the move count minus it, both over successful "
        "episodes only, so a cell with no success has neither.")
    note = (
        r"\emph{Tests} (exploratory, uncorrected). Quality Score across the four "
        r"levels, Kruskal--Wallis, and S3 against S4, Mann--Whitney: "
        f"{strong}. Success, smallest pairwise Fisher exact $p$ over the six "
        f"level pairs: {ps}; no pair of levels differs in success within any model.")
    return table(caption, "tab:solo", "llrrrr",
                 r"Model & & S1 & S2 & S3 & S4", body, note)


def scratchpad_table(opt, check):
    """S7.2's table: solo \\textsc{silent} against \\textsc{thinking} by level.

    Quality Score first, as clembench reports, with the paired permutation
    test per level and pooled (board is the pair); successes and the
    efficiency difference over each protocol's own successes are the
    game-specific rows beneath.
    """
    eps = all_episodes(opt, ("solo",))
    cells, perm = {}, {}
    for model in ARMS:
        for level in ALL_LEVELS + ("all",):
            lv = ALL_LEVELS if level == "all" else (level,)
            rows = {p: [e for e in eps if e["model"] == model and e["level"] in lv
                        and e["cond"] == "solo_" + p] for p in ("silent", "thinking")}
            for p in rows:
                cells[(model, level, p)] = cell(rows[p])
            d = qpairs(rows["thinking"], rows["silent"],
                       lambda e: (e["level"], e["game_id"]))
            perm[(model, level)] = dict(d=st.mean(d) if d else float("nan"),
                                        p=perm_p(d), n=len(d))
        (a, n), (b, m) = ((cells[(model, "all", p)]["k"], cells[(model, "all", p)]["n"])
                          for p in ("silent", "thinking"))
        perm[(model, "fisher")] = fisher_exact([[a, n - a], [b, m - b]]).pvalue

    def d_eff(model, level):
        s_, t_ = (cells[(model, level, p)]["eff"] for p in ("silent", "thinking"))
        return float("nan") if s_ != s_ or t_ != t_ else t_ - s_

    if check:
        for model in ARMS:
            for l in ALL_LEVELS + ("all",):
                s_, t_ = (cells[(model, l, p)] for p in ("silent", "thinking"))
                r = perm[(model, l)]
                print(f"{model:<20} {l:<3} Q {f2(s_['q']):>6} -> {f2(t_['q']):>6}  "
                      f"dQ {r['d']:+6.2f} (n={r['n']}, perm p={r['p']:.3g})  "
                      f"succ {s_['k']}/{s_['n']} -> {t_['k']}/{t_['n']}  "
                      f"eff d {d_eff(model, l):+.3f}")
            print(f"{model:<20} Fisher(success, pooled) p = {perm[(model, 'fisher')]:.3f}")
        print()
        return None

    def fd(x, dp):
        return "--" if x != x else signed(x, dp)

    body = []
    levels = ALL_LEVELS + ("all",)
    for model in ARMS:
        if body:
            body.append(r"\midrule")
        body += [
            f"{MODEL_NAMES[model]} & Quality, \\textsc{{silent}} & "
            + " & ".join(f2(cells[(model, l, 'silent')]["q"]) for l in levels),
            " & Quality, \\textsc{thinking} & "
            + " & ".join(f2(cells[(model, l, 'thinking')]["q"]) for l in levels),
            " & $\\Delta$ Quality & " + " & ".join(fd(perm[(model, l)]["d"], 2) for l in levels),
            " & permutation $p$ & " + " & ".join(
                "--" if perm[(model, l)]["p"] != perm[(model, l)]["p"] else fp(perm[(model, l)]["p"])
                for l in levels),
            " & Solved, \\textsc{silent} & "
            + " & ".join(f"{cells[(model, l, 'silent')]['k']}/{cells[(model, l, 'silent')]['n']}"
                         for l in levels),
            " & Solved, \\textsc{thinking} & "
            + " & ".join(f"{cells[(model, l, 'thinking')]['k']}/{cells[(model, l, 'thinking')]['n']}"
                         for l in levels),
            " & $\\Delta$ efficiency & " + " & ".join(fd(d_eff(model, l), 3) for l in levels)]
    gains = "; ".join(f"{MODEL_NAMES[m]} {signed(perm[(m, 'all')]['d'], 2)}, "
                      f"$p {psci(perm[(m, 'all')]['p'])}$" for m in STRONG)
    fis = ", ".join(f"${sci(perm[(m, 'fisher')])}$ ({MODEL_NAMES[m]})" for m in ARMS)
    sig = [m for m in STRONG if perm[(m, "all")]["p"] < 0.05]
    verdict = ("significant in " + " and ".join(MODEL_NAMES[m] for m in sig)
               if sig else "significant in neither")
    caption = (
        "Solo \\textsc{silent} against \\textsc{thinking} by level, 15 boards per "
        "cell. The scratchpad raises the mean Quality Score by about six points "
        f"in both strong models, pooled over the four levels ({gains}), "
        f"{verdict}: the mean moves on the few boards one protocol solves and "
        "the other does not, each of which moves the paired difference by the "
        "full score of the solved episode. There is no success gap to close --- silent play is already "
        "near ceiling at every level --- and the gain is not concentrated at "
        "S3 and S4. $\\Delta$ is \\textsc{thinking} minus \\textsc{silent}.")
    note = (
        r"\emph{Tests} (exploratory, uncorrected). Quality: two-sided paired "
        r"permutation test on the per-board differences. $\Delta$ Quality is over "
        r"the boards played under both protocols, so where episodes aborted "
        r"(the weak tier) it is not the difference of the two rows above it. "
        r"Success, pooled Fisher exact: "
        f"{fis}; no success contrast is significant. $\\Delta$ efficiency is over "
        r"each protocol's own successes, so a cell with no success under either "
        r"protocol has none.")
    return table(caption, "tab:scratchpad", "llrrrrr",
                 r"Model & & S1 & S2 & S3 & S4 & All", body, note)


GAMES = ("solo",) + SUITES


DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
MOVED = re.compile(r"moved ([A-Z]) (up|down|left|right)")


def objects_home(path):
    """(objects on target when the episode ended, objects in the instance).

    The games log no partial-goal score, so it is reconstructed by replay:
    every legal move is in the interaction log, and applying them to the
    instance's start positions reproduces the final board.  Blocked and
    illegal attempts never reach the log, so the replay is exact -- verified
    on all 1,552 successful episodes of the three arms, every one of which
    replays to 6/6.  This is the Goal rate row of tab:clembench, the analogue
    of Plant-in-Cupboard's Goal Success Rate (Jordan et al., 2025).
    """
    inst = json.load(open(os.path.join(path, "instance.json")))
    pos = {k: tuple(v) for k, v in inst["start_positions"].items()}
    for turn in json.load(open(os.path.join(path, "interactions.json")))["turns"]:
        for e in turn:
            a = e.get("action", {})
            if a.get("type") != "move":
                continue
            m = MOVED.search(a["content"])
            if not m:
                raise ValueError(f"unparsable move in {path}: {a['content']!r}")
            o, d = m.groups()
            dr, dc = DIRS[d]
            pos[o] = (pos[o][0] + dr, pos[o][1] + dc)
    tgt = inst["target_positions"]
    return (sum(pos[o] == tuple(tgt[o]) for o in inst["objects"]),
            len(inst["objects"]))


def _selfcheck_replay():
    """Every successful episode must replay to all objects home."""
    bad = []
    for model, prefix in ARMS.items():
        for game in GAMES:
            for f in sorted(glob.glob(os.path.join(ROOT, tree(prefix, game), "**",
                                            "scores.json"), recursive=True)):
                if not json.load(open(f))["episode scores"]["Success"]:
                    continue
                d = os.path.dirname(f)
                h, t = objects_home(d)
                if h != t:
                    bad.append((d, h, t))
    assert not bad, f"replay disagrees on {len(bad)} successes: {bad[:3]}"
    return True


def clembench_table(opt, check):
    """tab:clembench: the framework's own scores, per model and game.

    Aggregated the way clemcore's clemeval.save_clem_table does it: per game
    (one package = one suite), % Played is the mean of 1 - Aborted and the
    Quality Score the mean `Main Score` with aborted episodes (nan) skipped;
    both are rounded to 2 dp, macro-averaged over the four games, and
    clemscore = Average % Played / 100 * Average Quality Score.  Verified
    2026-09-24 against clemeval.save_clem_table on these trees: identical up to
    0.01 of half-even rounding in one All cell.  (`clem eval` itself refuses them, because the games
    log `Played` and clemeval raises PlayedScoreError on that.)

    Only the aggregation is clemcore's.  The per-episode score is NOT the
    shipped `Main Score`, which divides by the stored per-object BFS bound and
    understates S2-S4 (S6.1); it is recomputed on each suite's attainable optimum,
    BENCH_SCORE as S5.4 defines it.  The cross-check above was run on the
    shipped scores, so it verifies the aggregation, not these values.
    """
    cells = {}
    for model, prefix in ARMS.items():
        for game in GAMES:
            n = aborted = succ = home = goals = 0
            q = []
            for f in sorted(glob.glob(os.path.join(ROOT, tree(prefix, game), "**",
                                            "scores.json"), recursive=True)):
                level = re.search(r"S\d", f.split(os.sep)[-3]).group(0)
                gid = int(re.search(r"instance_(\d+)", f).group(1))
                s = json.load(open(f))["episode scores"]
                n += 1
                aborted += s["Aborted"]
                succ += s["Success"]
                h, t = objects_home(os.path.dirname(f))
                home += h
                goals += t
                if s["Aborted"]:
                    continue
                q.append(min(100.0, 100.0 * opt[(game, level, gid)] / s["Move Count"])
                         if s["Success"] else 0.0)
            cells[(model, game)] = dict(
                n=n, played=round(100.0 * (n - aborted) / n, 2),
                q=round(st.mean(q), 2) if q else float("nan"),
                succ=100.0 * succ / n, gsr=round(100.0 * home / goals, 2))
        cs = [cells[(model, g)] for g in GAMES]
        p = round(st.mean(c["played"] for c in cs), 2)
        q = round(st.mean(c["q"] for c in cs), 2)
        cells[(model, "all")] = dict(
            played=p, q=q, succ=st.mean(c["succ"] for c in cs),
            gsr=round(st.mean(c["gsr"] for c in cs), 2),
            clem=round(p / 100 * q, 2))

    if check:
        print("replay self-check:", "ok" if _selfcheck_replay() else "FAILED")
        for model in ARMS:
            for g in GAMES + ("all",):
                c = cells[(model, g)]
                extra = (f"  clemscore {c['clem']:.2f}"
                         if g == "all" else f"  n={c['n']}")
                print(f"{model:<20} {g:<5} played {c['played']:6.2f}  quality "
                      f"{c['q']:6.2f}  success "
                      f"{c['succ']:5.1f}  goals {c['gsr']:6.2f}{extra}")
        print()
        return None

    def f2(x):
        return "--" if x != x else f"{x:.2f}"

    body = []
    for model in ARMS:
        cs = [cells[(model, g)] for g in GAMES + ("all",)]
        a = cs[-1]
        if body:
            body.append(r"\midrule")
        body += [
            f"{MODEL_NAMES[model]} & \\% Played & "
            + " & ".join(f2(c["played"]) for c in cs),
            " & Quality Score & " + " & ".join(f2(c["q"]) for c in cs),
            " & Success (all episodes) & " + " & ".join(f"{c['succ']:.2f}" for c in cs),
            " & Goal rate & " + " & ".join(f2(c["gsr"]) for c in cs),
            f" & \\textbf{{clemscore}} & & & & & \\textbf{{{a['clem']:.2f}}}"]
    caption = (
        "The clembench scores, per model and game, aggregated as the "
        "framework aggregates them \\citep{chalamalasetti-etal-2023-clembench}. "
        "\\emph{\\% Played} is the share of episodes not aborted; the "
        "\\emph{Quality Score} is the mean \\texttt{BENCH\\_SCORE} "
        "(Equation~\\ref{eq:benchscore}) over played episodes only, so an aborted "
        "episode leaves the Quality Score untouched and lowers \\% Played instead. "
        "\\emph{All} is the macro-average over the four games and "
        "\\emph{clemscore} is Average \\% Played / 100 $\\times$ Average Quality "
        "Score. \\texttt{BENCH\\_SCORE} is taken on each game's attainable "
        "optimum (\\S5.4, \\S6.4), the fewest moves a flawless team of that "
        "size needs under that game's rules, so a perfect play-through scores "
        "100 in every game; not on the per-object bound the games store, which "
        "understates S2--S4. \\emph{Success} is successes over "
        "all episodes, the denominator used everywhere else in this thesis, where "
        "an abort counts as a failure. \\emph{Goal rate} is objects on target "
        "when the episode ended, over all objects in all episodes: the "
        "partial-credit measure the three outcome scores above it cannot "
        "express, since each of them is 0 unless all six objects are home. It "
        "is the analogue of the Goal Success Rate that "
        "Jordan et al.\\ \\rw{2025} report beside the framework scores, and "
        "it is reconstructed by replaying the logged moves (\\S5.4.1). It is "
        "reported as a diagnostic beside the outcome metrics and is deliberately "
        "not folded into the Quality Score: five of six objects home is not "
        "five-sixths of a solved board, because the object left behind is "
        "usually the one the level was built around. The weak tier shows why "
        "the scores cannot be read apart: a Quality Score of 41 at 3p rests on "
        "the 5\\% of episodes that were played at all, and its goal rate over "
        "all episodes is a small fraction of that.")
    return table(caption, "tab:clembench", "llrrrrr",
                 r"Model & & Solo & Dyad & 3p & 6p & All", body)


MODEL_NAMES = {"Qwen3.8": "Qwen3.8", "DeepSeek-V4.1-Flash": "DeepSeek",
               "Nemotron": "Nemotron"}


def goals_table(check):
    """tab:goals: objects on target at termination, by outcome and model.

    The distribution behind the Goal rate row of tab:clembench.  Successes are
    6/6 by definition, so that row is the replay's self-check rather than a
    finding: all 1,552 of them come back 6/6.
    """
    dist, per_level = {}, {}
    for model, prefix in ARMS.items():
        for game in GAMES:
            for f in sorted(glob.glob(os.path.join(ROOT, tree(prefix, game), "**",
                                            "scores.json"), recursive=True)):
                s = json.load(open(f))["episode scores"]
                h, _ = objects_home(os.path.dirname(f))
                out = ("aborted" if s["Aborted"] else
                       "success" if s["Success"] else "lost")
                dist.setdefault((model, out), []).append(h)
                if out == "lost" and model in STRONG:
                    lvl = re.search(r"S\d", f.split(os.sep)[-3]).group(0)
                    per_level.setdefault(lvl, []).append(h)

    if check:
        for model in ARMS:
            for out in ("success", "lost", "aborted"):
                v = dist.get((model, out), [])
                c = [sum(1 for x in v if x == k) for k in range(6, -1, -1)]
                m = st.mean(v) if v else float("nan")
                print(f"{model:<20} {out:<8} n={len(v):<4} "
                      + " ".join(f"{k}/6:{n:<4}" for k, n in zip(range(6, -1, -1), c))
                      + f" mean {m:.2f}")
        for lvl in sorted(per_level):
            v = per_level[lvl]
            print(f"  lost, strong tier, {lvl}: n={len(v)} at 5/6 "
                  f"{sum(1 for x in v if x == 5)}  mean {st.mean(v):.2f}")
        print()
        return None

    # Successes are all 6/6 by definition and carry no distribution; the replay
    # check that they do is _selfcheck_replay(), not a row of the table.  6/6 is
    # therefore an empty column here -- asserted rather than assumed, since an
    # abort on the turn after the last object lands would be one.
    assert not [x for (m, o), v in dist.items() if o != "success"
                for x in v if x == 6], "a non-success episode ended 6/6"
    body = []
    for model in ARMS:
        if body:
            body.append(r"\midrule")
        for label, out in (("Lost", "lost"), ("Aborted", "aborted")):
            v = dist.get((model, out), [])
            cells = " & ".join(str(sum(1 for x in v if x == k))
                               for k in range(5, -1, -1))
            mean = f"{st.mean(v):.2f}" if v else "--"
            name = MODEL_NAMES[model] if out == "lost" else ""
            body.append(f"{name} & {label} & {len(v)} & {cells} & {mean}")

    lost = [x for m in STRONG for x in dist.get((m, "lost"), [])]
    near5 = 100.0 * sum(1 for x in lost if x >= 5) / len(lost)
    near4 = 100.0 * sum(1 for x in lost if x >= 4) / len(lost)
    caption = (
        "Objects on target when the episode ended, for the episodes that did not "
        "succeed: the distribution behind the Goal rate row of "
        "Table~\\ref{tab:clembench}, reconstructed by replaying every logged move "
        "against the instance's start positions (\\S5.4.1). Successes are omitted "
        "because they carry no distribution --- an episode counts as a success "
        "only when all six objects are home, so all 1\\,552 of them replay to 6/6, "
        "which is how the reconstruction is verified rather than something the "
        "table reports. What is left is exactly what \\texttt{BENCH\\_SCORE} "
        "cannot distinguish, since it is 0 for all of it alike. \\emph{In the "
        f"strong tier a loss is a near miss}}: {near5:.0f}\\% of the 123 lost "
        f"episodes end one object short and {near4:.0f}\\% end at four of six or "
        "better, the means being 4.03 and 4.14. \\emph{An abort is not}: the same models average "
        "1.57 and 1.39, and the weak tier's 749 aborts average 0.31, with 619 of "
        "them at 0/6. The two non-success outcomes are different failures, and "
        "the binary score charges them the same.")
    note = (r"\emph{Strong-tier losses by level} (Qwen3.8 and DeepSeek pooled, "
            r"objects home): S1 $n=4$, mean 4.50; S2 $n=13$, 3.92; S3 $n=73$, "
            r"4.07; S4 $n=33$, 4.12. At S3, 34 of the 73 losses end one object "
            r"short --- the level where \S7.1 puts the efficiency floor is also "
            r"where the failures come closest to solving the board.")
    return table(caption, "tab:goals", "llrrrrrrrr",
                 r"Model & Outcome & $n$ & 5 & 4 & 3 & 2 & 1 & 0 & Mean",
                 body, note=note)


TEAMSIZE_STEPS = (("dyad", "3p"), ("3p", "6p"), ("dyad", "6p"))


def teamsize_cells(opt, tests=True):
    """(suite, level) -> cell() over the two strong models pooled, and the
    paired per-model Quality drops (m, a, b) -> (mean, n, perm p), on the
    per-suite denominator (`paired`) and on the joint optimum (`paired_j`, the
    cost including the rules).  Shared by tab:teamsize and figures/teamsize.py
    so the two cannot drift apart."""
    eps = [e for e in all_episodes(opt) if e["model"] in STRONG]
    cells = {(s, l): cell([e for e in eps if e["suite"] == s and e["level"] == l])
             for s in ("solo",) + SUITES for l in ALL_LEVELS}
    paired, paired_j = {}, {}
    for m in STRONG if tests else ():
        for a, b in TEAMSIZE_STEPS:
            for out, field in ((paired, "q"), (paired_j, "q_j")):
                d = qpairs([e for e in eps if e["model"] == m and e["suite"] == a],
                           [e for e in eps if e["model"] == m and e["suite"] == b],
                           lambda r: (r["level"], r["cond"], r["game_id"]), field)
                out[(m, a, b)] = (st.mean(d), len(d), perm_p(d))
    return cells, paired, paired_j


def teamsize_table(opt, check):
    """tab:teamsize (the inventory's tab 12): S9.1's cost of team size.

    The two strong models pooled, each suite by level, solo as the reference
    row.  Quality and efficiency are on each suite's attainable optimum; the
    Quality on the joint optimum is the extra row that carries the rules.  The
    paired dyad->3p->6p Quality contrasts are per model, over (level, protocol,
    board) played in both suites, with perm_p.  F-teamsize
    (figures/teamsize.py) draws the same cells.
    """
    cells, paired, paired_j = teamsize_cells(opt)
    steps = TEAMSIZE_STEPS

    if check:
        for (s, l), c in cells.items():
            print(f"{s:<5} {l} n={c['n']:>3} P {c['played']:6.1f} Q {c['q']:6.2f} "
                  f"QJ {c['q_j']:6.2f} k={c['k']:>3} eff {c['eff']:.3f} effJ {c['eff_j']:.3f}")
        for (m, a, b), (dq, n, p) in paired.items():
            dj, _, pj = paired_j[(m, a, b)]
            print(f"{m:<20} {a}->{b}  dQ {dq:+.2f} (n={n}, perm p={p:.2g})"
                  f"   on J {dj:+.2f} (p={pj:.2g})")
        print()
        return None

    # The prose of S9.1, asserted so that a rerun which breaks it breaks the
    # build: the third agent is a significant cost in both models, the 3->6
    # step is not, and on the joint optimum every step still is (the rules).
    for m in STRONG:
        assert paired[(m, "dyad", "3p")][0] > 0 and paired[(m, "dyad", "3p")][2] < 0.001
        assert paired[(m, "3p", "6p")][2] > 0.05, f"{m} 3p->6p now significant"
        assert all(paired_j[(m, a, b)][0] > 0 and paired_j[(m, a, b)][2] < 0.001
                   for a, b in steps)

    name = {"solo": "Solo (ref.)", "dyad": "Dyad", "3p": "3p", "6p": "6p"}
    body = []
    for label, fmt in (
            ("Quality / \\% Played",
             lambda c: f"{c['q']:.2f} / {c['played']:.1f}"),
            ("Quality on $J$", lambda c: f"{c['q_j']:.2f}"),
            ("Success", lambda c: f"{c['k']}/{c['n']} ({pct(c['rate'])})"),
            ("Efficiency", lambda c: f"{c['eff']:.2f}")):
        if body:
            body.append(r"\midrule")
        for i, s in enumerate(("solo",) + SUITES):
            body.append(f"{label if i == 0 else ''} & {name[s]} & "
                        + " & ".join(fmt(cells[(s, l)]) for l in ALL_LEVELS))

    def step(pr, a, b):
        (q1, n1, p1), (q2, n2, p2) = (pr[(m, a, b)] for m in STRONG)
        pp = lambda p: "p < 10^{-4}" if p < 1e-4 else f"p = {sci(p)}"
        return (f"{a}$\\to${b} {signed(q1, 2)} (${pp(p1)}$) and "
                f"{signed(q2, 2)} (${pp(p2)}$)")
    caption = (
        "The cost of team size at each level: Quality Score and \\% Played, "
        "success and move efficiency by suite, Qwen3.8 and DeepSeek-V4.1-Flash "
        "pooled ($n = 120$ per multiplayer cell, 60 for solo). Quality and "
        "efficiency divide by each suite's attainable optimum (\\S6.4), so a "
        "flawless team of any size scores 100; \\emph{Quality on $J$} divides "
        "by the free-order joint optimum instead and so also charges the moves "
        "the suite's rules force. Solo is the reference row, not a step of the "
        "ladder. The third agent costs Quality at every level, most at S3; the "
        "remaining three cost none beyond what the rules force, and below 6p "
        "every episode is played.")
    note = ("\\emph{Paired Quality drops}, per model (Qwen3.8, then DeepSeek), "
            "over (level, protocol, board) played in both suites, two-sided "
            "paired permutation test. On the attainable optimum: "
            + "; ".join(step(paired, a, b) for a, b in steps)
            + ". On $J$: " + "; ".join(step(paired_j, a, b) for a, b in steps)
            + ". Efficiency is over successes only.")
    return table(caption, "tab:teamsize", "llrrrr",
                 r"Measure & Suite & S1 & S2 & S3 & S4", body, note=note, tight=True)


def teamsize_steps_table(opt):
    """tab:teamsize-steps: what each added agent costs, split into the rules
    and the agents.  Per-step change in Quality on the joint optimum (the whole
    cost), on the attainable optimum (the agents), their difference (the
    rules), and % Played, by level, strong models pooled."""
    cells, _, _ = teamsize_cells(opt, tests=False)
    d = {}
    for a, b in (("dyad", "3p"), ("3p", "6p")):
        d[("q_j", a, b)] = [cells[(b, l)]["q_j"] - cells[(a, l)]["q_j"] for l in ALL_LEVELS]
        d[("q", a, b)] = [cells[(b, l)]["q"] - cells[(a, l)]["q"] for l in ALL_LEVELS]
        d[("rules", a, b)] = [x - y for x, y in zip(d[("q_j", a, b)], d[("q", a, b)])]
        d[("played", a, b)] = [cells[(b, l)]["played"] - cells[(a, l)]["played"]
                               for l in ALL_LEVELS]
    # The prose's shape: the third agent's cost to the agents peaks at S3; the
    # 3->6 step costs the agents nothing at any level beyond S1's rounding and
    # is carried by the rules; only the 3->6 step costs played episodes.
    s3 = ALL_LEVELS.index("S3")
    assert min(d[("q", "dyad", "3p")]) == d[("q", "dyad", "3p")][s3]
    assert all(x > -1.0 for x in d[("q", "3p", "6p")])
    assert all(x < 0 for x in d[("rules", "3p", "6p")])
    assert all(x == 0 for x in d[("played", "dyad", "3p")])
    body = []
    for f, label in (("q_j", "Quality on $J$"), ("rules", "\\quad rules"),
                     ("q", "\\quad agents"), ("played", "\\% Played")):
        if body and f in ("q_j", "played"):
            body.append(r"\midrule")
        for i, (a, b) in enumerate((("dyad", "3p"), ("3p", "6p"))):
            row = d[(f, a, b)]
            peak = min(row)
            body.append(f"{label if i == 0 else ''} & {a}$\\to${b} & " + " & ".join(
                ("0.0" if abs(x) < 0.05 else f"\\textbf{{{signed(x, 1)}}}" if x == peak and x < -0.5
                 else signed(x, 1))
                for x in row))
    caption = (
        "What each added agent costs, by level, and who pays it: change in "
        "Quality Score on the free-order joint optimum $J$ (the whole cost), "
        "split into the part the suite's rules force (the change in what a "
        "flawless team scores on $J$) and the part the agents add (the change "
        "in Quality on each suite's attainable optimum), with the change in "
        "\\% Played beneath. Qwen3.8 and DeepSeek-V4.1-Flash pooled ($n = 120$ "
        "per cell). The third agent's cost is mostly the agents' and "
        "concentrated at S3; the remaining three cost the agents nothing, and "
        "their whole Quality cost is the rules, while from S2 up they also cost "
        "aborted episodes, which the Quality Score does not count. Largest cost "
        "per row in bold.")
    return table(caption, "tab:teamsize-steps", "llrrrr",
                 r"Measure & Step & S1 & S2 & S3 & S4", body)


def teamsize_tests_table(opt):
    """tab:teamsize-tests: the six paired Quality contrasts behind S9.1, one
    row each, on the attainable optimum, with Holm over the six."""
    _, paired, _ = teamsize_cells(opt)
    keys = sorted(paired, key=lambda k: paired[k][2])
    holm, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, paired[k][2] * (len(keys) - i)))
        holm[k] = run
    assert all(holm[(m, "dyad", "3p")] < 0.01 for m in STRONG)
    assert all(holm[(m, "3p", "6p")] > 0.05 for m in STRONG)
    fp = lambda p: "<10^{-4}" if p < 1e-4 else sci(p)
    body = []
    for a, b in TEAMSIZE_STEPS:
        body.append(f"{a}$\\to${b} & " + " & ".join(
            f"{signed(paired[(m, a, b)][0], 2)} & {paired[(m, a, b)][1]} & "
            f"${fp(paired[(m, a, b)][2])}$ & ${fp(holm[(m, a, b)])}$"
            for m in STRONG))
    caption = (
        "Paired Quality Score drops with team size, per model, on each suite's "
        "attainable optimum. Each pair is one (level, protocol, board) played in "
        "both suites; $\\Delta Q$ is the mean of smaller-team minus larger-team "
        "Quality. The third agent is a significant drop in both models after "
        "Holm correction; the step from three to six is not a drop in either.")
    note = ("Two-sided paired permutation (sign-flip) test on the mean "
            "difference, $10^5$ resamples at a fixed seed, and $p < 10^{-4}$ marks the resampling "
            "floor. $n$ falls below 240 where a 6p "
            "episode aborted and has no Quality Score. $p_\\mathrm{Holm}$ adjusts "
            "over the six tests.")
    head = " & ".join(f"\\multicolumn{{4}}{{c}}{{{MODEL_NAMES[m]}}}" for m in STRONG)
    sub = " & ".join(r"$\Delta Q$ & $n$ & $p$ & $p_\mathrm{Holm}$" for _ in STRONG)
    return table(caption, "tab:teamsize-tests", "lrrrrrrrr",
                 f" & {head} \\\\\n    \\cmidrule(lr){{2-5}}\\cmidrule(lr){{6-9}}\n"
                 f"    Step & {sub}", body, note=note, tight=True)


def seal_greedy_table():
    """tab:seal-greedy: what the sealing move of an enclosure abort was worth to
    the player that played it, over the 30 aborts whose sealer had a choice.

    The replay, the harmlessness test and the distance are
    analysis/qualitative_analysis/seal_alt.py's, the same module figures/aborts.py draws
    from, so the table and the two abort figures cannot disagree.
    """
    sys.path.insert(0, os.path.join(ROOT, "analysis/qualitative_analysis"))
    import seal_alt as sa

    keys, counts = sa.greedy_rows(sa.cases())
    cells = lambda f: " & ".join(str(counts[k][f]) for k in keys)
    total, costly = counts["all"]["n"], counts["all"]["costly"]
    caption = (
        "The sealing move was the locally optimal move in every one of the "
        f"{total} enclosure aborts whose sealer had a choice; the two whose "
        "sealing move was its only legal move are excluded, having no choice "
        "to characterise. Distance is the shortest path to the mover's own "
        "target over walls, with the other objects ignored because they move; "
        "the three moves a Manhattan metric would score as retreats are all "
        "detours around a wall column. A move is \\emph{harmless} if it would "
        f"have left every player a legal move. Each block partitions the {total}. "
        f"In every one of the {costly} cases of the last row the sealing move "
        "was also the sealer's only progress-making move of any kind, so "
        "avoiding the enclosure there meant giving up ground on its own object "
        "for a teammate it cannot see.")
    body = [
        f"Aborts whose sealer had a harmless alternative & {cells('n')}",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\emph{What the sealing move did for the "
        r"object that played it}}",
        f"\\quad took it nearer its own target & {cells('nearer')}",
        f"\\quad left its distance unchanged & {cells('level')}",
        f"\\quad took it further from its own target & {cells('further')}",
        r"\midrule",
        r"\multicolumn{4}{@{}l}{\emph{What the sealer could have played "
        r"instead}}",
        f"\\quad a harmless move that also made progress & {cells('free')}",
        f"\\quad harmless moves, every one of them losing ground & {cells('costly')}",
    ]
    return table(caption, "tab:seal-greedy", "p{0.55\\textwidth}rrr",
                 r"& Qwen3.8 & DeepSeek & all", body)


def cut_shape_table():
    """tab:cut-shape: what the permanent cut of S9.3.2 was made of, by level.

    The replay, the reachability test and the minimum vertex cut are
    analysis/qualitative_analysis/cut.py's, the same module figures/cuts.py draws from,
    so the table and the two cut figures cannot disagree.
    """
    sys.path.insert(0, os.path.join(ROOT, "analysis/qualitative_analysis"))
    import cut

    everything = cut.cases()
    # the pockets are a different mechanism, not a different place: S9.3.2's
    # bullet describes them, and mixing them in here would make a column headed
    # "where" answer a question three of its rows are not about
    rows = [r for r in everything if r["kind"] in cut.CHOKEPOINTS]
    levels = ("S1", "S2", "S3", "S4")
    groups = (("Where the corridor device is involved",
               (cut.ENTRANCE, cut.ENTRANCE_OTHER, cut.INSIDE, cut.APPROACH)),
              ("Where it is not",
               (cut.OPEN,)))
    labels = {
        cut.ENTRANCE: "a seal on the entrance, target in the corridor",
        cut.ENTRANCE_OTHER: "a seal on the entrance, target elsewhere",
        cut.INSIDE: "the cut object locked inside the corridor",
        cut.APPROACH: "a seal on the approach to the entrance",
        cut.OPEN: "an open-board chokepoint",
    }
    per = collections.Counter((r["kind"], r["meta"]["level"]) for r in rows)
    eps = collections.defaultdict(set)
    for r in rows:
        eps[r["kind"]].add(r["meta"]["path"])
    mincut = {k: sorted({r["mincut"] for r in rows if r["kind"] == k})
              for k in labels}
    total, episodes = len(rows), len({r["meta"]["path"] for r in rows})
    pockets = len(everything) - total
    assert (total, pockets, episodes) == (97, 4, 74), (total, pockets, episodes)
    assert all(r["mincut"] == 1 for r in rows if r["kind"] in
               (cut.ENTRANCE, cut.INSIDE, cut.APPROACH)), "a corridor cut is not 1"
    assert all(r["mincut"] >= 2 for r in rows if r["kind"] == cut.OPEN), \
        "an open-board cut is a single cell"

    def line(kind):
        n = sum(per[(kind, lv)] for lv in levels)
        cells = " & ".join(str(per[(kind, lv)] or "") for lv in levels)
        cuts = ", ".join(str(c) for c in mincut[kind])
        return (f"\\quad {labels[kind]} & {cells} & {n} & "
                f"{len(eps[kind])} & {cuts}")

    def group(title):
        return r"\multicolumn{8}{@{}l}{\emph{" + title + r"}}"

    body = []
    for title, kinds in groups:
        body.append(r"\multicolumn{8}{@{}l}{\emph{" + title + r"}}")
        body += [line(k) for k in kinds]
        body.append(r"\midrule")
    body[-1] = ("\\textbf{All} & "
                + " & ".join(f"\\textbf{{{sum(per[(k, lv)] for k in labels)}}}"
                             for lv in levels)
                + f" & \\textbf{{{total}}} & \\textbf{{{episodes}}} & ")
    body.insert(-1, r"\midrule")

    caption = (
        f"Where the chokepoint was, over the {total} of the "
        f"{len(everything)} permanent cuts of \\S9.3.2 that pinch a route "
        f"rather than seal a destination. All {total} are severed by a "
        "\\texttt{done} declaration rather than by a move, and in every one "
        "the sealing object is standing on its own target, so the rows "
        "classify where that target happens to sit. The corridor device "
        f"accounts for {sum(per[(k, lv)] for t, ks in groups[:1] for k in ks for lv in levels)} "
        "of them and its entrance alone for "
        f"{sum(per[(k, lv)] for k in (cut.ENTRANCE, cut.ENTRANCE_OTHER) for lv in levels)}: "
        "the S3 generator places a target on the corridor's only door "
        "(\\S6.2), so the player that finishes there correctly is the one "
        f"that kills the board. The remaining {pockets} cuts seal the target "
        "cell's own neighbourhood instead and are the only ones S1 and S2 "
        "produce; they are described in the text rather than placed here, "
        "having no chokepoint to locate.")
    note = (
        "$n$ counts cut objects. \\emph{Cut} is the smallest number of free cells whose removal would "
        "separate that object from its target, searched exhaustively over the "
        "board. It is never 1 away from the corridor: the generator certifies "
        "the free space 2-connected, which forbids a one-cell cut, and the "
        "certificate is checked with the corridor treated as blocked, so the "
        "entrance is an articulation point by design rather than by accident. "
        "Objects, not episodes, are the unit: an episode with more than one cut "
        "object contributes to each row its cuts fall in, which is why the "
        "episode column sums past its total.")
    # ragged right and a column wide enough for the longest label: justified
    # p-columns hyphenate these across two lines and the continuation loses the
    # \quad, which puts it under the group header instead of under its own row
    head = r"Shape of the cut & S1 & S2 & S3 & S4 & $n$ & Episodes & Cut"
    return table(caption, "tab:cut-shape",
                 ">{\\raggedright\\arraybackslash}p{0.50\\textwidth}rrrrrrr",
                 head, body, note=note, tight=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="print the cells as plain text instead of LaTeX")
    args = ap.parse_args()

    opt = load_optima()
    eps_all, missing = episodes(opt)
    eps = [e for e in eps_all if e["level"] in LEVELS]
    by = {c: [e for e in eps if e["cond"] == c]
          for c in ("solo_silent", "solo_thinking") + PROTOCOLS}
    by["solo"] = by["solo_silent"] + by["solo_thinking"]
    cells = {c: cell(v) for c, v in by.items()}

    solo, none = cells["solo"], cells["none"]
    gap_s = solo["rate"] - none["rate"]
    gap_e = solo["eff"] - none["eff"]
    gap_q = solo["q"] - none["q"]

    idx = {(e["model"], e["level"], e["game_id"], e["cond"]): e for e in eps}
    keys = sorted({(m, l, g) for (m, l, g, c) in idx})
    res = {cc: contrast(idx, keys, *cc) for _, cc in CONTRASTS}

    # The decomposition, on cell means so that it adds up against the pooled gap
    # tab:recovery reports.  Coordination is SILENT -> NONE; the residual is the
    # half-share of deliberation the pooled baseline carries, since half its
    # episodes are THINKING.  The paired tests of tab:decomposition sit beside
    # these, not inside them: efficiency there is over boards solved under both
    # conditions, so it does not decompose additively.
    coord_s = cells["solo_silent"]["rate"] - none["rate"]
    coord_e = cells["solo_silent"]["eff"] - none["eff"]
    coord_q = cells["solo_silent"]["q"] - none["q"]
    resid_s, resid_e, resid_q = gap_s - coord_s, gap_e - coord_e, gap_q - coord_q

    if args.check:
        clembench_table(opt, True)
        goals_table(True)
        teamsize_table(opt, True)
        solo_table(opt, True)
        scratchpad_table(opt, True)
        for c in ("solo_silent", "solo_thinking", "solo") + PROTOCOLS:
            d = cells[c]
            print(f"{c:<12} n={d['n']:>4} P={d['played']:6.2f} Q={d['q']:6.2f} "
                  f"k={d['k']:>4} {d['rate']:>6.1f}% eff={d['eff']:.3f} (n={d['eff_n']})")
        print(f"\ngap: quality {gap_q:.2f}, success {gap_s:.1f} points, efficiency {gap_e:.3f}")
        print(f"  coordination (silent -> none)  quality {coord_q:+.2f}, residual {resid_q:+.2f}")
        print(f"  coordination (silent -> none)  {coord_s:+.1f} pts  {coord_e:+.3f}")
        print(f"  deliberation (residual)        {resid_s:+.1f} pts  {resid_e:+.3f}"
              f"   [half of {cells['solo_thinking']['rate'] - cells['solo_silent']['rate']:+.1f} pts,"
              f" {cells['solo_thinking']['eff'] - cells['solo_silent']['eff']:+.3f}]")
        for label, cc in CONTRASTS:
            r = res[cc]
            plain = label.replace(r"\textsc", "").translate(
                str.maketrans("", "", "{}\\"))
            print(f"  {plain:<40} dQ {r['d_q']:+6.2f} (n={r['n_q']}, perm p={r['pq']:.3g})  "
                  f"n={r['pairs']} {r['d_succ']:+.1f} pts (b/c {r['n01']}/{r['n10']}, "
                  f"p={r['pm']:.4f})  {r['d_eff']:+.3f} (n={r['n_eff']}, p={r['pw']:.4f})")
        if missing:
            print(f"\nWARNING: no exact optimum for {sorted(missing)}")
        return

    def row(name, d):
        return (f"{name} & {d['n']} & {d['played']:.2f} & {d['q']:.2f} & "
                f"{d['k']} & {pct(d['rate'])} & {d['eff']:.3f}")

    body = [row("Solo baseline (pooled)", solo),
            row(r"\quad \textsc{silent}", cells["solo_silent"]),
            row(r"\quad \textsc{thinking}", cells["solo_thinking"]),
            r"\midrule"]
    body += [row(f"Dyad, {NICE[p]}", cells[p]) for p in PROTOCOLS]

    caption = (
        "The dyad against its own solo baseline, S2--S4 pooled, on the clembench "
        "scores first and their two game-specific factors after. "
        f"The gap each protocol is asked to recover is {gap_q:.2f} points of Quality "
        f"Score, made of {gap_s:.1f} points of success and {gap_e:.3f} of move "
        "efficiency: what splitting six objects between two masked agents costs "
        "the same models on the same boards. Every episode in scope is played, so "
        "the Quality Score is over all of them. "
        "The two solo conditions are shown beneath the pooled "
        "baseline they make up: \\textsc{silent} is deliberation-free, as "
        "dyad-\\textsc{none} is, and \\textsc{thinking} is the deliberation-matched "
        "ceiling the three protocol conditions should be read against. "
        "S1 is omitted because it is at ceiling in all eight dyad "
        "cells of both models, leaving no gap to recover. The gap is computed before "
        "rounding, so it will not reproduce exactly from the printed cells.")
    note = (
        r"\emph{Scope.} The two strong models (Qwen3.8-27B-FP8 and "
        r"DeepSeek-V4.1-Flash), grid rendering, S2--S4, so each level contributes "
        r"15 boards. Solo pools "
        r"\textsc{silent} and \textsc{thinking} as the baseline the dyad is "
        r"measured against. "
        r"\emph{Efficiency} is the attainable optimum of the episode's game over "
        r"its move count (the exact joint optimum for solo, where the two "
        r"coincide), the scoring denominator of \S5.4 --- not the stored per-object "
        r"BFS bound, which is for budgeting and understates efficiency here by "
        r"0.07--0.08. It is conditioned on success, so the cells rest on "
        r"different board sets; the one contrast whose direction turns on this is "
        r"\textsc{structured} against \textsc{none}, which on the 75 boards solved "
        r"under both is $-0.027$ at a paired permutation $p = 0.063$. That is "
        r"contrast A4 of the confirmatory set, which Holm retains as a null "
        r"($p_\mathrm{Holm} = 0.56$): the direction is reported throughout and "
        r"the cost is not established.")

    print("% Generated by analysis/thesis/tables_comm.py -- do not hand-edit.")
    print("% Floats: tab:clembench, tab:goals, tab:solo (the inventory's tab 6), tab:scratchpad (tab 7),")
    print("% tab:recovery (tab 8),")
    print("% tab:decomposition, tab:interaction (tab 9), tab:dialogue,")
    print("% tab:carries (tab 10), tab:ladder (tab 11),")
    print("% tab:teamsize (tab 12), tab:teamsize-steps, tab:teamsize-tests,")
    print("% tab:seal-greedy, tab:cut-shape; then Appendix B,")
    print("% the annotation lexicon, which is a section rather than a float.")
    if missing:
        print(f"% WARNING: no exact joint optimum for {sorted(missing)}; "
              "those episodes are excluded from the efficiency columns.")
    print()
    print(clembench_table(opt, False))
    print(goals_table(False))
    print(solo_table(opt, False))
    print(scratchpad_table(opt, False))
    print(table(caption, "tab:recovery", "lrrrrrr",
                r"Condition & $n$ & \% Played & Quality & Solved & Success & Efficiency",
                body, note))

    # ---- tab:decomposition -------------------------------------------------
    body2 = [f"{label} & {signed(res[cc]['d_q'], 2)} & {fp(res[cc]['pq'])} & "
             f"{signed(res[cc]['d_succ'], 1)} & "
             f"{res[cc]['n01']}\\,/\\,{res[cc]['n10']} & {fp(res[cc]['pm'])} & "
             f"{signed(res[cc]['d_eff'], 3)} & {fp(res[cc]['pw'])}"
             for label, cc in CONTRASTS]

    ch = {p: res[(p, "solo_thinking")] for p in PROTOCOLS if p != "none"}
    best = max(ch.values(), key=lambda r: r["d_eff"])
    worst = min(ch.values(), key=lambda r: r["d_eff"])
    delib = res[("solo_thinking", "solo_silent")]
    # The claims cap2 words, checked rather than asserted.
    assert ch["freeform"]["pq"] > 0.05 and ch["hybrid"]["pq"] > 0.05, ch
    assert ch["structured"]["pq"] < 0.05 and ch["structured"]["d_succ"] <= 0, ch
    assert ch["freeform"]["d_succ"] >= 0 and ch["hybrid"]["d_succ"] >= 0, ch
    cap2 = (
        "The deliberation-matched contrasts, on the Quality Score first and its "
        "two game-specific factors after. Each row holds the writing surface "
        "fixed and varies one thing, so that what a channel buys is not confused "
        "with what any place to write buys: \\textsc{silent} and dyad-\\textsc{none} "
        "both lack one, \\textsc{thinking} and the three protocols both have one. "
        f"Pairs are (model, level, board), S2--S4, {res[CONTRASTS[0][1]]['pairs']} "
        "per contrast. "
        "The first row, the middle three and the last are the three quantities the "
        "design was built to separate. "
        f"\\emph{{Coordination}}: splitting the board costs {abs(coord_q):.2f} points of "
        "Quality Score with no channel to repair it "
        f"({abs(coord_s):.1f} of success and {abs(coord_e):.3f} of efficiency). "
        f"\\emph{{Deliberation}}: of the {gap_q:.2f} points that "
        "Table~\\ref{tab:recovery} measures against the pooled baseline, "
        f"{abs(coord_q):.2f} are coordination and the remaining {abs(resid_q):.2f} "
        "the half-share of the scratchpad that baseline carries, being half "
        "\\textsc{thinking} episodes; the scratchpad on its own is worth "
        f"{signed(delib['d_q'], 2)} ($p {psci(delib['pq'])}$). "
        "\\emph{What a channel recovers}: the middle block. Prose recovers the "
        "coordination penalty entire on the Quality Score --- "
        f"\\textsc{{freeform}} ends {signed(ch['freeform']['d_q'], 2)} and "
        f"\\textsc{{hybrid}} {signed(ch['hybrid']['d_q'], 2)} against its matched "
        "solo baseline, neither distinguishable from it --- and the schema "
        f"recovers none of it ({signed(ch['structured']['d_q'], 2)}, "
        f"$p {psci(ch['structured']['pq'])}$). The factors show how: prose "
        "trades a little efficiency "
        f"({signed(best['d_eff'], 3)} at best, {signed(worst['d_eff'], 3)} at worst "
        "across the three protocols) for success at or above the baseline, and "
        "the schema loses on efficiency without a success gain to pay for it.")
    note2 = (
        r"\emph{Tests.} $\Delta$ is the first condition minus the second. "
        r"Quality: two-sided paired permutation test on the "
        r"per-board differences, all pairs, since every episode here is played. "
        r"Success: exact conditional McNemar on the discordant pairs "
        r"(the $b\,/\,c$ column). Efficiency: two-sided paired permutation test "
        r"on the mean difference over the boards solved under \emph{both} "
        r"conditions, 73--87 per contrast. It is the test \S5.5 declares, and "
        r"it counts a tied board rather than dropping it. "
        r"\emph{The success column is underpowered by construction.} With $b+c$ "
        r"discordant pairs the smallest attainable two-sided exact $p$ is "
        r"$2^{1-(b+c)}$, so the freeform row could not have fallen below $0.25$ "
        r"however the pairs had gone; read those rows as absence of evidence, not "
        r"evidence of absence. "
        r"\emph{Exploratory, with one exception.} The first row is contrasts "
        r"A1 and A2 of Table~\ref{tab:confirmatory} --- the same pairs, the "
        r"same tests --- and carries the Holm correction there ($p_\mathrm{Holm} "
        r"= 0.56$ on success, $0.0056$ on efficiency). The other four are "
        r"outside the confirmatory set, carry no correction and support no "
        r"confirmatory claim.")
    print(table(cap2, "tab:decomposition", "lrrrrrrr",
                r"& \multicolumn{2}{c}{Quality Score} & \multicolumn{3}{c}{Success} & "
                r"\multicolumn{2}{c}{Efficiency} \\ \cmidrule(lr){2-3} "
                r"\cmidrule(lr){4-6} \cmidrule(lr){7-8} "
                r"Contrast & $\Delta$ & $p$ & $\Delta$ & $b\,/\,c$ & $p$ & $\Delta$ & $p$",
                body2, note2, tight=True))

    # ---- tab:interaction ---------------------------------------------------
    # Dyad only, and S1 included: the argument is that the protocol effect has
    # no purchase where the board hides nothing, which is only visible if the
    # level at ceiling is on the page.
    def lvcell(p, lv):
        return cell([e for e in eps_all if e["cond"] == p and e["level"] == lv])

    def lvrow(name, c, levels):
        return (f"{name} & " + " & ".join(
            f"{lvcell(c, lv)['q']:.2f} ({lvcell(c, lv)['k']}/{lvcell(c, lv)['n']})"
            for lv in levels))

    def lvbody(levels):
        return ([lvrow(r"Solo, \textsc{silent}", "solo_silent", levels),
                 lvrow(r"Solo, \textsc{thinking}", "solo_thinking", levels),
                 r"\midrule"]
                + [lvrow(f"Dyad, \\textsc{{{p}}}", p, levels) for p in PROTOCOLS])

    # Solo first, at the same denominator: 15 boards x 2 models = 30, as each
    # dyad cell.  SILENT is the baseline dyad-NONE is matched to and THINKING
    # the one the three protocols are, which is the pairing tab:decomposition
    # uses; pooling them here would put a /60 next to seven /30 columns.
    body3 = lvbody(ALL_LEVELS)
    assert all(e["q"] is not None for e in eps_all), "an abort in solo/dyad"

    Q = {(p, lv): lvcell(p, lv)["q"] for p in PROTOCOLS + ("solo_silent",)
         for lv in ALL_LEVELS}
    spread = {lv: max(Q[(p, lv)] for p in PROTOCOLS) - min(Q[(p, lv)] for p in PROTOCOLS)
              for lv in ALL_LEVELS}
    prose = {lv: (Q[("freeform", lv)] + Q[("hybrid", lv)]) / 2 for lv in ALL_LEVELS}
    assert max(spread, key=spread.get) == "S3" and spread["S1"] < 3, spread

    cap3 = (
        "The protocol ladder by spatial level in the dyad, pooled over the two "
        "strong models: Quality Score, with successes of 30 in parentheses. "
        r"\emph{The protocol effect exists only where spatial complexity does.} "
        f"At S1 the four protocols lie within {spread['S1']:.2f} points of each "
        f"other and the channel is worth nothing; the spread is {spread['S2']:.2f} "
        f"at S2, {spread['S3']:.2f} at S3 and {spread['S4']:.2f} at S4. At S3 and S4 "
        f"silence scores {Q[('none', 'S3')]:.2f} and {Q[('none', 'S4')]:.2f} where "
        f"prose scores {prose['S3']:.2f} and {prose['S4']:.2f}. The solo rows say why: "
        "splitting the board costs little at S1 and S2 "
        f"(dyad-\\textsc{{none}} {Q[('none', 'S1')]:.2f} against solo-\\textsc{{silent}} "
        f"{Q[('solo_silent', 'S1')]:.2f}, and {Q[('none', 'S2')]:.2f} against "
        f"{Q[('solo_silent', 'S2')]:.2f}) and most at S3 and S4 "
        f"({Q[('none', 'S3')]:.2f} against {Q[('solo_silent', 'S3')]:.2f}, "
        f"{Q[('none', 'S4')]:.2f} against {Q[('solo_silent', 'S4')]:.2f}). The channel "
        "is worth nothing until the board contains a structure a masked agent "
        "cannot see, and it is worth something exactly where the partition starts "
        "hiding one.")
    note3 = (
        r"\emph{Scope.} Grid rendering, the two strong models (Qwen3.8-27B-FP8 and "
        r"DeepSeek-V4.1-Flash), 15 boards per model per cell, so every cell in the "
        r"table has the same denominator: the two solo conditions are shown "
        r"separately rather than pooled for that reason. "
        r"Every episode in scope is played (\% Played is 100 in every cell), so "
        r"the Quality Score is over all 30 episodes. "
        r"\emph{Prose} is \textsc{freeform} and \textsc{hybrid} pooled; they are the "
        r"two conditions that carry unconstrained natural language. At the dyad the "
        r"interaction is real but small, because three objects per agent leaves each "
        r"player enough of the board to absorb most of what a channel would supply; "
        r"it grows as ownership gets finer, which is where this manipulation meets "
        r"the team-size one. "
        r"\emph{The two S1 non-successes} are solo-\textsc{silent} episodes that "
        r"the cycle detector ended after 30 and 32 moves of a 144- and 164-move "
        r"budget, with four of six objects already home and no illegal or blocked "
        r"move in either --- a termination failure on a board that was being "
        r"solved, not a spatial one.")
    print(table(cap3, "tab:interaction", "lrrrr",
                r"Condition & S1 & S2 & S3 & S4", body3, note3))

    # ---- tab:dialogue ------------------------------------------------------
    # The only table here whose unit is the message.  Protocol is held constant
    # by restricting to the two prose protocols: pooled over protocols the same
    # comparison is dominated by the fact that STRUCTURED both lacks most of
    # these features and succeeds less often, which is a protocol effect
    # wearing a dialogue-quality costume.
    msg = messages()
    su = [r for r in msg if r["su"]]
    fa = [r for r in msg if not r["su"]]

    def keyset(sel):
        return {(r["model"], r["suite"], r["lvl"], r["proto"], r["gid"]) for r in sel}

    ep_su, ep_fa = len(keyset(su)), len(keyset(fa))
    boards = len({(r["lvl"], r["gid"]) for r in msg})

    feats, delta = [], {}
    for label, k in DIALOGUE:
        a, b = sum(r[k] for r in su), sum(r[k] for r in fa)
        ra, rb = 100.0 * a / len(su), 100.0 * b / len(fa)
        pv = fisher_exact([[a, len(su) - a], [b, len(fa) - b]])[1]
        delta[k] = ra - rb
        feats.append((ra - rb, f"{label} & {pct(ra)} & {pct(rb)} & "
                               f"{signed(ra - rb, 1)} & {fp(pv)}"))
    body5 = [r for _, r in sorted(feats, key=lambda t: -t[0])] + [r"\midrule"]
    for label, k, dp in MEANS:
        a, b = [r[k] for r in su], [r[k] for r in fa]
        body5.append(f"{label} & {st.mean(a):.{dp}f} & {st.mean(b):.{dp}f} & "
                     f"{signed(st.mean(a) - st.mean(b), dp)} & "
                     f"{fp(mannwhitneyu(a, b).pvalue)}")

    cap5 = (
        "What separates successful from failed dialogue, once protocol is held "
        r"constant: almost nothing, and that is the finding. \emph{Two features "
        r"are higher in successes} --- disclosing one's own target and "
        r"acknowledging a teammate --- and \emph{three are higher in failures}: "
        "asking a teammate to move, naming a teammate's object, and reporting "
        "oneself blocked. Sequencing and corridor talk, the two features that "
        "carry the whole between-protocol contrast of "
        r"Table~\ref{tab:interaction}, do not separate outcomes at all "
        f"({signed(delta['seq'], 1)} and {signed(delta['corr'], 1)} points). "
        "Extra talk under pressure is a distress signal, not a repair. Rows are "
        r"ordered by $\Delta$, which is percentage points of messages in the "
        "upper block and words or coordinates in the lower two.")
    note5 = (
        r"\emph{Scope.} The two strong models (Qwen3.8-27B-FP8 and "
        r"DeepSeek-V4.1-Flash), grid rendering, S2--S4, the \textsc{freeform} and "
        r"\textsc{hybrid} protocols, and all three multiplayer suites --- the "
        r"corpus of \S8.4, not the dyad-only scope of the tables above. "
        r"\emph{Unit.} The row is the message: "
        f"{thousands(len(su))} from successes and {thousands(len(fa))} from failures, nested in "
        f"{ep_su} and {ep_fa} episodes on {boards} boards. Fisher exact on the "
        r"upper block, Mann--Whitney on the lower; neither accounts for the "
        r"nesting, so the exponents are an artefact of the message count and "
        r"nothing below $p = 0.0001$ should be read as episode-level evidence. "
        r"Read the ordering and the sizes of the differences, not the $p$ column. "
        r"\emph{The range restriction is the real limit on this table:} "
        f"the failure class is {ep_fa} episodes. "
        r"\emph{Length and coordinate density resolve as significant and are "
        r"negligible}, "
        f"{abs(st.mean([r['words'] for r in su]) - st.mean([r['words'] for r in fa])):.1f} "
        "words and "
        f"{abs(st.mean([r['ncoord'] for r in su]) - st.mean([r['ncoord'] for r in fa])):.1f} "
        "coordinates apart, in the same direction as the three distress "
        r"features. \emph{Correlational, and conditioned on a collider:} "
        r"failures concentrate on harder boards, so the three features higher in "
        r"failures are what a blocked board state elicits regardless of how well "
        r"the dialogue is going. The between-protocol contrast makes the same "
        r"point without conditioning on outcome, and it is the one the argument "
        r"rests on.")
    print(table(cap5, "tab:dialogue", "lrrrr",
                r"Message feature & Success & Failure & $\Delta$ & $p$",
                body5, note5))

    # ---- tab:carries -------------------------------------------------------
    # The inventory's tab 10.  Dyad, S1-S4, all three models: exactly the cells
    # qual_corpus.py verifies as Part II Table 12, so the table and R5's check
    # cannot disagree.  NONE is absent because it carries no messages at all.
    qc, allrows = corpus()
    dy = [r for r in allrows if r["suite"] == "dyad"]
    assert not [r for r in dy if r["proto"] == "none"], "a message under NONE"
    TALK = ("structured", "freeform", "hybrid")
    body6 = []
    for m in ("qwen38", "dsv41", "nemotron"):
        if body6:
            body6.append(r"\midrule")
        body6.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{qc.MODEL_LABEL[m]}}}}}")
        for p in TALK:
            sel = [r for r in dy if r["model"] == m and r["proto"] == p]
            rn = "---" if p == "freeform" else f"{qc.rate(sel, 'reqnone'):.0f}"
            body6.append(
                f"\\quad \\textsc{{{p}}} & {thousands(len(sel))} & "
                f"{st.mean(r['words'] for r in sel):.0f} & "
                f"{st.mean(r['ncoord'] for r in sel):.1f} & "
                + " & ".join(f"{qc.rate(sel, k):.0f}" for k in
                             ("tgt_disclosed", "names_other"))
                + f" & {rn} & {sum(r['temp'] for r in sel)}")
    sp = [r for r in dy if r["model"] in qc.STRONG and r["proto"] in qc.PROSE]
    ss = [r for r in dy if r["model"] in qc.STRONG and r["proto"] == "structured"]
    tgt = [qc.rate([r for r in dy if r["model"] == m and r["proto"] == p], "tgt_disclosed")
           for m in qc.STRONG for p in TALK]
    cap6 = (
        "What a dyad message carries, by model and protocol. "
        r"\emph{The protocols do not differ in information content}: every "
        f"strong-model cell discloses the speaker's own target in "
        f"{min(tgt):.0f}--{max(tgt):.0f}\\,\\% of messages. "
        r"\emph{They differ in the ability to order access to the board in time} "
        "--- reference to a teammate's object is routine in prose and all but "
        "absent from the schema, whose request slot is most often filled with "
        "\\texttt{none}; the sequencing and acknowledgement rates behind the same "
        "contrast are in the text. The word ``temporarily'' "
        f"occurs in {sum(r['temp'] for r in sp)} of {thousands(len(sp))} strong-model "
        f"prose messages and in {sum(r['temp'] for r in ss)} of "
        f"{thousands(len(ss))} structured ones.")
    note6 = (
        r"\emph{Scope.} Dyad, grid rendering, S1--S4, all messages of the three "
        r"talking protocols; \textsc{none} carries none. "
        r"\emph{Unit.} The message. Percentages are of messages in the cell; "
        r"\emph{words} and \emph{coords} are per-message means. "
        r"\emph{Annotation} is automatic: a fixed lexicon of regular expressions "
        r"plus a board replay that checks positional claims against the state at "
        r"the moment of utterance; no message was classified by hand. \emph{Target} "
        r"is a coordinate naming one of the speaker's own targets; \emph{other} "
        r"names a teammate's object; "
        r"\texttt{none} is the schema's slot filled as \texttt{REQUEST: none}, so it "
        r"is undefined under \textsc{freeform}. \emph{Temp.}\ counts messages "
        r"containing ``temporary'' or ``temporarily''. Lexicon with a worked "
        r"example each in Appendix~B. Detection is lexical and undercounts "
        r"paraphrase.")
    print(table(cap6, "tab:carries", "lrrrrrrr",
                r"Protocol & Msgs & Words & Coords & Target & Other & "
                r"\texttt{none} & Temp.",
                body6, note6))

    # ---- tab:ladder --------------------------------------------------------
    # The inventory's tab 11.  All levels, all three suites, all three models:
    # the generalisation claim is about the whole design, so nothing is scoped
    # out of it.
    lep = all_episodes(opt, SUITES)

    def lc(m, suites, protos, levels=ALL_LEVELS):
        return cell([e for e in lep if e["model"] == m and e["suite"] in suites
                     and e["cond"] in protos and e["level"] in levels])

    def qp(c):
        return f"{f2(c['q'])} / {c['played']:.2f}"

    body7 = []
    for m in ARMS:
        if body7:
            body7.append(r"\midrule")
        for s in SUITES + ("all",):
            ss_ = SUITES if s == "all" else (s,)
            cells_ = " & ".join(qp(lc(m, ss_, (p,))) for p in PROTOCOLS)
            name = MODEL_NAMES[m] if s == "dyad" else ""
            body7.append(f"{name} & {'all three' if s == 'all' else s} & {cells_}")

    # The claims the caption makes, computed rather than asserted.
    prose_top = all(max(lc(m, (s,), (p,))["q"] for p in ("none", "structured"))
                    < min(lc(m, (s,), (p,))["q"] for p in ("freeform", "hybrid"))
                    for m in STRONG for s in SUITES)
    assert prose_top
    below = [(m, s) for m in STRONG for s in SUITES
             if lc(m, (s,), ("structured",))["q"] < lc(m, (s,), ("none",))["q"]]
    inversions = [(m, s, lv) for m in STRONG for s in SUITES for lv in ALL_LEVELS
                  if lc(m, (s,), ("structured",), (lv,))["q"]
                  > min(lc(m, (s,), (q,), (lv,))["q"] for q in ("freeform", "hybrid"))]
    assert {lv for _, _, lv in inversions} <= {"S1"}, inversions
    assert all((lc(m, (s,), (p,))["played"] < 100) == (s == "6p")
               and lc(m, ("6p",), ("none",))["played"] <= lc(m, ("6p",), (p,))["played"]
               for m in STRONG for s in SUITES for p in PROTOCOLS)

    def key(e):
        return (e["suite"], e["level"], e["game_id"])

    tests = []
    for m in STRONG:
        rows = {p: [e for e in lep if e["model"] == m and e["cond"] == p] for p in PROTOCOLS}
        parts = []
        for a, b in (("freeform", "structured"), ("hybrid", "structured"),
                     ("structured", "none"), ("freeform", "hybrid")):
            d = qpairs(rows[a], rows[b], key)
            parts.append(f"\\textsc{{{a}}} against \\textsc{{{b}}} "
                         f"{signed(st.mean(d), 2)}, $p {psci(perm_p(d))}$")
        tests.append(f"{MODEL_NAMES[m]}: " + "; ".join(parts))

    def nm(m, s):
        return f"{MODEL_NAMES[m]} at {'the dyad' if s == 'dyad' else s}"

    below_txt = ("everywhere" if not below else "everywhere except "
                 + " and ".join(nm(m, s) for m, s in below)
                 + ", where the two are within "
                 + " and ".join(f"{lc(m, (s,), ('none',))['q'] - lc(m, (s,), ('structured',))['q']:.2f}"
                                for m, s in below) + " points")
    cap7 = (
        "The protocol ladder by suite and model, S1--S4 pooled, in the "
        "clembench format: each cell is Quality Score / \\% Played. "
        "Both prose protocols score above both \\textsc{structured} and "
        "\\textsc{none} in both strong models and every suite, and "
        f"\\textsc{{structured}} scores above \\textsc{{none}} {below_txt}. "
        f"Per level, \\textsc{{structured}} tops a prose protocol in {len(inversions)} "
        "strong-model cells, all at S1, where every protocol is within a few "
        "points of the ceiling. \\% Played falls below 100 only at 6p, and "
        "furthest under \\textsc{none}. The weak tier plays too few episodes for "
        "its Quality Score to carry the order: a cell's score is an average over "
        "the handful of episodes it played.")
    note7 = (
        r"\emph{Scope.} Grid rendering, 15 boards per level, so each suite cell "
        r"is 60 episodes and each \emph{all three} cell 180. The Quality Score is "
        r"over played episodes only (Table~\ref{tab:clembench}), which is why it "
        r"must be read with \% Played beside it. "
        r"\emph{Tests} (two-sided paired permutation test on the per-board Quality "
        r"differences, pairs (suite, level, board) played under both protocols, "
        r"all suites pooled, mean difference given): "
        + ". ".join(tests) + ". "
        r"\emph{Exploratory.} These are the generalisation of the dyad result, "
        r"not the confirmatory contrasts of Table~\ref{tab:confirmatory}.")
    print(table(cap7, "tab:ladder", "llrrrr",
                r"Model & Suite & \textsc{none} & \textsc{structured} & "
                r"\textsc{freeform} & \textsc{hybrid}",
                body7, note7))
    print(teamsize_table(opt, False))
    print(teamsize_steps_table(opt))
    print(teamsize_tests_table(opt))
    print(seal_greedy_table())
    print(cut_shape_table())

    # ---- Appendix B --------------------------------------------------------
    print()
    appendix_lexicon(qc, allrows)


# ---------------------------------------------------------------------------
#  Appendix B: the annotation lexicon
#
#  The table lists, per feature, every term the regular expression matches,
#  because a reader checking whether a rate is credible wants the word list and
#  not the pattern.  The words are written out here rather than parsed out of
#  the regex: several patterns are cross-products of optional groups and
#  mechanical expansion produces two dozen unreadable variants of "i will wait".
#
#  Drift is caught instead of prevented.  Every term below is probed against
#  the compiled pattern at generation time, so a term deleted from the regex
#  fails the build; for the plain alternations the branch count is checked too,
#  so a term ADDED to the regex fails the build as well.  The structural
#  patterns -- those that match a shape rather than a vocabulary -- carry a
#  description in place of a word list and are exempt from both checks.
# ---------------------------------------------------------------------------
S = "structural"
DISPLAY = {"WAIT": "blocked"}
NUM = {4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
       9: "Nine", 10: "Ten", 11: "Eleven", 12: "Twelve"}

#   name, what it marks, terms, alternation?
#   A term is either a string, or (printed, probed) where the two differ.
LEXICON = [
    ("COORD", "a grid coordinate, in every spelling the models use",
     (S, r"\texttt{R5,C3}, \texttt{(R5,C3)}, \texttt{R5 C3}, \texttt{(R5, C3)} "
         r"--- the parentheses, the comma and the spaces are each optional"), False),
    ("CLAIM", "an object asserted to be somewhere: the unit the replay checks",
     (S, r"an object letter \texttt{A}--\texttt{F}, then at most 24 further "
         r"characters not crossing a sentence boundary, then ``at'' and a "
         r"coordinate"), False),
    ("REQ", "a politeness form addressed to the teammate",
     ["please", "could you", "can you", "kindly",
      (r"\textit{x\=uy\`ao} (Chinese ``need'')", "需要")], True),
    ("MOVEREQ", "a concrete instruction to move a named object in a named direction",
     (S, r"one of \emph{move}, \emph{shift}, \emph{push}, \emph{vacate}, "
         r"\emph{step}, then either an object letter or \texttt{X at (R5,C3)}, "
         r"then one of \emph{up}, \emph{down}, \emph{left}, \emph{right} "
         r"within twenty characters"), False),
    ("WAIT", "the speaker reports that it cannot act",
     [("i'm waiting / i am waiting / i will wait", "i'm wait"), "waiting",
      "cannot move", "can't move", "no legal move", "blocked", "stuck",
      "trapped", ("hold position / holding position", "holding position"),
      "stay put", "pass this turn"], False),
    ("ACK", "the speaker registers a teammate's prior contribution",
     ["thank", "thanks", ("acknowledg-", "acknowledg"), "as requested",
      "as you asked", "got it", "understood", "confirmed", "noted"], True),
    ("SEQ", "a marker that places two actions in an order",
     ["after", "then", "once", "first", "before", "next turn", "meanwhile",
      "afterwards", "subsequently"], True),
    ("CORR", "the corridor or the dead end named as such",
     ["corridor", ("dead end / dead-end / deadend", "dead-end"), "shaft",
      "cul-de-sac", ("column C\\textit{n}", "column C3"), "blind alley",
      "pocket"], True),
    ("THINK", "a reasoning trace that has leaked into the channel",
     [(r"\texttt{<think>}, \texttt{</think>}", "</think>"),
      ("okay let me / okay, let me", "okay, let me"), "let me think",
      ("first i need to figure / first, i need to figure",
       "first, i need to figure")], False),
    ("CJK", "a message written in a non-Latin script",
     (S, r"any character in \texttt{U+3000--U+9FFF} or "
         r"\texttt{U+FF00--U+FFEF}, the CJK and full-width blocks"), False),
    ("STAY", "the pseudo-action the move grammar does not offer",
     [(r"\texttt{move: <A--F> stay}", "move: B stay"),
      (r"\texttt{move: <A--F> wait}", "move: B wait")], False),
    ("STAYARG", "with a \\textsc{stay} match, a speaker who plays the "
                "pseudo-action while saying it is illegal",
     ["not a valid", "invalid", "must move", "rules require"], True),
    ("TEMP", "the vocabulary of the temporary detour",
     ["temporary", "temporarily"], False),
    ("TEMPRET", "the two-clause request: vacate a cell \\emph{and} return to it",
     (S, r"\emph{temporary} or \emph{temporarily} within ninety characters of "
         r"\emph{return}, \emph{restore}, \emph{back}, \emph{then you can} or "
         r"\emph{and back}, in either order, without crossing a sentence "
         r"boundary"), False),
    ("REQNONE", "the schema's request slot left empty",
     [(r"\texttt{REQUEST: none}", "REQUEST: none")], False),
    ("QUESTION", "a question anywhere in the message",
     (S, r"a question mark"), False),
    ("TARGET DISCLOSED", "the speaker states where one of its own objects has "
                         "to end up",
     (S, r"\emph{not a word list}: true when a coordinate matched by "
         r"\textsc{coord} equals the target of one of the speaker's own "
         r"objects, by the instance file. A speaker who names its target in "
         r"words alone does not count"), False),
    ("NAMES OTHER", "the speaker refers to an object it does not control",
     (S, r"\emph{not a word list}: true when a bare object letter in the "
         r"message belongs to another player, by the ownership partition of "
         r"the instance file"), False),
    ("OWN-CLAIM ACCURACY", "whether what a speaker says about its own object "
                           "is true",
     (S, r"\emph{not a word list}: each \textsc{claim} match naming an object "
         r"the speaker controls is compared with the board as the replay holds "
         r"it at that turn. \emph{The unit is the claim, not the message}, and "
         r"one message can carry several"), False),
    ("TEAMMATE-CLAIM ACCURACY", "whether what a speaker says about an object "
                                "it does not control is true",
     (S, r"\emph{not a word list}: the same comparison for a claim about "
         r"another player's object, whose position a masked speaker can only "
         r"have learnt from what was said. Kept apart from the own-object rate "
         r"throughout: the gap between the two is what separates the tiers"),
     False),
    ("REQUEST COMPLIANCE", "whether a request to move was acted on",
     (S, r"\emph{not a word list}: each \textsc{movereq} match opens a window "
         r"on the object and direction it names, and the request counts as "
         r"honoured if the replay sees that exact move within six turns. The "
         r"unit is the request"), False),
    ("WORDS", "message length",
     (S, r"\emph{not a word list}: whitespace-separated tokens, reported as a "
         r"per-message mean"), False),
]

# The two rows above are computed from the instance file rather than matched,
# so they have no pattern to probe and no match to quote; the boolean the
# extraction already stored is what locates an example message.
DERIVED_KEYS = {"TARGET DISCLOSED": "tgt_disclosed", "NAMES OTHER": "names_other",
                "OWN-CLAIM ACCURACY": "own_bad", "TEAMMATE-CLAIM ACCURACY": "oth_bad",
                "REQUEST COMPLIANCE": "movereq", "WORDS": "words"}

# Where each feature surfaces, and its worked example.  Separate from the table
# because neither fits a column beside the word list.
SURFACES = {
    "COORD":   r"the \emph{Coords} column of Table~\ref{tab:carries}, and "
               r"\emph{mean coordinates} in Table~\ref{tab:dialogue}",
    "CLAIM":   r"the detector behind the two claim-accuracy rows below",
    "REQ":     r"the \emph{request} row of Table~\ref{tab:dialogue}",
    "MOVEREQ": r"the detector behind the \textsc{request compliance} row below",
    "WAIT":    r"the \emph{reports itself blocked} row of Table~\ref{tab:dialogue}",
    "ACK":     r"the \emph{acknowledgement} row of Table~\ref{tab:dialogue}",
    "SEQ":     r"the \emph{sequencing marker} row of Table~\ref{tab:dialogue}",
    "CORR":    r"the \emph{corridor or dead end named} row of Table~\ref{tab:dialogue}",
    "THINK":   r"the sanity check of \S8.4",
    "CJK":     r"the sanity check of \S8.4",
    "STAY":    r"\S9.3.1",
    "STAYARG": r"\S9.3.1",
    "TEMP":    r"the \emph{Temp.} column of Table~\ref{tab:carries}",
    "TEMPRET": r"\S8.4's account of what the schema cannot carry",
    "REQNONE": r"the \texttt{none} column of Table~\ref{tab:carries}",
    "QUESTION": r"nothing: it is computed for every message and reported "
                r"nowhere, and is listed so that the annotation is complete",
    "TARGET DISCLOSED": r"the \emph{Target} column of Table~\ref{tab:carries} "
                        r"and the \emph{target disclosed} row of "
                        r"Table~\ref{tab:dialogue}",
    "NAMES OTHER": r"the \emph{Other} column of Table~\ref{tab:carries} and "
                   r"the \emph{names a teammate's object} row of "
                   r"Table~\ref{tab:dialogue}",
    "OWN-CLAIM ACCURACY": r"the own-object accuracies of \S8.5 and \S10.2",
    "TEAMMATE-CLAIM ACCURACY": r"the teammate-object accuracies of \S8.5 and "
                               r"\S10.2 --- the measurement those sections "
                               r"turn on",
    "REQUEST COMPLIANCE": r"the compliance rates of \S8.4",
    "WORDS": r"the \emph{Words} column of Table~\ref{tab:carries} and "
             r"\emph{mean words} in Table~\ref{tab:dialogue}",
}

TEX = {"\\": r"\textbackslash{}", "$": r"\$", "&": r"\&", "#": r"\#",
       "%": r"\%", "_": r"\_", "^": r"\textasciicircum{}",
       "~": r"\textasciitilde{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def esc(txt):
    """Model output into LaTeX.  Curly braces are not in TEX because the terms
    above are written as LaTeX already; message text has none."""
    txt = txt.translate(str.maketrans(
        {"’": "'", "‘": "`", "“": "``", "”": "''",
         "—": "---", "–": "--", "→": " to ", " ": " ",
         "{": "", "}": ""}))
    return "".join(TEX.get(c, c) if c in TEX else (c if c.isascii() else "")
                   for c in txt)


def check(qc):
    """Every printed term must still match the pattern it claims to come from,
    and a plain alternation must have exactly as many branches as terms."""
    for name, _, terms, alt in LEXICON:
        if terms[0] is S:          # a shape, or not lexical at all: nothing to probe
            continue
        rx = getattr(qc, name)
        for t in terms:
            probe = t[1] if isinstance(t, tuple) else t
            assert rx.search(probe), f"{name}: lexicon lists {probe!r}, pattern does not match it"
        if alt:
            assert len(terms) == rx.pattern.count("|") + 1, (
                f"{name}: {len(terms)} terms listed, pattern has "
                f"{rx.pattern.count('|') + 1} branches -- a term was added to the regex")


def appendix_lexicon(qc, rows):
    check(qc)
    print(r"\section*{B\quad The annotation lexicon}")
    print(r"\addcontentsline{toc}{section}{B\quad The annotation lexicon}")
    print()
    print(r"Every message of the three talking protocols is annotated in two "
          r"steps, and the two tables below correspond to them. "
          r"Table~\ref{tab:lexicon} is what a fixed set of regular expressions "
          r"finds in the text of a message; Table~\ref{tab:derived} is what a "
          r"replay of the board, and the ownership partition of the instance "
          r"file, then make of those matches. No message was classified by "
          r"hand and no model was asked to judge another's output, so every "
          r"figure in \S8.4, \S8.5 and \S10.2 is a function of the released "
          r"episode directories and of these two pages. Matching is "
          r"case-insensitive and unanchored: a term matches wherever it occurs "
          r"in the message.")
    print()

    # Two tables, split where the appendix already draws the line: what a
    # pattern matches in the text, and what the replay computes about it.
    SPEC = (r"@{}>{\raggedright\arraybackslash}p{0.15\textwidth}"
            r">{\raggedright\arraybackslash}p{0.225\textwidth}"
            r">{\raggedright\arraybackslash}p{0.45\textwidth}@{}")

    def rows_for(want_derived):
        out = []
        for name, marks, terms, _ in LEXICON:
            if (name in DERIVED_KEYS) is not want_derived:
                continue
            if out:                      # a rule between features: the cells wrap
                out.append(r"\midrule")  # over several lines and run together otherwise
            cell = terms[1] if terms[0] is S else ", ".join(
                t[0] if isinstance(t, tuple) else t for t in terms)
            out.append(f"\\textsc{{{DISPLAY.get(name, name.lower())}}} & {marks} & {cell}")
        return out

    shapes = sum(1 for e in LEXICON if e[2][0] is S) - len(DERIVED_KEYS)
    print(table(
        "The annotation lexicon: what each pattern marks in a message and every "
        f"term that marks it. {NUM[shapes]} of the patterns match a shape "
        "rather than a word list and are described in place of being "
        "enumerated. What the replay then computes from these matches is "
        "Table~\\ref{tab:derived}.",
        "tab:lexicon", SPEC,
        r"Pattern & What it marks & Terms matched",
        rows_for(False),
        r"\emph{Terms} are the alternatives of the regular expression, "
        r"written out. A trailing hyphen marks a stem: \emph{acknowledg-} "
        r"matches \emph{acknowledge}, \emph{acknowledged} and "
        r"\emph{acknowledgement} alike. Where a term admits spelling variants "
        r"the variants are given in full. Each term is probed against the "
        r"compiled pattern when this table is generated, and for the "
        f"{NUM[sum(1 for e in LEXICON if e[3])].lower()} plain alternations the "
        r"number of terms is checked against the number of branches, so the "
        r"table cannot drift from the code in either direction."))
    print()
    print(table(
        "The measures computed from the matches: what the board replay and the "
        "ownership partition of the instance file make of what the patterns of "
        "Table~\\ref{tab:lexicon} find. None of these is a word list, and none "
        "can be read off a message alone.",
        "tab:derived", SPEC,
        r"Measure & What it captures & How it is computed",
        rows_for(True),
        r"\emph{Unit.} The first two are properties of a message, as every "
        r"percentage in Tables~\ref{tab:carries} and \ref{tab:dialogue} is. "
        r"The two claim accuracies are per \emph{claim} and compliance is per "
        r"\emph{request}, so a single message can contribute several "
        r"observations to them and none to the others."))
    print()

    print(r"\noindent Where each feature is used, with the first message in "
          r"the corpus that it matches:")
    print(r"\begingroup\raggedright")
    print(r"\begin{description}[leftmargin=0pt,style=nextline,itemsep=3pt]")
    for name, _, _, _ in LEXICON:
        rx = getattr(qc, name, None)
        print(f"\\item[\\textsc{{{DISPLAY.get(name, name.lower())}}}] {SURFACES[name]}.")
        if rx is None:
            hit = next((r for r in rows if r[DERIVED_KEYS[name]]), None)
            frag = esc(hit["text"].replace("\n", " ").strip()[:150])
            print(r"\par\vspace{2pt}\noindent \emph{Example message.} ``"
                  + frag + " \\dots''")
            continue
        hit = next((r for r in rows if rx.search(r["text"])), None)
        if hit:
            m = rx.search(hit["text"])
            a = max(0, m.start() - 36)
            b = min(len(hit["text"]), m.end() + 36)
            frag = esc(hit["text"][a:b].replace("\n", " ").strip())
            print(r"\par\vspace{2pt}\noindent \emph{Example.} ``"
                  + ("\\dots " if a else "") + frag
                  + (" \\dots" if b < len(hit["text"]) else "") + "''")
        else:
            print(r"\par\vspace{2pt}\noindent \emph{No message in the corpus "
                  r"matches, which is the sanity check itself.}")
    print(r"\end{description}")
    print(r"\endgroup")
    print()

    print(r"\subsection*{The board replay}")
    print()
    print(r"A positional claim is true or false only relative to a board, so "
          r"the annotation replays each episode rather than reading its "
          r"messages alone. Starting from the instance file's "
          r"\texttt{start\_positions}, every move event in the transcript is "
          r"applied in order, so that the replay holds the board the game "
          r"engine held. The state is snapshotted at each \emph{get message} "
          r"event --- the moment the speaker was handed the board it is about "
          r"to describe --- and every \textsc{claim} match in the message it "
          r"then produces is checked against that snapshot rather than against "
          r"the state at the end of the turn. Each claim is scored against the "
          r"ownership partition as a claim about the speaker's own object or "
          r"about a teammate's, which is the distinction \S8.5 rests on: the "
          r"first is visible to the speaker, the second can only have been "
          r"learnt from what was said.")
    print()
    print(r"\noindent Two consequences of the design belong here, where the "
          r"procedure is defined. Detection is lexical, so it undercounts "
          r"paraphrase and every rate in this thesis is a floor rather than an "
          r"estimate. And a claim is checked against the board as it is rather "
          r"than as it is about to be, so a request or a plan concerning a "
          r"teammate's object --- ``after you move X to (R3,C4)'' --- is "
          r"scored as an error although it asserts nothing false. Excluding "
          r"claims that carry an explicit future marker raises strong-tier "
          r"accuracy on teammates' objects by between one and six points "
          r"depending on how futurity is detected, so the reported figure "
          r"understates it.")



if __name__ == "__main__":
    main()
