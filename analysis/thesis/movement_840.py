#!/usr/bin/env python3
"""R4: every outcome number, on the 826-episode analysis set and on 840.

    python3 analysis/thesis/movement_840.py > thesis/movement_840.md

The 826 column is the published figure (S2 #10 dropped everywhere); the 840
column is the same computation with the replacement board's 42 episodes folded
in.  The 826 column is therefore also the check: if it does not reproduce
Table 2 of results_three_models_grid.md digit for digit, the loader is wrong
and the diff means nothing.  --check asserts exactly that and prints nothing
else.

Outcome numbers only.  Oscillation and deadlock moved for every episode in the
rescore (corrected metrics, S5.4.3 Path A) and are not this list's business.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics as st

from scipy.stats import binomtest, fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUITES = ("solo", "dyad", "3p", "6p")
MODELS = {                      # label -> (results prefix, model directory)
    "Nemotron-S3": ("results_nemotron_fixthink_grid_n15", "nemotron-super-3-120b-bedrock"),
    "Qwen3.8": ("results_qwen38_grid_n15", "Qwen3.8-27B-FP8"),
    "DeepSeek-V4.1": ("results_dsv41_grid_n15", "DeepSeek-V4.1-Flash"),
}
EXCLUDED = ("S2", 10)           # the replaced board: in 840, out of 826
Z = 1.959963984540054           # two-sided 95%


def wilson(k, n):
    if not n:
        return float("nan"), float("nan")
    p, z2 = k / n, Z * Z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def load():
    """One record per scored episode across all twelve trees."""
    out = []
    for label, (prefix, model_dir) in MODELS.items():
        for suite in SUITES:
            pat = os.path.join(ROOT, f"{prefix}_{suite}", model_dir, "*", "*", "instance_*", "scores.json")
            for f in glob.glob(pat):
                exp = f.split(os.sep)[-3]
                if suite == "solo":
                    _, level, cond, _ = exp.split("_")
                    cond = cond.lower()
                else:
                    level, cond = exp.split("_")[0], exp.split("_")[1]
                if not re.fullmatch(r"S[1-4]", level):
                    raise SystemExit(f"unparsed experiment name: {exp}")
                s = json.load(open(f))["episode scores"]
                out.append(dict(
                    model=label, suite=suite, level=level, cond=cond,
                    gid=int(re.search(r"instance_(\d+)", f).group(1)),
                    success=int(s.get("Success", 0)), aborted=int(s.get("Aborted", 0)),
                    played=int(s.get("Played", 0)), main=s.get("Main Score"),
                    moves=s.get("Move Count"),
                ))
    return out


def analysis_set(rows, full):
    return rows if full else [r for r in rows if (r["level"], r["gid"]) != EXCLUDED]


def cell(rows, opt):
    """The row of Table 2 for one group of episodes.

    Main score is the published one: min(100, 100 * joint optimum / moves) on a
    success and 0 on a played loss, averaged over played episodes.  NOT
    scores.json's `Main Score`, which divides by the stored per-object BFS sum
    (S5.4's budgeting bound) and runs 6 points low on the strong models.
    """
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    played = [r for r in rows if r["played"]]
    aborted = sum(r["aborted"] for r in rows)
    lo, hi = wilson(len(succ), n)
    mains = [min(100, 100 * opt[(r["level"], r["gid"])] / r["moves"]) if r["success"] else 0
             for r in played]
    return dict(
        n=n, succ=len(succ), rate=100 * len(succ) / n if n else float("nan"),
        lo=lo, hi=hi, aborted=aborted, lost=n - len(succ) - aborted,
        main=st.mean(mains) if mains else float("nan"),
        moves=st.median([r["moves"] for r in succ]) if succ else float("nan"),
    )


def key(r):
    return (r["suite"], r["level"], r["cond"], r["gid"])


def paired(rows, a, b):
    """McNemar over the episodes both models played on the same board+cell."""
    ma = {key(r): r["success"] for r in rows if r["model"] == a}
    mb = {key(r): r["success"] for r in rows if r["model"] == b}
    shared = sorted(set(ma) & set(mb))
    both = sum(1 for k in shared if ma[k] and mb[k])
    aonly = sum(1 for k in shared if ma[k] and not mb[k])
    bonly = sum(1 for k in shared if mb[k] and not ma[k])
    neither = len(shared) - both - aonly - bonly
    p = binomtest(aonly, aonly + bonly, 0.5).pvalue if aonly + bonly else 1.0
    n = len(shared)
    po = (both + neither) / n
    pa = ((both + aonly) * (both + bonly) + (neither + bonly) * (neither + aonly)) / (n * n)
    kappa = (po - pa) / (1 - pa) if pa != 1 else float("nan")
    return dict(n=n, both=both, aonly=aonly, bonly=bonly, neither=neither, p=p, kappa=kappa)


def suite_pair(rows, model, sa, sb, level=None):
    """3p vs 6p for one model: unpaired Fisher over all episodes, and exact
    McNemar over the board+protocol cells the two suites share."""
    sel = [r for r in rows if r["model"] == model and (level is None or r["level"] == level)]
    a = [r for r in sel if r["suite"] == sa]
    b = [r for r in sel if r["suite"] == sb]
    ka, kb = sum(r["success"] for r in a), sum(r["success"] for r in b)
    fisher = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
    ma = {(r["level"], r["cond"], r["gid"]): r["success"] for r in a}
    mb = {(r["level"], r["cond"], r["gid"]): r["success"] for r in b}
    shared = set(ma) & set(mb)
    x = sum(1 for k in shared if ma[k] and not mb[k])
    y = sum(1 for k in shared if mb[k] and not ma[k])
    mcnemar = binomtest(x, x + y, 0.5).pvalue if x + y else 1.0
    return dict(ka=ka, na=len(a), kb=kb, nb=len(b), fisher=fisher, mcnemar=mcnemar)


def optima():
    opt = {}
    for ln in open(os.path.join(ROOT, "analysis/efficiency_baseline/joint_optimum.jsonl")):
        d = json.loads(ln)
        if d.get("status") == "exact":
            opt[(d["level"], d["game_id"])] = d["opt"]
    return opt


# Table 2 of results_three_models_grid.md, as published on 826.  The loader has
# to reproduce these before any 840 figure is worth reading.
PUBLISHED = {   # (successes, n, aborted, lost, main, median moves)
    ("Nemotron-S3", "solo"): (4, 118, 81, 33, 10.3, 32), ("Nemotron-S3", "dyad"): (11, 236, 211, 14, 35.5, 33),
    ("Nemotron-S3", "3p"): (6, 236, 223, 7, 41.3, 30), ("Nemotron-S3", "6p"): (6, 236, 220, 10, 29.3, 40),
    ("Qwen3.8", "solo"): (112, 118, 0, 6, 86.3, 47), ("Qwen3.8", "dyad"): (226, 236, 0, 10, 81.6, 51),
    ("Qwen3.8", "3p"): (210, 236, 0, 26, 74.1, 51), ("Qwen3.8", "6p"): (198, 236, 14, 24, 71.4, 53),
    ("DeepSeek-V4.1", "solo"): (114, 118, 0, 4, 88.4, 47), ("DeepSeek-V4.1", "dyad"): (227, 236, 0, 9, 84.1, 49),
    ("DeepSeek-V4.1", "3p"): (216, 236, 0, 20, 77.6, 50), ("DeepSeek-V4.1", "6p"): (197, 236, 17, 22, 72.6, 53),
}


def check(rows, opt):
    bad = 0
    for (model, suite), want in PUBLISHED.items():
        c = cell([r for r in analysis_set(rows, False)
                  if r["model"] == model and r["suite"] == suite], opt)
        got = (c["succ"], c["n"], c["aborted"], c["lost"], round(c["main"], 1), round(c["moves"]))
        if got != want:
            print(f"MISMATCH {model} {suite}: published {want} != computed {got}")
            bad += 1
    total = sum(1 for r in analysis_set(rows, False) if r["success"])
    if total != 1527:
        print(f"MISMATCH total successes: published 1527 != computed {total}")
        bad += 1
    print("826 reproduces the published table" if not bad else f"{bad} mismatches")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the 826 column only")
    args = ap.parse_args()

    rows, opt = load(), optima()
    missing = {(r["level"], r["gid"]) for r in rows} - set(opt)
    if missing:
        raise SystemExit(f"no exact joint optimum for {sorted(missing)}")
    if args.check:
        raise SystemExit(check(rows, opt))

    old, new = analysis_set(rows, False), analysis_set(rows, True)
    print("# R4 — every outcome number re-derived on 840\n")
    print("Generated by `analysis/thesis/movement_840.py` from the twelve scored grid trees. "
          "**826** is the published analysis set (S2 #10 excluded); **840** folds in "
          "the replacement board's 42 episodes. `--check` asserts the 826 column "
          "against the published table before any of this is worth reading.\n")
    print(f"Episodes: {len(old)} -> {len(new)} ({len(new) - len(old)} new), "
          f"per model {len(old) // 3} -> {len(new) // 3}.\n")

    print("## 1. Table 2 — success, CI, aborts, losses, main score, median moves\n")
    print("| Model | Suite | Success 826 | Success 840 | CI 826 | CI 840 | Ab. | Lost | Main 826 | Main 840 | Mv |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in MODELS:
        for suite in list(SUITES) + ["total"]:
            sel = (lambda rs: [r for r in rs if r["model"] == model]) if suite == "total" else \
                  (lambda rs: [r for r in rs if r["model"] == model and r["suite"] == suite])
            o, n = cell(sel(old), opt), cell(sel(new), opt)
            print(f"| {model if suite == 'solo' else ''} | {'**total**' if suite == 'total' else suite} "
                  f"| {o['succ']}/{o['n']} ({o['rate']:.1f}%) | {n['succ']}/{n['n']} ({n['rate']:.1f}%) "
                  f"| [{o['lo']:.1f}, {o['hi']:.1f}] | [{n['lo']:.1f}, {n['hi']:.1f}] "
                  f"| {o['aborted']}->{n['aborted']} | {o['lost']}->{n['lost']} "
                  f"| {o['main']:.1f} | {n['main']:.1f} | {o['moves']:.0f}->{n['moves']:.0f} |")

    print("\n## 2. The S2 cells — the only cells whose n changes (14 -> 15)\n")
    print("| Model | Suite | Condition | 826 | 840 |")
    print("|---|---|---|---|---|")
    for model in MODELS:
        for suite in SUITES:
            conds = sorted({r["cond"] for r in new if r["model"] == model and r["suite"] == suite})
            for cond in conds:
                f = lambda rs: [r for r in rs if r["model"] == model and r["suite"] == suite
                                and r["cond"] == cond and r["level"] == "S2"]
                o, n = cell(f(old), opt), cell(f(new), opt)
                mark = "" if o["succ"] == n["succ"] else "  **moves**"
                print(f"| {model} | {suite} | {cond} | {o['succ']}/{o['n']} | {n['succ']}/{n['n']}{mark} |")

    print("\n## 3. Grand totals\n")
    for name, rs in (("826", old), ("840", new)):
        succ = sum(1 for r in rs if r["success"])
        ab = sum(r["aborted"] for r in rs)
        print(f"- **{name}**: {len(rs)} episodes, {succ} successes, {ab} aborts, "
              f"{len(rs) - succ - ab} played losses.")

    print()
    for name, rs in (("826", old), ("840", new)):
        succ = [r for r in rs if r["success"]]
        hit = [r for r in succ if opt.get((r["level"], r["gid"])) == r["moves"]]
        under = [r for r in succ if r["moves"] < opt.get((r["level"], r["gid"]), 0)]
        print(f"- **{name}**: {len(succ)} successes, {len(hit)} hit the exact joint optimum, "
              f"{len(under)} finish below it (must be 0).")

    print("\n## 4. Paired model comparisons (matched episodes)\n")
    print("| Pair | Set | n | both | A only | B only | neither | McNemar p | kappa |")
    print("|---|---|---|---|---|---|---|---|---|")
    pairs = [("Qwen3.8", "DeepSeek-V4.1"), ("Qwen3.8", "Nemotron-S3"), ("DeepSeek-V4.1", "Nemotron-S3")]
    for a, b in pairs:
        for name, rs in (("826", old), ("840", new)):
            r = paired(rs, a, b)
            print(f"| {a} vs {b} | {name} | {r['n']} | {r['both']} | {r['aonly']} | {r['bonly']} "
                  f"| {r['neither']} | {r['p']:.3f} | {r['kappa']:.3f} |")

    print("\n## 5. Per-suite McNemar, Qwen vs DeepSeek\n")
    print("| Suite | 826 p | 840 p | 826 kappa | 840 kappa |")
    print("|---|---|---|---|---|")
    for suite in SUITES:
        po = paired([r for r in old if r["suite"] == suite], "Qwen3.8", "DeepSeek-V4.1")
        pn = paired([r for r in new if r["suite"] == suite], "Qwen3.8", "DeepSeek-V4.1")
        print(f"| {suite} | {po['p']:.3f} | {pn['p']:.3f} | {po['kappa']:.3f} | {pn['kappa']:.3f} |")

    print("\n## 6. The contrasts the old exclusion table flagged as conclusion-bearing\n")
    print("These are the four rows printed in bold or near it in *What the exclusion "
          "changes*; that table compares the old board against its own removal and is "
          "obsolete, but the contrasts are not.\n")
    print("| Contrast | 826 | 840 |")
    print("|---|---|---|")
    for lbl, model, level, kind in (
            ("Qwen 3p vs 6p, unpaired", "Qwen3.8", None, "fisher"),
            ("Qwen 3p vs 6p, paired", "Qwen3.8", None, "mcnemar"),
            ("DeepSeek 3p vs 6p, paired", "DeepSeek-V4.1", None, "mcnemar"),
            ("Qwen S2, 3p vs 6p, unpaired", "Qwen3.8", "S2", "fisher")):
        o = suite_pair(old, model, "3p", "6p", level)
        n = suite_pair(new, model, "3p", "6p", level)
        print(f"| {lbl} | p = {o[kind]:.4g} | p = {n[kind]:.4g} |")


# Old string -> new string, for the one pass over the prose.  The script greps
# results_three_models_grid.md for each left-hand side: the chapters were
# written from that document, so a hit there names a sentence to patch even
# though the .tex itself lives in Overleaf.
SUBSTITUTIONS = [
    ("826", "840", "per-model episode count"),
    ("2 478", "2 520", "episodes in the comparison"),
    ("1 527", "1 552", "successes in the analysis set"),
    ("1 546", "--", "successes on the old full board set; no second set exists now"),
    ("n = 14", "n = 15", "the S2 cells are full again"),
    ("746/826 (90.3%)", "760/840 (90.5%)", "Qwen total"),
    ("754/826 (91.3%)", "765/840 (91.1%)", "DeepSeek total"),
    ("27/826 (3.3%)", "27/840 (3.2%)", "Nemotron total"),
    ("[88.1, 92.1]", "[88.3, 92.3]", "Qwen total CI"),
    ("[89.2, 93.0]", "[89.0, 92.8]", "DeepSeek total CI"),
    ("[2.3, 4.7]", "[2.2, 4.6]", "Nemotron total CI"),
    ("735 (89.0%)", "749 (89.2%)", "Nemotron aborts"),
    ("17 (2.1%)", "18 (2.1%)", "DeepSeek aborts"),
    ("p = 0.358", "p = 0.609", "Qwen vs DeepSeek, McNemar"),
    ("kappa 0.580", "kappa 0.566", "Qwen vs DeepSeek, overall"),
    ("both succeed on 721", "both succeed on 732", "paired cell"),
    ("Qwen-only 25", "Qwen-only 28", "paired cell"),
    ("719 to 0", "733 to 0", "Nemotron discordant, vs Qwen"),
    ("727 to 0", "738 to 0", "Nemotron discordant, vs DeepSeek"),
    ("3p p = 0.238", "3p p = 0.359", "per-suite"),
    ("solo 0.166", "solo 0.167", "per-suite kappa"),
    ("dyad 0.397", "dyad 0.374", "per-suite kappa"),
    ("3p 0.567", "3p 0.552", "per-suite kappa"),
    ("**6p 0.674**", "**6p 0.663**", "per-suite kappa"),
]


def patch_list():
    doc = os.path.join(ROOT, "results_three_models_grid.md")
    lines = open(doc).read().splitlines()
    print("\n## 7. The patch list\n")
    print("There is no `% PATCH-840` marker anywhere in this repository: the chapters "
          "are written in Overleaf and only `results_three_models_grid.md` is here. "
          "So the list below is by string, not by marker -- search the .tex for the "
          "left-hand side. Line numbers are in `results_three_models_grid.md`, whose "
          "sentences the chapters were written from.\n")
    print("| Search for | Replace with | What it is | Lines in results_three_models_grid.md |")
    print("|---|---|---|---|")
    for old_s, new_s, note in SUBSTITUTIONS:
        hits = [str(i + 1) for i, ln in enumerate(lines) if old_s in ln]
        where = ", ".join(hits[:12]) + (" ..." if len(hits) > 12 else "") or "--"
        print(f"| `{old_s}` | `{new_s}` | {note} | {where} |")
    print("\nCode, not prose:\n")
    print("- `analysis/thesis/tables_comm.py`: `EXCLUDED = (\"S2\", 10)` -> `None`, then re-run; "
          "two captions say \"with S2~\\#10 excluded\".")
    print("- `analysis/efficiency_baseline/tables_ch6.py`: line 19 still does "
          "`boards.pop(ia.EXCLUDED, None)`, and the tab:4 caption still says S2~\\#10 "
          "has no stuck-free solution. R3a step 2 has not landed.")
    print("- `analysis/efficiency_baseline/stuckfree.jsonl` predates the splice (13 Sep), so the "
          "replacement board has no stuck-free result. R3a step 3, half done: "
          "`joint_optimum.jsonl` was regenerated here (S2 #10 exact optimum 47, "
          "stored 37, gap 10; the old board's was 31).")
    print("\n## 8. The communication chapter's own tables (W8a / W8b)\n")
    print("`analysis/thesis/tables_comm.py` had `EXCLUDED = (\"S2\", 10)`; it is now `None` and "
          "`thesis/tables_comm.tex` is regenerated on 840. The version it replaces is "
          "kept as `thesis/tables_comm.826.tex` -- `diff` them for the exact digit "
          "list, which is this chapter's patch list. What moves:\n")
    print("- tab:recovery, every cell: n 88 -> 90 per condition, solo 176 -> 180.")
    print("- The gap the protocols are asked to recover: **9.1 -> 10.0 points** of "
          "success, 0.087 -> 0.086 of efficiency. Dyad \\textsc{none} 86.4% -> 85.6% "
          "is the whole of it: the replacement board is a DeepSeek \\textsc{none} "
          "failure and a success under the other three protocols.")
    print("- tab:decomposition: \\textsc{none} vs solo \\textsc{silent} $-$8.0 -> $-$8.9, "
          "McNemar 0.1185 -> **0.0768**; the other four contrasts move in the third "
          "decimal and no McNemar p changes at all.")
    print("- **The one threshold crossing in the whole list.** The scope note's matched "
          "subset, \\textsc{structured} against \\textsc{none}: 74 -> 75 boards solved "
          "under both, $-0.025$ -> $-0.027$, Wilcoxon **$p = 0.057$ -> $p = 0.040$**. "
          "The caption read \"a direction, not an established cost\"; it now reads "
          "\"an exploratory contrast, not one of the confirmatory nineteen, and not an "
          "established cost\". Those three figures were hardcoded in the script, not "
          "computed, which is why they survived every earlier re-run unchanged -- "
          "check whether the chapter repeats them in prose.")

    print("\n## 9. The message corpus (R5)\n")
    print("`analysis/qualitative_analysis/qual_corpus.py --refresh --verify` was re-run on the "
          "replacement board. 109 of the asserted values moved; all were corrected in "
          "`results_three_models_grid.md` and in the script's `DOC` table, which now "
          "reproduces **301 of 301** (295 before, plus two claims the README said were "
          "checked and were not: 5.4's DeepSeek structured compliance by outcome and "
          "5.5's within-structured request comparison). The pre-replacement cache is "
          "`analysis/qualitative_analysis/corpus.826.json`; diff the two reports with\n")
    print("```\npython3 qual_corpus.py --cache corpus.826.json > old.txt\n"
          "python3 qual_corpus.py > new.txt\n```\n")
    print("- Corpus 66 007 -> **66 584** messages over the same 2 160 episodes. The "
          "replaced board contributes 1 137 messages against the defective board's 560, "
          "because its 6p episodes are playable instead of ending after one accepted move.")
    print("- **The corpus half of the stated inconsistency is retired.** 5.1 said the "
          "corpus was the one section computed on the full board set while every "
          "episode figure used the 826-episode set; that sentence and its restatement "
          "in the limits section are rewritten, and the corpus is now wholly on the "
          "sixty boards. The document is mixed until R6 moves the episode figures with "
          "section 7 above: 5.1 to 5.6 and Part II 2.3 are on 840, the rest still reads "
          "826. 5.1 deliberately no longer claims a scope for the document as a whole -- "
          "R6 puts that claim back once it is true.")
    print("- No claim changes. The three largest p-value movements are all in 5.5 and "
          "all in the same direction as before: names-a-teammate 1.1e-08 -> 1.9e-09, "
          "request 8.6e-14 -> 1.8e-13, sequencing 0.35 -> 0.30 (still n.s.). The "
          "within-structured request contrast moves 2.9e-40 -> 4e-39.")


main()
patch_list()
