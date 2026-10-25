"""Results-chapter message tables on the final arm, from the message analysis CSVs.

Point estimates only (the chapter is descriptive); per-episode means over the
cell, as in analysis/messages/out/<dir>/. Prints LaTeX tabular bodies and the
pooled numbers the results chapter cites.

    python3 analysis/thesis/results_messages.py [out_dir] [model]   (default test_dsv41 dsv41)
"""

import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "messages", "out")
TEAMS = ["dyad", "3p", "6p"]
MSG = ["structured", "hybrid", "freeform"]
SLOT = ["structured", "hybrid"]
SC = {p: r"\textsc{%s}" % p for p in MSG}


def rows(name):
    p = os.path.join(OUT, D, name + ".csv")
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def cell(rs, **kw):
    """est and n of the one row matching kw (strings), or None."""
    hit = [r for r in rs if all(r.get(k) == v for k, v in kw.items())]
    if not hit or hit[0]["est"] == "":
        return None
    return float(hit[0]["est"]), int(float(hit[0]["n"]))


def f(x, scale=100, d=1):
    return "--" if x is None else f"{x[0] * scale:.{d}f}"


def body(lines):
    print("\n".join("    " + x for x in lines) + "\n")


def episodes_n(suite, proto):
    eps = {r["episode"] for r in rows("d2_episodes") if r["model"] == M and r["suite"] == suite and r["proto"] == proto}
    return len(eps)


def union_mean(suite, proto):
    """D2 with the model's records (the primary version): mean of per-episode accuracy."""
    ps = [float(r["union_prop"]) for r in rows("d2_episodes")
          if r["model"] == M and r["suite"] == suite and r["proto"] == proto and r["union_n"] not in ("", "0")]
    return (statistics.mean(ps), len(ps)) if ps else None


def accuracy():
    pr, own, d2, cnt = rows("pr_cells"), rows("own_cells"), rows("d2_cells"), rows("teammate_counts_cells")
    out = []
    for s in TEAMS:
        for p in MSG:
            n = episodes_n(s, p)
            unres = [cell(pr, table="pr_resolvable", measure="unres", model=M, suite=s, proto=p, part=part)
                     for part in ("prose", "slots")]
            plan = cell(own, value="prop", channel="plan_from", model=M, suite=s, proto=p)
            prose = cell(own, value="prop", channel="prose_loc", model=M, suite=s, proto=p)
            un, gr = union_mean(s, p), cell(d2, value="all_prop", model=M, suite=s, proto=p)
            tr = cell(cnt, value="n_true", model=M, suite=s, proto=p)
            fa = cell(cnt, value="n_false", model=M, suite=s, proto=p)
            nsub = lambda x: "" if x is None or x[1] >= n else rf"\textsubscript{{{x[1]}}}"
            out.append(f"{s} & {SC[p]} & {n} & {f(unres[0])} & {f(unres[1])} & {f(plan)} & {f(prose)}{nsub(prose)}"
                       f" & {f(un)}{nsub(un)} & {f(gr)}{nsub(gr)} & {f(tr, 1, 2)} & {f(fa, 1, 2)} \\\\")
        out.append(r"\midrule")
    body(out[:-1])


BOARDREF = ["own_obj", "mate_active", "mate_finished", "empty", "device", "own_target", "mate_target"]


def boardref():
    pr = rows("pr_cells")
    out = []
    for s in TEAMS:
        for p, part in [("structured", "slots"), ("hybrid", "slots"), ("hybrid", "prose"), ("freeform", "prose")]:
            vals = [f(cell(pr, table="pr_boardref", measure=m, model=M, suite=s, proto=p, part=part)) for m in BOARDREF]
            out.append(f"{s} & {SC[p]} & {part} & " + " & ".join(vals) + r" \\")
        base = []
        for m in BOARDREF:
            b = [r["baseline"] for r in pr if r["table"] == "pr_boardref" and r["measure"] == m and r["model"] == M
                 and r["suite"] == s and r["baseline"] != ""]
            base.append(f"{float(b[0]) * 100:.1f}" if b else "--")
        out.append(rf"{s} & \multicolumn{{2}}{{l}}{{\emph{{cell share}}}} & " + " & ".join(base) + r" \\")
        out.append(r"\midrule")
    body(out[:-1])


def slot_table(metrics):
    rp = rows("rp_cells")
    out = []
    for s in TEAMS:
        for p in SLOT:
            vals = []
            for m in metrics:
                x = cell(rp, model=M, suite=s, proto=p, level="all", metric=m)
                vals.append(f(x, 1, 1) if m == "conf_latency" else f(x))
            out.append(f"{s} & {SC[p]} & " + " & ".join(vals) + r" \\")
        out.append(r"\midrule")
    body(out[:-1])


RUNS = ["run_reached", "run_unforced", "run_forced", "run_stale"]


def persistence():
    """How PLAN runs end; open is the rest (no later message of the seat before the first fatal decision)."""
    rp = rows("rp_cells")
    out = []
    for s in TEAMS:
        for p in SLOT:
            xs = [cell(rp, model=M, suite=s, proto=p, level="all", metric=m) for m in RUNS]
            rest = max(0.0, 1 - sum(x[0] for x in xs))
            out.append(f"{s} & {SC[p]} & {xs[0][1]} & " + " & ".join(f(x) for x in xs) + f" & {100 * rest:.1f} \\\\")
        out.append(r"\midrule")
    body(out[:-1])


BASE = ["V_silent_baseline", "V_unnamed_onpath", "K_unnamed_onpath", "uptake_silent_baseline"]


def baselines():
    """Observed against comparison per episode, board-paired (rp_baselines)."""
    rb = rows("rp_baselines")
    out = []
    for s in TEAMS:
        for p in SLOT:
            vals = []
            for c in BASE:
                hit = [r for r in rb if r["model"] == M and r["suite"] == s and r["proto"] == p and r["comparison"] == c]
                vals += ([f"{float(hit[0]['mean_observed']) * 100:.1f}", f"{float(hit[0]['mean_comparison']) * 100:.1f}"]
                         if hit else ["--", "--"])
            out.append(f"{s} & {SC[p]} & " + " & ".join(vals) + r" \\")
        out.append(r"\midrule")
    body(out[:-1])


def o1():
    """Named moves taken (alignment_cells o1), with the not-in-view and other-direction comparisons."""
    ac, co = rows("alignment_cells"), rows("alignment_contrasts")
    out = []
    for s in TEAMS:
        for p in MSG:
            x = cell(ac, model=M, suite=s, proto=p, level="all", metric="o1")
            vals = [f(x), str(x[1]) if x else "--"]
            for c in ("nc_exposure", "direction_specificity"):
                hit = [r for r in co if r["model"] == M and r["suite"] == s and r["proto"] == p and r["comparison"] == c]
                vals += ([f"{float(hit[0]['mean_observed']) * 100:.1f}", f"{float(hit[0]['mean_comparison']) * 100:.1f}"]
                         if hit and hit[0]["mean_observed"] else ["--", "--"])
            out.append(f"{s} & {SC[p]} & " + " & ".join(vals) + r" \\")
        out.append(r"\midrule")
    body(out[:-1])


E3 = ["correct_orig", "correct_nomsg", "named_share", "named_correct_orig", "unnamed_correct_orig", "unknown_orig",
      "unknown_nomsg"]
E4 = ["tau", "delta", "floor_tau", "floor_delta"]


def probe(name, cols, protos):
    """E3 or E4 per game and protocol: means over the sampled decisions (probe_*_decisions);
    a subscript gives the decisions where a column is defined in fewer than the cell holds."""
    pr = [r for r in rows(name) if r["model"] == M]
    out = []
    for s in TEAMS:
        cell_rows = {p: [r for r in pr if r["suite"] == s and r["proto"] == p] for p in protos}
        if not any(cell_rows.values()):
            continue
        for p in protos:
            rs, vals = cell_rows[p], []
            for c in cols:
                v = [float(r[c]) for r in rs if r[c] != ""]
                vals.append("--" if not v else f"{100 * statistics.mean(v):.1f}".replace("-", "$-$")
                            + ("" if len(v) >= len(rs) else rf"\textsubscript{{{len(v)}}}"))
            out.append(f"{s} & {SC[p]} & {len(rs)} & " + " & ".join(vals) + r" \\")
        out.append(r"\midrule")
    body(out[:-1])


DEST = ["commit_stated", "described_now", "next_progress", "reached"]


def dest():
    dc = rows("dest_cells")
    out = []
    for s in TEAMS:
        xs = [cell(dc, model=M, suite=s, level="all", metric=m) for m in DEST]
        units = [r["units"] for r in dc if r["model"] == M and r["suite"] == s and r["level"] == "all"
                 and r["metric"] == "next_progress"]
        out.append(f"{s} & {xs[2][1] if xs[2] else '--'} & {units[0] if units else '--'} & "
                   + " & ".join(f(x) for x in xs) + r" \\")
    body(out)


def pooled():
    """Numbers for the bullets: D2 by tier and version, sources, E5 checks, prose requests."""
    for r in rows("d2"):
        print(f"d2 {r['version']}: strong {float(r['strong']) * 100:.1f} (n {r['n_strong']}) primary {r['primary']}")
    for r in (r for r in rows("d2_splits") if r["model"] == M):
        print("split", r["tier"], r["split"], r["value"], r["n"], r["share_true"], r["share_of_statements"])
    for r in rows("e5_rejects"):
        if r["prompt"] == "primary":
            print("e5 reject", r["reason"], r["n"])
    for r in rows("e5_agreement"):
        print("e5 agree", {k: r[k] for k in ("prompt", "tier", "n_grammar", "n_model", "n_both", "disagree_share",
                                               "canonical_recall")})
    for r in rows("dest_meta"):
        print("dest meta", r)
    for r in rows("rp_summary"):
        if r["model"] == M and r["level"] == "all":
            print("rp pooled", r["suite"], r["proto"], "recall", r["recall"], "recall_or_shadow", r["recall_or_shadow"],
                  "req_nocell", r["req_nocell"], "silent_blocked", r["silent_blocked"])
    for r in rows("rp_baselines"):
        if r["model"] == M and r["reported"] == "True":
            print("baseline", r["suite"], r["proto"], r["comparison"], r["mean_observed"], r["mean_comparison"],
                  r["pairs"])


def main():
    print("% accuracy"); accuracy()
    print("% boardref"); boardref()
    print("% request"); slot_table(["req_filled", "ref_err", "relevant", "recall", "silent_blocked", "V_next", "V_H",
                                    "K_kept", "useful_req"])
    print("% plan"); slot_table(["saydo_progress", "to_valid", "to_goal", "wp_give_way", "conflict", "up_enter"])
    print("% persistence"); persistence()
    print("% baselines"); baselines()
    print("% o1"); o1()
    print("% e3"); probe("probe_e3_decisions", E3, MSG)
    print("% e4"); probe("probe_e4_decisions", E4, SLOT)
    print("% dest"); dest()
    pooled()


if __name__ == "__main__":
    D = sys.argv[1] if len(sys.argv) > 1 else "test_dsv41"
    M = sys.argv[2] if len(sys.argv) > 2 else "dsv41"
    main()
