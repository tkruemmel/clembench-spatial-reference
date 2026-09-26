#!/usr/bin/env python3
"""The three Chapter 6 tables, as LaTeX (booktabs). Item D-a on the stack.

    python3 analysis/efficiency_baseline/tables_ch6.py > analysis/efficiency_baseline/tables_ch6.tex

Numbers come from instance_audit.audit_board, i.e. the same rows as
`instance_audit.py`, joined with joint_optimum.jsonl and stuckfree.jsonl.
Scope is all sixty boards.  The original S2 #10 was unplayable at 6p and was
regenerated; the replacement has a stuck-free solution (55 against a joint
optimum of 47), so no board is excluded from any table.  The pre-replacement
solver outputs are kept as *.jsonl.pre840.
"""
import statistics as st
import instance_audit as ia

LEVELS = ia.LEVELS


def rows():
    boards = ia.load_boards(ia.SOLO)
    jo, sf = ia.load_solver_outputs()
    return [ia.audit_board(lv, gid, boards[(lv, gid)], jo, sf)
            for lv, gid in sorted(boards)]


def by_level(rs, key):
    return {lv: [r[key] for r in rs if r["level"] == lv and r[key] is not None]
            for lv in LEVELS}


def table(caption, label, spec, header, body, note=None, wide=False):
    """wide=True wraps the tabular in \resizebox: tab:audit is 16 columns and
    runs ~190pt past the 2.5cm-margin text block otherwise."""
    out = [r"\begin{table}[tbp]", r"  \centering", r"  \footnotesize",
           f"  \\caption{{{caption}}}", f"  \\label{{{label}}}"]
    if wide:
        out.append(r"  \resizebox{\textwidth}{!}{%")
    out += [f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule",
           "    " + header + r" \\", r"    \midrule"]
    out += ["    " + b + r" \\" for b in body]
    out += [r"    \bottomrule", r"  \end{tabular}" + ("}%" if wide else "")]
    if note:
        out += [r"  \\[2pt]", r"  \begin{minipage}{0.9\textwidth}\footnotesize",
                "  " + note, r"  \end{minipage}"]
    out += [r"\end{table}", ""]
    return "\n".join(out)


def main():
    rs = rows()
    m = lambda key: {lv: st.mean(v) for lv, v in by_level(rs, key).items() if v}
    n = {lv: sum(1 for r in rs if r["level"] == lv) for lv in LEVELS}
    h0, opt = m("h0"), m("opt")

    # tab:2 -- stored bound vs exact joint optimum. gap% is the ratio of the
    # means, which is how S6.1 reports it.
    body = [f"{lv} & {n[lv]} & {h0[lv]:.1f} & {opt[lv]:.1f} & "
            f"{opt[lv]-h0[lv]:.1f} & {100*(opt[lv]-h0[lv])/h0[lv]:.1f} \\%"
            for lv in LEVELS]
    t2 = table(
        "Stored move bound against the exact joint optimum, per level "
        "(all sixty boards). The stored bound is the per-object BFS sum "
        "used for budgeting; the joint optimum is the scoring denominator. "
        "The error is not constant across levels, and on the exact numbers S4 "
        "is the longer level where the stored bound makes S3 longer.",
        "tab:denominator", "lrrrrr",
        r"Level & $n$ & Stored $h_0$ & Exact optimum & Interaction gap & Error",
        body)

    # tab:3 -- the per-board audit, summarised per level.
    dt = {"S1": None, "S2": 1.5, "S3": 2.0, "S4": 2.0}
    body = []
    for lv in LEVELS:
        r = [x for x in rs if x["level"] == lv]
        g = lambda k: st.mean([x[k] for x in r])
        det = st.median([d for x in r for d in x["detour_objs"]])
        body.append(
            f"{lv} & {n[lv]} & {g('chain'):.0f} & {g('edges'):.0f} & "
            f"{g('chains'):.0f} & {sum(1 for x in r if x['swap'])}/{n[lv]} & "
            f"{g('dead_end_cells'):.0f} & {g('targets_in_dead_end'):.0f} & "
            f"{opt[lv]-h0[lv]:.1f} & {g('mob_mean'):.2f} & "
            f"{min(x['mob_min'] for x in r)} & {sum(x['forced_6p'] for x in r)} & "
            f"{sum(x['zero_6p'] for x in r)} & "
            f"{'--' if dt[lv] is None else f'{dt[lv]:.1f}'} & {det:.2f} & "
            f"{g('overhead_6p'):.2f}")
    t3 = table(
        "Per-board property audit, summarised by level (all sixty boards; "
        "means unless stated). Every column is a function of the "
        "instance file alone --- no model, no episode. Per-board rows are in "
        "Appendix~C.",
        "tab:audit", "lrrrrcrrrrrrrrrr",
        r"Level & $n$ & Chain & Edges & Chains & Swap & DE cells & T@DE & "
        r"Gap & Mob.\ & Min & F@6p & Z@6p & Detour tgt & Real.\ & 6p ovh.",
        body,
        note=r"\emph{Chain}: longest dependency chain in edges. "
             r"\emph{Swap}: boards carrying the forced non-monotonic swap. "
             r"\emph{DE}: dead-end corridor cells; \emph{T@DE}: targets "
             r"inside one. \emph{Gap}: interaction gap, exact optimum minus "
             r"stored bound. \emph{Mob.}/\emph{Min}: start mobility, mean "
             r"over objects and minimum over boards. \emph{F@6p}/\emph{Z@6p}: "
             r"objects with exactly one and with no legal move at the start, "
             r"summed over boards. \emph{Detour}: generator target against "
             r"the realised pooled median.",
        wide=True)

    # tab:4 -- what a flawless six-player team costs under the protocol.
    sfm, ovh = m("h0"), m("overhead_6p")
    body = []
    for lv in LEVELS:
        r = [x for x in rs if x["level"] == lv]
        sfree = st.mean([x["overhead_6p"] * x["opt"] for x in r])
        body.append(f"{lv} & {n[lv]} & {opt[lv]:.1f} & {sfree:.1f} & "
                    f"{ovh[lv]:.2f}$\\times$")
    allo = st.mean([x["overhead_6p"] for x in rs if x["overhead_6p"] is not None])
    body.append(r"\midrule All & " + f"{len(rs)} & "
                f"{st.mean([x['opt'] for x in rs]):.1f} & "
                f"{st.mean([x['overhead_6p']*x['opt'] for x in rs]):.1f} & "
                f"\\textbf{{{allo:.2f}$\\times$}}")
    t4 = table(
        "Six-way protocol overhead by level: the attainable six-player "
        "optimum against the free-order joint optimum (all sixty boards). The "
        "stuck-free solution is the cheapest play-through in which no player "
        "is ever enclosed, i.e.\\ what a flawless six-player team costs under "
        "strict round-robin, permanent \\texttt{done} and no \\texttt{pass}; "
        "every board has one.",
        "tab:overhead", "lrrrr",
        r"Level & $n$ & Joint optimum & Attainable 6p & Overhead",
        body)

    print("% Generated by analysis/efficiency_baseline/tables_ch6.py -- do not hand-edit.")
    print("% Chapter 6 floats: tab:denominator (S6.1), tab:audit (S6.2), "
          "tab:overhead (S6.4).\n")
    print(t2); print(t3); print(t4)


if __name__ == "__main__":
    main()
