"""Listener probe, target question (E3g), per team game and protocol: the table body and the
pooled counts the results prose needs. Splits the answers by whether the target cell appears
in a message in view, which probe_e3g_decisions.csv does not hold, so the corpus is loaded.
Writes the per-decision rows to analysis/messages/out/<dir>/probe_e3g_targets.csv, which the
table below and figures/results_messages_figs.py (res_msg_probe_target) both read.

    python3 analysis/thesis/probe_targets.py
"""

import collections
import csv
import os
import statistics
import sys

import results_messages as rm

sys.path.insert(0, os.path.dirname(os.path.dirname(rm.HERE)))
from analysis.messages import corpus, grammar  # noqa: E402

rm.D, rm.M = "test_dsv41", "dsv41"
COLS = ["correct", "stated", "in_view", "in_view_correct", "out_view_correct", "unknown"]  # the table's order


def answers():
    """Original-prompt answers of the objects asked on both prompts, with in_view: the target
    cell appears in some message in view at the decision (any speaker, any part)."""
    rows = [r for r in rm.rows("probe_e3g_answers") if r["model"] == rm.M]
    both = collections.Counter((r["episode"], r["ply"], r["letter"]) for r in rows)
    eps = {ep["meta"]["id"]: ep for ep in corpus.load("test", models=(rm.M,), protos=tuple(rm.MSG))}
    A = grammar.analyse_all(eps.values())
    out = []
    for r in rows:
        if r["arm"] != "orig" or both[(r["episode"], r["ply"], r["letter"])] != 2:
            continue
        ep, L = eps[r["episode"]], r["letter"]
        T, tc = ep["turns"], ep["targets"][L]
        vis = [p for p in T[int(r["ply"])]["vis"] if T[p]["is_msg"]]
        named = any(grammar.target_mentions(A[(r["episode"], p)], {L: tc})[L]["named"] for p in vis)
        plan = any(T[p]["seat"] == ep["owner"][L] and T[p]["slots"] and T[p]["slots"]["plan_obj"] == L
                   and T[p]["slots"]["plan_to"] == tc for p in vis)
        out.append(dict(r, in_view=named, plan_to=plan))
    return out


def decisions(ans):
    """Per decision, shares over its teammate objects; a split is None where it has no object."""
    by = collections.defaultdict(list)
    for a in ans:
        by[(a["suite"], a["proto"], a["episode"], a["ply"])].append(a)
    out = []
    for (s, p, e, ply), objs in by.items():
        share = lambda sel, f: statistics.mean(f(o) for o in sel) if sel else None
        ok = lambda o: o["correct"] == "True"
        inv = [o for o in objs if o["in_view"]]
        out.append(dict(model=rm.M, suite=s, proto=p, episode=e, ply=ply,
                        correct=share(objs, ok), in_view=share(objs, lambda o: o["in_view"]),
                        in_view_correct=share(inv, ok), out_view_correct=share([o for o in objs if not o["in_view"]], ok),
                        stated=share(objs, lambda o: o["stated"] == "True"),
                        unknown=share(objs, lambda o: o["unknown"] == "True")))
    return out


def write(dec):
    """The rows as rm.rows() reads them: a split without objects is an empty string."""
    with open(os.path.join(rm.OUT, rm.D, "probe_e3g_targets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dec[0]))
        w.writeheader()
        w.writerows({k: "" if v is None else v for k, v in d.items()} for d in dec)


def pooled(ans):
    n = lambda sel: f"{sum(o['correct'] == 'True' for o in sel)} of {len(sel)}"
    st, un = [o for o in ans if o["stated"] == "True"], [o for o in ans if o["stated"] == "False"]
    print("stated, answered with the target", n(st))
    print("not stated, cell in view", n([o for o in un if o["in_view"]]))
    print("cell in no message in view", n([o for o in ans if not o["in_view"]]))
    for home in ("False", "True"):
        sel = [o for o in un if o["home"] == home]
        print(f"not stated, on target {home}: {n(sel)}, cell in view {sum(o['in_view'] for o in sel)} of {len(sel)}")
    print("wrong cell", sum(o["correct"] == "False" and o["unknown"] == "False" for o in ans), "of", len(ans))
    for p in rm.SLOT:
        sel = [o for o in un if o["proto"] == p and o["correct"] == "True"]
        print(f"{p}: not stated and answered {len(sel)}, the cell is the owner's PLAN destination {sum(o['plan_to'] for o in sel)}")


def main():
    ans = answers()
    dec = decisions(ans)
    # the columns probe_e3g_decisions.csv also holds must agree with it
    for col, ref in (("correct", "correct_orig"), ("stated", "stated_share"), ("unknown", "unknown_orig")):
        for s in rm.TEAMS:
            for p in rm.MSG:
                mine = statistics.mean(d[col] for d in dec if d["suite"] == s and d["proto"] == p)
                theirs = statistics.mean(float(r[ref]) for r in rm.rows("probe_e3g_decisions")
                                         if r["model"] == rm.M and r["suite"] == s and r["proto"] == p)
                assert abs(mine - theirs) < 1e-5, (col, s, p, mine, theirs)  # the CSV keeps 6 digits
    write(dec)
    print("% e3g targets: " + " & ".join(COLS)); rm.probe("probe_e3g_targets", COLS, rm.MSG)
    pooled(ans)


if __name__ == "__main__":
    main()
