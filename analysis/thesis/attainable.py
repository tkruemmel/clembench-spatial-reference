"""Per-suite attainable optimum: split each team-size step into rules and agents.

Every suite is scored against the free-order joint optimum J (what one player
moving any object on any turn needs).  A team of k players cannot always reach
J: strict round-robin, a permanent `done` and no `pass` force extra moves.  The
attainable optimum A_k (analysis/efficiency_baseline/stuckfree.py --suite k) is what a
flawless k-player team needs under those rules.  Per successful episode

    eff_J = J / moves = (J / A_k) * (A_k / moves) = rules_k * eff_A

so a drop in eff_J between suites is a drop in rules_k (a board property, the
same for every model) times a drop in eff_A (agent error).  Nothing here enters
the confirmatory ladder.

    python3 analysis/thesis/attainable.py            # tables to stdout
    python3 analysis/thesis/attainable.py --check    # also fail on any episode below A_k
"""

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EB = os.path.join(ROOT, "analysis/efficiency_baseline")
ARMS = {"Qwen3.8": "results_qwen38_grid_n15",
        "DeepSeek-V4.1-Flash": "results_dsv41_grid_n15"}
SUITES = ("solo", "dyad", "3p", "6p")
LEVELS = ("S1", "S2", "S3", "S4")


def optima():
    """(suite, level, game_id) -> A_k, and (level, game_id) -> J.  Exact only."""
    J = {}
    for ln in open(os.path.join(EB, "joint_optimum.jsonl")):
        d = json.loads(ln)
        if d.get("status") == "exact":
            J[(d["level"], d["game_id"])] = d["opt"]
    A = {}
    for s in SUITES:
        p = os.path.join(EB, f"attainable_{s}.jsonl")
        if s == "solo" or not os.path.exists(p):
            continue
        for ln in open(p):
            d = json.loads(ln)
            if d["status"] == "exact":
                A[(s, d["level"], d["game_id"])] = d["stuckfree"]
    for (l, g), j in J.items():                  # solo attainable is J by definition
        A[("solo", l, g)] = j
    return A, J


def episodes(A, J):
    rows = []
    for model, prefix in ARMS.items():
        for suite in SUITES:
            for f in glob.glob(os.path.join(ROOT, f"{prefix}_{suite}", "**", "scores.json"),
                               recursive=True):
                exp = f.split(os.sep)[-3]
                m = re.search(r"(S\d)", exp)
                if not m:
                    continue
                level = m.group(1)
                gid = int(re.search(r"instance_(\d+)", f).group(1))
                s = json.load(open(f))["episode scores"]
                key = (suite, level, gid)
                if key not in A or (level, gid) not in J:
                    continue
                ok = bool(s.get("Success")) and bool(s.get("Move Count"))
                mv = s.get("Move Count") if ok else None
                proto = exp.split("_")[2].lower() if suite == "solo" else exp.split("_")[1]
                rows.append(dict(model=model, suite=suite, level=level, gid=gid, proto=proto,
                                 aborted=int(s["Aborted"]), success=int(bool(s.get("Success"))),
                                 moves=mv, J=J[(level, gid)], A=A[key]))
    return rows


def qs(rs, denom):
    """Quality Score over played episodes on the given denominator."""
    played = [r for r in rs if not r["aborted"]]
    if not played:
        return float("nan")
    return st.mean(100.0 * min(1.0, r[denom] / r["moves"]) if r["success"] else 0.0
                   for r in played)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    A, J = optima()
    have = {s: sum(1 for k in A if k[0] == s) for s in SUITES}
    print("boards with an exact attainable optimum:", have)

    # 1. Board property: overhead A_k / J per suite and level.
    print("\nRule overhead A_k / J, mean over boards (board property)")
    print(f"{'level':<6}" + "".join(f"{s:>8}" for s in SUITES))
    for l in LEVELS + ("all",):
        cells = []
        for s in SUITES:
            v = [A[k] / J[k[1:]] for k in A if k[0] == s and (l == "all" or k[1] == l)]
            cells.append(f"{st.mean(v):>8.3f}" if v else f"{'--':>8}")
        print(f"{l:<6}" + "".join(cells))

    rows = episodes(A, J)
    below = [r for r in rows if r["success"] and r["moves"] < r["A"]]
    print(f"\nsuccessful episodes below their suite's attainable optimum: {len(below)}")
    for r in below[:10]:
        print("   ", r)
    if a.check and below:
        raise SystemExit("attainable optimum violated by an observed episode")
    at = sum(1 for r in rows if r["success"] and r["moves"] == r["A"])
    print(f"successful episodes exactly at it: {at}")

    # 2. Efficiency and Quality Score on both denominators, strong tier pooled.
    by = defaultdict(list)
    for r in rows:
        by[(r["suite"], r["level"])].append(r)
        by[(r["suite"], "all")].append(r)
    print("\nStrong tier pooled.  eff_J = J/moves, eff_A = A_k/moves (successes);"
          " QS on each denominator (played episodes)")
    print(f"{'suite':<6}{'level':<6}{'n':>5}{'succ':>6}{'eff_J':>8}{'eff_A':>8}"
          f"{'rules':>8}{'QS_J':>8}{'QS_A':>8}")
    for s in SUITES:
        for l in LEVELS + ("all",):
            rs = by.get((s, l), [])
            ok = [r for r in rs if r["success"]]
            if not rs:
                continue
            eJ = st.mean(r["J"] / r["moves"] for r in ok)
            eA = st.mean(r["A"] / r["moves"] for r in ok)
            rule = st.mean(r["J"] / r["A"] for r in ok)
            print(f"{s:<6}{l:<6}{len(rs):>5}{len(ok):>6}{eJ:>8.3f}{eA:>8.3f}"
                  f"{rule:>8.3f}{qs(rs, 'J'):>8.2f}{qs(rs, 'A'):>8.2f}")

    # 3. Each team-size step in log efficiency: total = rules + agents.
    print("\nTeam-size steps, mean log efficiency over successes"
          " (total = rules + agents; share of total that is rules)")
    print(f"{'step':<12}{'level':<6}{'total':>8}{'rules':>8}{'agents':>8}{'rules%':>8}")
    for a_, b_ in (("solo", "dyad"), ("dyad", "3p"), ("3p", "6p"), ("dyad", "6p")):
        for l in LEVELS + ("all",):
            ra = [r for r in by.get((a_, l), []) if r["success"]]
            rb = [r for r in by.get((b_, l), []) if r["success"]]
            if not ra or not rb:
                continue
            lJ = lambda rs: st.mean(math.log(r["J"] / r["moves"]) for r in rs)
            lR = lambda rs: st.mean(math.log(r["J"] / r["A"]) for r in rs)
            tot = lJ(rb) - lJ(ra)
            rul = lR(rb) - lR(ra)
            share = f"{100 * rul / tot:>7.0f}%" if abs(tot) > 1e-3 else f"{'--':>8}"
            print(f"{a_ + '->' + b_:<12}{l:<6}{tot:>8.3f}{rul:>8.3f}{tot - rul:>8.3f}{share}")

    # 4. The same steps on matched boards: (model, level, board, protocol) solved
    # in both suites.  Solo has no protocols, so solo steps are not matched.
    print("\nTeam-size steps on matched boards (solved in both suites; log efficiency)")
    print(f"{'step':<12}{'level':<6}{'pairs':>6}{'total':>8}{'rules':>8}{'agents':>8}{'rules%':>8}")
    idx = {(r["suite"], r["model"], r["level"], r["gid"], r["proto"]): r for r in rows if r["success"]}
    for a_, b_ in (("dyad", "3p"), ("3p", "6p"), ("dyad", "6p")):
        for l in LEVELS + ("all",):
            pr = [(idx[k], idx[(b_,) + k[1:]]) for k in idx
                  if k[0] == a_ and (l == "all" or k[2] == l) and (b_,) + k[1:] in idx]
            if not pr:
                continue
            tot = st.mean(math.log(y["J"] / y["moves"]) - math.log(x["J"] / x["moves"]) for x, y in pr)
            rul = st.mean(math.log(y["J"] / y["A"]) - math.log(x["J"] / x["A"]) for x, y in pr)
            share = f"{100 * rul / tot:>7.0f}%" if abs(tot) > 1e-3 else f"{'--':>8}"
            print(f"{a_ + '->' + b_:<12}{l:<6}{len(pr):>6}{tot:>8.3f}{rul:>8.3f}{tot - rul:>8.3f}{share}")


if __name__ == "__main__":
    main()
