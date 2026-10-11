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
import sys
import statistics as st
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "analysis/qualitative_analysis"))
from qual_corpus import tree                                     # noqa: E402
EB = os.path.join(ROOT, "analysis/efficiency_baseline")
ARMS = {"Qwen3.8": "results_qwen38_grid_n15",
        "DeepSeek-V4.1-Flash": "results_dsv41_grid_n15"}
SUITES = ("solo", "dyad", "3p", "6p")
LEVELS = ("S1", "S2", "S3", "S4")
BOOT_B = 10_000


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
            for f in sorted(glob.glob(os.path.join(ROOT, tree(prefix, suite), "**", "scores.json"),
                               recursive=True)):
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

    per_model(rows)


def boot(per_board, stat, b=BOOT_B, seed=0):
    """Percentile 95% CI of stat(summed columns) over boards resampled with
    replacement.  per_board: array (boards x columns) of per-board sums."""
    x = np.asarray(per_board, float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(b, len(x)))
    v = stat(x[idx].sum(axis=1).T)
    return np.nanpercentile(v, [2.5, 97.5])


def per_model(rows):
    """Who pays, per model, on matched boards, with a board bootstrap.

    Pairs are (level, board, protocol) solved in both suites; per pair, the log
    efficiency step on J splits into rules (log A_a/A_b, a board property) and
    agents (log eff_A_b - log eff_A_a).  The bootstrap resamples the 60 boards,
    each carrying all its pairs, BOOT_B times at seed 0; shares are the
    component's mean over the total's mean, so a share's CI is wide wherever
    the total is near zero.  Exploratory."""
    print(f"\nPer model, matched boards, log efficiency; 95% board-bootstrap CI"
          f" (B = {BOOT_B}, seed 0)")
    idx = {(r["suite"], r["model"], r["level"], r["gid"], r["proto"]): r for r in rows if r["success"]}
    boards = sorted({(r["level"], r["gid"]) for r in rows})
    for m in ARMS:
        for a_, b_ in (("dyad", "3p"), ("3p", "6p"), ("dyad", "6p")):
            sums = {bd: [0.0, 0.0, 0] for bd in boards}       # total, rules, pairs
            for k, x in idx.items():
                if k[0] != a_ or k[1] != m or (b_,) + k[1:] not in idx:
                    continue
                y = idx[(b_,) + k[1:]]
                s = sums[(k[2], k[3])]
                s[0] += math.log(x["moves"] / y["moves"])  # = log eff_J,b - log eff_J,a
                s[1] += math.log(x["A"] / y["A"])
                s[2] += 1
            M = np.array(list(sums.values()))
            n = int(M[:, 2].sum())
            tot, rul = M[:, 0].sum() / n, M[:, 1].sum() / n
            ci = lambda f: "[{:+.3f}, {:+.3f}]".format(*boot(M, f))
            sh = lambda f: "[{:.0f}, {:.0f}]%".format(*(100 * boot(M, f)))
            print(f"  {m:<20}{a_ + '->' + b_:<9} pairs {n:>3}  total {tot:+.3f} {ci(lambda c: c[0] / c[2])}"
                  f"  rules {rul:+.3f} {ci(lambda c: c[1] / c[2])}"
                  f"  agents {tot - rul:+.3f} {ci(lambda c: (c[0] - c[1]) / c[2])}"
                  f"  rules share {100 * rul / tot:.0f}% {sh(lambda c: c[1] / c[0])}"
                  f"  agents share {100 * (tot - rul) / tot:.0f}% {sh(lambda c: (c[0] - c[1]) / c[0])}")
    print("\nPer model, 6p successes: efficiency on J and on A_6p; rules share of the"
          " log inefficiency on J, with the same bootstrap")
    for m in ARMS:
        sums = {bd: [0.0, 0.0, 0] for bd in boards}           # log J/moves, log J/A, n
        ok = [r for r in rows if r["model"] == m and r["suite"] == "6p" and r["success"]]
        for r in ok:
            s = sums[(r["level"], r["gid"])]
            s[0] += math.log(r["J"] / r["moves"])
            s[1] += math.log(r["J"] / r["A"])
            s[2] += 1
        M = np.array(list(sums.values()))
        sh = boot(M, lambda c: c[1] / c[0])
        print(f"  {m:<20} n {len(ok)}  eff_J {st.mean(r['J'] / r['moves'] for r in ok):.3f}"
              f"  eff_A {st.mean(r['A'] / r['moves'] for r in ok):.3f}"
              f"  rules share {100 * M[:, 1].sum() / M[:, 0].sum():.0f}% [{100 * sh[0]:.0f}, {100 * sh[1]:.0f}]%")


if __name__ == "__main__":
    main()
