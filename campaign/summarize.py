#!/usr/bin/env python3
"""Per-suite summary over scored episodes: coverage, aborts, success, main score."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LEVEL = re.compile(r"S([1-4])")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def condition(name: str) -> str:
    """Protocol / thinking condition, i.e. the experiment name minus its level."""
    parts = [p for p in name.split("_") if not LEVEL.fullmatch(p)]
    return "_".join(parts) or name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--expected", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.results) / args.model / args.game
    rows = []
    for scores_file in sorted(root.glob("*/instance_*/scores.json")):
        try:
            episode = json.loads(scores_file.read_text())["episode scores"]
        except Exception:
            continue
        experiment = scores_file.parent.parent.name
        rows.append((experiment, episode))

    played = sum(1 for _, e in rows if e.get("Played"))
    aborted = sum(1 for _, e in rows if e.get("Aborted"))
    success = sum(1 for _, e in rows if e.get("Success"))

    print(f"\n===== SUITE SUMMARY: {args.label} ({args.game}) =====")
    print(f"model            {args.model}")
    print(f"episodes scored  {len(rows)}" + (f" / {args.expected} expected" if args.expected else ""))
    print(f"played / aborted {played} / {aborted}"
          + (f"  (abort rate {aborted / len(rows):.1%})" if rows else ""))
    print(f"successes        {success}"
          + (f"  ({success / played:.1%} of played)" if played else ""))
    if played:
        print(f"mean main score  {mean([e.get('Main Score', 0) for _, e in rows if e.get('Played')]):.2f}")

    by_level = defaultdict(list)
    by_cond = defaultdict(list)
    for experiment, episode in rows:
        m = LEVEL.search(experiment)
        by_level[m.group(0) if m else "?"].append(episode)
        by_cond[condition(experiment)].append(episode)

    def table(title, groups):
        print(f"\n{title:<18}{'n':>4}{'abort':>7}{'succ':>6}{'succ%':>7}{'main':>8}{'moves':>7}")
        for key in sorted(groups):
            eps = groups[key]
            p = [e for e in eps if e.get("Played")]
            s = [e for e in eps if e.get("Success")]
            moves = mean([e.get("Move Count", 0) for e in s]) if s else float("nan")
            print(f"{key:<18}{len(eps):>4}{sum(1 for e in eps if e.get('Aborted')):>7}{len(s):>6}"
                  f"{(len(s) / len(p) * 100 if p else 0):>6.0f}%{mean([e.get('Main Score', 0) for e in p]):>8.1f}"
                  f"{moves:>7.0f}")

    table("by spatial level", by_level)
    table("by condition", by_cond)

    # Level x condition: the cell the analysis actually compares. Aggregating a
    # condition across levels mixes a saturated level with an unsolved one.
    cells = defaultdict(list)
    for experiment, episode in rows:
        m = LEVEL.search(experiment)
        cells[(m.group(0) if m else "?", condition(experiment))].append(episode)
    conds = sorted({c for _, c in cells})
    print(f"\n{'level x cond':<14}" + "".join(f"{c[:12]:>14}" for c in conds))
    for level in sorted({l for l, _ in cells}):
        line = f"{level:<14}"
        for c in conds:
            eps = cells.get((level, c), [])
            if not eps:
                line += f"{'-':>14}"
            else:
                s_ = sum(1 for e in eps if e.get("Success"))
                line += f"{f'{s_}/{len(eps)}':>9}{f'{s_/len(eps)*100:.0f}%':>5}"
        print(line)
    print(f"===== END {args.label} =====\n")


main()
