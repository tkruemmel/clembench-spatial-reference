#!/usr/bin/env python3
"""Propose per-suite worker counts for a worker-based grid campaign.

Weights each suite by the remaining episodes PER LEVEL x a reference minutes-
per-episode for that level, so the four suites finish at about the same time,
and hands out --total streams greedily to the suite with the largest hours per
stream (min-max optimal, at least --min per unfinished suite). The reference is the finished compact Qwen3.8 campaign
(--ref-prefix, per-suite per-level medians: solo S2-S4 cost ~10x S1, dyad ~4.5x),
used only for levels this campaign has not yet measured (fewer than --min-n
finished episodes), rescaled by the lowest measured level's minutes over the
reference's. Measured levels use the median over all their finished episodes. Prints the SOLO=.. DYAD=.. P3=.. P6=.. line for campaign/qwen38_grid.sh
(or dsflash_grid.sh with --prefix/--model).

    python3 campaign/rebalance.py --total 24 [--window-hours 6]
"""
import argparse
import statistics as st
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_status import SUITES, episodes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", default="results_qwen38_grid_n15")
ap.add_argument("--model", default="Qwen3.8-27B-FP8")
ap.add_argument("--total", type=int, default=24)
ap.add_argument("--window-hours", type=float, default=6.0)
ap.add_argument("--min", type=int, default=1, help="minimum workers per unfinished suite")
ap.add_argument("--ref-prefix", default="results_qwen38_reason_n15", help="finished campaign giving per-level reference minutes")
ap.add_argument("--ref-model", default="Qwen3.8-27B-FP8")
ap.add_argument("--min-n", type=int, default=5, help="episodes needed before a level counts as measured")
ap.add_argument("--min-frac", type=float, default=1/3, help="share of a level that must be finished before its median is trusted")
ap.add_argument("--ref-dir", action="append", default=[], metavar="SHORT=DIR",
                help="reference results dir for one suite, e.g. solo=results_dsflash_n15_solo_markov")
ap.add_argument("--ref-root", action="append", default=[], metavar="SHORT=RESULTS_ROOT",
                help="per-suite reference tree of the SAME game and model (e.g. a parked earlier run of this "
                     "campaign); its measured levels override the compact reference and the levels it lacks are "
                     "the compact medians scaled at the highest shared level")
args = ap.parse_args()
REPO = Path(__file__).resolve().parent.parent
window = datetime.now() - timedelta(hours=args.window_hours)

LEVELS = ("S1", "S2", "S3", "S4")
work = {}
for short, suite, game, expected in SUITES:
    rows = episodes(REPO / f"{args.prefix}_{short}", args.model, game)
    left = expected - len(rows)
    per_level = expected // len(LEVELS)
    left_by_level = {lv: per_level - sum(r["level"] == lv for r in rows) for lv in LEVELS}
    # Measured minutes per episode per level in THIS campaign (window first, then
    # all finished); a level counts as measured once it has --min-n episodes.
    meas = {}
    for lv in LEVELS:
        # All finished episodes of the level, never a time window: a level's slow
        # episodes finish last, so a recent window over-estimates a completed level.
        lv_rows = [r["dur"] for r in rows if r["level"] == lv]
        # A level in progress reports its fastest episodes first, so its median is
        # biased low until a good share of it is done: require --min-frac of the
        # level's episodes (and --min-n) before trusting the measurement.
        if len(lv_rows) >= max(args.min_n, args.min_frac * per_level):
            meas[lv] = (st.median(lv_rows) / 60, len(lv_rows))
    ref_dir = dict(kv.split("=", 1) for kv in args.ref_dir).get(short, f"{args.ref_prefix}_{short}")
    ref_rows = episodes(REPO / ref_dir, args.ref_model, game.split("_grid")[0])
    ref = {lv: st.median([r["dur"] for r in ref_rows if r["level"] == lv]) / 60
           for lv in LEVELS if any(r["level"] == lv for r in ref_rows)}
    ref_note = "ref=compact"
    own_root = dict(kv.split("=", 1) for kv in args.ref_root).get(short)
    if own_root:
        own_rows = episodes(REPO / own_root, args.model, game)
        own = {lv: st.median([r["dur"] for r in own_rows if r["level"] == lv]) / 60
               for lv in LEVELS if sum(r["level"] == lv for r in own_rows) >= args.min_n}
        if own and len(ref) == len(LEVELS):
            hi = [lv for lv in LEVELS if lv in own][-1]
            k = own[hi] / ref[hi]
            ref = {lv: own.get(lv, ref[lv] * k) for lv in LEVELS}
            ref_note = f"ref=own run up to {hi}, compact x{k:.2f} above"
    # Unmeasured levels: reference minutes rescaled by the lowest measured level.
    base = next((lv for lv in LEVELS if lv in meas), None)
    if base and len(ref) == len(LEVELS) and ref.get(base):
        scale = meas[base][0] / ref[base]
    elif len(ref) == len(LEVELS):
        scale = 1.0   # nothing measured yet (fresh tree): trust the reference as is
    else:
        scale = None
    hours = 0.0
    parts = []
    for lv in LEVELS:
        if lv in meas:
            m = meas[lv][0]; parts.append(f"{lv}:{m:.0f}m*")
        elif scale is not None:
            m = ref[lv] * scale; parts.append(f"{lv}:{m:.0f}m")
        else:
            m = meas[base][0] if base else float("nan"); parts.append(f"{lv}:{m:.0f}m?")
        hours += left_by_level[lv] * m / 60
    med = meas[base][0] if base else (ref["S1"] * scale if scale else float("nan"))
    work[short] = (left, med, sum(n for _, n in meas.values()), hours)
    print(f"{short:<5} remaining {left:3d} ({' '.join(f'{lv}:{left_by_level[lv]}' for lv in LEVELS)})  "
          f"min/ep {' '.join(parts)} (*=measured, n={work[short][2]}, {ref_note})  -> {hours:6.0f} ep-h")

active = {k: v for k, v in work.items() if v[0] > 0}
tot = sum(v[3] for v in active.values())
# Minimise the long pole: start every unfinished suite at --min, then hand each
# further stream to the suite whose hours-per-stream is currently largest. This
# is optimal for min-max(hours/streams) and, unlike proportional rounding, never
# leaves a cheap-per-call suite with many expensive episodes (solo) starved.
alloc = {k: args.min for k in active}
while sum(alloc.values()) < args.total:
    k = max(active, key=lambda k: active[k][3] / alloc[k])
    alloc[k] += 1
while sum(alloc.values()) > args.total:  # --min alone exceeds --total
    alloc[max(alloc, key=alloc.get)] -= 1
names = dict(solo="SOLO", dyad="DYAD", **{"3p": "P3", "6p": "P6"})
eta = max(work[k][3] / alloc[k] for k in alloc if alloc[k])
print(f"\ntotal {sum(alloc.values())} streams, ~{tot:.0f} episode-hours, ETA ~{eta:.0f} h (long pole: "
      f"{max(alloc, key=lambda k: work[k][3] / alloc[k])})")
wrapper = "dsflash_grid.sh" if "dsflash" in args.prefix else "qwen38_grid.sh"
print(" ".join(f"{names[k]}={alloc.get(k, 0)}" for k in ("solo", "dyad", "3p", "6p")) + f" bash campaign/{wrapper}")
