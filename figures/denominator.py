"""Draws F5: solo move efficiency by level under both denominators.

One panel per strong model. Each shows mean efficiency over solved solo
episodes (both solo protocols pooled) against S1-S4, once on the stored
per-object bound and once on the exact joint optimum. On the stored bound the
levels fall S1 > S2 > S3 > S4, the intended ladder; on the exact optimum S4
recovers above S3. The S3-S4 segment is the figure: it slopes down on one
denominator and up on the other.

Episodes and optima come from analysis/thesis/tables_comm.py's all_episodes() and
load_optima(), the loaders tab:solo is built from, so the figure and the table
cannot drift apart.

    python3 figures/denominator.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

from matplotlib.lines import Line2D

from style import CUE, FAINT, GRID, INK, MUTED, OWNER, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "thesis"))
import tables_comm as tc  # noqa: E402

EXACT, STORED = OWNER["A"], OWNER["B"]
TITLES = {"Qwen3.8": "Qwen3.8", "DeepSeek-V4.1-Flash": "DeepSeek-V4.1-Flash"}


def stored_bounds():
    """(level, game_id) -> stored optimal_moves, from the file load_optima reads."""
    path = os.path.join(REPO_ROOT, "analysis/efficiency_baseline/joint_optimum.jsonl")
    return {(d["level"], d["game_id"]): d["stored"]
            for d in map(json.loads, open(path)) if d.get("status") == "exact"}


def series(opt, sto):
    """model -> (exact per level, stored per level), solved solo episodes."""
    eps = tc.all_episodes(opt, ("solo",))
    out = {}
    for m in tc.STRONG:
        ex, so = [], []
        for lv in tc.ALL_LEVELS:
            solved = [e for e in eps if e["model"] == m and e["level"] == lv
                      and e["success"]]
            ex.append(st.mean(e["eff"] for e in solved))
            # eff is opt / moves, so stored / moves is eff rescaled
            so.append(st.mean(e["eff"] * sto[(lv, e["game_id"])]
                              / opt[(lv, e["game_id"])] for e in solved))
        out[m] = (ex, so)
    return out


def render(ser, out_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.5), sharey=True)
    xs = list(range(len(tc.ALL_LEVELS)))
    s3, s4 = tc.ALL_LEVELS.index("S3"), tc.ALL_LEVELS.index("S4")
    for ax, m in zip(axes, tc.STRONG):
        ex, so = ser[m]
        # the S3-S4 segment, where the two denominators disagree on direction
        ax.axvspan(s3, s4, color=CUE, alpha=0.09, linewidth=0, zorder=0)
        ax.plot(xs, ex, color=EXACT, marker="o", markersize=5.0,
                markeredgecolor="white", linewidth=1.6, zorder=3)
        ax.plot(xs, so, color=STORED, marker="s", markersize=5.0,
                markeredgecolor="white", linewidth=1.6, linestyle="--", zorder=3)
        for x in (s3, s4):
            ax.text(x, ex[x] + 0.018, f"{ex[x]:.2f}", ha="center", va="bottom",
                    fontsize=6.6, color=INK)
            ax.text(x, so[x] - 0.022, f"{so[x]:.2f}", ha="center", va="top",
                    fontsize=6.6, color=MUTED)
        ax.set_title(TITLES[m], fontsize=7.8, color=INK, pad=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(tc.ALL_LEVELS, fontsize=7.2, color=MUTED)
        ax.set_xlim(-0.35, len(xs) - 0.65)
        ax.set_ylim(0.65, 1.04)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels([f"{t:.1f}" for t in (0.7, 0.8, 0.9, 1.0)],
                           fontsize=7.2, color=MUTED)
        ax.tick_params(length=0, pad=3)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(FAINT)
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("move efficiency (solved)", fontsize=7.6, color=INK,
                       labelpad=4)

    handles = [Line2D([], [], color=EXACT, marker="o", markersize=5.0,
                      markeredgecolor="white", linewidth=1.6,
                      label="exact joint optimum"),
               Line2D([], [], color=STORED, marker="s", markersize=5.0,
                      markeredgecolor="white", linewidth=1.6, linestyle="--",
                      label="stored per-object bound")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=7.4, handletextpad=0.5,
               columnspacing=2.0, labelcolor=MUTED)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.80, bottom=0.11, wspace=0.10)
    return save(fig, out_dir, "denominator", ("pdf", "png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    ser = series(tc.load_optima(), stored_bounds())
    # the numbers S7.1 quotes, so a changed tree fails loudly
    ex = {m: [round(x, 2) for x in ser[m][0]] for m in tc.STRONG}
    assert ex == {"Qwen3.8": [1.00, 0.90, 0.85, 0.89],
                  "DeepSeek-V4.1-Flash": [1.00, 0.88, 0.85, 0.93]}, ex
    for m in tc.STRONG:
        ex_, so = ser[m]
        assert so == sorted(so, reverse=True), (m, so)  # stored: the intended ladder
        assert ex_[3] > ex_[2], (m, ex_)                 # exact: S4 recovers above S3

    for path in render(ser, args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
