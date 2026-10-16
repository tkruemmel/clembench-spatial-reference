"""Draws the protocol ladder: tab:ladder's cells as a slope chart.

One panel per model, the four protocols in ladder order on x, one line per
suite. Each segment is one of the three prices -- existence, expressiveness,
shared frame -- so the claim "prose > schema > silence in every suite" reads as
every line rising and then going flat. Suite is ordered by team size, so it is
a light-to-dark ink ramp with a marker per suite, not a categorical hue: blue
and orange already mean prose and silence in F3.

Cells come from analysis/thesis/tables_comm.py's ladder(), the loader tab:ladder is
built from.

    python3 figures/protocol_ladder.py
"""

from __future__ import annotations

import argparse
import os
import sys

from matplotlib.lines import Line2D

from style import FAINT, GRID, INK, MUTED, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "thesis"))
import tables_comm as tc  # noqa: E402

# lightest step is the lightest grey that still clears 3:1 on white
SUITE_STYLE = {"dyad": ("#7D8995", "o", "dyad"),
               "3p": ("#56616D", "s", "3 players"),
               "6p": (INK, "^", "6 players")}
NAMES = {"Qwen3.8": "Qwen3.8", "DeepSeek-V4.1-Flash": "DeepSeek-V4.1-Flash",
         "Nemotron": "Nemotron"}
# successes of 60 per (model, suite), in PROTOCOLS order (none, structured,
# freeform, hybrid); pinned to the ledger (results_numbers.md A120/A121,
# ledger_extra ladder.succ.*), so a changed tree fails loudly
PINNED = {
    "Qwen3.8": {"dyad": (54, 58, 60, 59), "3p": (46, 55, 58, 59), "6p": (39, 44, 56, 53)},
    "DeepSeek-V4.1-Flash": {"dyad": (54, 58, 60, 60), "3p": (43, 50, 60, 58),
                            "6p": (39, 48, 55, 56)},
    "Nemotron": {"dyad": (0, 3, 6, 3), "3p": (0, 2, 3, 2), "6p": (0, 2, 4, 3)},
}


def check(lad):
    """The plotted counts, and the property the figure shows: in the strong
    models every suite orders the protocols prose >= schema >= silence."""
    for model, suites in PINNED.items():
        for suite, want in suites.items():
            got = tuple(sum(x for lv in tc.ALL_LEVELS for x in lad[(model, suite, lv, p)])
                        for p in tc.PROTOCOLS)
            assert got == want, (model, suite, got)
            if model in tc.STRONG:
                none, schema, free, hyb = got
                assert min(free, hyb) >= schema >= none, (model, suite, got)


def render(out_dir):
    import matplotlib.pyplot as plt

    lad = tc.ladder()
    check(lad)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.5), sharey=True)
    xs = range(len(tc.PROTOCOLS))
    for ax, model in zip(axes, tc.ARMS):
        for suite, (colour, marker, _) in SUITE_STYLE.items():
            ys = []
            for p in tc.PROTOCOLS:
                v = [x for lv in tc.ALL_LEVELS for x in lad[(model, suite, lv, p)]]
                assert len(v) == 60, (model, suite, p, len(v))
                ys.append(100.0 * sum(v) / len(v))
            ax.plot(xs, ys, color=colour, linewidth=1.5, marker=marker,
                    markersize=5.0, markeredgecolor="white", markeredgewidth=0.9,
                    zorder=3)
        ax.set_title(NAMES[model], fontsize=7.8, color=INK, pad=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(tc.PROTOCOLS, fontsize=6.8, color=MUTED, rotation=30,
                           ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.3, len(xs) - 0.7)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 25))
        ax.set_yticklabels([f"{t}%" for t in range(0, 101, 25)], fontsize=7.2,
                           color=MUTED)
        ax.tick_params(length=0, pad=3)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(FAINT)
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("episodes solved, S1–S4", fontsize=7.6, color=INK,
                       labelpad=4)

    handles = [Line2D([], [], color=c, marker=m, markersize=5.0,
                      markeredgecolor="white", linewidth=1.5, label=lab)
               for c, m, lab in SUITE_STYLE.values()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, frameon=False, fontsize=7.4, handletextpad=0.5,
               columnspacing=2.0, labelcolor=MUTED)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.22, wspace=0.12)
    return save(fig, out_dir, "protocol_ladder", ("pdf", "png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    for path in render(args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
