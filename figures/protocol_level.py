"""Draws F3: the protocol x level interaction, prose against silence.

Four small multiples on one y-scale: all three multiplayer suites pooled, then
the dyad, 3p and 6p. Each panel is success by spatial level under NONE and
under prose (FREEFORM and HYBRID pooled); the band between them is the
prose-minus-silence gap. The pooled panel labels the gap, the suite panels
label S3, the level the chapter's argument rests on.

Cells come from analysis/thesis/tables_comm.py's ladder(), the loader tab:ladder is
built from, so the figure and the tables cannot drift apart.

    python3 figures/protocol_level.py
"""

from __future__ import annotations

import argparse
import os
import sys

from matplotlib.lines import Line2D

from style import FAINT, GRID, INK, MUTED, OWNER, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "thesis"))
import tables_comm as tc  # noqa: E402

PROSE, SILENCE = OWNER["A"], OWNER["B"]
PANELS = (("All three suites", tc.SUITES), ("Dyad", ("dyad",)),
          ("3 players", ("3p",)), ("6 players", ("6p",)))


def rates(lad, suites):
    """Per level: (silence %, prose %), strong models only."""
    out = []
    for lv in tc.ALL_LEVELS:
        def pct(protos):
            v = [x for m in tc.STRONG for s in suites for p in protos
                 for x in lad[(m, s, lv, p)]]
            return 100.0 * sum(v) / len(v)
        out.append((pct(("none",)), pct(("freeform", "hybrid"))))
    return out


def render(out_dir):
    import matplotlib.pyplot as plt

    lad = tc.ladder()
    fig, axes = plt.subplots(1, 4, figsize=(6.4, 2.45), sharey=True)
    xs = range(len(tc.ALL_LEVELS))
    for i, (ax, (title, suites)) in enumerate(zip(axes, PANELS)):
        r = rates(lad, suites)
        sil, pro = [a for a, _ in r], [b for _, b in r]
        ax.fill_between(xs, sil, pro, color=PROSE, alpha=0.10, linewidth=0, zorder=1)
        for ys, colour, marker in ((sil, SILENCE, "s"), (pro, PROSE, "o")):
            ax.plot(xs, ys, color=colour, linewidth=1.6, marker=marker,
                    markersize=5.0, markeredgecolor="white", markeredgewidth=1.0,
                    zorder=3)
        if i == 0:
            # a row of its own: at S1-S2 the gap is too thin to label in place
            ax.text(-0.3, 19, "prose $-$ silence", ha="left", va="center",
                    fontsize=6.4, color=MUTED)
            for x, (a, b) in zip(xs, r):
                ax.text(x, 9, f"+{b - a:.1f}", ha="center", va="center",
                        fontsize=6.8, color=INK)
        else:
            s3 = tc.ALL_LEVELS.index("S3")
            # below-left of the point: the line into S3 runs above it there
            ax.text(s3 - 0.12, sil[s3] - 1.5, f"{sil[s3]:.0f}", ha="right",
                    va="top", fontsize=6.6, color=MUTED)
            ax.text(s3, pro[s3] - 4.5, f"{pro[s3]:.0f}", ha="center", va="top",
                    fontsize=6.6, color=MUTED)
        ax.set_title(title, fontsize=7.8, color=INK, pad=4,
                     fontweight="bold" if i == 0 else "normal")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(tc.ALL_LEVELS, fontsize=7.2, color=MUTED)
        ax.set_xlim(-0.35, len(xs) - 0.65)
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
    axes[0].set_ylabel("episodes solved", fontsize=7.6, color=INK, labelpad=4)

    handles = [Line2D([], [], color=PROSE, marker="o", markersize=5.0,
                      markeredgecolor="white", linewidth=1.6,
                      label=r"prose (freeform, hybrid)"),
               Line2D([], [], color=SILENCE, marker="s", markersize=5.0,
                      markeredgecolor="white", linewidth=1.6,
                      label="silence (none)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=False, fontsize=7.4, handletextpad=0.5,
               columnspacing=2.0, labelcolor=MUTED)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.11, wspace=0.12)
    return save(fig, out_dir, "protocol_level", ("pdf", "png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    lad = tc.ladder()
    # the numbers the figure prints, pinned to the ledger (results_numbers.md
    # A123: success gap by level; A80/A125: S3 by suite), so a changed tree fails
    gap = [round(b - a, 1) for a, b in rates(lad, tc.SUITES)]
    assert gap == [1.1, 14.4, 38.9, 25.6], gap
    s3 = [rates(lad, (s,))[2] for s in tc.SUITES]
    assert [round(a) for a, _ in s3] == [87, 47, 30], s3
    assert [round(b) for _, b in s3] == [100, 98, 82], s3

    for path in render(args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
