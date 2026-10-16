"""Draws F-teamsize: the cost of team size at each level (S9.1), as two
separate figures, teamsize_quality (the Quality Score) and teamsize_efficiency
(move efficiency over successes), both on each suite's attainable optimum. The x-axis is players on the team (dyad 2,
3p, 6p), spaced by headcount so the 3p->6p step reads as three added agents.
One line per level, labelled at its right end; S3 is the level the section's
argument rests on and is drawn in the cue colour. 6p points below 100 % Played
are hollow, since part of their fall is aborts. Solo is left out: it has no
teammate, so it is not a step of the team-size ladder (tab:teamsize keeps it).

Cells come from analysis/thesis/tables_comm.py's teamsize_cells(), the loader
tab:teamsize is built from, so the figure and the table cannot drift apart.

    python3 figures/teamsize.py
"""

from __future__ import annotations

import argparse
import os
import sys

from style import FAINT, GRID, INK, MUTED, OBJECT, OWNER, PAPER, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "thesis"))
import tables_comm as tc  # noqa: E402

PLAYERS = {"dyad": 2, "3p": 3, "6p": 6}
COLOUR = {"S1": FAINT, "S2": MUTED, "S3": OWNER["B"], "S4": OBJECT}
# stem, cell key, scale, y-limits, y-ticks, y-label, min label gap (y units)
FIGURES = (("teamsize_quality", "q", 1.0, (40, 100), range(40, 101, 20),
            "Quality Score", 4.0),
           ("teamsize_efficiency", "eff", 100.0, (65, 100), range(70, 101, 10),
            "move efficiency (% of optimum)", 2.2))


def render(cells, out_dir):
    import matplotlib.pyplot as plt

    paths = []
    for stem, key, scale, ylim, yticks, ylabel, gap in FIGURES:
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        for lv in tc.ALL_LEVELS:
            c, lw = COLOUR[lv], 2.0 if lv == "S3" else 1.4
            xs = [PLAYERS[s] for s in tc.SUITES]
            ys = [scale * cells[(s, lv)][key] for s in tc.SUITES]
            ax.plot(xs, ys, color=c, linewidth=lw, zorder=3)
            for s, x, y in zip(tc.SUITES, xs, ys):
                full = cells[(s, lv)]["played"] == 100.0
                ax.plot(x, y, marker="o", markersize=4.6, color=c, zorder=4,
                        markerfacecolor=c if full else PAPER,
                        markeredgecolor=PAPER if full else c,
                        markeredgewidth=0.9 if full else 1.2)
        ends = sorted(((scale * cells[("6p", lv)][key], lv) for lv in tc.ALL_LEVELS),
                      reverse=True)
        placed = []
        for y, lv in ends:
            if placed and placed[-1] - y < gap:
                y = placed[-1] - gap
            placed.append(y)
            ax.text(6.2, y, lv, ha="left", va="center", fontsize=7.0,
                    color=COLOUR[lv], fontweight="bold" if lv == "S3" else "normal")
        ax.set_xticks(list(PLAYERS.values()))
        ax.set_xticklabels(["dyad", "3p", "6p"], fontsize=7.2, color=MUTED)
        ax.set_xlim(1.7, 6.7)
        ax.set_xlabel("players on the team", fontsize=7.2, color=MUTED, labelpad=3)
        ax.set_ylim(*ylim)
        ax.set_yticks(list(yticks))
        ax.set_yticklabels([str(t) for t in yticks], fontsize=7.2, color=MUTED)
        ax.set_ylabel(ylabel, fontsize=7.4, color=INK, labelpad=4)
        ax.tick_params(length=0, pad=3)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(FAINT)
        ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
        fig.subplots_adjust(left=0.13, right=0.94, top=0.96, bottom=0.17)
        paths += save(fig, out_dir, stem, ("pdf", "png"))
        plt.close(fig)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    cells, _, _ = tc.teamsize_cells(tc.load_optima(), tests=False)
    # the plotted Quality Scores, pinned to the ledger (results_numbers.md A130),
    # so a changed tree fails loudly
    q = [tuple(round(cells[(s, l)]["q"], 2) for s in tc.SUITES) for l in tc.ALL_LEVELS]
    assert q == [(97.32, 95.60, 96.03), (83.35, 77.36, 80.35),
                 (76.19, 62.89, 53.49), (80.74, 74.14, 77.48)], q
    # hollow 6p points: % Played below 100 at S2-S4 (A132)
    assert [round(cells[("6p", l)]["played"], 1) for l in tc.ALL_LEVELS] == [100.0, 95.8, 90.0, 89.2]

    for path in render(cells, args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
