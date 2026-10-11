"""Draws the grounding figure: whose object a positional claim is about.

One paired row per model and prose protocol. The left dot is the model's
accuracy about its own object, the right dot its accuracy about a teammate's,
and the gap between them is the whole result: self-localisation does not
separate the models, claims about others do.

Counts come from the annotated corpus cache written by
analysis/qualitative_analysis/qual_corpus.py, so the figure and Sections 5.3 / 2.3 of
results_three_models_grid.md cannot drift apart.

    python3 figures/grounding.py                # whole multiplayer corpus
    python3 figures/grounding.py --scope dyad   # the Table 12 cell
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os

from matplotlib.lines import Line2D

from style import FAINT, GRID, INK, MUTED, OWNER, OBJECT, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
CACHE = os.path.join(REPO_ROOT, "analysis/qualitative_analysis", "corpus.json")

OWN, OTHER = OBJECT, OWNER["B"]
MODELS = (("qwen38", "Qwen3.8"), ("dsv41", "DeepSeek-V4.1"), ("nemotron", "Nemotron"))
PROTOCOLS = ("freeform", "hybrid")


def wilson(ok: int, n: int, z: float = 1.96):
    """95% interval on a proportion. Normal approximation is useless at n = 4."""
    if not n:
        return 0.0, 0.0, 0.0
    p = ok / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def tally(scope: str):
    """(own_ok, own_n, other_ok, other_n) per model and protocol."""
    rows = json.load(open(CACHE))["rows"]
    agg = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if scope != "all" and r["suite"] != scope:
            continue
        a = agg[(r["model"], r["proto"])]
        a[0] += r["own_ok"]
        a[1] += r["own_ok"] + r["own_bad"]
        a[2] += r["oth_ok"]
        a[3] += r["oth_ok"] + r["oth_bad"]
    return agg


def render(scope: str, out_dir: str):
    import matplotlib.pyplot as plt

    agg = tally(scope)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    labels, y = [], 0.0
    ticks = []

    for mi, (key, name) in enumerate(MODELS):
        for proto in PROTOCOLS:
            own_ok, own_n, oth_ok, oth_n = agg[(key, proto)]
            po, lo, hi = wilson(own_ok, own_n)
            pt, lt, ht = wilson(oth_ok, oth_n)
            ax.plot([po * 100, pt * 100], [y, y], color=GRID, linewidth=2.0, zorder=2,
                    solid_capstyle="round")
            for p, lo_, hi_, colour in ((po, lo, hi, OWN), (pt, lt, ht, OTHER)):
                ax.plot([lo_ * 100, hi_ * 100], [y, y], color=colour, alpha=0.35,
                        linewidth=2.0, zorder=3, solid_capstyle="round")
                ax.plot([p * 100], [y], marker="o", markersize=7.0, color=colour,
                        markeredgecolor="white", markeredgewidth=1.4, zorder=4)
            ax.text(101.5, y, f"{oth_n:,}".replace(",", " "), va="center", ha="left",
                    fontsize=7.0, color=MUTED)
            ticks.append(y)
            labels.append(proto)
            y -= 1.0
        ax.text(-2.0, ticks[-2] + 0.5, name, ha="right", va="center", fontsize=8.2,
                color=INK, fontweight="bold")
        y -= 0.45

    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.6, color=MUTED)
    ax.set_ylim(y + 0.6, 0.9)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 25)], fontsize=7.6, color=MUTED)
    ax.set_xlabel("positional claims true of the board when uttered", fontsize=8.0,
                  color=INK, labelpad=6)
    ax.text(101.5, 0.95, "claims about\na teammate's", fontsize=7.0, color=MUTED,
            ha="left", va="center", linespacing=1.25)
    ax.tick_params(length=0, pad=4)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(FAINT)
    ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=1)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                      color=OWN, markeredgecolor="white", markeredgewidth=1.2,
                      label="about its own object"),
               Line2D([], [], marker="o", linestyle="none", markersize=6.5,
                      color=OTHER, markeredgecolor="white", markeredgewidth=1.2,
                      label="about a teammate's object")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=2, frameon=False, fontsize=7.6, handletextpad=0.5,
              columnspacing=1.8, labelcolor=MUTED)

    fig.subplots_adjust(left=0.20, right=0.86, top=0.88, bottom=0.20)
    stem = "grounding" if scope == "all" else f"grounding_{scope}"
    return save(fig, out_dir, stem, ("pdf", "png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="all", choices=("all", "dyad", "3p", "6p"))
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    agg = tally("all")
    # the two figures Section 5.3 leads with, so a changed cache fails loudly
    assert agg[("nemotron", "freeform")][2:] == [168, 560], agg[("nemotron", "freeform")]
    assert agg[("nemotron", "hybrid")][2:] == [3, 50], agg[("nemotron", "hybrid")]

    for path in render(args.scope, args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
