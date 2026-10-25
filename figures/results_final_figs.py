"""Figures of the results outline (thesis/results_outline.tex), one model at a time.

    res_clem_level        clemscore by spatial level, one panel per team game, one line per
                          protocol, solo by level as the dashed baseline
    res_clem_teamsize     clemscore by team size, one line per protocol, solo as reference
    res_clem_heatmap      clemscore per level x protocol, one panel per team game,
                          unsuccessful episodes printed under each value
    res_move_level        optimal-move rate and oscillation by level, one panel per team game,
                          one line per protocol, solo as the dashed baseline
    res_detour            forced-detour and free-state accuracy against their chance levels,
                          one row per condition, solo baseline on top

Every number comes from analysis/thesis/results_final.py (episodes, clem) and
move_metrics.final.jsonl (pool), the loaders the outline's tables are printed
from, so the figures and the tables cannot drift apart.

    python3 figures/results_final_figs.py [--model dsv41]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from style import CUE, FAINT, GRID, INK, MOVE, MUTED, OWNER, PAPER, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "thesis"))
import results_final as rf  # noqa: E402

GAME_STYLE = {"solo": (INK, "D"), "dyad": (OWNER["A"], "o"), "3p": (MOVE, "^"), "6p": (OWNER["B"], "s")}
PROTO_STYLE = {"none": (OWNER["B"], "s"), "structured": (CUE, "D"), "hybrid": (MOVE, "^"),
               "freeform": (OWNER["A"], "o")}
GAME_NAME = {"solo": "solo", "dyad": "dyad", "3p": "3p", "6p": "6p"}

# Drawn at the printed size, so a point in the file is a point on the page: the thesis is
# 12pt report on A4 with 2.5 cm margins, a 16.0 cm (6.30 in) text block. \thesisfig places
# full-width figures at \linewidth and the team-size figure at 0.55\linewidth.
TEXT_W = 6.30
HALF_W = 0.55 * TEXT_W
# No text below 8 pt on the page; the body is 12 pt.
FS_TICK, FS_LEGEND, FS_LABEL, FS_TITLE, FS_VALUE, FS_SMALL = 9.0, 9.0, 10.0, 10.0, 9.0, 8.0
# The solo baseline is the reference in every panel: MUTED keeps it at 4.8:1 on white
# (FAINT, 2.6:1, is below the 3:1 a mark the reading depends on needs).
BASE = MUTED
LW, MS = 1.7, 5.6


def publish(fig, out, stem):
    """save(), after checking that no visible text on the page is below 8 pt."""
    from matplotlib.text import Text
    fig.canvas.draw()
    small = [(t.get_text(), t.get_fontsize()) for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FS_SMALL]
    assert not small, f"{stem}: text below {FS_SMALL} pt: {small[:5]}"
    return save(fig, out, stem, ("pdf", "png"))


def sel(X, **k):
    return [r for r in X if all(r[a] == b for a, b in k.items())]


def frame(ax, xlabels, ylim, yticks, fmt="{:g}"):
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=FS_TICK, color=MUTED)
    ax.set_xlim(-0.3, len(xlabels) - 0.7)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([fmt.format(t) for t in yticks], fontsize=FS_TICK, color=MUTED)
    ax.tick_params(length=0, pad=3)
    for side, spine in ax.spines.items():
        spine.set_visible(side == "bottom")
        spine.set_color(FAINT)
    ax.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def line(ax, ys, colour, marker, label=None, dashed=False, dx=0.0):
    ax.plot([i + dx for i in range(len(ys))], ys, color=colour, linewidth=LW, marker=marker, markersize=MS,
            markeredgecolor=PAPER, markeredgewidth=0.9, zorder=3, label=label,
            linestyle="--" if dashed else "-")


def baseline(ax, ys):
    ax.plot(range(len(ys)), ys, color=BASE, linewidth=1.3, linestyle=(0, (3, 2)), marker="D",
            markersize=MS - 1.2, markerfacecolor=PAPER, markeredgecolor=BASE, markeredgewidth=1.0, zorder=2)


def baseline_legend(fig, y=1.0):
    handles = [Line2D([], [], color=BASE, linestyle=(0, (3, 2)), marker="D", markersize=MS - 1.2,
                      markerfacecolor=PAPER, markeredgecolor=BASE, linewidth=1.3, label="solo baseline")]
    handles += [Line2D([], [], color=c, marker=m, markersize=MS, markeredgecolor=PAPER, linewidth=LW, label=p)
                for p, (c, m) in PROTO_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=len(handles), frameon=False,
               fontsize=FS_LEGEND, handlelength=2.2, handletextpad=0.5, columnspacing=1.4, labelcolor=INK)


def title(ax, text):
    ax.set_title(text, fontsize=FS_TITLE, color=INK, pad=5)


def clem_level(R, out):
    """One panel per team game: clemscore by level under each protocol, solo as the dashed baseline."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.9), sharey=True)
    solo = [rf.clem(sel(R, suite="solo", level=lv))[0] for lv in rf.LEVELS]
    for ax, s in zip(axes, rf.TEAMS):
        baseline(ax, solo)
        for p, (c, m) in PROTO_STYLE.items():
            line(ax, [rf.clem(sel(R, suite=s, level=lv, proto=p))[0] for lv in rf.LEVELS], c, m)
        frame(ax, rf.LEVELS, (25, 101), range(30, 101, 10))
        title(ax, GAME_NAME[s])
    axes[0].set_ylabel("clemscore", fontsize=FS_LABEL, color=INK, labelpad=4)
    baseline_legend(fig)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.79, bottom=0.10, wspace=0.10)
    return publish(fig, out, "res_clem_level")


def clem_teamsize(R, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(HALF_W, 2.9))
    solo = rf.clem(sel(R, suite="solo"))[0]
    ax.axhline(solo, color=BASE, linewidth=1.3, linestyle=(0, (3, 2)), zorder=2)
    ax.text(-0.25, solo - 0.8, f"solo baseline {solo:.1f}", ha="left", va="top", fontsize=FS_SMALL, color=INK)
    for p, (c, m) in PROTO_STYLE.items():
        line(ax, [rf.clem(sel(R, suite=s, proto=p))[0] for s in rf.TEAMS], c, m)
    frame(ax, ["dyad", "3p", "6p"], (50, 95), range(50, 96, 5))
    ax.set_ylabel("clemscore", fontsize=FS_LABEL, color=INK, labelpad=4)
    handles = [Line2D([], [], color=c, marker=m, markersize=MS, markeredgecolor=PAPER, linewidth=LW, label=p)
               for p, (c, m) in PROTO_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2, frameon=False,
               fontsize=FS_LEGEND, handlelength=2.2, handletextpad=0.5, columnspacing=1.2, labelcolor=INK)
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.10)
    return publish(fig, out, "res_clem_teamsize")


def ink_on(rgb):
    """INK or PAPER, whichever contrasts more with a cell colour (WCAG relative luminance)."""
    def lum(c):
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c[:3]]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    from matplotlib.colors import to_rgb
    cell, ink, paper = lum(rgb), lum(to_rgb(INK)), lum(to_rgb(PAPER))
    ratio = lambda a, b: (max(a, b) + 0.05) / (min(a, b) + 0.05)
    return INK if ratio(cell, ink) >= ratio(cell, paper) else PAPER


def clem_heatmap(R, out):
    """Every cell carries its value, so there is no colour bar: the shading only groups."""
    import matplotlib.pyplot as plt
    cmap = LinearSegmentedColormap.from_list("clem", ["#F4F7FA", OWNER["A"]])
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W, 2.9), sharey=True)
    for ax, s in zip(axes, rf.TEAMS):
        grid = [[sel(R, suite=s, level=lv, proto=p) for p in rf.PROTOS] for lv in rf.LEVELS]
        q = [[rf.clem(c)[0] for c in row] for row in grid]
        ax.imshow(q, cmap=cmap, vmin=25, vmax=100, aspect="auto")
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                fails = sum(not r["success"] for r in cell)
                ink = ink_on(cmap((q[i][j] - 25) / 75))
                ax.text(j, i - (0.12 if fails else 0), f"{q[i][j]:.1f}", ha="center", va="center",
                        fontsize=FS_VALUE, color=ink)
                if fails:
                    ax.text(j, i + 0.24, f"({fails})", ha="center", va="center", fontsize=FS_SMALL, color=ink)
        ax.set_xticks(range(len(rf.PROTOS)))
        ax.set_xticklabels(["none", "struct.", "hybrid", "free."], fontsize=FS_TICK, color=MUTED)
        ax.set_yticks(range(len(rf.LEVELS)))
        ax.set_yticklabels(rf.LEVELS, fontsize=FS_TICK, color=MUTED)
        ax.tick_params(length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        title(ax, GAME_NAME[s])
    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.10, wspace=0.06)
    return publish(fig, out, "res_clem_heatmap")


def move_level(M, out):
    """Optimal-move rate (top) and oscillation (bottom) by level, one panel per team game,
    one line per protocol, solo as the dashed baseline."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 5.0), sharey="row")
    rows = (("opt", "optimal-move rate", (0.84, 1.0), [0.85, 0.9, 0.95, 1.0], "{:.2f}"),
            ("osc", "oscillation rate", (0, 0.24), [0, 0.05, 0.1, 0.15, 0.2], "{:.2f}"))
    for r, (key, ylab, ylim, ticks, fmt) in enumerate(rows):
        solo = [rf.pool(sel(M, suite="solo", level=lv))[key] for lv in rf.LEVELS]
        for ax, s in zip(axes[r], rf.TEAMS):
            baseline(ax, solo)
            for p, (c, m) in PROTO_STYLE.items():
                line(ax, [rf.pool(sel(M, suite=s, level=lv, proto=p))[key] for lv in rf.LEVELS], c, m)
            frame(ax, rf.LEVELS, ylim, ticks, fmt)
            if r == 0:
                title(ax, GAME_NAME[s])
        axes[r][0].set_ylabel(ylab, fontsize=FS_LABEL, color=INK, labelpad=4)
    baseline_legend(fig)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.06, wspace=0.10, hspace=0.30)
    return publish(fig, out, "res_move_level")


def detour(M, out):
    """Forced (left) and free states (right): chance (open) joined to accuracy (filled),
    one row per condition, solo baseline on top."""
    import matplotlib.pyplot as plt
    conds = [("solo", None)] + [(s, p) for s in rf.TEAMS for p in rf.PROTOS]
    ys, y = [], 0.0
    for i, (s, p) in enumerate(conds):
        if i and conds[i - 1][0] != s:
            y += 0.6  # gap between games
        ys.append(y)
        y += 1
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 4.4), sharey=True)
    for ax, (head, acc, ch, n) in zip(axes, (("forced-detour states", "accF", "chF", "nF"),
                                            ("free states", "accG", "chG", "nG"))):
        for yy, (s, p) in zip(ys, conds):
            q = rf.pool(sel(M, suite=s) if p is None else sel(M, suite=s, proto=p))
            colour = BASE if p is None else PROTO_STYLE[p][0]
            if q[acc] is None:
                continue
            ax.plot([q[ch], q[acc]], [yy, yy], color=colour, linewidth=LW, zorder=2)
            ax.plot(q[ch], yy, marker="o", markersize=MS, markerfacecolor=PAPER, markeredgecolor=colour,
                    markeredgewidth=1.3, linestyle="none", zorder=3)
            ax.plot(q[acc], yy, marker="o", markersize=MS, color=colour, markeredgecolor=PAPER,
                    markeredgewidth=0.8, linestyle="none", zorder=3)
            if acc == "accF":
                ax.text(1.03, yy, f"{q[n]}", ha="left", va="center", fontsize=FS_SMALL, color=INK,
                        transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=FS_TICK, color=MUTED)
        title(ax, head)
        ax.tick_params(length=0, pad=3)
        for side, spine in ax.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(FAINT)
        ax.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)
    axes[0].text(1.03, ys[0] - 1.0, "$|F|$", ha="left", va="center", fontsize=FS_SMALL, color=INK,
                 transform=axes[0].get_yaxis_transform())
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(["solo baseline" if p is None else f"{GAME_NAME[s]}, {p}" for s, p in conds],
                            fontsize=FS_TICK, color=INK)
    axes[0].invert_yaxis()
    fig.supxlabel("share of actions that are optimal", fontsize=FS_LABEL, color=INK, y=0.015)
    handles = [Line2D([], [], marker="o", markersize=MS, markerfacecolor=PAPER, markeredgecolor=INK,
                      markeredgewidth=1.3, linestyle="none", label="chance"),
               Line2D([], [], marker="o", markersize=MS, color=INK, linestyle="none", label="accuracy")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.62, 1.0), ncol=2, frameon=False,
               fontsize=FS_LEGEND, labelcolor=INK)
    fig.subplots_adjust(left=0.24, right=0.985, top=0.86, bottom=0.11, wspace=0.16)
    return publish(fig, out, "res_detour")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dsv41")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    R = rf.episodes(args.model)
    M = [r for r in map(json.loads, open(os.path.join(REPO_ROOT, "analysis", "thesis", "move_metrics.final.jsonl")))
         if r["model"] == args.model]
    if args.model == "dsv41":
        # pinned to tab:res-clem and tab:res-move-level, so a changed tree fails here
        assert [round(rf.clem(sel(R, suite=s))[0], 1) for s in rf.SUITES] == [82.1, 86.7, 82.7, 77.9]
        assert round(rf.pool(sel(M, suite="solo"))["accF"], 3) == 0.700
        assert round(rf.pool(sel(M, suite="6p", proto="freeform"))["accF"], 3) == 0.913
    for path in (clem_level(R, args.out) + clem_teamsize(R, args.out) + clem_heatmap(R, args.out)
                 + move_level(M, args.out) + detour(M, args.out)):
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
