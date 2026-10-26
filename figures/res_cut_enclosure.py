"""fig:res-cut-enclosure: the two ways a team loses its board, side by side.

DeepSeek-V4.1, six players, S2, none, unshuffled 6p run. (a) A cut on board 12.
At action 30 Yellow declares done with D on its target, which freezes D beside
A's target. That target then has a wall or a frozen object on every side, so A
can no longer reach it, and the team plays on until a position repeats.
(b) An enclosure on board 10. At action 6 Purple moves F up into the cell B had
just left. B's other sides are two walls and C, so Orange has no legal move on
its next turn, tries three blocked moves and the episode aborts.

The pair is an illustration chosen after inspection, with both episodes in one
team game, level and protocol. Positions are replayed from the logs with
analysis/thesis/move_metrics.replay, the action the h* replay marks as the
first one after which the board has no solution is checked to be the one drawn,
and the facts the caption states are asserted in main().

    python3 figures/res_cut_enclosure.py
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from res_cut_s3 import load, region
from results_final_figs import FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE, OUT_DIR, REPO_ROOT, TEXT_W, publish
from style import FAINT, INK, MOVE, OWNER, WALL, at, draw_arrow, draw_grid, draw_target, marker, square_axes

RUN = "results/DeepSeek-V4.1-Flash/matrixgame_6p"
CUT = f"{RUN}/S2_none/instance_00012"
ENCLOSURE = f"{RUN}/S2_none/instance_00010"
# Only the two objects a panel is about are coloured, each in its owner's seat colour.
SEAT = {"Green": MOVE, "Orange": OWNER["B"], "Yellow": "#B08D00", "Purple": "#6A3D9A"}
OTHER = FAINT

SIDE, GAP, LEFT = 2.55, 0.62, 0.38
COLS = (LEFT, LEFT + SIDE + GAP)
HEAD = 0.78


def fatal(path):
    """The first action after which the h* replay finds no solution."""
    with open(os.path.join(REPO_ROOT, "analysis", "thesis", "move_metrics.final.jsonl")) as f:
        return next(r["fatal"] for r in map(json.loads, f) if r["path"] == path)


def sides(ep, cell, pos):
    """What lies on each side of a cell: 'wall', an object letter, or None when free."""
    n, at_cell, out = ep["n"], {c: o for o, c in pos.items()}, []
    r, c = cell
    for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        inside = 0 <= nb[0] < n and 0 <= nb[1] < n
        out.append("wall" if not inside or nb in ep["walls"] else at_cell.get(nb))
    return out


def panel(ax, ep, k, coloured, victim, arrow=None):
    pos, frozen = ep["board"](k)
    draw_grid(ax, ep["n"], ep["walls"], labels=FS_SMALL)
    draw_target(ax, ep["targets"][victim], victim, SEAT[ep["seat_of"][victim]], nested=False, lw=1.3, letter=8.7)
    if arrow:
        draw_arrow(ax, *arrow, SEAT[ep["seat_of"][coloured[0]]], lw=2.2, scale=11.0, shrink=(6.0, 8.5), zorder=9)
    for o, c in sorted(pos.items()):
        marker(ax, c, o, SEAT[ep["seat_of"][o]] if o in coloured else OTHER, frozen=o in frozen)


def main():
    cut, enc = load(CUT), load(ENCLOSURE)

    # (a) Yellow's done at action 30 freezes D beside A's target and closes it
    k = fatal(CUT)
    assert k == 30 and cut["acts"][k - 1][2] == "done" and cut["seat_of"]["D"] == "Yellow"
    assert cut["seat_of"]["A"] == "Green"
    pos, frozen = cut["board"](k)
    assert "D" in frozen and "D" not in cut["board"](k - 1)[1]
    walls_now = {pos[o] for o in frozen}
    assert cut["targets"]["A"] in region(cut, pos["A"], walls_now - {pos["D"]})
    assert cut["targets"]["A"] not in region(cut, pos["A"], walls_now)
    around = sides(cut, cut["targets"]["A"], pos)
    assert all(s == "wall" or s in frozen for s in around) and "D" in around, around
    assert cut["reason"] == "lose:cycle"
    cut_after = len(cut["acts"]) - k

    # (b) Purple's F up at action 6 walls in B, and Orange moves before Blue or Purple can clear it
    j = fatal(ENCLOSURE)
    assert j == 6 and enc["acts"][j - 1][2] == "F up" and enc["seat_of"]["F"] == "Purple"
    assert enc["seat_of"]["B"] == "Orange" and enc["seat_of"]["C"] == "Blue"
    before, _ = enc["board"](j - 1)
    pos_b, frozen_b = enc["board"](j)
    assert not frozen_b
    assert sorted(map(str, sides(enc, pos_b["B"], pos_b))) == ["C", "F", "wall", "wall"]
    assert pos_b["F"] == enc["board"](1)[0]["B"]          # the cell B left with action 2
    assert [enc["seat_of"][a[2][0]] for a in enc["acts"][j:]] == ["Green"]   # A down, then Orange's turn
    tried = [e["action"]["content"] for t in enc["it"]["turns"] for e in t if e["action"]["type"] == "invalid_move"]
    assert len(tried) == 3 and all(t.startswith("Player Orange: [occupied] Cannot move B") for t in tried), tried
    assert enc["reason"] == "abort:moves"

    H = HEAD + SIDE + 0.30 + 2 * 0.19 + 0.20 + 0.42
    fig = plt.figure(figsize=(TEXT_W, H))
    bottom = H - HEAD - SIDE
    heads = [("(a) cut", f"action {k}: Yellow declares done"),
             ("(b) enclosure", f"action {j}: Purple moves F up")]
    notes = [["D freezes beside A's target, which is now closed",
              f"on all four sides. The team plays on for {cut_after} actions."],
             ["B's other sides are two walls and C. On its next",
              "turn Orange has no legal move, and the game aborts."]]
    for x, (t1, t2), lines in zip(COLS, heads, notes):
        fig.text(*at(fig, x - 0.05, H - 0.10), t1, fontsize=FS_TITLE, color=INK, va="top", fontweight="bold")
        fig.text(*at(fig, x - 0.05, H - 0.33), t2, fontsize=FS_TICK, color=INK, va="top")
        for i, line in enumerate(lines):
            fig.text(*at(fig, x, bottom - 0.20 - i * 0.19), line, fontsize=FS_SMALL, color=INK, va="top")
    panel(square_axes(fig, COLS[0], bottom, SIDE), cut, k, coloured=("D", "A"), victim="A")
    panel(square_axes(fig, COLS[1], bottom, SIDE), enc, j, coloured=("F", "B"), victim="B",
          arrow=(before["F"], pos_b["F"]))

    keys = [
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=OTHER,
               markeredgecolor=INK, label="object of an uninvolved player"),
        Patch(facecolor="none", edgecolor=INK, linewidth=1.3, linestyle=(0, (2.2, 1.5)),
              label="target of A (a) and B (b)"),
        Line2D([], [], marker="s", linestyle="none", markersize=7, markerfacecolor=OTHER,
               markeredgecolor=INK, label="object frozen by a done"),
        Patch(facecolor=WALL, edgecolor=WALL, linewidth=0.5, label="wall"),
    ]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=at(fig, COLS[0] - 0.10, 0.04), ncol=2,
               frameon=False, fontsize=FS_LEGEND, handletextpad=0.6, columnspacing=1.0, labelspacing=0.5,
               borderpad=0.0)
    print(publish(fig, OUT_DIR, "res_cut_enclosure"))
    print(dict(cut_action=k, cut_after=cut_after, enclosure_action=j))


if __name__ == "__main__":
    main()
