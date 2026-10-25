"""The proposed S4 variant: the inner object's target moved into the corridor.

The corridor of S4 board 11 (fig:level-s4), rows 3-8 and columns 3-5, with
only the two objects of the swap drawn. B, the inner object, starts at the dead
end as in S4, but its target is now A's start cell, so A and B each stand on
the other's target. (a) The start. (b) On its way out B stands on its own
target while A waits beside the entrance. A done by B's owner there shuts A
out of the dead end for good.

The exact solver (matrixgame/utils/solver.py, solo rules) solves the whole
modified board alone in 48 moves, against 44 before the change.

    python3 figures/s4_inner_target.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from results_final_figs import FS_TICK, FS_TITLE, OUT_DIR, publish
from spatial_levels import load_instances
from style import FRAME, INK, OWNER, at, draw_grid, draw_objects

ROWS, COLS = (2, 8), (2, 5)  # R3-R8 and C3-C5: the corridor, its entrance and the cells beside it
CELL = 0.38  # inches on the page
PARK = (3, 4)  # R4C5, beside the entrance, where A waits in (b)
COLOURS = {"A": OWNER["A"], "B": OWNER["B"]}
LEFT, GAP, HEAD = 0.12, 0.55, 0.58


def panel(fig, x, y, walls, pos, targets):
    w, h = COLS[1] - COLS[0], ROWS[1] - ROWS[0]
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, y / H, w * CELL / W, h * CELL / H])
    draw_grid(ax, 8, walls, labels=0)
    ax.set_xlim(*COLS)
    ax.set_ylim(ROWS[1], ROWS[0])
    ax.add_patch(Rectangle((COLS[0], ROWS[0]), w, h, facecolor="none", edgecolor=FRAME, linewidth=0.9, zorder=5))
    draw_objects(ax, {"start_positions": pos, "target_positions": targets}, colours=COLOURS,
                 radius=0.30, letter=9.0, corner=8.0)


def main():
    board = next(i for i in load_instances()["S4"] if str(i["game_id"]) == "11")
    dev = board["device"]
    corridor, dead_end = {tuple(c) for c in dev["corridor"]}, tuple(dev["dead_end"])
    walls = {tuple(w) for w in board["walls"]}
    a0, b0 = tuple(board["start_positions"]["A"]), tuple(board["start_positions"]["B"])
    assert b0 == dead_end == tuple(board["target_positions"]["A"]) and a0 == tuple(dev["inner_adjacent"])
    entrance = tuple(dev["entrance"])
    crop = {(r, c) for r in range(*ROWS) for c in range(*COLS)}
    assert corridor | {entrance, PARK} <= crop and PARK not in walls | corridor
    assert abs(PARK[0] - entrance[0]) + abs(PARK[1] - entrance[1]) == 1

    targets = {"A": dead_end, "B": a0}  # the one change: B's target is A's start cell
    w, h = (COLS[1] - COLS[0]) * CELL, (ROWS[1] - ROWS[0]) * CELL
    W, H = LEFT + 2 * w + GAP + 0.35, HEAD + h + 0.08
    fig = plt.figure(figsize=(W, H))
    heads = [("(a) start", "each on the other's target"),
             ("(b) B on its way out", "a done here shuts A out")]
    for x, pos, (t1, t2) in zip((LEFT, LEFT + w + GAP), ({"A": a0, "B": b0}, {"A": PARK, "B": a0}), heads):
        fig.text(*at(fig, x - 0.05, H - 0.10), t1, fontsize=FS_TITLE, color=INK, va="top", fontweight="bold")
        fig.text(*at(fig, x - 0.05, H - 0.32), t2, fontsize=FS_TICK, color=INK, va="top")
        panel(fig, x, 0.08, walls, pos, targets)
    print(publish(fig, OUT_DIR, "s4_inner_target"), f"width {W:.2f} in")


if __name__ == "__main__":
    main()
