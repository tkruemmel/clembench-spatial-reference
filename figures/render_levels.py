"""Draws the spatial-level figures and the board-model figure.

  spatial_level_S<n>  the board: start cells, target cells, walls
  board_model         the S2 board drawn, beside the grid Player Green is sent

    python3 figures/render_levels.py
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

from spatial_levels import LEVELS, OUT_DIR, REPO_ROOT, check_chains, load_instances, pick_representative
from style import (
    FAINT,
    INK,
    MONO,
    MUTED,
    OBJECT,
    PAPER,
    SEAT,
    WALL,
    WALL_HI,
    Line2D,
    board_legend,
    draw_grid,
    draw_objects,
    plt,
    save,
)

# The S3 board the thesis shows; the others are pick_representative's choice.
PICKS = {"S3": 10}
MODEL_LEVEL = "S2"  # the level of the board-model figure


def render_board(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """The board and nothing else: start cells, target cells, walls.

    Drawn at the printed width (0.75 of the text width), so the font sizes are
    the sizes on the page. The caption names the board.
    """
    instance = spec["instance"]
    level = spec["level"]

    # positions in inches on the page
    W, H = 4.70, 5.00
    side, left, bottom = 3.95, 0.50, 0.42  # the board
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
    draw_grid(ax, instance["grid_size"], instance["walls"], labels=9.0)
    draw_objects(ax, instance, letter=10.0, corner=8.0)

    fig.text(0.22 / W, 4.75 / H, level, fontsize=15.0, color=OBJECT,
             fontweight="bold", va="baseline")
    board_legend(fig, 0.06 / H, x=0.22 / W, walls=bool(instance["walls"]),
                 size=9.0)
    return save(fig, out_dir, f"spatial_level_{level}", formats)


def render_board_model(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """The board drawn for the reader beside the grid Player Green of the dyad is sent.

    The grid comes from the game's own Board class. Drawn at the printed
    width, so the font sizes are the sizes on the page.
    """
    from matplotlib.patches import Patch
    from matplotlib.transforms import blended_transform_factory

    sys.path.insert(0, REPO_ROOT)
    from matrixgame.utils.board import Board

    instance = spec["instance"]
    size = instance["grid_size"]
    own = set(instance["player_objects"][0])
    by_owner = {obj: SEAT["Green" if obj in own else "Orange"] for obj in instance["objects"]}
    board = Board(size, {tuple(w) for w in instance["walls"]})
    for obj, (r, c) in instance["start_positions"].items():
        board.place_object(obj, r, c)
    lines = board.render(own).split("\n")
    assert all(lines[0].index(f"C{c + 1}") == 7 + 5 * c for c in range(size))
    assert all(lines[2 + 2 * r].startswith(f"R{r + 1}   |") for r in range(size))

    # positions in inches on the page, 6.30 in = the text width
    W, H = 6.30, 3.95
    side, left, top = 2.60, 0.30, 0.48  # the drawn board
    text_x = 3.14  # the grid Player Green is sent, 8 pt mono, 46 chars wide
    bottom = H - top - side

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, side / W, side / H])
    draw_grid(ax, size, instance["walls"], labels=9.0)
    draw_objects(ax, instance, colours=by_owner, radius=0.29, letter=9.0,
                 corner=8.0)

    # the text rows share the drawn board's row coordinates, so R3 lines up with R3
    tx = fig.add_axes([text_x / W, bottom / H, (W - text_x) / W, side / H])
    tx.set_xlim(0, 1)
    tx.set_ylim(size, 0)
    tx.axis("off")
    blend = blended_transform_factory(tx.transAxes, tx.transData)
    tx.text(0, -0.30, lines[0], transform=blend, fontsize=8.0, color=MUTED,
            va="center", **MONO)
    for k, line in enumerate(lines[1:]):
        tx.text(0, k / 2, line, transform=blend, fontsize=8.0, va="center",
                color=FAINT if k % 2 == 0 else INK, **MONO)

    for x, title in [(left, "The board"), (text_x, "What Player Green is sent")]:
        fig.text(x / W, 1 - 0.20 / H, title, fontsize=10.0, color=INK,
                 fontweight="bold", va="baseline")

    target = Patch(facecolor="none", edgecolor=MUTED, linewidth=1.1,
                   linestyle=(0, (2.2, 1.5)), label="target cell")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=8.0,
               markerfacecolor=SEAT["Green"], markeredgecolor=PAPER,
               label="Player Green's object"),
        Line2D([], [], marker="o", linestyle="none", markersize=8.0,
               markerfacecolor=SEAT["Orange"], markeredgecolor=PAPER,
               label="Player Orange's object"),
        target,
        Line2D([], [], marker="s", linestyle="none", markersize=8.0,
               markerfacecolor=WALL, markeredgecolor=WALL_HI, label="wall"),
    ]
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=((left - 0.12) / W, (bottom - 0.12) / H),
               ncol=2, frameon=False, fontsize=9.0, handlelength=1.1,
               handleheight=1.1, handletextpad=0.6, columnspacing=1.2,
               labelspacing=0.55, borderaxespad=0)

    symbols = [(".", "empty cell"),
               (min(own), f"Player Green's own object ({', '.join(sorted(own))})"),
               ("X", "a teammate's object"), ("##", "wall")]
    assert all(f" {m:^2} " in "\n".join(lines[2::2]) for m, _ in symbols)
    for i, (mark, label) in enumerate(symbols):
        y = (bottom - 0.24 - i * 0.18) / H
        fig.text(text_x / W, y, mark, fontsize=9.0, color=INK, va="baseline",
                 **MONO)
        fig.text((text_x + 0.32) / W, y, label, fontsize=9.0, color=INK,
                 va="baseline")
    return save(fig, out_dir, "board_model", formats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    boards = load_instances()
    specs = {}
    for level in LEVELS:
        candidates = boards[level]
        instance = (next(c for c in candidates if c["game_id"] == PICKS[level]) if level in PICKS
                    else pick_representative(candidates))
        check_chains(instance)
        specs[level] = {"instance": instance, "level": level}
        for path in render_board(specs[level], args.out):
            print(f"wrote {os.path.relpath(path)}")
    for path in render_board_model(specs[MODEL_LEVEL], args.out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
