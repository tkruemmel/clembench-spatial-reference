"""The start position of S3 #10, for the worked example of S9.4.

The level figure of S5.2 draws the same board under the level-figure rule of
marks only. This one is allowed three annotations, because the worked example
is about them: the dead-end corridor, its entrance, and the order in which the
chain D -> C -> B has to arrive. The second chain (E, F) and A, whose arrival
is not forced, are drawn but left unnumbered.

`s3_worked_views` is the decision point of the same board in DeepSeek's hybrid
episode: the state in which B, already on the entrance, declares `done`. It
shows that state with D's path to its target running through B's cell.

Board read from the released six-player instances, so it regenerates with them.

    python3 figures/s3_worked.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from style import (CUE, FAINT, HIDDEN, INK, MUTED, OBJECT, OWNER, PAPER, WALL,
                   WALL_HI, centre, draw_disc, draw_grid, draw_objects,
                   draw_target, save)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
INSTANCES = os.path.join(REPO_ROOT, "matrixgame_covered_comm_nplayer_grid",
                         "in", "instances_6p.json")
LEVEL, GAME_ID = "S3", 10
SIDE = FAINT  # objects outside the corridor chain
RUN = os.path.join(REPO_ROOT, "results_dsv41_grid_n15_6p", "*", "*",
                   "S3_hybrid_6p_grid", "instance_00010", "interactions.json")
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def load_instance() -> dict:
    for exp in json.load(open(INSTANCES))["experiments"]:
        if exp["name"].startswith(LEVEL + "_"):
            for inst in exp["game_instances"]:
                if inst["game_id"] == GAME_ID:
                    return inst
    raise SystemExit(f"{LEVEL} #{GAME_ID} not in {INSTANCES}")


def corridor(inst: dict):
    """The dead-end cells and the entrance, derived rather than typed in."""
    walls = {tuple(w) for w in inst["walls"]}
    col = inst["target_positions"]["D"][1]
    cells, r = [], inst["target_positions"]["D"][0]
    while (r, col - 1) in walls and (r, col + 1) in walls:  # walk down from the dead end
        cells.append((r, col))
        r += 1
    entrance = (cells[-1][0] + 1, col)
    assert cells[0] == tuple(inst["target_positions"]["D"])
    assert entrance == tuple(inst["start_positions"]["C"])
    assert entrance == tuple(inst["target_positions"]["B"])
    return cells, entrance


def order(inst: dict):
    """D, C, B: each target is the previous object's start cell."""
    start = {k: tuple(v) for k, v in inst["start_positions"].items()}
    target = {k: tuple(v) for k, v in inst["target_positions"].items()}
    assert target["C"] == start["D"] and target["B"] == start["C"]
    return [("D", target["D"]), ("C", target["C"]), ("B", target["B"])]


def badge(ax, cell, n):
    x, y = cell[1] + 0.86, cell[0] + 0.14
    ax.add_patch(Circle((x, y), 0.155, facecolor=CUE, edgecolor=PAPER,
                        linewidth=0.8, zorder=9))
    ax.text(x, y + 0.01, str(n), ha="center", va="center", fontsize=6.4,
            color=PAPER, fontweight="bold", zorder=10)


def render(out_dir: str = OUT_DIR):
    inst = load_instance()
    cells, entrance = corridor(inst)
    start = {k: tuple(v) for k, v in inst["start_positions"].items()}
    target = {k: tuple(v) for k, v in inst["target_positions"].items()}
    colour = {k: (OBJECT if k in "ABCD" else SIDE) for k in start}

    fig = plt.figure(figsize=(3.7, 4.05))
    ax = fig.add_axes([0.075, 0.085, 0.905, 0.86])
    draw_grid(ax, inst["grid_size"], inst["walls"], labels=6.6)

    # corridor tint and entrance frame sit under every other mark
    for c in cells:
        ax.add_patch(Rectangle((c[1], c[0]), 1, 1, facecolor=to_rgba(CUE, 0.13),
                               edgecolor="none", zorder=1.5))
    ax.add_patch(Rectangle((entrance[1] + 0.02, entrance[0] + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor=CUE, linewidth=1.6,
                           zorder=6.5))

    occupied = {cell: obj for obj, cell in start.items()}
    for obj, cell in sorted(target.items()):
        blocking = occupied.get(cell)
        draw_target(ax, cell, obj, colour[obj],
                    nested=bool(blocking) and blocking != obj, letter=11.0)
    for obj, cell in sorted(start.items()):
        draw_disc(ax, cell, obj, colour[obj], letter=8.6)
    for n, (_, cell) in enumerate(order(inst), 1):
        badge(ax, cell, n)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6, mfc=OBJECT, mec=PAPER,
               label="start"),
        Line2D([], [], marker="s", ls="none", ms=6.2, mfc="none", mec=OBJECT,
               label="target"),
        Line2D([], [], marker="s", ls="none", ms=6.2, mfc=WALL, mec=WALL_HI,
               label="wall"),
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc=CUE, mec=PAPER,
               label="forced arrival order"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
               ncol=4, frameon=False, fontsize=6.6, handletextpad=0.35,
               columnspacing=1.1, labelcolor=INK)
    return save(fig, out_dir, "s3_worked_start", ("pdf", "png"))


def state_at_done(inst: dict, who: str = "B"):
    """Replay the recorded moves up to `who`'s done declaration."""
    d = json.load(open(glob.glob(RUN)[0]))
    pos = {k: tuple(v) for k, v in inst["start_positions"].items()}
    n = 0
    for rnd in d["turns"]:
        for e in rnd:
            a = e["action"]
            if e["from"] != "GM":
                continue
            if a["type"] == "move":
                obj, way = re.search(r"moved (\w) (\w+)", a["content"]).groups()
                pos[obj] = (pos[obj][0] + STEP[way][0], pos[obj][1] + STEP[way][1])
                n += 1
            elif a["type"] == "player_done" and f"Player {who} " in a["content"]:
                assert pos[who] == tuple(inst["target_positions"][who])
                return pos, n
    raise SystemExit(f"{who} never declares done in {RUN}")


def shortest_route(inst: dict, src, dst, blocked=()):
    """Shortest path around walls and the `blocked` cells."""
    walls = {tuple(w) for w in inst["walls"]} | set(blocked)
    size = inst["grid_size"]
    prev, todo = {src: None}, deque([src])
    while todo:
        cur = todo.popleft()
        if cur == dst:
            break
        for dr, dc in STEP.values():
            nxt = (cur[0] + dr, cur[1] + dc)
            if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                    and nxt not in walls and nxt not in prev):
                prev[nxt] = cur
                todo.append(nxt)
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return path[::-1]


def render_views(out_dir: str = OUT_DIR):
    inst = load_instance()
    _, entrance = corridor(inst)
    pos, n = state_at_done(inst)
    assert pos["B"] == entrance
    now = dict(inst, start_positions={k: list(v) for k, v in pos.items()})
    others = "#8FAED0"
    truth = {k: others for k in pos}
    truth.update(B=OWNER["B"], D=OBJECT)
    # around every object except B, so the only thing in D's way is B
    route = shortest_route(inst, pos["D"], tuple(inst["target_positions"]["D"]),
                           blocked=[c for k, c in pos.items() if k not in "BD"])
    assert entrance in route

    fig = plt.figure(figsize=(3.7, 4.0))
    ax = fig.add_axes([0.075, 0.1, 0.905, 0.84])
    draw_grid(ax, inst["grid_size"], inst["walls"], labels=6.6)
    ax.add_patch(Rectangle((entrance[1] + 0.02, entrance[0] + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor=CUE, linewidth=1.6,
                           zorder=6.5))
    xs, ys = map(list, zip(*(centre(c) for c in route)))
    ys[-1] += 0.5  # stop at the target's edge, clear of its letter
    ax.plot(xs, ys, color=OBJECT, lw=1.2, ls=(0, (2.4, 1.6)), zorder=4)
    for k in "BD":  # every object, but only B's and D's targets
        draw_target(ax, tuple(inst["target_positions"][k]), k, truth[k],
                    nested=False, letter=11.0)
    for k, c in sorted(pos.items()):
        draw_disc(ax, c, k, truth[k], letter=8.6)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6, mfc=OWNER["B"], mec=PAPER,
               label="B, declaring done"),
        Line2D([], [], color=OBJECT, lw=1.2, ls=(0, (2.4, 1.6)),
               label="D's path to its target"),
        Line2D([], [], marker="s", ls="none", ms=6.2, mfc="none", mec=CUE,
               mew=1.4, label="entrance"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
               ncol=3, frameon=False, fontsize=6.6, handletextpad=0.4,
               columnspacing=1.2, labelcolor=INK)
    return save(fig, out_dir, "s3_worked_views", ("pdf", "png"))


if __name__ == "__main__":
    for path in render() + render_views():
        print(path)
