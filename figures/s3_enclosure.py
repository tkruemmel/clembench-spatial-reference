"""How B is enclosed on S3 #10, DeepSeek freeform, six players (S9.4).

One panel per move from the move that pushes B into (R7,C5) to the move that
takes its last exit. In each panel the object that just moved carries an arrow
from its previous cell, and B's free neighbours are marked, so the count of
exits can be read off the board. Positions, the mover and the legal moves of
the final mover are replayed from the recorded log.

The panels are cropped to columns C3-C8; the script asserts that no object
stands outside that window, so the crop hides nothing.

    python3 figures/s3_enclosure.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch

from s3_promise import read_log
from style import (CUE, INK, MOVE, MUTED, OBJECT, OWNER, PAPER, WALL, WALL_HI,
                   centre, draw_arrow, draw_disc, draw_grid, draw_target, save)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
RUN = os.path.join(REPO_ROOT, "results_dsv41_grid_n15_6p", "*", "*",
                   "S3_freeform_6p_grid", "instance_00010", "interactions.json")
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
COLS = range(2, 8)  # C3..C8, zero-based

# (speaker, fragments shown, joined by an ellipsis); each is checked verbatim
MESSAGES = [
    ("B", ["D has priority up C5. I'll wait around C3/C4 until D passes"]),
    ("D", ["R4C5 is now free."]),
    ("A", ["B,", "please wait at R8C5 until I leave R7C5."]),
    ("A", ["Moving A down to R6C5. B, I need R7C5 next",
           "please vacate left once E clears R7C4."]),
    ("B", ["B is trapped at R7C5 by A at R6C5, E at R7C4, C at R8C5, and wall "
           "at R7C6."]),
]


def replay():
    path = glob.glob(RUN)[0]
    d = json.load(open(path))
    inst = json.load(open(path.replace("interactions", "instance")))
    pos = {k: tuple(v) for k, v in inst["start_positions"].items()}
    states = [(0, None, None, dict(pos))]
    for rnd in d["turns"]:
        for e in rnd:
            a = e["action"]
            if e["from"] == "GM" and a["type"] == "move":
                obj, way = re.search(r"moved (\w) (\w+)", a["content"]).groups()
                pos[obj] = (pos[obj][0] + STEP[way][0], pos[obj][1] + STEP[way][1])
                states.append((len(states), obj, way, dict(pos)))
    return inst, states


def free_neighbours(inst, pos, obj):
    walls = {tuple(w) for w in inst["walls"]}
    taken = set(pos.values())
    r, c = pos[obj]
    out = []
    for way, (dr, dc) in STEP.items():
        q = (r + dr, c + dc)
        if (0 <= q[0] < inst["grid_size"] and 0 <= q[1] < inst["grid_size"]
                and q not in walls and q not in taken):
            out.append((way, q))
    return out


def pick_messages():
    """Locate each quoted message; return (moves played, speaker, text)."""
    msgs, _ = read_log(RUN)
    out, after = [], -1
    for who, parts in MESSAGES:
        for n, speaker, body in msgs:
            if speaker == who and n > after and all(p in body for p in parts):
                idx = [body.index(p) for p in parts]
                assert idx == sorted(idx), f"fragments out of order: {parts}"
                out.append((n, who, " \u2026 ".join(parts)))
                after = n
                break
        else:
            raise SystemExit(f"no message from {who} containing {parts}")
    return out


def draw_messages(fig, W, H, *, x0, top, bottom):
    ax = fig.add_axes([x0 / W, bottom / H, (W - x0 - 0.05) / W,
                       (H - top - bottom) / H])
    w, h_total = W - x0 - 0.05, H - top - bottom
    ax.set_xlim(0, w)
    ax.set_ylim(h_total, 0)
    ax.axis("off")
    rows = pick_messages()
    wrap = 58
    lines = [textwrap.wrap(f"\u201c{t}\u201d", wrap) for _, _, t in rows]
    heights = [0.14 + 0.13 * len(ls) for ls in lines]
    gap = (h_total - 0.2 - sum(heights)) / (len(rows) - 1)
    ax.text(0.14, 0.08, "moves", ha="center", va="center", fontsize=5.8,
            color=MUTED)
    y = 0.2
    for (n, who, _), ls, h in zip(rows, lines, heights):
        mid = y + h / 2
        colour = OWNER["B"] if who == "B" else OBJECT
        ax.text(0.14, mid, str(n), ha="center", va="center", fontsize=7.0,
                color=INK)
        ax.add_patch(Circle((0.5, mid), 0.12, facecolor=colour,
                            edgecolor=PAPER, lw=0.8))
        ax.text(0.5, mid, who, ha="center", va="center", fontsize=7.0,
                color=PAPER, fontweight="bold")
        ax.add_patch(FancyBboxPatch((0.72, y), w - 0.74, h,
                                    boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=PAPER, edgecolor=colour, lw=0.9))
        ax.text(0.8, mid, "\n".join(ls), ha="left", va="center", fontsize=6.8,
                color=INK, linespacing=1.3)
        y += h + gap


def render(out_dir: str = OUT_DIR):
    inst, states = replay()
    # the enclosing window: from B's push into the pocket to its last exit going
    last = next(i for i, s in enumerate(states)
                if i and not free_neighbours(inst, s[3], "B"))
    first = max(i for i in range(last) if states[i][1] == "B")
    assert states[last][1] == "A"
    a_had = free_neighbours(inst, states[last - 1][3], "A")
    assert len(a_had) == 3, a_had
    final = states[last][3]
    assert all(c in COLS for _, c in final.values()), "crop would hide an object"

    # which move took each of B's exits: an exit closes when an object enters it
    sealed = {}
    for k in range(first + 1, last + 1):
        before = {q for _, q in free_neighbours(inst, states[k - 1][3], "B")}
        after = {q for _, q in free_neighbours(inst, states[k][3], "B")}
        for q in before - after:
            sealed[states[k][1]] = k
    assert set(sealed) == {"A", "C", "E"}, sealed

    walls = [w for w in inst["walls"] if w[1] in COLS]
    shift = lambda cell: (cell[0], cell[1] - COLS[0])
    colour = {k: OBJECT for k in final}
    colour["B"] = OWNER["B"]

    W, aw = 6.3, 2.5
    ah, bottom = aw * 8 / 6, 0.62
    H = bottom + ah + 0.28
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.3 / W, bottom / H, aw / W, ah / H])
    draw_grid(ax, 8, [shift(w) for w in walls], labels=0)
    ax.set_xlim(0, 6)
    for c in COLS:
        ax.text(c - COLS[0] + 0.5, -0.3, f"C{c + 1}", ha="center", va="center",
                fontsize=6.4, color=MUTED)
    for r in range(8):
        ax.text(-0.35, r + 0.5, f"R{r + 1}", ha="center", va="center",
                fontsize=6.4, color=MUTED)

    draw_target(ax, shift(tuple(inst["target_positions"]["D"])), "D", OBJECT,
                nested=False, letter=10.0)
    for way, q in a_had:
        if q != final["A"]:
            ax.add_patch(Circle(centre(shift(q)), 0.3, facecolor="none",
                                edgecolor=OBJECT, lw=1.0, ls=(0, (1.8, 1.4)),
                                zorder=7))
    was = states[last - 1][3]["A"]
    ax.add_patch(Circle(centre(shift(was)), 0.32, facecolor=to_rgba(OBJECT, 0.22),
                        edgecolor="none", zorder=7))
    ax.text(*centre(shift(was)), "A", ha="center", va="center", fontsize=8.4,
            color=OBJECT, fontweight="bold", zorder=8)
    draw_arrow(ax, shift(was), shift(final["A"]), OBJECT, lw=1.1, scale=8,
               shrink=(10.0, 10.0), zorder=8)
    for o, c in sorted(final.items()):
        draw_disc(ax, shift(c), o, colour[o], radius=0.32, letter=8.4)
    for o, k in sealed.items():
        x, y = centre(shift(final[o]))
        ax.add_patch(Circle((x + 0.3, y - 0.3), 0.19, facecolor=CUE,
                            edgecolor=PAPER, lw=0.8, zorder=9))
        ax.text(x + 0.3, y - 0.29, str(k), ha="center", va="center",
                fontsize=5.8, color=PAPER, fontweight="bold", zorder=10)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc=CUE, mec=PAPER,
               label="move on which it took one of B's exits"),
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc=to_rgba(OBJECT, 0.22),
               mec="none", label="A before move 54"),
        Line2D([], [], marker="o", ls="none", ms=6.4, mfc="none", mec=OBJECT,
               mew=1.0, label="A's two other legal moves"),
    ]
    draw_messages(fig, W, H, x0=aw + 0.55, top=0.28, bottom=bottom)
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.2 / W, 0.0),
               ncol=1, frameon=False, fontsize=6.6, handletextpad=0.5,
               labelcolor=INK)
    print(f"final state after move {last}; exits sealed by {sealed}")
    return save(fig, out_dir, "s3_enclosure", ("pdf", "png"))


if __name__ == "__main__":
    for path in render():
        print(path)
