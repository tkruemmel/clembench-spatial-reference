"""B's promise and its `done` on S3 #10, Qwen3.8 hybrid, six players
(fig:s3-10-views, written as s3_worked_views).

Left, the board at the instant B declares `done`: B stands on its own target,
the corridor entrance, and D's only way to its target runs through B's cell.
Right, the channel around it: B's promise to step aside, D's request, the
`done` on B's next turn, the teammates who go on asking, and the end.

Asserted before drawing, all replayed from the move log: B steps onto the
entrance with the promise and declares `done` on its next turn without having
moved; with B frozen D has no route to its target, with B movable it has one;
cut.py records the same cut (victim D, sealer B, on the entrance); the episode
is a cycle loss. Every quote is cut from the transcript, never reworded.

    python3 figures/s3_promise.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from s3_worked import STEP, corridor, load_instance, shortest_route
from style import (CUE, FAINT, INK, MUTED, OBJECT, OWNER, PAPER, centre,
                   draw_disc, draw_grid, draw_target, save)
from warning import marked

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis/qualitative_analysis"))
import cut  # noqa: E402

RUN = os.path.join(REPO_ROOT, "results_qwen38_grid_updated_prompt_n15_6p", "*", "*",
                   "S3_hybrid_6p_grid", "instance_00010")
KEY = ("qwen38", "6p", "S3", "hybrid", "instance_00010")
MOVED = re.compile(r"^Player(?: \w+)? moved ([A-F]) (up|down|left|right)$")

# (speaker, fragments shown, joined by an ellipsis); each is checked verbatim
QUOTES = [
    ("B", ["D, take (R6,C5) next; I’ll then step right to (R5,C6) so you can "
           "pass, then return to my goal."]),
    ("D", ["B, please step right to (R5,C6) after I enter so I can pass, then "
           "return to (R5,C5)."]),
    ("D", ["D is boxed by walls and the X at (R5,C5), so I must step down to "
           "(R7,C5) to wait."]),
    ("A", ["B, your done at (R5,C5) still blocks D"]),
    ("E", ["B, if you are done at (R5,C5), please reconsider"]),
]


# ------------------------------------------------------------ the transcript
# Shared with s3_enclosure.py. A row is (moves played, kind, speaker, text,
# turn index): `moves played` counts the accepted moves when the line appears
# in the transcript, so a message carries the number of the move it rides on.

def transcript(run_dir):
    it = json.load(open(os.path.join(run_dir, "interactions.json")))
    n, rows = 0, []
    for ti, rnd in enumerate(it["turns"]):
        for e in rnd:
            a = e["action"]
            if e["from"] != "GM":
                continue
            kind, content = a["type"], str(a.get("content", ""))
            if kind == "move" and MOVED.match(content):
                n += 1
            elif kind == "message":
                spk, text = content.split(": ", 1)
                if text != "done":
                    rows.append((n, "msg", spk[-1], text, ti))
            elif kind == "player_done":
                rows.append((n, "done", content.split()[1], "done", ti))
            elif kind == "invalid_move":
                rows.append((n, "invalid", content[7], content.split("] ", 1)[-1], ti))
            elif kind in ("success", "lose", "abort"):
                rows.append((n, kind, "GM", content, ti))
    return rows


def moves_of(run_dir):
    it = json.load(open(os.path.join(run_dir, "interactions.json")))
    return [MOVED.match(e["action"]["content"]).groups()
            for rnd in it["turns"] for e in rnd
            if e["from"] == "GM" and e["action"]["type"] == "move"
            and MOVED.match(str(e["action"]["content"]))]


def board_at(inst, moves, upto):
    pos = {k: tuple(v) for k, v in inst["start_positions"].items()}
    for obj, way in moves[:upto]:
        pos[obj] = (pos[obj][0] + STEP[way][0], pos[obj][1] + STEP[way][1])
    return pos


def quote(rows, speaker, frags, after=-1):
    """The first message by `speaker` after move `after` that holds every
    fragment in order: (moves played, turn index, the fragments as drawn)."""
    for n, kind, who, text, ti in rows:
        if kind == "msg" and who == speaker and n > after:
            try:
                return n, ti, marked(text, frags)
            except ValueError:
                continue
    raise SystemExit(f"no message from {speaker} containing {frags}")


def draw_timeline(fig, x0, y0, w, h, rows, *, wrap=56, text_size=6.8):
    """Rows of (moves played, speaker, text, kind) stacked top to bottom in the
    box (x0, y0, w, h), in inches from the bottom-left. kind: quote, done, gm."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x0 / W, y0 / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    # a monospace glyph is wider than the serif average, so GM lines wrap sooner
    lines = [[t] if k == "done" else textwrap.wrap(t, wrap * 3 // 4 if k == "gm" else wrap)
             for _, _, t, k in rows]
    heights = [0.14 + 0.128 * len(ls) for ls in lines]
    gap = (h - 0.2 - sum(heights)) / (len(rows) - 1)
    assert gap > 0.03, f"timeline overflows its box by {-gap:.2f} in"
    ax.text(0.16, 0.07, "moves\nplayed", ha="center", va="center", fontsize=5.6,
            color=MUTED, linespacing=1.0)
    y = 0.2
    for (n, who, _, kind), ls, hh in zip(rows, lines, heights):
        mid = y + hh / 2
        colour = OWNER["B"] if who == "B" else OBJECT
        ax.text(0.16, mid, str(n), ha="center", va="center", fontsize=7.0, color=INK)
        box = dict(boxstyle="round,pad=0,rounding_size=0.05")
        if kind == "gm":
            ax.add_patch(FancyBboxPatch((0.66, y), w - 0.68, hh, **box,
                                        facecolor="#F3F4F6", edgecolor=FAINT, lw=0.7))
            ax.text(0.46, mid, "GM", ha="center", va="center", fontsize=6.0,
                    color=MUTED, fontweight="bold")
            ax.text(0.74, mid, "\n".join(ls), ha="left", va="center",
                    fontsize=text_size - 0.6, color=INK, family="monospace",
                    linespacing=1.3)
        else:
            ax.add_patch(Circle((0.46, mid), 0.115, facecolor=colour,
                                edgecolor=PAPER, lw=0.8))
            ax.text(0.46, mid, who, ha="center", va="center", fontsize=6.8,
                    color=PAPER, fontweight="bold")
            filled = kind == "done"
            width = 0.5 if filled else w - 0.68
            ax.add_patch(FancyBboxPatch((0.66, y), width, hh, **box,
                                        facecolor=colour if filled else PAPER,
                                        edgecolor=colour, lw=0.9))
            ax.text(0.66 + (width / 2 if filled else 0.08), mid, "\n".join(ls),
                    ha="center" if filled else "left", va="center",
                    fontsize=text_size, color=PAPER if filled else INK,
                    fontweight="bold" if filled else "normal",
                    family="monospace" if filled else None, linespacing=1.3)
        y += hh + gap


# ------------------------------------------------------------------ the figure

def facts(inst, run_dir):
    """Everything the figure claims, checked against the rules and the log."""
    ep = json.load(open(os.path.join(run_dir, "instance.json")))
    assert all(ep[k] == inst[k] for k in ("grid_size", "walls", "start_positions",
                                          "target_positions", "player_objects"))
    _, entrance = corridor(inst)
    rows, moves = transcript(run_dir), moves_of(run_dir)
    done = next(n for n, k, who, _, _ in rows if k == "done" and who == "B")
    promise, _, _ = quote(rows, *QUOTES[0])
    assert moves[promise - 1] == ("B", "up"), moves[promise - 1]
    assert board_at(inst, moves, promise)["B"] == entrance  # the promise rides on it
    assert all(o != "B" for o, _ in moves[promise:done])    # done on B's next turn
    pos = board_at(inst, moves, done)
    assert pos["B"] == entrance == tuple(inst["target_positions"]["B"])
    walls, n = {tuple(w) for w in inst["walls"]}, inst["grid_size"]
    seat = {chr(65 + i): objs for i, objs in enumerate(inst["player_objects"])}
    done_by = {who for m, k, who, _, _ in rows if k == "done" and m <= done}
    assert done_by == {"B"} and seat["B"] == ["B"], (done_by, seat["B"])
    frozen = {pos[o] for p in done_by for o in seat[p]}
    d_tgt = tuple(inst["target_positions"]["D"])
    assert cut.path(pos["D"], d_tgt, n, walls, set()) is not None
    assert cut.path(pos["D"], d_tgt, n, walls, frozen) is None  # the done cuts D
    end = rows[-1]
    assert end[1] == "lose" and "cycle" in end[3] and end[0] == len(moves)
    recorded = [r for r in cut.cases()
                if (r["meta"]["model"], r["meta"]["suite"], r["meta"]["level"],
                    r["meta"]["proto"], r["meta"]["inst"]) == KEY]
    assert [(r["victim"], r["sealer"], r["ply"], r["kind"]) for r in recorded] == \
        [("D", "B", done, cut.ENTRANCE)], recorded
    return rows, pos, entrance, promise, done, end


def render(out_dir: str = OUT_DIR):
    inst = load_instance()
    run_dir, = glob.glob(RUN)
    rows, pos, entrance, promise, done, end = facts(inst, run_dir)

    timeline, after = [], -1
    for who, frags in QUOTES:
        n, _, text = quote(rows, who, frags, after)
        timeline.append((n, who, text, "quote"))
        after = n
    timeline = sorted(timeline + [(done, "B", "done", "done")], key=lambda t: t[0])
    timeline.append((end[0], "GM", end[3], "gm"))

    W, side, bottom, top = 6.3, 2.5, 0.5, 0.42
    H = bottom + side + top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.3 / W, bottom / H, side / W, side / H])
    draw_grid(ax, inst["grid_size"], inst["walls"], labels=5.6)
    ax.add_patch(Rectangle((entrance[1] + 0.02, entrance[0] + 0.02), 0.96, 0.96,
                           facecolor="none", edgecolor=CUE, linewidth=1.5,
                           zorder=6.5))
    # around every object but B, so the only thing in D's way is B
    route = shortest_route(inst, pos["D"], tuple(inst["target_positions"]["D"]),
                           blocked=[c for k, c in pos.items() if k not in "BD"])
    assert entrance in route
    xs, ys = map(list, zip(*(centre(c) for c in route)))
    ys[-1] += 0.5
    ax.plot(xs, ys, color=OBJECT, lw=1.1, ls=(0, (2.4, 1.6)), zorder=4)
    colour = {k: "#8FAED0" for k in pos}
    colour.update(B=OWNER["B"], D=OBJECT)
    for k in "BD":
        draw_target(ax, tuple(inst["target_positions"][k]), k, colour[k],
                    nested=False, letter=9.0)
    for k, c in sorted(pos.items()):
        draw_disc(ax, c, k, colour[k], letter=7.6)
    fig.text(0.3 / W, (bottom + side + 0.24) / H,
             f"after move {done}: B declares done", fontsize=7.4, color=INK)

    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.6, mfc=OWNER["B"], mec=PAPER,
               label="B, declaring done"),
        Line2D([], [], color=OBJECT, lw=1.1, ls=(0, (2.4, 1.6)),
               label="D's path to its target"),
        Line2D([], [], marker="s", ls="none", ms=5.8, mfc="none", mec=CUE,
               mew=1.3, label="entrance"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.2 / W, 0.0),
               ncol=2, frameon=False, fontsize=6.2, handletextpad=0.4,
               columnspacing=0.9, labelcolor=INK)
    draw_timeline(fig, side + 0.5, 0.08, W - side - 0.55, H - 0.16, timeline,
                  wrap=54)
    print(f"B on the entrance after move {promise}, done after move {done}, "
          f"{end[1]} after move {end[0]}")
    return save(fig, out_dir, "s3_worked_views", ("pdf", "png"))


if __name__ == "__main__":
    for path in render():
        print(path)
