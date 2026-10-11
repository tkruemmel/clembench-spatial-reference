"""Draws the one heeded enclosure warning of S9.3.3.

DeepSeek, S4 board 10, six players, freeform. A warns C off the one cell that
would leave A no legal move, B relays the warning, and C, which had announced
the route through that cell itself, goes the other way. The episode is won.

The board is replayed from the recorded move log, the pinch is checked with the
rules (C up leaves A no legal move, C down leaves every player one), and every
quoted fragment is checked to occur verbatim, in order, in its message.

    python3 figures/warning.py
"""

from __future__ import annotations

import argparse
import json
import os
import string

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from aborts import OUT_DIR, REPO_ROOT, at, cell_name, square_axes
from style import (CUE, FAINT, INK, MOVE, MUTED, OBJECT, OWNER, PAPER,
                   centre, draw_arrow, draw_disc, draw_grid, draw_target, save,
                   wrap_words)
import seal_alt as sa

RUN = os.path.join(
    REPO_ROOT, "results_dsv41_grid_n15_6p", "DeepSeek-V4.1-Flash",
    "matrixgame_covered_comm_6p_grid", "S4_freeform_6p_grid", "instance_00010")
VICTIM, ACTOR, RELAY = "A", "C", "B"
COLOUR = {VICTIM: OWNER["B"], ACTOR: OBJECT, RELAY: "#6B4C9A"}

# (ply the message rides on, fragments cut from it, never reworded)
QUOTES = [
    (3, ["I’m taking the top-right route to my goal (R6,C7): R2C5, R1C5–C8"]),
    (7, ["C, please move down to (R4,C5) this turn, NOT up into (R2,C5); "
         "if you take C5 I’ll be trapped and B can’t pass."]),
    (8, ["C, please move down to (R4,C5), not up into (R2,C5), so A isn’t "
         "trapped and I can follow."]),
    (9, ["Moving C down to (R4,C5) as requested", "A, R2C5 stays open."]),
]


def load():
    ins = json.load(open(os.path.join(RUN, "instance.json")))
    it = json.load(open(os.path.join(RUN, "interactions.json")))
    moves, said, outcome = [], {}, None
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            if a.get("type") == "move":
                m = sa.MOVE.match(a.get("content", ""))
                if m:
                    moves.append((m.group(2), m.group(3)))
            elif a.get("type") == "message":
                # the broadcast that follows a move belongs to that move's ply
                said.setdefault(len(moves), a["content"].split(": ", 1)[1])
            elif a.get("type") == "success":
                outcome = "won"
    return ins, moves, said, outcome


def board_at(ins, moves, upto):
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    for o, d in moves[:upto]:
        pos = sa.step(pos, o, d)
    return pos


TXT, LEAD, CHARS = 6.9, 0.135, 34     # message text: size, line pitch (in), wrap


def bubble(fig, x_in, top_in, width_in, who, head, text, *, faded=False,
           chars=CHARS, colour=None):
    """One message: tinted box, speaker chip, the ply and move, the words."""
    colour = colour or COLOUR[who]
    lines = wrap_words(text, chars)
    h = 0.21 + LEAD * len(lines) + 0.05
    w, H = fig.get_size_inches()
    ax = fig.add_axes([x_in / w, (top_in - h) / H, width_in / w, h / H])
    ax.set_xlim(0, width_in); ax.set_ylim(h, 0); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), width_in, h,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc=to_rgba(colour, 0.04 if faded else 0.08),
                                ec=to_rgba(colour, 0.30 if faded else 0.45),
                                lw=0.6))
    ax.add_patch(FancyBboxPatch((0.06, 0.05), 0.14, 0.12,
                                boxstyle="round,pad=0,rounding_size=0.025",
                                fc=colour, ec="none"))
    ax.text(0.13, 0.11, who, fontsize=6.4, color=PAPER, fontweight="bold",
            ha="center", va="center")
    ax.text(0.27, 0.11, head, fontsize=6.2, color=colour, va="center",
            family="monospace", fontweight="bold")
    for j, line in enumerate(lines):
        ax.text(0.06, 0.21 + LEAD * (j + 0.45), line, fontsize=TXT,
                color=MUTED if faded else INK, va="center")
    return h


def marked(full, frags):
    """Fragments as drawn: in quotation marks, joined by an ellipsis, with one
    more at either end wherever the message goes on, so every cut is visible.
    Raises unless every fragment occurs verbatim, in order, in `full`."""
    at_ = 0
    for f in frags:
        at_ = full.index(f, at_) + len(f)
    first = full.index(frags[0])
    return ("\u201c" + ("\u2026 " if full[:first].strip() else "")
            + " \u2026 ".join(frags)
            + (" \u2026" if full[at_:].strip() else "") + "\u201d")


VERBATIM = ("Messages are quoted verbatim from the episode transcript; "
            "\u2026 marks omitted text.")


def height_of(text, chars=CHARS):
    return 0.21 + LEAD * len(wrap_words(text, chars)) + 0.05


def exits(pos, walls, n):
    return [sa.step(pos, VICTIM, d)[VICTIM]
            for _, d in sa.legal_moves(pos, [VICTIM], walls, n)]


def panel(ax, n, walls, pos, prev, moved, goal, *, declined=None):
    """The board after one move: the move just played as an arrow from a ghost
    of where the object stood, A's free neighbours as dashed cells, and C's
    target, since whether C's choice cost it anything is read off that."""
    draw_grid(ax, n, walls, labels=4.6)
    draw_target(ax, goal, ACTOR, COLOUR[ACTOR], nested=False, letter=6.6,
                lw=1.1)
    for cell in exits(pos, walls, n):
        draw_target(ax, cell, "", COLOUR[VICTIM], nested=False, lw=1.1)
    ghost = prev[moved]
    ax.add_patch(Circle(centre(ghost), 0.22, fc="none",
                        ec=COLOUR.get(moved, FAINT), lw=0.9, ls=(0, (1.2, 1.2)),
                        zorder=7))
    for obj, cell in sorted(pos.items()):
        draw_disc(ax, cell, obj, COLOUR.get(obj, FAINT), radius=0.31,
                  letter=6.6, ring=MUTED if obj == "F" else None, ring_lw=1.2)
    if declined:
        draw_arrow(ax, ghost, declined, CUE, lw=1.4, scale=8.0,
                   shrink=(4.0, 2.0), ls=(0, (1.8, 1.4)), zorder=8.5)
    draw_arrow(ax, ghost, pos[moved], MOVE, lw=1.9, scale=9.0,
               shrink=(4.0, 5.5), zorder=8.5)


def render(out_dir):
    ins, moves, said, outcome = load()
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    owner = {o: string.ascii_uppercase[i]
             for i, g in enumerate(ins["player_objects"]) for o in g}
    ply = 9                                   # C's decision
    pre = board_at(ins, moves, ply - 1)
    played = moves[ply - 1]
    alts = sa.legal_moves(pre, [ACTOR], walls, n)
    sealing = [(o, d) for o, d in alts
               if not sa.legal_moves(sa.step(pre, o, d), [VICTIM], walls, n)]

    # what the figure asserts, checked against the rules before it is drawn
    assert outcome == "won"
    assert played == (ACTOR, "down") and sealing == [(ACTOR, "up")], (played, sealing)
    boards = {p: board_at(ins, moves, p) for p in (6, 7, 8, 9)}
    counts = {p: len(exits(boards[p], walls, n)) for p in (7, 8, 9)}
    assert counts == {7: 2, 8: 1, 9: 1}, counts   # A writes with two exits
    last = exits(pre, walls, n)[0]
    assert last == sa.step(pre, *sealing[0])[ACTOR]
    # no move in the episode leaves anyone without a legal move; the one
    # immobile object is B at the start, the walled-in opening every S4 board has
    stuck = [(p, o) for p in range(len(moves) + 1)
             for o, pos in [(o, board_at(ins, moves, p)) for o in pre]
             if not sa.legal_moves(pos, [o], walls, n)]
    assert stuck == [(0, "B")], stuck
    for p, frags in QUOTES:
        at_ = 0
        for f in frags:
            at_ = said[p].index(f, at_)       # verbatim and in order, or raise
    assert owner[moves[2][0]] == ACTOR and moves[2][1] == "up"

    W, M, GAP = 6.3, 0.22, 0.20
    side = (W - 2 * M - 2 * GAP) / 3
    xs = [M + i * (side + GAP) for i in range(3)]
    texts = {p: marked(said[p], f) for p, f in QUOTES}
    tall = max(height_of(texts[p]) for p in (7, 8, 9))
    foot = height_of(texts[3], chars=108)
    H = 0.66 + 0.22 + 0.24 + side + 0.44 + tall + 0.16 + foot + 0.50
    fig = plt.figure(figsize=(W, H))

    y = H - 0.12
    fig.text(*at(fig, M, y), "The warning that was heeded", fontsize=10.0,
             color=INK, va="top", fontweight="bold")
    fig.text(*at(fig, M, y - 0.25),
             "DeepSeek, six players, S4 board 10, freeform. Won in "
             f"{len(moves)} moves; no move in it leaves a player without a legal move.",
             fontsize=7.2, color=MUTED, va="top")
    fig.text(*at(fig, M, y - 0.40), VERBATIM, fontsize=7.2, color=MUTED,
             va="top", style="italic")
    y -= 0.66

    bubble(fig, M, y, W - 2 * M, ACTOR, "earlier, move 3 · C up",
           texts[3], faded=True, chars=108)
    y -= foot + 0.22

    heads = {7: "A warns", 8: "B relays the warning", 9: "C complies"}
    notes = {p: f"A can move to {counts[p]} cell{'s' * (counts[p] > 1)}"
             for p in (7, 8, 9)}
    goal = tuple(ins["target_positions"][ACTOR])
    dist = sa.dist_map(walls, n, goal)        # walls only; objects move
    steps = {p: dist[boards[p][ACTOR]] for p in (6, 7, 8, 9)}
    up = dist[sa.step(pre, *sealing[0])[ACTOR]]
    # complying cost C a step: the declined move was its only progress
    assert (steps[8], up, steps[9]) == (11, 10, 12), (steps, up)
    assert all(dist[sa.step(pre, *m)[ACTOR]] >= steps[8]
               for m in alts if m != sealing[0])
    far = {7: f"C is {steps[7]} moves from its goal",
           8: f"C is {steps[8]} moves from its goal",
           9: f"C: {steps[8]} \u2192 {steps[9]} moves from its goal (up: {up})"}
    for x, p in zip(xs, (7, 8, 9)):
        obj, d = moves[p - 1]
        fig.text(*at(fig, x + side / 2, y), f"after move {p}: {heads[p]}",
                 fontsize=7.6, color=INK, ha="center", va="top",
                 fontweight="bold")
        ax = square_axes(fig, x, y - 0.24 - side, side)
        panel(ax, n, walls, boards[p], boards[p - 1], obj, goal,
              declined=last if p == ply else None)
        fig.text(*at(fig, x + side / 2, y - 0.26 - side), notes[p],
                 fontsize=6.8, color=COLOUR[VICTIM], ha="center", va="top")
        fig.text(*at(fig, x + side / 2, y - 0.40 - side), far[p],
                 fontsize=6.8, color=COLOUR[ACTOR], ha="center", va="top")
        bubble(fig, x, y - 0.40 - side - 0.20, side, owner[obj],
               f"move {p} · {obj} {d}", texts[p])
    y -= 0.40 + side + 0.20 + tall + 0.16

    # key: each mark drawn as itself, so colour is never the only cue
    k = fig.add_axes([0, 0, 1, 1], facecolor="none"); k.axis("off")
    k.set_xlim(0, W); k.set_ylim(0, H)
    rows = ((y - 0.075, (
                (M, "-", MOVE, "move just played"),
                (M + 1.30, "ring", MUTED, "where it stood"),
                (M + 2.45, (0, (1.8, 1.4)), CUE,
                 "C up: legal, not played, leaves A with no move"))),
            (y - 0.265, (
                (M, "box", COLOUR[VICTIM], "a cell A can move to"),
                (M + 1.30, "goal", COLOUR[ACTOR],
                 "C's goal; distances are shortest paths round the walls"))))
    for yk, items in rows:
        for kx, style, colour, label in items:
            if style in ("box", "goal"):
                k.add_patch(FancyBboxPatch((kx + 0.05, yk - 0.06), 0.12, 0.12,
                                           boxstyle="round,pad=0,rounding_size=0.02",
                                           fc=to_rgba(colour, 0.10), ec=colour,
                                           lw=0.9, ls=(0, (2.2, 1.5))))
                if style == "goal":
                    k.text(kx + 0.11, yk, ACTOR, fontsize=5.6, color=colour,
                           ha="center", va="center", fontweight="bold")
            elif style == "ring":
                k.add_patch(Circle((kx + 0.11, yk), 0.05, fc="none", ec=colour,
                                   lw=0.9, ls=(0, (1.2, 1.2))))
            else:
                k.add_patch(FancyArrowPatch((kx, yk), (kx + 0.22, yk),
                                            arrowstyle="-|>", mutation_scale=7,
                                            color=colour, lw=1.4, ls=style))
            k.text(kx + 0.28, yk, label, fontsize=6.6, color=INK, va="center")

    paths = save(fig, out_dir, "warning_heeded", ("pdf", "png"))
    tex = os.path.join(out_dir, "warning.tex")
    with open(tex, "w") as fh:
        fh.write(FLOAT % dict(ply=ply, last=cell_name(last), total=len(moves)))
    return paths + [tex]


FLOAT = r"""%% Figure float for the heeded warning, written by figures/warning.py.
\begin{figure}[tbp]
    \centering
    \thesisfig{warning_heeded.pdf}
    \caption{The only enclosure warning in the strong tier that names both the
    cell and the immobility, and the only one heeded. DeepSeek, S4 board 10,
    six players, \textsc{freeform}; each board is the full board after the
    move named above it. \texttt{A} warns at move 7 while it still has two free
    neighbours; \texttt{B}'s move 8 leaves it one, \texttt{%(last)s}, and
    repeats the warning. At move %(ply)d \texttt{C} could have played up into
    that cell, the route it had announced at move 3 and its only move towards
    its goal; it plays down instead, from 11 to 12 moves from its target,
    back onto the cell it started on.
    No move in the episode leaves a player without a legal move, and it is
    won in %(total)d moves. Messages are quoted verbatim from the transcript;
    \ldots{} marks omitted text.}
    \label{fig:warning-heeded}
\end{figure}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    for path in render(ap.parse_args().out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
