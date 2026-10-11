"""Draws the strong tier's failure at the pinch: the stranded player invites it.

The twin of figures/warning.py. Under the prose protocols the strong tier played
30 sealing moves that had a harmless alternative. In 29 the player about to be
stranded had named the cell and the mover in the last round of messages; 12
fall in episodes that fail, and in 10 of those the message invites the move.
Two of the ten are drawn: the player vacates a cell, offers it to the teammate
behind, and the teammate's acceptance leaves it with no legal move.

The counts are recomputed from analysis/qualitative_analysis/seal_averted.py except the
last, which is a hand reading of the 12 and is pinned below; every board is
replayed from the move log and every quoted fragment is checked verbatim.

    python3 figures/invitation.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from aborts import OUT_DIR, REPO_ROOT, at, cell_name, square_axes
from style import (INK, MOVE, MUTED, OBJECT, OWNER, PAPER, centre, draw_arrow,
                   draw_disc, draw_grid, draw_target, save)
import seal_alt as sa
import warning as wn

sys.path.insert(0, os.path.join(REPO_ROOT, "analysis/qualitative_analysis"))
import seal_averted as sv  # noqa: E402

VICTIM, ACTOR = OWNER["B"], OBJECT
BYSTANDER = MUTED        # FAINT is 2.6:1 on white, below 3:1 for a mark that matters

# The hand reading of the 12 failing pinches: the stranded player's message
# offers the mover the cell (one of the ten only hedges: "free if you need it,
# but I may need it"). The other two are a request to a player already done
# and a neutral "wait for D to clear".
INVITED = {
    ("qwen38", "S3", "instance_00010", "freeform", 56),
    ("qwen38", "S3", "instance_00010", "hybrid", 82),
    ("qwen38", "S4", "instance_00008", "freeform", 17),
    ("qwen38", "S4", "instance_00008", "freeform", 21),
    ("qwen38", "S4", "instance_00011", "freeform", 19),
    ("dsv41", "S3", "instance_00006", "freeform", 24),
    ("dsv41", "S3", "instance_00010", "freeform", 47),
    ("dsv41", "S3", "instance_00010", "freeform", 53),
    ("dsv41", "S4", "instance_00005", "freeform", 33),
    ("dsv41", "S4", "instance_00008", "hybrid", 22),
}

# (results root, level, instance, victim, mover, invitation move, seal move,
#  {move: fragments cut from its message, never reworded})
CASES = [
    ("results_qwen38_grid_n15_6p", "Qwen3.8", "S4", "instance_00011", "A", "B",
     19, 20,
     {19: ["I’m moving A right to (R4,C5) to free (R4,C4) for B. "
           "B, take (R4,C4) next"],
      20: ["A, thanks; I'm taking (R4,C4).",
           "A, hold (R4,C5) until I'm at (R3,C4)."]}),
    ("results_dsv41_grid_n15_6p", "DeepSeek", "S3", "instance_00006", "C", "A",
     21, 25,
     {21: ["C up to (R3,C3), freeing (R4,C3) for A. A, take (R4,C3) now"],
      25: ["A right to (R4,C3) to stage behind C.",
           "I need (R3,C3) free next after C moves."]}),
]


def key(r):
    m = r["meta"]
    return m["model"], m["level"], m["inst"], m["proto"], r["ply"]


def counts():
    """30 -> 29 -> 12 -> 10, the first three recomputed from the pinches."""
    rs, _ = sv.rows(models=sv.TIERS["strong"])
    taken = [r for r in rs if r["taken"] and r["safe"]]
    named = []
    for r in taken:
        cell = sv.step(r["pos"], *r["played"])[r["played"][0]]
        rx = re.compile(sv.CELL % (cell[0] + 1, cell[1] + 1), re.I)
        if any(r["ply"] - 6 <= p <= r["ply"] and rx.search(t)
               and re.search(r"\b%s\b" % r["player"], t)
               for q in r["victims"] for p, t in r["said"].get(q, [])):
            named.append(r)
    failing = {key(r) for r in named if r["meta"]["outcome"] != "success"}
    assert (len(taken), len(named), len(failing)) == (30, 29, 12), \
        (len(taken), len(named), len(failing))
    assert INVITED <= failing and len(INVITED) == 10, INVITED - failing
    return len(taken), len(named), len(failing), len(INVITED)


def load(root, level, inst):
    wn.RUN, = glob.glob(os.path.join(REPO_ROOT, root, "*", "*",
                                     f"{level}_freeform_6p_grid", inst))
    return wn.load()


def exits(pos, obj, walls, n):
    return [sa.step(pos, obj, d)[obj] for _, d in sa.legal_moves(pos, [obj], walls, n)]


def panel(ax, n, walls, pos, prev, moved, victim, mover, offered=None):
    """The board after one move. Roles are carried by colour *and* mark: the
    stranded player's disc is ringed, bystanders are grey, and once the offer
    is accepted the offered cell keeps a solid outline under the taker."""
    draw_grid(ax, n, walls, labels=5.4)
    for cell in exits(pos, victim, walls, n):
        draw_target(ax, cell, "", VICTIM, nested=False, lw=1.1)
    if offered:
        r, c = offered
        ax.add_patch(Rectangle((c + 0.06, r + 0.06), 0.88, 0.88, fc="none",
                               ec=VICTIM, lw=1.5, zorder=6))
    ghost = prev[moved]
    ax.add_patch(Circle(centre(ghost), 0.22, fc="none", ec=MUTED, lw=0.9,
                        ls=(0, (1.2, 1.2)), zorder=7))
    for obj, cell in sorted(pos.items()):
        colour = VICTIM if obj == victim else ACTOR if obj == mover else BYSTANDER
        draw_disc(ax, cell, obj, colour, radius=0.31, letter=6.6,
                  ring=INK if obj == victim else None, ring_lw=1.6)
    draw_arrow(ax, ghost, pos[moved], MOVE, lw=1.9, scale=9.0,
               shrink=(4.0, 5.5), zorder=8.5)


def render(out_dir):
    total, named, failing, invited = counts()
    rows = []
    for root, model, level, inst, v, m, inv, seal, quotes in CASES:
        ins, moves, said, outcome = load(root, level, inst)
        n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
        b = {p: wn.board_at(ins, moves, p) for p in (inv - 1, inv, seal - 1, seal)}
        offered = b[inv - 1][v]                     # the cell the victim vacates
        # what the figure asserts, checked against the rules before drawing
        assert moves[inv - 1][0] == v and moves[seal - 1][0] == m
        assert b[seal][m] == offered, (b[seal][m], offered)
        assert exits(b[seal - 1], v, walls, n) == [offered]
        assert exits(b[seal], v, walls, n) == []
        assert outcome is None                      # the episode is not won
        stuck = [k for k in range(seal, len(moves) + 1)
                 if not exits(wn.board_at(ins, moves, k), v, walls, n)]
        run = len([k for i, k in enumerate(stuck) if k == seal + i])
        gap = ""
        if seal - inv > 1:
            gap = (f"Moves {inv + 1}\u2013{seal - 1} leave {cell_name(offered)} "
                   f"as {v}'s last free cell. ")
        if seal + run > len(moves):
            fate = (f"{v} has no legal move; its next turn aborts the episode "
                    f"after {len(moves)} moves.")
        else:
            fate = (f"{v} has no legal move for {run} moves until a third "
                    f"player frees it; the episode is lost.")
        rows.append(dict(src=("transcript:", f"{root}/…/", f"{level}_freeform_6p_grid/{inst}"),
                         model=model, level=level, inst=int(inst[-2:]), v=v,
                         m=m, inv=inv, seal=seal, n=n, walls=walls, b=b,
                         moves=moves, cell=offered, fate=gap + fate,
                         texts={p: wn.marked(said[p], f) for p, f in quotes.items()},
                         left={p: len(exits(b[p], v, walls, n)) for p in (inv, seal)}))

    W, M, GAP = 6.3, 0.22, 0.14
    side = 1.78
    col = M + 2 * side + GAP + 0.22
    colw = W - col - M
    chars = 33
    row_h = 0.22 + 0.34 + side + 0.30          # title, panel heads, board, note
    H = 0.46 + len(rows) * (row_h + 0.16) + 0.50
    fig = plt.figure(figsize=(W, H))

    y = H - 0.12
    fig.text(*at(fig, M, y), wn.VERBATIM, fontsize=7.2, color=MUTED,
             va="top", style="italic")
    y -= 0.34

    for r in rows:
        fig.text(*at(fig, M, y), f"{r['model']}, six players, {r['level']} board "
                 f"{r['inst']}, freeform", fontsize=7.8, color=INK, va="top",
                 fontweight="bold")
        top = y - 0.22
        heads = {r["inv"]: f"after move {r['inv']}: {r['v']} offers {cell_name(r['cell'])}",
                 r["seal"]: f"after move {r['seal']}: {r['m']} takes it"}
        for i, p in enumerate((r["inv"], r["seal"])):
            x = M + i * (side + GAP)
            fig.text(*at(fig, x + side / 2, top), heads[p], fontsize=7.0,
                     color=INK, ha="center", va="top")
            ax = square_axes(fig, x, top - 0.34 - side, side)
            panel(ax, r["n"], r["walls"], r["b"][p], r["b"][p - 1],
                  r["moves"][p - 1][0], r["v"], r["m"],
                  offered=r["cell"] if p == r["seal"] else None)
            k = r["left"][p]
            fig.text(*at(fig, x + side / 2, top - 0.38 - side),
                     f"{r['v']} can move to {k} cell{'s' * (k != 1)}",
                     fontsize=6.8, color=VICTIM, ha="center", va="top")
        fig.text(*at(fig, col, top), "What the players wrote, verbatim",
                 fontsize=7.0, color=INK, va="top")
        yb = top - 0.24
        for p, who, colour in ((r["inv"], r["v"], VICTIM), (r["seal"], r["m"], ACTOR)):
            obj, d = r["moves"][p - 1]
            h = wn.bubble(fig, col, yb, colw, who, f"move {p} · {obj} {d}",
                          r["texts"][p], chars=chars, colour=colour)
            yb -= h + 0.10
        fate = wn.wrap_words(r["fate"], 40)
        for j, line in enumerate(fate):
            fig.text(*at(fig, col, yb - 0.02 - j * 0.15), line, fontsize=6.9,
                     color=INK, va="top")
        for j, line in enumerate(r["src"]):
            fig.text(*at(fig, col, yb - 0.08 - (len(fate) + j * 0.72) * 0.15),
                     line, fontsize=5.8, color=MUTED, va="top",
                     family="monospace")
        y -= row_h + 0.16

    # key: each mark drawn as itself, so colour is never the only cue
    k = fig.add_axes([0, 0, 1, 1], facecolor="none"); k.axis("off")
    k.set_xlim(0, W); k.set_ylim(0, H)

    def disc(x, y, colour, ring=None):
        k.add_patch(Circle((x + 0.11, y), 0.065, fc=colour,
                           ec=ring or PAPER, lw=1.2 if ring else 0.5))

    def box(x, y, ls, fill):
        k.add_patch(Rectangle((x + 0.05, y - 0.06), 0.12, 0.12,
                              fc=to_rgba(VICTIM, 0.10) if fill else "none",
                              ec=VICTIM, lw=1.1, ls=ls))

    def ghost(x, y):
        k.add_patch(Circle((x + 0.11, y), 0.05, fc="none", ec=MUTED, lw=0.9,
                           ls=(0, (1.2, 1.2))))

    def arrow(x, y):
        k.add_patch(FancyArrowPatch((x, y), (x + 0.22, y), arrowstyle="-|>",
                                    mutation_scale=7, color=MOVE, lw=1.4))

    rows_k = ((0.40, ((lambda x, y: disc(x, y, VICTIM, INK), "player left stranded"),
                      (lambda x, y: disc(x, y, ACTOR), "teammate it invites"),
                      (lambda x, y: disc(x, y, BYSTANDER), "other players"))),
              (0.20, ((arrow, "move just played"),
                      (ghost, "where it stood"),
                      (lambda x, y: box(x, y, (0, (2.2, 1.5)), True),
                       "cell the stranded player can move to"),
                      (lambda x, y: box(x, y, "-", False), "the offered cell"))))
    for yk, items in rows_k:
        x = M
        for mark, label in items:
            mark(x, yk)
            t = k.text(x + 0.28, yk, label, fontsize=6.6, color=INK, va="center")
            wide = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
            x += 0.28 + wide + 0.30
        assert x <= W - M + 0.30, f"key row overflows by {x - W + M - 0.30:.2f} in"
    paths = save(fig, out_dir, "invitation_sealed", ("pdf", "png"))
    tex = os.path.join(out_dir, "invitation.tex")
    with open(tex, "w") as fh:
        fh.write(FLOAT % dict(total=total, named=named, failing=failing,
                              invited=invited))
    return paths + [tex]


FLOAT = r"""%% Figure float for the strong tier's invitations, written by figures/invitation.py.
\begin{figure}[tbp]
    \centering
    \thesisfig{invitation_sealed.pdf}
    \caption{How the strong tier fails at the pinch. Of the %(total)d sealing
    moves it played under the prose protocols with a harmless alternative
    available, %(named)d were preceded by the stranded player naming the cell
    and the mover in the last round of messages; %(failing)d fall in failing
    episodes, and in %(invited)d of those the message offers the cell. Two are
    drawn, each as the full board after the move beneath it: the player
    vacates a cell, offers it to the teammate behind, and the teammate's
    acceptance leaves it with no legal move. Messages are quoted verbatim
    from the episode transcripts; \ldots{} marks omitted text.}
    \label{fig:invitation-sealed}
\end{figure}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    for path in render(ap.parse_args().out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
