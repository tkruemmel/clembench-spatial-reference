"""Draws the strong tier's failure at the pinch: the stranded player invites it
(fig:invitation-sealed).

Under the prose protocols the strong tier played
30 sealing moves that had a harmless alternative. In 22 the player about to be
stranded had named the cell and the mover in the six plies before; 8 fall in
episodes that fail, and in 6 of those the message invites the move under the
written coding rule of analysis/qualitative_analysis/failures.py. Two of the six
are drawn, one per model: the player names a cell and the teammate who should
take it, and the teammate's move into it leaves the player with no legal move.

All four counts come from failures.invitation_chain; every board is replayed
from the move log, each drawn case is asserted to be one of the six and to
satisfy the rule, and every quoted fragment is checked verbatim.

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
import failures  # noqa: E402
import seal_averted as sv  # noqa: E402

VICTIM, ACTOR = OWNER["B"], OBJECT
BYSTANDER = MUTED        # FAINT is 2.6:1 on white, below 3:1 for a mark that matters

# one per model, from the six invited failing seals; S3 #10 is left to the
# worked example. inv = the victim's move whose message offers the cell,
# seal = the mover's move into it; {move: fragments, never reworded}
CASES = [
    dict(root="results_qwen38_grid_updated_prompt_n15_6p", key="qwen38",
         model="Qwen3.8", level="S4", proto="hybrid", inst="instance_00005",
         v="A", m="C", inv=33, seal=34,
         quotes={33: ["I\u2019m stepping left to (R4,C3) so C can occupy (R4,C4)",
                      "REQUEST: C move up to (R4,C4) now"],
                 34: ["I\u2019ll follow A\u2019s handoff: take (R4,C4) now",
                      "A, hold (R4,C3) until I reach (R3,C4), then you can "
                      "reclaim (R4,C4)."]}),
    dict(root="results_dsv41_grid_updated_prompt_n15_6p", key="dsv41",
         model="DeepSeek", level="S4", proto="freeform", inst="instance_00008",
         v="A", m="F", inv=19, seal=23,
         quotes={19: ["F, I'll hold R4C5 only until B vacates R5C5; please move "
                      "to R3C5 and wait, then R4C5 is yours."],
                 23: ["Moving F right to R3C5.",
                      "I\u2019ll wait at R3C5 until A vacates R4C5"]}),
]


def counts():
    """30 -> 22 -> 8 -> 6, and the keys of the six."""
    taken, named = failures.invitation_chain(
        sv.rows(models=failures.STRONG, protos=failures.PROSE)[0])
    failing = [x for x in named if x["fail"]]
    invited = {x["key"] for x in failing if x["invited"]}
    got = (len(taken), len(named), len(failing), len(invited))
    assert got == (30, 22, 8, 6), got     # results_numbers.md B22-B24, B31
    return got, invited


def load(root, level, proto, inst):
    wn.RUN, = glob.glob(os.path.join(REPO_ROOT, root, "*", "*",
                                     f"{level}_{proto}_6p_grid", inst))
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
    (total, named, failing, invited), six = counts()
    rows = []
    for c in CASES:
        v, m, inv, seal = c["v"], c["m"], c["inv"], c["seal"]
        ins, moves, said, outcome = load(c["root"], c["level"], c["proto"], c["inst"])
        n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
        b = {p: wn.board_at(ins, moves, p) for p in range(inv - 1, seal + 1)}
        offered = b[seal][m]                        # the cell the mover takes
        # what the figure asserts, checked against the rules before drawing
        assert (c["key"], c["level"], c["inst"], c["proto"], seal - 1) in six
        assert moves[inv - 1][0] == v and moves[seal - 1][0] == m
        assert exits(b[seal - 1], v, walls, n) == [offered]
        assert exits(b[seal], v, walls, n) == []
        assert outcome is None                      # the episode is not won
        rx = re.compile(sv.CELL % (offered[0] + 1, offered[1] + 1), re.I)
        assert failures.invites(said[inv], rx, m), said[inv]
        stuck = [k for k in range(seal, len(moves) + 1)
                 if not exits(wn.board_at(ins, moves, k), v, walls, n)]
        run = len([k for i, k in enumerate(stuck) if k == seal + i])
        gap = ""
        if seal - inv > 1:
            last = min(k for k in range(inv, seal)
                       if all(exits(b[j], v, walls, n) == [offered]
                              for j in range(k, seal)))
            gap = (f"From move {last} on, {cell_name(offered)} is {v}'s last "
                   f"free cell. ")
        if seal + run > len(moves):
            fate = (f"{v} has no legal move; its next turn aborts the episode "
                    f"after {len(moves)} moves.")
        else:
            fate = (f"{v} has no legal move for {run} moves until a third "
                    f"player frees it; the episode is lost.")
        rows.append(dict(src=("transcript:", f"{c['root']}/…/",
                              f"{c['level']}_{c['proto']}_6p_grid/{c['inst']}"),
                         model=c["model"], level=c["level"], proto=c["proto"],
                         inst=int(c["inst"][-2:]), v=v, m=m, inv=inv, seal=seal,
                         n=n, walls=walls, b=b, moves=moves, cell=offered,
                         fate=gap + fate,
                         texts={p: wn.marked(said[p], f) for p, f in c["quotes"].items()},
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
                 f"{r['inst']}, {r['proto']}", fontsize=7.8, color=INK, va="top",
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

    rows_k = ((0.40, ((lambda x, y: disc(x, y, VICTIM, INK), "player left trapped"),
                      (lambda x, y: disc(x, y, ACTOR), "teammate it invites"),
                      (lambda x, y: disc(x, y, BYSTANDER), "other players"))),
              (0.20, ((arrow, "move just played"),
                      (ghost, "where it stood"),
                      (lambda x, y: box(x, y, (0, (2.2, 1.5)), True),
                       "cell the trapped player can move to"),
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
    \caption{The trapped player invites the seal. Of the %(total)d sealing
    moves the strong tier played under the prose protocols with a harmless
    alternative available, %(named)d were preceded by the trapped player
    naming the cell and the mover in the six plies before; %(failing)d fall in
    failing episodes, and in %(invited)d of those the message offers the cell.
    Two of the %(invited)d are drawn, chosen after inspection, one per model,
    each as the full board after the move beneath it: the player names a cell
    and the teammate who should take it, and the teammate's move into it
    leaves the player with no legal move. Messages are quoted verbatim from
    the episode transcripts; \ldots{} marks omitted text.}
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
