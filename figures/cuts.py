"""fig:cut-corridor: one fork on S3 #14, two choices (D10).

DeepSeek, six players, S3 board 14. In both episodes B steps onto its own
target, the corridor entrance, with move 50, and D still needs to pass it.
(a) Under freeform B declares `done` on its next turn: the cell becomes a wall
and D (and E) can no longer reach their targets. (b) Under hybrid B steps off
on its next turn instead, and the episode is won. Panel (a) is the recorded
cut from analysis/qualitative_analysis/cut.py; panel (b) is replayed and
checked by figures/giveway.py.

The population counts of the caption are recomputed from cut.py and pinned to
the ledger; the pair is an illustration chosen after inspection. Every
position, route and count is replayed from the recorded move logs; nothing
here is hand-placed.

    python3 figures/cuts.py
"""

from __future__ import annotations

import argparse
import os
import sys

import json

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle

from style import (CUE, FAINT, INK, MUTED, OBJECT, OWNER, PAPER, WALL,
                   draw_arrow, draw_grid, draw_target, save)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis/qualitative_analysis"))
import cut  # noqa: E402

VICTIM = OWNER["B"]   # the object that can no longer reach its target
ACTOR = OBJECT        # the player whose declaration this panel is about

# The episode drawn in panel (a), named rather than searched for so the figure
# is stable, with the property that made it the right example asserted below.
CORRIDOR = ("dsv41", "6p", "S3", "freeform", "instance_00014", "D")
# B's promise before it settles, and its declaration; quoted verbatim, checked
# against the transcript before drawing
FREEFORM_SAID = [
    (44, ["D: R5C2 stays clear this turn, and I\u2019ll yield it for your",
          "transit before settling."]),
]


def cell_name(cell):
    return f"R{cell[0] + 1}C{cell[1] + 1}"


# --------------------------------------------------------------- geometry
#
# Placed in inches from the bottom-left rather than through a layout engine,
# because these two figures have to arrive at the exact width of the thesis
# text block (\textwidth = 455.24411 pt) so that \thesisfig scales them 1:1
# and the point sizes below are the point sizes on the page.

WIDTH = 455.24411 / 72.27          # the document's \linewidth, in inches
SIDE, GAP, LEFT = 2.56, 0.5592, 0.42
COLS = (LEFT, LEFT + SIDE + GAP)
HEAD = 1.16                        # title, subtitle, panel titles, column labels
LEAD, LEG_LEAD, CLOSE_LEAD = 0.205, 0.168, 0.21


def square_axes(fig, left_in, bottom_in, side_in):
    w, h = fig.get_size_inches()
    return fig.add_axes([left_in / w, bottom_in / h, side_in / w, side_in / h])


def at(fig, x_in, y_in):
    w, h = fig.get_size_inches()
    return x_in / w, y_in / h


def notes(fig, x_in, y_in, lines, *, size=7.4, leading=0.205):
    """Draw a stack of coloured lines; returns the y (inches) after the last."""
    for i, (colour, text) in enumerate(lines):
        x, y = at(fig, x_in, y_in - i * leading)
        fig.text(x, y, text, fontsize=size, color=colour, va="top")
    return y_in - len(lines) * leading


# ------------------------------------------------------------------- marks
#
# Colour is never the only carrier here. An object's own target is the only
# target drawn, so the cut object is identified by the letter it shares with
# the dashed cell; a declaration turns its object from a circle into a square
# and cross-hatches the cell it now walls off; the region the cut object can
# still move in is outlined as well as tinted; and the cells it loses between
# the two panels are hatched. Every one of those survives a grayscale print,
# which the palette audit says the house colours alone do not: VICTIM and
# ACTOR sit 9.85 CIE L* apart and VICTIM and MUTED 0.06 apart.


def tint(ax, cells, colour, alpha, *, zorder=1.5):
    for r, c in cells:
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=to_rgba(colour, alpha),
                               edgecolor="none", zorder=zorder))


def outline(ax, cells, walls, n, colour, *, lw=1.3):
    """Trace where the region ends against something that is not a wall.

    Edges against walls are left undrawn: the wall already shows them, and
    outlining every one of them buries the informative part of the boundary,
    which is the handful of edges where a frozen object closed the way.
    """
    cells = set(cells)
    for r, c in cells:
        for (dr, dc), (x0, y0, x1, y1) in (
                ((-1, 0), (c, r, c + 1, r)), ((1, 0), (c, r + 1, c + 1, r + 1)),
                ((0, -1), (c, r, c, r + 1)), ((0, 1), (c + 1, r, c + 1, r + 1))):
            nb = (r + dr, c + dc)
            if nb in cells or nb in walls or not (0 <= nb[0] < n and 0 <= nb[1] < n):
                continue
            ax.plot([x0, x1], [y0, y1], color=colour, linewidth=lw,
                    solid_capstyle="round", zorder=4.5)


def hatch_cells(ax, cells, colour, pattern, *, lw=0.5, alpha=0.0, edge=None):
    for r, c in cells:
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=to_rgba(colour, alpha),
                               edgecolor=edge or colour, linewidth=lw,
                               hatch=pattern, zorder=2.2))


def ink_on(colour):
    """Dark letters on a pale fill, pale on a dark one: keeps the letter legible."""
    r, g, b = mpl_colors.to_rgb(colour)
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else PAPER


def marker(ax, cell, label, colour, *, frozen, cue=False):
    """An object: a circle while it can still move, a square once it cannot."""
    edge, lw = (CUE, 2.0) if cue else (INK, 0.7)
    if frozen:
        side = 0.54
        ax.add_patch(Rectangle((cell[1] + (1 - side) / 2, cell[0] + (1 - side) / 2),
                               side, side, facecolor=colour, edgecolor=edge,
                               linewidth=lw, zorder=7.5))
    else:
        ax.add_patch(Circle((cell[1] + 0.5, cell[0] + 0.5), 0.295,
                            facecolor=colour, edgecolor=edge, linewidth=lw,
                            zorder=7.5))
    ax.text(cell[1] + 0.5, cell[0] + 0.5, label, ha="center", va="center",
            fontsize=8.4, color=ink_on(colour), fontweight="bold", zorder=8)


def reachable(case, blocked):
    """Every cell the victim can still move to, given `blocked` frozen cells."""
    seen, queue = {case["pre"][case["victim"]]}, [case["pre"][case["victim"]]]
    while queue:
        r, c = queue.pop()
        for dr, dc in cut.DIRS.values():
            nb = (r + dr, c + dc)
            if not (0 <= nb[0] < case["grid"] and 0 <= nb[1] < case["grid"]):
                continue
            if nb in case["walls"] or nb in blocked or nb in seen:
                continue
            seen.add(nb)
            queue.append(nb)
    return seen


OPEN_DASH = (0, (1.3, 1.5))    # a route that still exists
DEAD_DASH = (0, (4.5, 1.8))    # one already closed by an earlier declaration


def route(ax, cells, colour, dash, *, lw=1.2):
    for src, dst in zip(cells, cells[1:]):
        draw_arrow(ax, src, dst, colour, lw=lw, scale=7.5, shrink=(4.0, 1.0),
                   ls=dash, zorder=3.5)


def board(ax, case, *, sealed, routes=(), lost=(), labels=5.0):
    """One panel. `sealed` switches the declaration on; nothing else changes."""
    frozen = dict(case["frozen_post"] if sealed else case["frozen_pre"])
    draw_grid(ax, case["grid"], case["walls"], labels=labels)
    region = reachable(case, set(frozen.values()))
    tint(ax, region, VICTIM, 0.13)
    outline(ax, region, case["walls"], case["grid"], VICTIM)
    hatch_cells(ax, lost, VICTIM, "/", lw=0.3, edge=to_rgba(VICTIM, 0.5))
    hatch_cells(ax, frozen.values(), WALL, "xxx", alpha=0.20)
    for cells, colour, dash in routes:
        route(ax, cells, colour, dash)
    draw_target(ax, case["targets"][case["victim"]], case["victim"], VICTIM,
                nested=False, lw=1.3, letter=8.4)
    for obj, cell in sorted(case["pre"].items()):
        colour = (VICTIM if obj == case["victim"]
                  else ACTOR if obj in case["culprits"] else FAINT)
        marker(ax, cell, obj, colour, frozen=obj in frozen,
               cue=sealed and obj in case["seals"])
    return region


def render(case, gw, out_dir):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from style import MOVE
    from warning import bubble, height_of, marked

    v, holder, ent = case["victim"], case["sealer"], case["entrance"]
    path_f = os.path.join(REPO_ROOT, case["meta"]["path"])
    import giveway as gwm
    _, _, _, said_f, _ = gwm.replay(path_f)
    texts_a = [(p, marked(said_f[p], f)) for p, f in FREEFORM_SAID]
    texts_b = [(p, marked(gw["said"][p], f)) for p, f in gwm.SAID]
    chars = 46
    col_h = max(sum(height_of(t, chars) + 0.06 for _, t in texts_a) + height_of("done", chars),
                sum(height_of(t, chars) + 0.06 for _, t in texts_b))
    H = HEAD + SIDE + 0.30 + col_h + 0.42 + 0.66      # + notes + key
    fig = plt.figure(figsize=(WIDTH, H))
    fig.text(*at(fig, COLS[0], H - 0.20), "One fork, two choices", fontsize=10.4,
             color=INK, va="top", fontweight="bold")
    fig.text(*at(fig, COLS[0], H - 0.52),
             "DeepSeek, six players, S3 board %s: B's next turn after stepping onto "
             "its own target, the corridor entrance" % case["meta"]["inst"].split("_")[1].lstrip("0"),
             fontsize=7.5, color=MUTED, va="top")
    bottom = H - HEAD - SIDE
    ax_a, ax_b = (square_axes(fig, x, bottom, SIDE) for x in COLS)
    before = reachable(case, set(case["frozen_pre"].values()))
    after = reachable(case, set(case["frozen_post"].values()))
    board(ax_a, case, sealed=True,
          lost=before - after - set(case["frozen_post"].values()))
    board(ax_b, gw["case"], sealed=False, routes=[(gw["live"], VICTIM, OPEN_DASH)])
    pos = gw["case"]["pre"]
    draw_arrow(ax_b, pos[holder], cut.step(pos, holder, "left")[holder], MOVE,
               lw=2.2, scale=11.0, shrink=(6.0, 0.0), zorder=9)
    for x, tag, text in zip(COLS, "ab", (
            "freeform, after move %d: %s declares done" % (case["ply"], holder),
            "hybrid, move %d: %s steps off its target" % (gw["step_off"], holder))):
        fig.text(*at(fig, x, bottom + SIDE + 0.26), "(%s)" % tag, fontsize=8.6,
                 color=INK, va="bottom", fontweight="bold")
        fig.text(*at(fig, x + 0.24, bottom + SIDE + 0.26), text, fontsize=8.2,
                 color=INK, va="bottom")

    cut_objs = sorted(r["victim"] for r in gw["cuts"])
    y = bottom - 0.24
    for p, t in texts_a:
        y -= bubble(fig, COLS[0], y, SIDE, holder, "move %d" % p, t, chars=chars,
                    colour=ACTOR) + 0.06
    y -= bubble(fig, COLS[0], y, SIDE, holder, "after move %d" % case["ply"],
                "done", chars=chars, colour=ACTOR) + 0.04
    notes(fig, COLS[0], y - 0.02, [
        (INK, "%s can no longer reach their targets; the team" % " and ".join(cut_objs)),
        (INK, "plays %d more moves and loses after move %d." % (case["tail"],
                                                              case["meta"]["moves"]))],
        size=7.2, leading=0.17)
    y = bottom - 0.24
    for p, t in texts_b:
        y -= bubble(fig, COLS[1], y, SIDE, holder, "move %d" % p, t, chars=chars,
                    colour=ACTOR) + 0.06
    notes(fig, COLS[1], y - 0.02, [
        (INK, "%s declares done after move %d; the episode" % (holder, gw["done"])),
        (INK, "is won in %d moves." % gw["total"])], size=7.2, leading=0.17)

    keys = [
        Patch(facecolor=to_rgba(WALL, 0.20), edgecolor=WALL, hatch="xxx",
              linewidth=0.5, label="declared done: object frozen (square), cell walled"),
        Patch(facecolor=to_rgba(VICTIM, 0.13), edgecolor=VICTIM, linewidth=1.1,
              label="where %s can still move" % v),
        Patch(facecolor="none", edgecolor=to_rgba(VICTIM, 0.5), hatch="//",
              linewidth=0.3, label="cells %s loses to %s's done" % (v, holder)),
        Patch(facecolor="none", edgecolor=VICTIM, linewidth=1.3,
              linestyle=(0, (2.2, 1.5)), label="%s's target" % v),
        Line2D([], [], color=VICTIM, linestyle=OPEN_DASH, linewidth=1.2,
               label="%s's route, open while %s can move" % (v, holder)),
        Line2D([], [], color=MOVE, linewidth=2.0, marker=">", markersize=4,
               label="%s's move off its target" % holder),
    ]
    fig.legend(handles=keys, loc="lower left",
               bbox_to_anchor=at(fig, COLS[0] - 0.06, 0.08), ncol=2,
               frameon=False, fontsize=6.6, handletextpad=0.6,
               columnspacing=1.2, labelspacing=0.45, borderpad=0.0)
    return save(fig, out_dir, "cut_corridor", ("pdf", "png"))


FLOAT = r"""%% Figure float for fig:cut-corridor, written by figures/cuts.py.
%% Every number in the caption is read off the replayed episodes or the cut
%% population, both asserted in main().
\begin{figure}[tbp]
    \centering
    \thesisfig{cut_corridor.pdf}
    \caption[One fork, two choices]{%(cap)s}
    \label{fig:cut-corridor}
\end{figure}
"""


def write_float(rows, case, gw, out_dir):
    ent = sum(r["kind"] in (cut.ENTRANCE, cut.ENTRANCE_OTHER) for r in rows)
    cap = (
        "One fork, two choices, on one board: %s, S3~\\#%s, six players, "
        "chosen after inspection. In both episodes \\texttt{%s} steps onto its "
        "own target \\texttt{%s}, the corridor's single entrance, with move %d, "
        "and \\texttt{%s} still has to pass it to reach \\texttt{%s}. "
        "(a)~Under \\textsc{freeform}, \\texttt{%s} declares \\texttt{done} on "
        "its next turn, after move %d; the cell becomes a wall, \\texttt{%s} and "
        "\\texttt{%s} can no longer reach their targets, and the team plays %d "
        "more moves and loses after move %d. (b)~Under \\textsc{hybrid}, "
        "\\texttt{%s} steps off on its next turn, move %d, declares "
        "\\texttt{done} after move %d, and the episode is won in %d moves. "
        "Of the %d permanent cuts recorded across the %d cycle losses, %d are "
        "seals on a corridor entrance. Messages are quoted verbatim; "
        "\\ldots{} marks omitted text."
        % (cut.MODELS[case["meta"]["model"]],
           case["meta"]["inst"].split("_")[1].lstrip("0"), case["sealer"],
           cell_name(case["entrance"]), gw["arrive"], case["victim"],
           cell_name(tuple(case["device"]["dead_end"])), case["sealer"],
           case["ply"], *sorted(r["victim"] for r in gw["cuts"]), case["tail"],
           case["meta"]["moves"], case["sealer"], gw["step_off"], gw["done"],
           gw["total"], len(rows), len({r["meta"]["path"] for r in rows}), ent))
    path = os.path.join(out_dir, "cuts.tex")
    with open(path, "w") as fh:
        fh.write(FLOAT % dict(cap=cap))
    return [path]


def write_provenance(rows, case, gw, out_dir):
    """What each panel was built from, so the figure can be checked or redrawn."""
    record = {
        "figure": "cut_corridor",
        "population": {
            "scope": "cycle losses in 3p and 6p, Qwen3.8 and DeepSeek-V4.1, "
                     "updated-prompt arm",
            "episodes": len({r["meta"]["path"] for r in rows}),
            "cut_objects": len(rows),
            "by_shape": {k: sum(1 for r in rows if r["kind"] == k)
                         for k in sorted({r["kind"] for r in rows})},
        },
        "panel_a": {
            "episode": case["meta"]["path"], "protocol": case["meta"]["proto"],
            "accepted_moves_before_the_declaration": case["ply"],
            "episode_length_moves": case["meta"]["moves"],
            "declaring_player": case["sealer_player"],
            "cut_objects": sorted(r["victim"] for r in gw["cuts"]),
            "shape": case["kind"], "minimum_vertex_cut": case["mincut"],
            "moves_played_after_the_cut": case["tail"],
        },
        "panel_b": {
            "episode": gw["path"], "protocol": "hybrid",
            "holder_on_entrance_after_move": gw["arrive"],
            "holder_steps_off_at_move": gw["step_off"],
            "holder_done_after_move": gw["done"],
            "episode_length_moves": gw["total"], "outcome": "success",
        },
        "method": {
            "replay": "accepted moves parsed from interactions.json and applied "
                      "to instance.json start positions",
            "reachability": "breadth-first search over free cells, with walls and "
                            "the objects of players that have declared done "
                            "treated as blocked and all other objects passable",
            "analysis_module": "analysis/qualitative_analysis/cut.py",
            "renderer": "figures/cuts.py, figures/giveway.py",
        },
    }
    path = os.path.join(out_dir, "cuts.provenance.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return [path]


def pick(rows, key):
    model, suite, level, proto, inst, victim = key
    for r in rows:
        m = r["meta"]
        if ((m["model"], m["suite"], m["level"], m["proto"], m["inst"], r["victim"])
                == key):
            return r
    raise SystemExit(f"no recorded cut matching {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    rows = cut.cases()
    # the population the caption quotes, pinned to the ledger (results_numbers.md
    # B33, B68, B73, B74, B42): a changed tree fails here rather than in a caption
    assert len({r["meta"]["path"] for r in rows}) == 88, "episode count moved"
    assert len(rows) == 120, len(rows)
    assert sum(r["kind"] in (cut.ENTRANCE, cut.ENTRANCE_OTHER) for r in rows) == 62
    assert sum(r["kind"] == cut.OPEN for r in rows) == 34
    assert sum(r["kind"] == cut.POCKET for r in rows) == 5

    case = pick(rows, CORRIDOR)
    # the property panel (a) shows: B on its own target at the entrance, and its
    # done alone severs D (minimum cut 1, one culprit); E goes with it
    assert case["pre"][case["sealer"]] == case["entrance"] == case["targets"][case["sealer"]]
    assert case["kind"] == cut.ENTRANCE and case["mincut"] == 1
    assert len(case["culprits"]) == 1
    same = [r for r in rows if r["meta"]["path"] == case["meta"]["path"]]
    assert {(r["victim"], r["sealer"], r["ply"]) for r in same} == \
        {("D", "B", case["ply"]), ("E", "B", case["ply"])}, same
    assert (case["ply"], case["tail"], case["meta"]["moves"]) == (54, 30, 84)  # B67
    import giveway as gwm
    gw = gwm.giveway(case)
    gw["cuts"] = same
    assert (gw["arrive"], gw["step_off"], gw["done"], gw["total"]) == (50, 56, 103, 110)
    _, _, _, said, _ = gwm.replay(os.path.join(REPO_ROOT, case["meta"]["path"]))
    for p, frags in FREEFORM_SAID:
        at_ = 0
        for f in frags:
            at_ = said[p].index(f, at_)                  # verbatim, in order

    paths = render(case, gw, args.out)
    paths += write_float(rows, case, gw, args.out)
    paths += write_provenance(rows, case, gw, args.out)
    for path in paths:
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
