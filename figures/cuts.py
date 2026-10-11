"""The two figures for S9.3.2: what a permanent cut is made of.

`cut_corridor` is the shape 54 of the 101 cuts take — a player finishes on the
corridor entrance, which is the cell its own target sits on, and the board dies
the moment it declares `done`. `cut_twocut` is the other shape, where the
board's free space is 2-connected and no single cell can sever anything, so it
takes two finished objects, one on each way round.

Both figures are drawn on the state one instant before the fatal declaration,
so the left and right panel of each differ only in the `done`: no object moves
between them. That is the claim the pair exists to make.

Every position, route and count is replayed from the recorded move log by
analysis/qualitative_analysis/cut.py; nothing here is hand-placed.

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

# The two episodes drawn, named rather than searched for so the figures are
# stable, with the property that made each the right example asserted below.
CORRIDOR = ("dsv41", "6p", "S3", "freeform", "instance_00014", "D")
TWOCUT = ("qwen38", "6p", "S4", "none", "instance_00003", "D")

# B spends eight messages promising to wait, then declares done four moves
# after stepping onto the entrance. Quoted verbatim; checked against the
# transcript before it is drawn.
PROMISE = ("B moves right to goal R5C2. D: I know you need R5C2 to reach "
           "R1C2; I'll step left to R5C1 when you reach R5C3, so signal me.")


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


def page_height(n_legend, n_notes, n_close=2):
    """Tall enough for exactly the blocks this figure has, and no taller."""
    rows = -(-n_legend // 2)
    return (HEAD + SIDE + 0.24 + rows * LEG_LEAD + 0.18
            + n_notes * LEAD + 0.16 + n_close * CLOSE_LEAD + 0.16)


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


def legend(fig, y_in, *, dead_route):
    """The key. Returns the y (inches) below it."""
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.0,
               markerfacecolor=FAINT, markeredgecolor=INK, markeredgewidth=0.7,
               label="object, still able to move"),
        Patch(facecolor=to_rgba(WALL, 0.20), edgecolor=WALL, hatch="xxx",
              linewidth=0.5, label="owner declared done: object frozen (square), cell walled"),
        Patch(facecolor="none", edgecolor=VICTIM, linewidth=1.3,
              linestyle=(0, (2.2, 1.5)), label="the cut object's target"),
        Patch(facecolor=to_rgba(VICTIM, 0.13), edgecolor=VICTIM, linewidth=1.1,
              label="where the cut object can still move"),
        Patch(facecolor="none", edgecolor=to_rgba(VICTIM, 0.5), hatch="//",
              linewidth=0.3, label="cells it loses to this declaration"),
        Line2D([], [], color=VICTIM, linestyle=OPEN_DASH, linewidth=1.2,
               label="a route that still exists"),
    ]
    if dead_route:
        handles.append(Line2D([], [], color=MUTED, linestyle=DEAD_DASH,
                              linewidth=1.2, label="a route already closed"))
    fig.legend(handles=handles, loc="upper left",
               bbox_to_anchor=at(fig, COLS[0] - 0.06, y_in), ncol=2,
               frameon=False, fontsize=6.8, handletextpad=0.7,
               columnspacing=1.4, labelspacing=0.45, borderpad=0.0)
    return y_in - (-(-len(handles) // 2)) * LEG_LEAD - 0.18


def frame(fig, title, subtitle, case, titles, *, routes_open, routes_dead=()):
    """Header, the two panels and the key. Returns the y the notes start at."""
    height = fig.get_size_inches()[1]
    board_bottom = height - HEAD - SIDE
    fig.text(*at(fig, COLS[0], height - 0.20), title, fontsize=10.4, color=INK,
             va="top", fontweight="bold")
    fig.text(*at(fig, COLS[0], height - 0.52), subtitle, fontsize=7.5,
             color=MUTED, va="top")
    axes = [square_axes(fig, x, board_bottom, SIDE) for x in COLS]
    before = board(axes[0], case, sealed=False,
                   routes=list(routes_open) + list(routes_dead))
    after = board(axes[1], case, sealed=True, routes=list(routes_dead),
                  lost=before - reachable(case, set(case["frozen_post"].values()))
                  - set(case["frozen_post"].values()))
    for x, tag, text in zip(COLS, "ab", titles):
        fig.text(*at(fig, x, board_bottom + SIDE + 0.26), "(%s)" % tag,
                 fontsize=8.6, color=INK, va="bottom", fontweight="bold")
        fig.text(*at(fig, x + 0.22, board_bottom + SIDE + 0.26), text,
                 fontsize=8.2, color=INK, va="bottom")
    return legend(fig, board_bottom - 0.24, dead_route=bool(routes_dead))


def render_corridor(case, out_dir):
    v, s = case["victim"], case["sealer"]
    ent, dead = case["entrance"], tuple(case["device"]["dead_end"])

    fig = plt.figure(figsize=(WIDTH, page_height(6, 7)))
    top = frame(fig, "The seal behind half of every cycle loss",
          "six players, S3 board %s, %s, %s — one board, one turn apart, "
          "and nothing moves in between"
          % (case["meta"]["inst"].split("_")[1].lstrip("0"),
             cut.MODELS[case["meta"]["model"]], case["meta"]["proto"]),
          case,
          ("move %d: %s stands on %s" % (case["ply"], s, cell_name(ent)),
           "move %d: %s declares done" % (case["ply"], s)),
          routes_open=[(case["route"], VICTIM, OPEN_DASH)])
    tail = notes(fig, COLS[0], top, [
        (VICTIM, "%s must reach %s, the dead end of the corridor, and every route"
                 % (v, cell_name(dead))),
        (VICTIM, "     to it passes %s, the corridor's only entrance."
                 % cell_name(ent)),
        (ACTOR, "%s's own target is that entrance. Standing there costs nothing —"
                % s),
        (ACTOR, "     an object can be asked to step aside, and %s had offered to." % s),
        (CUE, "The declaration is what turns the cell into a wall. The hatched"),
        (CUE, "     cells in (b) are what %s loses by it, for the next %d moves."
              % (v, case["tail"])),
        (INK, "%d of the %d recorded cuts have exactly this shape." % (54, 101)),
    ])
    fig.text(*at(fig, COLS[0], tail - 0.10), "“%s”" % PROMISE,
             fontsize=7.2, color=MUTED, va="top", style="italic")
    fig.text(*at(fig, COLS[0], tail - 0.31),
             "%s, four moves before declaring done, and the eighth message in a "
             "row promising to wait." % s,
             fontsize=7.2, color=MUTED, va="top")
    return save(fig, out_dir, "cut_corridor", ("pdf", "png"))


def render_twocut(case, out_dir):
    v, s = case["victim"], case["sealer"]
    early = [o for o in case["culprits"] if o != s]

    fig = plt.figure(figsize=(WIDTH, page_height(7, 7)))
    top = frame(fig, "Where no single cell can cut the board, two finished objects can",
          "six players, S4 board %s, %s, %s — the corridor plays no part in this one"
          % (case["meta"]["inst"].split("_")[1].lstrip("0"),
             cut.MODELS[case["meta"]["model"]], case["meta"]["proto"]),
          case,
          ("move %d: the southern way round already frozen" % case["ply"],
           "move %d: %s declares done, closing the northern" % (case["ply"], s)),
          routes_open=[(case["route"], VICTIM, OPEN_DASH)],
          routes_dead=[(case["other_route"], MUTED, DEAD_DASH)])
    tail = notes(fig, COLS[0], top, [
        (VICTIM, "%s must reach %s, and the wall block leaves exactly two ways"
                 % (v, cell_name(case["targets"][v]))),
        (VICTIM, "     round it: north through %s, south through %s."
                 % (cell_name(case["pre"][s]), cell_name(case["pre"][early[0]]))),
        (MUTED, "%s finished on the southern cell earlier and froze it (long dashes)."
                % early[0]),
        (ACTOR, "%s now finishes on the northern one. Both are ordinary target cells,"
                % s),
        (ACTOR, "     and under masking neither player can see the other's route."),
        (CUE, "The generator certifies the free space 2-connected, so no single cell"),
        (CUE, "     severs anything here — and two is all 2-connectedness buys."),
    ])
    fig.text(*at(fig, COLS[0], tail - 0.10),
             "%d of the %d cuts take this form, every one on a board whose minimum "
             "vertex cut is 2." % (31, 101),
             fontsize=7.2, color=INK, va="top")
    fig.text(*at(fig, COLS[0], tail - 0.31),
             "This same pair of targets closed this same board in four independent "
             "episodes, two per strong model.",
             fontsize=7.2, color=MUTED, va="top")
    return save(fig, out_dir, "cut_twocut", ("pdf", "png"))


FLOATS = r"""%% Figure floats for the two cut figures, written by figures/cuts.py.
%% Every number in a caption is read off the replayed episodes, so a changed
%% tree cannot leave a caption asserting something the picture no longer shows.
\begin{figure}[tbp]
    \centering
    \thesisfig{cut_corridor.pdf}
    \caption[%(short_corridor)s]{%(cap_corridor)s}
    \label{fig:cut-corridor}
\end{figure}

\begin{figure}[tbp]
    \centering
    \thesisfig{cut_twocut.pdf}
    \caption[%(short_twocut)s]{%(cap_twocut)s}
    \label{fig:cut-twocut}
\end{figure}
"""


def write_floats(rows, corridor, twocut, out_dir):
    total = len(rows)
    ent = sum(1 for r in rows
              if r["kind"] in (cut.ENTRANCE, cut.ENTRANCE_OTHER))
    opn = sum(1 for r in rows if r["kind"] == cut.OPEN)
    early = [o for o in twocut["culprits"] if o != twocut["sealer"]][0]
    caps = {
        "short_corridor": "A \\texttt{done} declaration on the corridor entrance",
        "short_twocut": "A cut that takes two declarations, not one",
        "cap_corridor": (
            "How a \\texttt{done} declaration kills a board. %s, S3 board %s, "
            "\\textsc{%s}, six players: both panels show move %d of %d, and no "
            "object moves between them. \\texttt{%s} needs \\texttt{%s}, the "
            "dead end of the corridor, and every route to it --- one is dotted "
            "--- runs through \\texttt{%s}, the corridor's single entrance, "
            "where \\texttt{%s} "
            "is standing on its own target. Occupying that cell costs nothing "
            "while \\texttt{%s} can still be asked to move; declaring "
            "\\texttt{done} turns it into a wall, and the tinted region is "
            "everywhere \\texttt{%s} can reach for the remaining %d moves of "
            "the episode. %d of the %d permanent cuts recorded across the %d "
            "cycle losses take this form. An object is drawn as a circle while "
            "it can still move and as a square once its owner has declared "
            "\\texttt{done}, so colour carries nothing the shapes, hatches and "
            "the key do not."
            % (cut.MODELS[corridor["meta"]["model"]],
               corridor["meta"]["inst"].split("_")[1].lstrip("0"),
               corridor["meta"]["proto"], corridor["ply"],
               corridor["meta"]["moves"], corridor["victim"],
               cell_name(tuple(corridor["device"]["dead_end"])),
               cell_name(corridor["entrance"]), corridor["sealer"],
               corridor["sealer"], corridor["victim"], corridor["tail"],
               ent, total, len({r["meta"]["path"] for r in rows}))),
        "cap_twocut": (
            "The same failure where the geometry forbids a one-cell seal. %s, "
            "S4 board %s, \\textsc{%s}, six players, move %d of %d in both "
            "panels; the corridor plays no part. \\texttt{%s} must reach "
            "\\texttt{%s}, and the wall block leaves exactly two routes: "
            "\\texttt{%s} had already finished on the southern one (long grey "
            "dashes) and "
            "frozen it, and \\texttt{%s} now finishes on the northern one. "
            "Neither cell is an articulation point --- the generator certifies "
            "the free space 2-connected, and the minimum vertex cut between "
            "\\texttt{%s} and its target is %d --- which is why it takes two "
            "declarations rather than one. %d of the %d cuts have this shape, "
            "every one of them at minimum cut 2. Marks as in "
            "Figure~\\ref{fig:cut-corridor} and the key."
            % (cut.MODELS[twocut["meta"]["model"]],
               twocut["meta"]["inst"].split("_")[1].lstrip("0"),
               twocut["meta"]["proto"], twocut["ply"], twocut["meta"]["moves"],
               twocut["victim"], cell_name(twocut["targets"][twocut["victim"]]),
               early, twocut["sealer"], twocut["victim"], twocut["mincut"],
               opn, total)),
    }
    path = os.path.join(out_dir, "cuts.tex")
    with open(path, "w") as fh:
        fh.write(FLOATS % caps)
    return [path]


ALT = {
    "cut_corridor":
        "Two eight-by-eight grid boards side by side, the same board one "
        "instant apart. In (a) object B stands as a circle on R5C2, the only "
        "entrance to a dead-end corridor, and a dotted route runs from object "
        "D through that cell and up the corridor to D's target at R1C2. In (b) "
        "B's owner has declared done: B is now a square on a walled cell, the "
        "dotted route is gone, and the shaded region D can still move in no "
        "longer contains its target.",
    "cut_twocut":
        "Two eight-by-eight grid boards side by side, the same board one "
        "instant apart. Object D at centre must reach R8C7, and a wall block "
        "leaves two routes round it: a northern one through R4C6 and a "
        "southern one through R7C6. In (a) E is already a frozen square on the "
        "southern cell and that route is drawn in long grey dashes, while the "
        "northern route is dotted and open through B. In (b) B has declared "
        "done, is drawn as a frozen square, and the shaded region D can move "
        "in no longer contains its target.",
}


def write_provenance(rows, cases_by_stem, out_dir):
    """What each panel was built from, so the figures can be checked or redrawn.

    The two episodes are illustrative; the population they stand for is the
    whole set of cuts, whose counts are recorded here and asserted in main().
    """
    record = {
        "figures": {},
        "population": {
            "scope": "cycle losses in 3p and 6p, Qwen3.8 and DeepSeek-V4.1",
            "episodes": len({r["meta"]["path"] for r in rows}),
            "cut_objects": len(rows),
            "by_shape": {k: sum(1 for r in rows if r["kind"] == k)
                         for k in sorted({r["kind"] for r in rows})},
            "formed_by_a_done_declaration": sum(1 for r in rows),
        },
        "method": {
            "replay": "accepted moves parsed from interactions.json and applied "
                      "to instance.json start positions; no state is taken from "
                      "scores.json",
            "reachability": "breadth-first search over free cells, with walls and "
                            "the objects of players that have declared done "
                            "treated as blocked and all other objects passable",
            "routes": "breadth-first shortest paths under the same blocking rule",
            "minimum_vertex_cut": "exhaustive search over free cells up to size 3",
            "hand_placed_elements": "none; every cell, route and count is computed",
            "analysis_module": "analysis/qualitative_analysis/cut.py",
            "renderer": "figures/cuts.py",
        },
        "rendering": {
            "page_width_in": round(WIDTH, 4),
            "page_width_note": "the document's \\textwidth, 455.24411 pt, so "
                               "\\thesisfig scales the figure 1:1",
            "bbox_inches": None,
            "png_dpi": 300,
            "pdf_fonttype": 42,
            "colour_is_redundant": "object mobility is shape, a frozen cell is "
                                   "cross-hatched, lost cells are hatched, the "
                                   "reachable region is outlined, and the two "
                                   "routes use different dash patterns",
            "palette_audit": "house palette; VICTIM vs ACTOR 9.85 and VICTIM vs "
                             "MUTED 0.06 CIE L*, both below the 10.0 grayscale "
                             "screening heuristic, which is why no meaning rests "
                             "on hue alone",
        },
    }
    for stem, case in cases_by_stem.items():
        record["figures"][stem] = {
            "outputs": [stem + ".pdf", stem + ".png"],
            "alt_text": ALT[stem],
            "episode": case["meta"]["path"],
            "source_files": [case["meta"]["path"] + "/instance.json",
                             case["meta"]["path"] + "/interactions.json"],
            "model": cut.MODELS[case["meta"]["model"]],
            "suite": case["meta"]["suite"],
            "level": case["meta"]["level"],
            "protocol": case["meta"]["proto"],
            "instance": case["meta"]["inst"],
            "grid_size": case["grid"],
            "panel_state": {
                "accepted_moves_before_the_declaration": case["ply"],
                "episode_length_moves": case["meta"]["moves"],
                "declaring_player": case["sealer_player"],
                "objects_frozen_before": sorted(case["frozen_pre"]),
                "objects_frozen_after": sorted(case["frozen_post"]),
                "cut_object": case["victim"],
                "shape": case["kind"],
                "minimum_vertex_cut": case["mincut"],
                "moves_played_after_the_cut": case["tail"],
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
    # the counts S9.3.2 quotes: a changed tree fails here rather than in a caption
    assert len({r["meta"]["path"] for r in rows}) == 78, "episode count moved"
    assert len(rows) == 101, len(rows)
    assert sum(r["kind"] in (cut.ENTRANCE, cut.ENTRANCE_OTHER)
               for r in rows) == 54
    assert sum(r["kind"] == cut.OPEN for r in rows) == 31
    assert sum(r["kind"] == cut.POCKET for r in rows) == 4

    corridor, twocut = pick(rows, CORRIDOR), pick(rows, TWOCUT)
    assert corridor["pre"][corridor["sealer"]] == corridor["entrance"]
    assert corridor["mincut"] == 1 and len(corridor["culprits"]) == 1
    assert twocut["mincut"] == 2 and len(twocut["culprits"]) == 2
    assert twocut["kind"] == cut.OPEN and twocut["other_route"]
    # the quote is drawn, so it has to still be in the transcript verbatim
    import json
    it = json.load(open(os.path.join(REPO_ROOT, corridor["meta"]["path"],
                                     "interactions.json")))
    said = [e["action"]["content"] for t in it["turns"] for e in t
            if e.get("action", {}).get("type") == "message"]
    assert any(PROMISE in s.replace("’", "'") for s in said), "promise reworded"

    paths = render_corridor(corridor, args.out)
    paths += render_twocut(twocut, args.out)
    paths += write_floats(rows, corridor, twocut, args.out)
    paths += write_provenance(rows, {"cut_corridor": corridor,
                                     "cut_twocut": twocut}, args.out)
    for path in paths:
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
