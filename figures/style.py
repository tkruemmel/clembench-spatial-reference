"""The visual kit every figure in this directory draws with.

Palette, matplotlib defaults and the board primitives live here so that the
level figures and the episode figures cannot drift apart. A reader who learns
the vocabulary on one figure reads the others for free, which only holds if
there is exactly one definition of each mark.

Nothing in here knows about a particular board or a particular episode.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb, to_rgba  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

Cell = Tuple[int, int]

# STIX math renders these; most text serifs carry no arrow glyphs.
LARR = r"$\leftarrow$"
RARR = r"$\rightarrow$"

# ------------------------------------------------------------------- palette

INK = "#22272E"
MUTED = "#69747F"
FAINT = "#98A2AD"
GRID = "#DEE3E9"
FRAME = "#A7B0BA"
WALL = "#4A5361"
WALL_HI = "#5C6675"
OBJECT = "#1C5C9E"  # every object, on the plain level boards
OWNER = {"A": "#1C5C9E", "B": "#B45A15"}
# The dyad's seats by their game names. Blue-green rather than green, so Green
# stays apart from Orange under red-green colour blindness.
SEAT = {"Green": "#0F7B5F", "Orange": "#B45A15"}
HIDDEN = "#96A0AB"  # a teammate's object, seen only as X
MOVE = "#2E7D46"  # the move played this turn
CUE = "#B8860B"  # what the caption wants the eye to land on
PAPER = "#FFFFFF"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,  # embed TrueType, so the PDF travels without the font
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # A list here rather than "serif" so matplotlib falls back per glyph:
        # quoted model text can contain characters Charter has no glyph for
        # (arrows, dashes), and without fallback those render as blanks.
        # STIX Two Text is deliberately not in this chain: it ships no bold, so
        # every bold draw would probe it and log a findfont miss.
        "font.family": ["Charter", "DejaVu Serif"],
        "font.serif": ["Charter", "STIX Two Text", "DejaVu Serif"],
        "font.monospace": ["DejaVu Sans Mono"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)

MONO = {"family": "monospace"}


# ------------------------------------------------------------------- helpers


def centre(cell: Cell) -> Tuple[float, float]:
    return cell[1] + 0.5, cell[0] + 0.5


def wrap_pairs(items: Sequence[str], per_line: int) -> List[str]:
    return ["  ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)]


def wrap_words(text: str, width: int) -> List[str]:
    lines, row = [], ""
    for word in text.split():
        if len(row) + len(word) + 1 > width:
            lines.append(row)
            row = word
        else:
            row = f"{row} {word}".strip()
    if row:
        lines.append(row)
    return lines


def wrap_mono(text: str, width: int, indent: str = "") -> List[str]:
    """Wrap one logical line, continuing under a hanging indent.

    Used for the listing lines, whose wraps have to look like continuations of
    the same line rather than new fields.
    """
    lines, row = [], ""
    for word in text.split(" "):
        # no strip(): it would eat the hanging indent as soon as a second word
        # joined the continuation row
        candidate = f"{row} {word}" if row else word
        if row and len(candidate) > width:
            lines.append(row)
            row = indent + word
        else:
            row = candidate
    if row:
        lines.append(row)
    return lines


def save(fig, out_dir: str, stem: str, formats: Sequence[str]) -> List[str]:
    written = []
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None, facecolor=PAPER)
        written.append(path)
    plt.close(fig)
    return written


# ----------------------------------------------------------------- the board


def draw_grid(ax, size: int, walls: Iterable[Cell], *, labels: float = 6.2) -> None:
    """The empty board: lattice, frame, wall cells, coordinate labels."""
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)  # row 1 at the top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(size + 1):
        ax.plot([i, i], [0, size], color=GRID, linewidth=0.55, zorder=1)
        ax.plot([0, size], [i, i], color=GRID, linewidth=0.55, zorder=1)
    ax.add_patch(
        Rectangle((0, 0), size, size, facecolor="none", edgecolor=FRAME,
                  linewidth=0.9, zorder=5)
    )
    for cell in {tuple(w) for w in walls}:
        ax.add_patch(
            Rectangle((cell[1], cell[0]), 1, 1, facecolor=WALL,
                      edgecolor=WALL_HI, linewidth=0.4, zorder=2)
        )
    if not labels:
        return
    for c in range(size):
        ax.text(c + 0.5, -0.30, f"C{c + 1}", ha="center", va="center",
                fontsize=labels, color=MUTED)
    for r in range(size):
        ax.text(-0.32, r + 0.5, f"R{r + 1}", ha="center", va="center",
                fontsize=labels, color=MUTED)


def draw_target(ax, cell: Cell, label: str, colour: str, *, nested: bool,
                letter: float = 8.0, corner: Optional[float] = None,
                lw: float = 1.1,
                dashes: Tuple[float, float] = (2.2, 1.5)) -> None:
    """A target cell: dashed rounded outline, tinted unless something sits on it.

    `nested` means the cell is currently occupied by a different object, so the
    outline shrinks and its letter moves to the corner rather than fighting the
    disc for the middle.
    """
    inset = 0.055 if nested else 0.15
    # alpha on the patch would dim the dashed edge too, so tint the fill
    ax.add_patch(
        FancyBboxPatch(
            (cell[1] + inset, cell[0] + inset), 1 - 2 * inset, 1 - 2 * inset,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor="none" if nested else to_rgba(colour, 0.10),
            edgecolor=colour, linewidth=lw,
            linestyle=(0, dashes), zorder=6,
        )
    )
    if nested:
        ax.text(cell[1] + 0.145, cell[0] + 0.175, label, ha="center",
                va="center", fontsize=corner or letter * 0.66, color=colour, zorder=8,
                fontweight="bold",
                # a white halo breaks the dashed outline around the letter
                path_effects=[withStroke(linewidth=1.8, foreground=PAPER)])
    else:
        ax.text(*centre(cell), label, ha="center", va="center",
                fontsize=letter * 0.92, color=colour, zorder=7,
                fontweight="bold")


def draw_disc(ax, cell: Cell, label: str, colour: str, *, radius: float = 0.285,
              letter: float = 8.0, ring: Optional[str] = None,
              ring_lw: float = 1.7) -> None:
    """An object where it currently stands. `ring` marks it for the caption."""
    ax.add_patch(
        Circle(centre(cell), radius, facecolor=colour,
               edgecolor=ring or PAPER, linewidth=ring_lw if ring else 0.8,
               zorder=7.5)
    )
    ax.text(*centre(cell), label, ha="center", va="center", fontsize=letter,
            color=PAPER, fontweight="bold", zorder=8)


def draw_objects(ax, instance: dict, *, visible: Optional[Set[str]] = None,
                 colours: Optional[Dict[str, str]] = None,
                 radius: float = 0.285, letter: float = 8.0,
                 corner: Optional[float] = None) -> None:
    """Start cells as filled discs, target cells as dashed outlines.

    A letter appearing in both a disc and an outline is the only link drawn
    between an object's start and its target. Nothing else is overlaid.
    """
    start = {k: tuple(v) for k, v in instance["start_positions"].items()}
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    occupied = {cell: obj for obj, cell in start.items()}

    def colour(obj: str) -> str:
        return OBJECT if colours is None else colours.get(obj, OBJECT)

    for obj, cell in sorted(target.items()):
        if visible is not None and obj not in visible:
            continue  # a teammate's target is omitted, not masked
        blocking = occupied.get(cell)
        draw_target(ax, cell, obj, colour(obj),
                    nested=bool(blocking) and blocking != obj, letter=letter,
                    corner=corner)

    for obj, cell in sorted(start.items()):
        shown = visible is None or obj in visible
        draw_disc(ax, cell, obj if shown else "X",
                  colour(obj) if shown else HIDDEN, radius=radius, letter=letter)


def draw_arrow(ax, src: Cell, dst: Cell, colour: str, *, rad: float = 0.0,
               lw: float = 1.5, scale: float = 9.5, zorder: float = 4.0,
               shrink: Tuple[float, float] = (11.0, 13.0),
               ls: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            centre(src), centre(dst), arrowstyle="-|>", mutation_scale=scale,
            linewidth=lw, color=colour, zorder=zorder, linestyle=ls,
            shrinkA=shrink[0], shrinkB=shrink[1],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def clear_bend(src: Cell, dst: Cell, avoid: Iterable[Cell],
               magnitude: float = 0.27) -> float:
    """Pick the arc side whose midpoint sits furthest from everything else.

    Straight arrows collide with discs and walls on a crowded board, and a
    hand-tuned bend per object does not survive changing the instance. Both
    candidate arcs are scored on clearance and the roomier one wins; ties break
    positive so the choice is deterministic.
    """
    (x0, y0), (x1, y1) = centre(src), centre(dst)
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    dx, dy = x1 - x0, y1 - y0
    others = [centre(c) for c in avoid if c != src and c != dst]
    if not others:
        return magnitude

    def clearance(sign: float) -> float:
        # matplotlib's arc3 offsets the chord midpoint by rad/2 perpendicular
        px, py = mx - sign * magnitude * dy / 2.0, my + sign * magnitude * dx / 2.0
        return min((px - ox) ** 2 + (py - oy) ** 2 for ox, oy in others)

    return magnitude if clearance(1.0) >= clearance(-1.0) else -magnitude


def board_legend(fig, y: float, x: float = 0.085, *, masked: bool = False,
                 walls: bool = True, size: float = 7.0) -> None:
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=6.0,
               markerfacecolor=OBJECT, markeredgecolor=PAPER,
               label="start cell"),
        Line2D([], [], marker="s", linestyle="none", markersize=6.2,
               markerfacecolor="none", markeredgecolor=OBJECT,
               label="target cell"),
    ]
    if walls:
        handles.append(
            Line2D([], [], marker="s", linestyle="none", markersize=6.2,
                   markerfacecolor=WALL, markeredgecolor=WALL_HI, label="wall")
        )
    if masked:
        handles.append(
            Line2D([], [], marker="o", linestyle="none", markersize=6.0,
                   markerfacecolor=HIDDEN, markeredgecolor=PAPER,
                   label="a teammate's object, unidentifiable")
        )
    fig.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(x, y),
        ncol=len(handles), frameon=False, fontsize=size, handletextpad=0.6,
        columnspacing=2.0,
    )


# ------------------------------------------------------- monospaced passages


def mono_advance(ax, size: float) -> float:
    """Width of one monospaced character, as a fraction of the axes width.

    Measured by differencing two runs: `get_text_width_height_descent` reports
    the ink box, which is narrower than the advance by the glyph's side
    bearings, and differencing cancels them. Requires a drawn canvas.
    """
    renderer = ax.figure.canvas.get_renderer()
    prop = FontProperties(family="monospace", size=size)
    long_run = renderer.get_text_width_height_descent("M" * 41, prop, False)[0]
    one = renderer.get_text_width_height_descent("M", prop, False)[0]
    return (long_run - one) / 40.0 / ax.get_window_extent().width


def find_span(lines: Sequence[str], needle: str) -> Tuple[int, int, int]:
    """Locate `needle` in a wrapped block, as (line index, start col, end col).

    Highlights are anchored to the text they mark rather than to hand-counted
    offsets, so rewrapping or a changed transcript cannot silently shift them.
    """
    for i, line in enumerate(lines):
        col = line.find(needle)
        if col >= 0:
            return i, col, col + len(needle)
    raise ValueError(f"{needle!r} does not occur in the rendered block")


# ------------------------------------------------- episode panels
#
# Placed in inches from the bottom-left rather than through a layout engine, so
# a figure arrives at the exact width of the thesis text block and its point
# sizes are the sizes on the page.

def square_axes(fig, left_in, bottom_in, side_in):
    w, h = fig.get_size_inches()
    return fig.add_axes([left_in / w, bottom_in / h, side_in / w, side_in / h])


def at(fig, x_in, y_in):
    w, h = fig.get_size_inches()
    return x_in / w, y_in / h


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
    r, g, b = to_rgb(colour)
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


OPEN_DASH = (0, (1.3, 1.5))    # a route that still exists


def route(ax, cells, colour, dash, *, lw=1.2):
    for src, dst in zip(cells, cells[1:]):
        draw_arrow(ax, src, dst, colour, lw=lw, scale=7.5, shrink=(4.0, 1.0),
                   ls=dash, zorder=3.5)


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


# Re-exported so figure modules import their drawing vocabulary from one place.
__all__ = [n for n in dir() if not n.startswith("_")]
