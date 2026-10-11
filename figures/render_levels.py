"""Draws the spatial-level figures.

Four kinds, deliberately kept apart so no single image carries more than one
job. Every figure is titled by its level alone.

  spatial_level_S<n>          the board: start cells, target cells, walls
  spatial_level_S<n>_reads    the listing a player is given
  spatial_level_S<n>_infers   what a player has to work out from that listing
  masking_dyad                ground truth beside each player's masked view
  renderings                  the masking board, compact / grid / drawn (F1)
  board_model                 the masking board drawn, beside Player A's grid

    python3 figures/render_levels.py                    # all of the above
    python3 figures/render_levels.py --list S3          # candidate boards
    python3 figures/render_levels.py --pick S3=7        # override the board
    python3 figures/render_levels.py --masking-level S4 # board for the masking figure
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import Dict, List, Optional, Sequence, Set, Tuple

from spatial_levels import (  # noqa: E402
    LEVELS,
    OUT_DIR,
    cellname,
    describe,
    free_path,
    load_instances,
    manhattan,
    pick_representative,
)
from style import (  # noqa: E402
    FAINT,
    GRID,
    INK,
    LARR,
    MONO,
    MUTED,
    OBJECT,
    OWNER,
    RARR,
    board_legend,
    draw_grid,
    draw_objects,
    plt,
    save,
    wrap_pairs,
    wrap_words,
)

ROLE = {
    "S1": "S1 measures everything that is not planning. The rendering, the "
          "response grammar, the reprompting machinery and, in the cooperative "
          "games, the masking and the channel all run over a puzzle whose "
          "spatial content is trivial, so a failure here is a failure of "
          "format or rule compliance and cannot be charged to spatial "
          "reasoning.",
    "S2": "S2 adds the two elementary forms of structure the upper levels "
          "compound: walls that lengthen paths, and a dependency that orders "
          "them. The chain fixes which target must be vacated before which, "
          "but leaves the model free to choose where the displaced object "
          "parks.",
    "S3": "S3 removes that discretion. Because the corridor entrance is its "
          "only access, the geometry rather than the model's judgement fixes "
          "the order in which targets can be occupied, and a second chain "
          "outside the corridor must be interleaved with the corridor plan.",
    "S4": "S4 isolates the one demand the lower levels never impose: a correct "
          "plan whose prefix moves an object away from its own target. It is "
          "the adversarial case for a stepwise-greedy policy.",
}

# ------------------------------------------------------------------- helpers


def median_detour(spec: dict) -> Optional[float]:
    instance = spec["instance"]
    walls = {tuple(w) for w in instance["walls"]}
    if not walls:
        return None
    ratios = []
    for obj in instance["objects"]:
        start = tuple(instance["start_positions"][obj])
        goal = tuple(instance["target_positions"][obj])
        base = manhattan(start, goal)
        path = free_path(start, goal, walls, instance["grid_size"])
        if base and path:
            ratios.append((len(path) - 1) / base)
    return statistics.median(ratios) if ratios else None


def compact_listing(
    instance: dict, visible: Optional[Set[str]] = None
) -> Tuple[List[str], List[str], List[str]]:
    """The three listing lines, mirroring Board.render_compact.

    Objects come in letter order with masked ones replaced by X; targets cover
    the visible objects only, since a player cannot move a teammate's object.
    """
    start = {k: tuple(v) for k, v in instance["start_positions"].items()}
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    objects = [
        f"{obj if (visible is None or obj in visible) else 'X'}@{cellname(start[obj])}"
        for obj in sorted(start)
    ]
    walls = [cellname(tuple(w)) for w in sorted(tuple(w) for w in instance["walls"])]
    targets = [
        f"{obj}->{cellname(target[obj])}"
        for obj in sorted(target)
        if visible is None or obj in visible
    ]
    return objects, walls, targets


# ------------------------------------------------------------- text layout


class Layout:
    """Collects rows of text, then renders them into a figure sized to fit.

    Content length varies a lot across the ladder, so height is measured from
    what was added rather than guessed up front; a fixed height silently
    truncated the longest level.
    """

    WIDTH = 4.6
    LINE = 0.166
    TOP = 0.62
    BOTTOM = 0.30
    LEFT_IN = 0.30

    def __init__(self) -> None:
        self.ops: List[dict] = []

    def text(self, body="", *, size=7.5, colour=INK, mono=False,
             weight="normal", indent=0.0, height=1.0) -> None:
        self.ops.append(dict(kind="text", text=body, size=size, colour=colour,
                             mono=mono, weight=weight, indent=indent,
                             height=height * self.LINE))

    def gap(self, amount: float) -> None:
        self.ops.append(dict(kind="gap", height=amount * self.LINE))

    def rule(self) -> None:
        self.ops.append(dict(kind="rule", height=0.62 * self.LINE))

    def heading(self, body: str) -> None:
        self.text(body, size=7.2, colour=MUTED, weight="bold", height=0.6)
        self.rule()
        self.gap(0.35)

    def paragraph(self, body: str, *, size=7.0, colour=INK,
                  width=94) -> None:
        for row in wrap_words(body, width):
            self.text(row, size=size, colour=colour, height=0.94)

    def bullet(self, body: str, *, size=7.4, width=88) -> None:
        rows = wrap_words(body, width)
        self.text(f"·   {rows[0]}", size=size)
        for row in rows[1:]:
            self.text(row, size=size, indent=0.028, height=0.94)

    def render(self, level: str, stem: str, out_dir: str,
               formats: Sequence[str]) -> List[str]:
        height = self.TOP + sum(op["height"] for op in self.ops) + self.BOTTOM
        fig = plt.figure(figsize=(self.WIDTH, height))
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        left = self.LEFT_IN / self.WIDTH
        right = 1 - left

        fig.text(left, 1 - 0.30 / height, level, fontsize=15.0, color=OBJECT,
                 fontweight="bold", va="baseline")

        cursor = height - self.TOP
        for op in self.ops:
            cursor -= op["height"]
            y = cursor / height
            if op["kind"] == "gap":
                continue
            if op["kind"] == "rule":
                ax.plot([left, right], [y, y], transform=ax.transAxes,
                        color=GRID, linewidth=0.6)
                continue
            ax.text(left + op["indent"], y, op["text"],
                    transform=ax.transAxes, fontsize=op["size"],
                    color=op["colour"], va="baseline", ha="left",
                    fontweight=op["weight"], **(MONO if op["mono"] else {}))

        return save(fig, out_dir, stem, formats)


# ------------------------------------------------------- figure 1: the board


def render_board(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """The board and nothing else: start cells, target cells, walls."""
    instance = spec["instance"]
    level = spec["level"]

    fig = plt.figure(figsize=(4.7, 5.15))
    ax = fig.add_axes([0.115, 0.135, 0.845, 0.750])
    draw_grid(ax, instance["grid_size"], instance["walls"])
    draw_objects(ax, instance)

    fig.text(0.085, 0.945, level, fontsize=15.0, color=OBJECT,
             fontweight="bold", va="baseline")

    board_legend(fig, 0.048, walls=bool(instance["walls"]))
    fig.text(
        0.085, 0.018,
        f"instance {instance.get('game_id')}   ·   walls "
        f"{len(instance['walls'])}   ·   optimal_moves "
        f"{instance['optimal_moves']}",
        fontsize=6.2, color=FAINT, va="baseline",
    )
    return save(fig, out_dir, f"spatial_level_{level}", formats)


# ------------------------------------------------- figure 2: what is read


def render_reads(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """The listing a player is given, and nothing more."""
    instance = spec["instance"]
    level = spec["level"]
    objects, walls, targets = compact_listing(instance)

    lay = Layout()
    lay.heading("WHAT A PLAYER READS")

    for i, chunk in enumerate(wrap_pairs(objects, 3)):
        lay.text(f"{'Objects:' if i == 0 else '        '}  {chunk}", mono=True)
    for i, chunk in enumerate(wrap_pairs(walls, 6) or ["(none)"]):
        lay.text(f"{'Walls:  ' if i == 0 else '        '}  {chunk}", mono=True)
    for i, chunk in enumerate(wrap_pairs(targets, 3)):
        lay.text(f"{'Targets:' if i == 0 else '        '}  {chunk}", mono=True)

    lay.gap(0.45)
    lay.paragraph(
        "That listing is the whole of a player's view. There is no grid and no "
        "picture. Under masking a player reads only its own objects, a "
        "teammate's object arrives as X, and a teammate's targets are left "
        "out entirely.",
        colour=MUTED,
    )
    return lay.render(level, f"spatial_level_{level}_reads", out_dir, formats)


# --------------------------------------------- figure 3: what is inferred


def render_infers(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """What a player has to work out from the listing alone."""
    level = spec["level"]

    facts: List[str] = []
    if spec["extent"]:
        mouth, dead_end = spec["extent"]
        entrance = tuple(spec["device"]["entrance"])
        facts.append(
            f"that {cellname(mouth)}–{cellname(dead_end)} is a dead-end "
            f"corridor, reachable only through {cellname(entrance)}"
        )
    if spec["swap"]:
        outer, inner = spec["swap"]
        entrance = cellname(tuple(spec["device"]["entrance"]))
        facts.append(
            f"that {outer} is aiming for the cell {inner} occupies, so {outer} "
            f"must retreat clear of {entrance} and give up ground before "
            f"{inner} can leave"
        )
    for chain in spec["plain_chains"]:
        facts.append(
            f"that {f' {LARR} '.join(chain)} is a dependency chain, resolved "
            f"in the order {f' {RARR} '.join(reversed(chain))}"
        )
    ratio = median_detour(spec)
    if ratio is not None:
        facts.append(
            f"that the walls put a median detour of {ratio:.2f}× the direct "
            "distance on the six objects"
        )
    if not facts:
        facts.append(
            "nothing beyond the listing itself: the board is open and every "
            "target is directly reachable"
        )

    lay = Layout()
    lay.heading("WHAT A PLAYER MUST INFER")
    for i, fact in enumerate(facts):
        if i:
            lay.gap(0.25)
        lay.bullet(fact)

    lay.gap(0.7)
    lay.rule()
    lay.gap(0.3)
    lay.paragraph(ROLE[level])
    return lay.render(level, f"spatial_level_{level}_infers", out_dir, formats)


# ------------------------------------------------------- figure 4: masking


def render_masking(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """Ground truth beside what each player of the dyad can actually see."""
    instance = spec["instance"]
    level = spec["level"]
    own = {
        "A": set(instance["player_a_objects"]),
        "B": set(instance["player_b_objects"]),
    }
    by_owner = {
        obj: OWNER["A" if obj in own["A"] else "B"] for obj in instance["objects"]
    }

    fig = plt.figure(figsize=(7.5, 4.9))
    columns = [
        (0.045, None, "The board as it is", "blue Player A, orange Player B"),
        (0.360, "A", "What Player A reads",
         f"owns {', '.join(sorted(own['A']))}"),
        (0.675, "B", "What Player B reads",
         f"owns {', '.join(sorted(own['B']))}"),
    ]

    for x, viewer, title, note in columns:
        ax = fig.add_axes([x, 0.345, 0.278, 0.395])
        draw_grid(ax, instance["grid_size"], instance["walls"], labels=4.9)
        visible = own[viewer] if viewer else None
        colours = {o: OWNER[viewer] for o in own[viewer]} if viewer else by_owner
        draw_objects(ax, instance, visible=visible, colours=colours,
                     radius=0.30, letter=6.6)

        fig.text(x, 0.800, title, fontsize=9.0, color=INK, fontweight="bold",
                 va="baseline")
        fig.text(x, 0.773, note, fontsize=6.8, color=MUTED, va="baseline")

        objects, _walls, targets = compact_listing(instance, visible)
        rows = wrap_pairs(objects, 2)
        ty = 0.263
        for i, chunk in enumerate(rows):
            fig.text(x, ty - i * 0.029,
                     f"{'Objects:' if i == 0 else '        '} {chunk}",
                     fontsize=5.9, color=INK, va="baseline", **MONO)
        ty -= 0.029 * len(rows) + 0.014
        for i, chunk in enumerate(wrap_pairs(targets, 2)):
            fig.text(x, ty - i * 0.029,
                     f"{'Targets:' if i == 0 else '        '} {chunk}",
                     fontsize=5.9, color=INK, va="baseline", **MONO)

    fig.text(0.045, 0.950, "Masking in the dyad", fontsize=12.5, color=INK,
             fontweight="bold", va="baseline")
    for i, row in enumerate(wrap_words(
        "The same board, three views. Each player holds a complete "
        "specification of its own three objects and no direct knowledge of "
        "its teammate's. A teammate's object shows as X, enough to judge "
        "whether a move is legal but not to tell which object it is or who "
        "controls it.", 108
    )):
        fig.text(0.045, 0.912 - i * 0.026, row, fontsize=7.4, color=MUTED,
                 va="baseline")

    # The ground-truth column carries three target lines, the widest case.
    board_legend(fig, 0.036, x=0.045, masked=True,
                 walls=bool(instance["walls"]))
    fig.text(0.045, 0.014,
             f"{level} · instance {instance.get('game_id')} · the wall list is "
             "identical in all three views and is omitted here for space",
             fontsize=6.0, color=FAINT, va="baseline")

    return save(fig, out_dir, "masking_dyad", formats)


# ------------------------------------------------ figure F1: the renderings


def render_renderings(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """One board as Player A of the dyad receives it: compact, grid, drawn.

    The two text panels come from the game's own Board class, so they are
    what the model is sent, not a transcription of it. The board is the
    masking figure's, so the two cannot show different boards.
    """
    import sys
    import textwrap

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matrixgame_covered_comm.utils.board import Board

    instance = spec["instance"]
    own = set(instance["player_a_objects"])
    board = Board(instance["grid_size"], {tuple(w) for w in instance["walls"]})
    for obj, (r, c) in instance["start_positions"].items():
        board.place_object(obj, r, c)
    for obj, (r, c) in instance["target_positions"].items():
        board.set_target(obj, r, c)

    compact = "\n".join(
        textwrap.fill(line, 30, subsequent_indent="  ")
        for line in ("CURRENT BOARD:\n" + board.render_compact(own) + "\n\n"
                     "GOAL BOARD:\n" + board.render_targets_compact(own)).split("\n"))
    grid = "CURRENT BOARD:\n" + board.render_for_player(own)

    fig = plt.figure(figsize=(7.5, 3.3))
    panels = [(0.020, "Compact", "what the compact games send"),
              (0.300, "Grid", "what the grid games send"),
              (0.685, "Drawn", "the same view, for the reader")]
    for x, title, note in panels:
        fig.text(x, 0.905, title, fontsize=9.0, color=INK, fontweight="bold",
                 va="baseline")
        fig.text(x, 0.868, note, fontsize=6.8, color=MUTED, va="baseline")
    fig.text(0.020, 0.815, compact, fontsize=7.0, color=INK, va="top", **MONO)
    fig.text(0.300, 0.815, grid, fontsize=6.2, color=INK, va="top",
             linespacing=1.05, **MONO)
    ax = fig.add_axes([0.685, 0.09, 0.30, 0.73])
    draw_grid(ax, instance["grid_size"], instance["walls"], labels=4.9)
    draw_objects(ax, instance, visible=own, colours={o: OWNER["A"] for o in own},
                 radius=0.30, letter=6.6)
    fig.text(0.020, 0.025,
             f"{spec['level']} · instance {instance.get('game_id')} · Player A, "
             f"owns {', '.join(sorted(own))} · the grid games send a GOAL BOARD "
             "grid as well, omitted here",
             fontsize=6.0, color=FAINT, va="baseline")
    return save(fig, out_dir, "renderings", formats)


# ------------------------------------------------ figure: the physical model


def render_board_model(spec: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    """The board drawn for the reader beside the grid Player A of the dyad is sent.

    The grid rows come from the game's own Board class. Its header centres the
    labels in 4-char columns over 5-char cells, so they drift one char per
    column; here they are re-centred over their cells, and the caption says so.
    Same board as the masking and renderings figures.
    """
    import sys

    from matplotlib.transforms import blended_transform_factory

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from matrixgame_covered_comm.utils.board import Board

    instance = spec["instance"]
    size = instance["grid_size"]
    own = set(instance["player_a_objects"])
    board = Board(size, {tuple(w) for w in instance["walls"]})
    for obj, (r, c) in instance["start_positions"].items():
        board.place_object(obj, r, c)
    lines = board.render_for_player(own).split("\n")
    lines[0] = " " * 6 + "".join(f"{'C' + str(c + 1):^4} " for c in range(size))
    assert all(lines[0].index(f"C{c + 1}") == 7 + 5 * c for c in range(size))
    assert all(lines[2 + 2 * r].startswith(f"R{r + 1}   |") for r in range(size))

    fig = plt.figure(figsize=(6.3, 3.9))
    bottom, height = 0.215, 0.64
    ax = fig.add_axes([0.055, bottom, height * 3.9 / 6.3, height])
    draw_grid(ax, size, instance["walls"], labels=6.0)
    draw_objects(ax, instance, radius=0.30, letter=6.8)

    # the text rows share the drawn board's row coordinates, so R3 lines up with R3
    tx = fig.add_axes([0.53, bottom, 0.45, height])
    tx.set_xlim(0, 1)
    tx.set_ylim(size, 0)
    tx.axis("off")
    blend = blended_transform_factory(tx.transAxes, tx.transData)
    tx.text(0, -0.30, lines[0], transform=blend, fontsize=6.4, color=MUTED,
            va="center", **MONO)
    for k, line in enumerate(lines[1:]):
        tx.text(0, k / 2, line, transform=blend, fontsize=6.4, va="center",
                color=FAINT if k % 2 == 0 else INK, **MONO)

    for x, title in [(0.055, "The board"), (0.53, "What Player A is sent")]:
        fig.text(x, 0.935, title, fontsize=8.5, color=INK, fontweight="bold",
                 va="baseline")
    symbols = [(".", "empty cell"),
               (min(own), f"Player A's own object ({', '.join(sorted(own))})"),
               ("X", "a teammate's object"), ("##", "wall")]
    assert all(f" {m:^2} " in "\n".join(lines[2::2]) for m, _ in symbols)
    fig.text(0.53, 0.175, "\n".join(f"{m:<4}{label}" for m, label in symbols),
             fontsize=6.2, color=INK, va="top", linespacing=1.45, **MONO)
    board_legend(fig, 0.055, x=0.04)
    fig.text(0.055, 0.018, f"{spec['level']} · instance {instance.get('game_id')}",
             fontsize=5.8, color=FAINT, va="baseline")
    return save(fig, out_dir, "board_model", formats)


# -------------------------------------------------- figure 5: contact sheet


def render_contact_sheet(specs: Dict[str, dict], out_dir: str,
                         formats=("pdf", "png")) -> List[str]:
    fig = plt.figure(figsize=(7.0, 7.6))
    positions = {
        "S1": [0.055, 0.520, 0.385, 0.335],
        "S2": [0.560, 0.520, 0.385, 0.335],
        "S3": [0.055, 0.055, 0.385, 0.335],
        "S4": [0.560, 0.055, 0.385, 0.335],
    }
    for level, rect in positions.items():
        instance = specs[level]["instance"]
        ax = fig.add_axes(rect)
        draw_grid(ax, instance["grid_size"], instance["walls"], labels=5.2)
        draw_objects(ax, instance, radius=0.29, letter=6.6)
        top = rect[1] + rect[3]
        fig.text(rect[0], top + 0.048, level, fontsize=11.5, color=OBJECT,
                 fontweight="bold", va="baseline")
        fig.text(rect[0], top + 0.026,
                 f"walls {len(instance['walls'])}   ·   optimal_moves "
                 f"{instance['optimal_moves']}",
                 fontsize=6.0, color=MUTED, va="baseline")

    fig.text(0.055, 0.960, "The spatial ladder", fontsize=12.5, color=INK,
             fontweight="bold", va="baseline")
    fig.text(0.055, 0.938,
             "one representative board per level, S1 through S4",
             fontsize=7.0, color=MUTED, va="baseline")

    return save(fig, out_dir, "spatial_ladder", formats)


# ----------------------------------------------------------------------- cli


def resolve_picks(raw: Optional[Sequence[str]]) -> Dict[str, str]:
    picks = {}
    for item in raw or []:
        level, _, game_id = item.partition("=")
        if level not in LEVELS or not game_id:
            raise SystemExit(f"--pick expects LEVEL=game_id, got {item!r}")
        picks[level] = game_id
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--list", metavar="LEVEL",
                        help="print candidate boards for a level and exit")
    parser.add_argument("--pick", action="append", metavar="LEVEL=game_id",
                        help="use a specific board instead of the median one")
    parser.add_argument("--masking-level", default="S2", choices=list(LEVELS),
                        help="which level's board the masking figure uses")
    parser.add_argument("--levels", nargs="*", default=list(LEVELS))
    args = parser.parse_args()

    released = load_instances()

    if args.list:
        level = args.list
        if level not in released:
            raise SystemExit(f"unknown level {level!r}")
        chosen = pick_representative(released[level])
        print(f"{level}: {len(released[level])} distinct boards")
        for inst in sorted(released[level], key=lambda i: i["optimal_moves"]):
            ratio = median_detour(describe(inst))
            row = (f"  game_id {str(inst.get('game_id')):>3}   "
                   f"optimal_moves {inst['optimal_moves']:>3}   "
                   f"walls {len(inst['walls']):>3}")
            if ratio is not None:
                row += f"   median detour {ratio:.2f}x"
            print(row + ("   <- chosen" if inst is chosen else ""))
        return

    picks = resolve_picks(args.pick)
    specs: Dict[str, dict] = {}
    for level in sorted(set(args.levels) | {args.masking_level}):
        candidates = released[level]
        if level in picks:
            match = [c for c in candidates
                     if str(c.get("game_id")) == picks[level]]
            if not match:
                raise SystemExit(f"{level}: no board with game_id {picks[level]!r}")
            instance = match[0]
        else:
            instance = pick_representative(candidates)
        specs[level] = describe(instance)

    for level in args.levels:
        for render in (render_board, render_reads, render_infers):
            for path in render(specs[level], args.out):
                print(f"wrote {os.path.relpath(path)}")

    for render in (render_masking, render_renderings, render_board_model):
        for path in render(specs[args.masking_level], args.out):
            print(f"wrote {os.path.relpath(path)}")

    if set(LEVELS) <= set(specs):
        for path in render_contact_sheet(specs, args.out):
            print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
