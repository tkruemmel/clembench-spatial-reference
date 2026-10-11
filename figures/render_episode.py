"""Draws the episode figures: what the game is, and what playing it looks like.

Three kinds, in the order a reader should meet them.

  two_views        the same board as each player is shown it, with an arrow
                   from every object it owns to where that object must go
  turn_companion   one turn verbatim, beside the board that listing decodes to
  episode_strip    the board after each of the opening moves, under the channel

All three are drawn from one recorded episode and one released instance, and
share the palette and board marks of the spatial-level figures, so a reader who
learns the vocabulary on one reads the rest for free.

    python3 figures/render_episode.py                        # all three
    python3 figures/render_episode.py --figures views        # just one
    python3 figures/render_episode.py --run PATH --turn 2 --moves 4
    python3 figures/render_episode.py --list                 # candidate episodes
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

from episodes import (  # noqa: E402
    DEFAULT_RUN,
    Cell,
    cellname,
    check_abridged,
    exchange_for,
    find_episodes,
    listing,
    load_episode,
    on_target,
    owner,
    replay,
    sides,
)
from spatial_levels import OUT_DIR, dependency_chains  # noqa: E402
from style import (  # noqa: E402
    CUE,
    FAINT,
    HIDDEN,
    INK,
    MONO,
    MOVE,
    MUTED,
    OWNER,
    PAPER,
    WALL,
    WALL_HI,
    Circle,
    FancyBboxPatch,
    Line2D,
    Rectangle,
    centre,
    clear_bend,
    draw_arrow,
    draw_disc,
    draw_grid,
    draw_target,
    find_span,
    mono_advance,
    plt,
    save,
    to_rgba,
    wrap_mono,
    wrap_words,
)

FIGURES = ("overview", "views", "turn", "strip")

# Curated abridgements of channel messages, for the default run. Each is
# checked against the transcript before it is drawn, so these may cut but can
# never reword. Any turn without an entry falls back to abridge().
ABRIDGED: Dict[int, str] = {
    1: "…My A target is (R5,C3), so please clear that cell for me; I can adapt "
       "once your X moves. PLAN: move D from (R2,C1) to (R1,C5) — REQUEST: "
       "clear R5C3 for A",
    2: "Moving B down to R6C3 frees R5C3 for your A. Please move X at R6C2 left "
       "to R6C1 so B can finish at R6C2… PLAN: move B from (R5,C3) to (R6,C2) — "
       "REQUEST: move X from (R6,C2) to (R6,C1)",
    3: "Moving C left to R6C1, freeing R6C2 for B as you asked… PLAN: move C "
       "from (R6,C2) to (R7,C7) — REQUEST: move B to (R6,C2); clear (R4,C4) for A",
    4: "Taking B into R6C2 now, as planned. I’ll move E right to R4C5 next, "
       "freeing R4C4 so A can pass… PLAN: move B from (R6,C3) to (R6,C2) — "
       "REQUEST: hold D off (R3,C2)-(R3,C4) until E passes",
}


def curated(episode: dict) -> Dict[int, str]:
    """The hand-cut quotations, which belong to one run and no other."""
    return ABRIDGED if is_default(episode) else {}


def is_default(episode: dict) -> bool:
    return os.path.realpath(episode["path"]) == os.path.realpath(DEFAULT_RUN)


def abridge(message: str, index: Optional[int] = None, *,
            table: Optional[Dict[int, str]] = None, limit: int = 240) -> str:
    """Shorten a channel message to its opening claim and its PLAN/REQUEST tail.

    A curated cut is used where one exists for this run; otherwise the message
    is trimmed mechanically. Either way the result is verified to be a cut of
    the recorded text and nothing else, so a figure can shorten a quotation but
    never reword one.
    """
    table = ABRIDGED if table is None else table
    if index is not None and index in table:
        return check_abridged(message, table[index])
    if len(message) <= limit:
        return message
    head = message.split(". ")[0].strip()
    tail = message[message.index("PLAN:"):].strip() if "PLAN:" in message else ""
    shown = f"{head}… {tail}".strip() if tail else f"{head}…"
    return check_abridged(message, shown)


# ------------------------------------------------------------- what is drawn


def blocked_targets(instance: dict, positions: Dict[str, Cell],
                    side: str) -> List[Tuple[str, str, Cell]]:
    """(my object, the teammate object in its way, the contested cell).

    This is the coordination problem made concrete, and every caption about
    masking on these figures is derived from it rather than asserted.
    """
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    mine = set(sides(instance)[side])
    occupant = {cell: obj for obj, cell in positions.items()}
    out = []
    for obj in sorted(mine):
        holder = occupant.get(target[obj])
        if holder is not None and holder not in mine:
            out.append((obj, holder, target[obj]))
    return out


def chain_caption(instance: dict) -> Optional[str]:
    """'A <- B <- C: A's target is B's cell, B's target is C's cell'."""
    chains = dependency_chains(instance["start_positions"],
                               instance["target_positions"])
    if not chains:
        return None
    chain = chains[0]
    arrow = r" $\leftarrow$ ".join(chain)
    pairs = ", ".join(f"{a}'s target is {b}'s cell"
                      for a, b in zip(chain, chain[1:]))
    return f"{arrow}:  {pairs}"


def cue_span(text: str, cell: Cell) -> str:
    """The phrase in a message that refers to `cell`, for highlighting.

    Prefers a full 'X at R6C2' style reference so the highlight reads as the
    referring expression it is, and falls back to the bare coordinate.
    """
    name = cellname(cell)
    match = re.search(rf"X\s+(?:at|from)\s+\(?{name}", text)
    return match.group(0) if match else name


# ------------------------------------------- figure 0: how the game works


def quote(message: str, budget: int = 300) -> str:
    """As much of a player's own message as fits, cut only at sentence joins.

    Leading sentences are taken verbatim until the budget runs out; where the
    message carries the hybrid protocol's PLAN/REQUEST clause, room is reserved
    so that clause always survives, since it is what makes the message a
    protocol message. Every cut is marked and then verified against the
    transcript, so this shortens but never paraphrases.
    """
    tail = message[message.index("PLAN:"):].strip() if "PLAN:" in message else ""
    body = (message[:message.index("PLAN:")] if tail else message).strip()
    room = budget - (len(tail) + 2 if tail else 0)

    kept: List[str] = []
    for sentence in re.split(r"(?<=\.)\s+", body):
        if kept and len(" ".join(kept)) + 1 + len(sentence) > room:
            break
        kept.append(sentence)
    shown = " ".join(kept).rstrip()
    if len(shown) < len(body):
        shown += " …" if shown.endswith(".") else "…"
    if tail:
        shown = f"{shown} {tail}" if shown.endswith("…") else f"{shown} … {tail}"
    return check_abridged(message, shown)


def overview_turns(episode: dict) -> List[int]:
    """The opening exchange: request, comply-and-counter-request, comply, land."""
    return [i for i in (1, 2, 3, 4) if i <= len(episode["moves"])]


def render_overview(episode: dict, out_dir: str, view: str = "B",
                    formats=("pdf", "png")) -> List[str]:
    """One figure that teaches the game: the goal, what a player actually sees,
    what the players actually said, and the board before and after."""
    instance = episode["instance"]
    size = instance["grid_size"]
    walls = [tuple(w) for w in instance["walls"]]
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    own = sides(instance)
    start = episode["start"]
    states = replay(episode)
    quoted = overview_turns(episode)

    WRAP, TXT = 104, 7.2
    LINE, PAD, BOARD, FW = 0.128, 0.150, 1.90, 7.1

    said = [(episode["moves"][i - 1],
             wrap_words(quote(episode["moves"][i - 1]["message"] or ""), WRAP))
            for i in quoted]
    intro = wrap_words(
        f"Player A moves {' '.join(own['A'])} and Player B moves "
        f"{' '.join(own['B'])}. Neither can move the other's objects, and "
        "neither can see them: a teammate's object shows only as X, as in the "
        "middle board below. So the only way to get one out of your way is to "
        "ask for it, in words, and hope the teammate agrees.", 118)

    # every band's height, so the figure is exactly as tall as its content
    talk = sum(len(lines) * LINE + PAD for _m, lines in said)
    bands = [0.34, 0.26, 0.30, 0.26, len(intro) * 0.155, 0.20, talk,
             0.34, 0.32, BOARD, 0.24, 0.24, 0.16]
    FH = sum(bands)

    def top(inches: float) -> float:
        return 1 - inches / FH

    fig = plt.figure(figsize=(FW, FH))
    L, RIGHT = 0.042, 0.958
    cur = 0.34

    fig.text(L, top(cur), "How the game works", fontsize=13.0, color=INK,
             fontweight="bold", va="baseline")
    cur += 0.26

    ax = fig.add_axes([L, top(cur + 0.30), RIGHT - L, 0.30 / FH])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc="#EEF1F4", ec="#D5DBE1", lw=0.7,
                                transform=ax.transAxes))
    ax.text(0.014, 0.54, "Goal", fontsize=8.0, color=INK, fontweight="bold",
            va="center", transform=ax.transAxes)
    ax.text(0.068, 0.54,
            "get every object onto its target cell, moving one object one "
            "square per turn.", fontsize=8.0, color=INK, va="center",
            transform=ax.transAxes)
    cur += 0.30 + 0.26

    for line in intro:
        fig.text(L, top(cur), line, fontsize=7.6, color=MUTED, va="baseline")
        cur += 0.155
    cur += 0.20

    # what the players actually said
    for move, lines in said:
        colour = OWNER[move["side"]]
        height = len(lines) * LINE + PAD
        ax = fig.add_axes([L, top(cur + height * 0.90), RIGHT - L,
                           height * 0.90 / FH])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=to_rgba(colour, 0.07),
                                    ec=to_rgba(colour, 0.30), lw=0.65,
                                    transform=ax.transAxes))
        step = 1.0 / (len(lines) + 0.35)
        anchor = 1 - step * 0.62
        chip_w = 0.17 / ((RIGHT - L) * FW)
        chip_h = 0.15 / (height * 0.90)
        ax.add_patch(FancyBboxPatch((0.009, anchor - chip_h / 2), chip_w, chip_h,
                                    boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=colour, ec="none", transform=ax.transAxes))
        ax.text(0.009 + chip_w / 2, anchor, move["side"], fontsize=7.2, color=PAPER,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        for j, line in enumerate(lines):
            ax.text(0.052, 1 - step * (j + 0.62), line, fontsize=TXT, color=INK,
                    va="center", transform=ax.transAxes)
        ax.text(0.995, anchor, f"move: {move['object']} {move['direction']}",
                fontsize=TXT, color=colour, fontweight="bold", ha="right",
                va="center", family="monospace", transform=ax.transAxes)
        cur += height

    cur += 0.24
    fig.text(0.5, top(cur),
             f"{len(episode['moves']) - len(quoted)} more moves, the same way",
             fontsize=6.9, color=FAINT, ha="center", va="baseline")
    cur += 0.10 + 0.32

    # the truth, one player's view of it, and the finish
    bw, bh = BOARD / FW, BOARD / FH
    pair, arrow = 0.30 / FW, 0.80 / FW
    x0 = (1 - (3 * bw + pair + arrow)) / 2
    xs = [x0, x0 + bw + pair, x0 + 2 * bw + pair + arrow]
    by = top(cur + BOARD)
    mine = set(own[view])

    panels = (
        (states[0], None, "Initial state",
         "dashed square = where that object must get to"),
        (start, view, f"…as Player {view} is shown it",
         f"{', '.join(own[view])} are Player {view}'s own; the rest are just X"),
        (states[-1], None, "Final state",
         f"all six home in {len(episode['moves'])} moves"),
    )
    for x, (positions, masked, label, note) in zip(xs, panels):
        ax = fig.add_axes([x, by, bw, bh])
        draw_grid(ax, size, walls, labels=4.4)
        occupant = {cell: o for o, cell in positions.items()}
        if label != "Final state":
            for obj in (sorted(mine) if masked else sorted(target)):
                draw_target(ax, target[obj], obj,
                            OWNER[masked] if masked else OWNER[owner(instance, obj)],
                            nested=occupant.get(target[obj], obj) != obj,
                            letter=6.2, lw=1.0, dashes=(2.2, 1.5))
        for obj, cell in sorted(positions.items()):
            visible = (not masked) or obj in mine
            draw_disc(ax, cell, obj if visible else "X",
                      (OWNER[masked] if masked else OWNER[owner(instance, obj)])
                      if visible else HIDDEN,
                      radius=0.30, letter=6.4,
                      ring=MOVE if label == "Final state" else None, ring_lw=1.5)
        fig.text(x + bw / 2, top(cur - 0.24), label, fontsize=8.8,
                 color=OWNER[masked] if masked else INK, fontweight="bold",
                 ha="center", va="baseline")
        fig.text(x + bw / 2, top(cur + BOARD + 0.17), note, fontsize=6.5,
                 color=MOVE if label == "Final state" else MUTED,
                 ha="center", va="baseline")

    ax = fig.add_axes([xs[1] + bw, by, arrow, bh])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.annotate("", xy=(0.82, 0.5), xytext=(0.18, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=MOVE, lw=1.8,
                                mutation_scale=11))
    ax.text(0.5, 0.60, f"{len(episode['moves'])} moves", fontsize=6.8,
            color=MOVE, ha="center", va="baseline", transform=ax.transAxes)

    fig.text(L, top(FH - 0.14),
             "blue = Player A's objects  ·  orange = Player B's  ·  dark cells "
             "are walls  ·  messages are the players' own words, cut only where "
             f"marked …  ·  {episode['level']}, {episode['protocol']} protocol, "
             f"{', '.join(episode['models'])}",
             fontsize=6.3, color=FAINT, va="baseline")

    return save(fig, out_dir, "game_overview", formats)


# --------------------------------------------------- figure 1: the two views


def render_views(episode: dict, out_dir: str, formats=("pdf", "png")) -> List[str]:
    instance = episode["instance"]
    size = instance["grid_size"]
    walls = [tuple(w) for w in instance["walls"]]
    start = episode["start"]
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    own = sides(instance)

    FW, FH = 7.2, 5.36
    fig = plt.figure(figsize=(FW, FH))
    L = 0.0417

    fig.text(L, 0.947, "The same board, as each player is shown it",
             fontsize=13.0, color=INK, fontweight="bold", va="baseline")
    fig.text(L, 0.912,
             "Six objects, two players, three each. An arrow marks where one "
             "object must end up, not the route it takes. Neither player is "
             "ever shown the other panel.",
             fontsize=7.2, color=MUTED, va="baseline")
    fig.text(0.958, 0.947,
             f"{episode['level']}  ·  instance {instance['game_id']}  ·  "
             f"optimal {instance['optimal_moves']} moves",
             fontsize=7.0, color=FAINT, ha="right", va="baseline")

    pw, gap = 2.75 / FW, 0.50 / FW
    by, bh = 1.50 / FH, 2.75 / FH
    contested = set()

    for i, side in enumerate(("A", "B")):
        mine, colour = own[side], OWNER[side]
        x = L + i * (pw + gap)
        ax = fig.add_axes([x, by, pw, bh])
        draw_grid(ax, size, walls, labels=5.4)

        blocked = {cell for _o, _h, cell in blocked_targets(instance, start, side)}
        contested |= blocked

        for obj in mine:
            draw_target(ax, target[obj], obj, colour,
                        nested=target[obj] in blocked, letter=9.2, lw=1.25,
                        dashes=(2.4, 1.6))
        for obj in mine:
            bend = clear_bend(start[obj], target[obj], start.values())
            draw_arrow(ax, start[obj], target[obj], to_rgba(colour, 0.55),
                       rad=bend)
        for obj, cell in sorted(start.items()):
            draw_disc(ax, cell, obj if obj in mine else "X",
                      colour if obj in mine else HIDDEN, radius=0.315,
                      letter=9.0, ring=CUE if cell in blocked else None,
                      ring_lw=2.0)

        fig.text(x, 0.868, f"What Player {side} sees", fontsize=10.5,
                 color=colour, fontweight="bold", va="baseline")
        fig.text(x, 0.835,
                 f"owns {', '.join(mine)}   ·   the other three are only X",
                 fontsize=7.0, color=MUTED, va="baseline")
        lines = listing(instance, start, side)
        for j, field in enumerate(("Objects", "Targets")):
            fig.text(x, 0.239 - j * 0.034, f"{field}: {lines[field]}",
                     fontsize=6.2, color=INK, va="baseline", **MONO)

    fig.text(L, 0.164,
             "Each player has exactly one arrow ending on an X: the cell it "
             "must reach is held by an object it can neither identify nor move.",
             fontsize=7.2, color=CUE, va="baseline")
    fig.text(L, 0.131,
             "That is the whole game. The channel is the only way either player "
             "can learn what the other panel says, or get an X out of the way.",
             fontsize=7.2, color=MUTED, va="baseline")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=7.0, mfc=OWNER["A"], mec=PAPER,
               label="an object this player owns and may move"),
        Line2D([], [], marker="s", ls="none", ms=7.0,
               mfc=to_rgba(OWNER["A"], 0.12), mec=OWNER["A"],
               label="where one of its objects must end up"),
        Line2D([], [], marker=r"$\rightarrow$", ls="none", ms=8.5,
               mec=OWNER["A"], mfc=OWNER["A"],
               label=r"object $\rightarrow$ its destination"),
        Line2D([], [], marker="o", ls="none", ms=7.0, mfc=HIDDEN, mec=PAPER,
               label="occupied, by what and whose is not shown"),
        Line2D([], [], marker="o", ls="none", ms=7.0, mfc=HIDDEN, mec=CUE,
               mew=1.8, label="an X standing on a cell this player needs"),
        Line2D([], [], marker="s", ls="none", ms=7.0, mfc=WALL, mec=WALL_HI,
               label="wall, impassable"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(L, 0.018),
               ncol=3, frameon=False, fontsize=7.0, handletextpad=0.7,
               columnspacing=2.0, labelspacing=0.55)

    return save(fig, out_dir, "two_views", formats)


# -------------------------------------------------- figure 2: one turn, whole


MONO_PT = 5.0
LEAD_PT = 1.45 * MONO_PT


def _mono_panel(fig, rect, lines, *, colour, tint, title, highlight=None,
                bold_last=False):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0,rounding_size=0.015",
                                fc=to_rgba(tint, 0.055), ec=to_rgba(tint, 0.30),
                                lw=0.6, transform=ax.transAxes, zorder=1))
    ax.add_patch(Rectangle((0, 0), 0.006, 1, fc=tint, ec="none",
                           transform=ax.transAxes, zorder=2))
    ax.text(0.028, 1.055, title, fontsize=7.0, color=colour, fontweight="bold",
            va="baseline", transform=ax.transAxes)

    h_pt = rect[3] * fig.get_figheight() * 72.0
    lead = LEAD_PT / h_pt
    x0 = 0.028
    for i, line in enumerate(lines):
        last = bold_last and i == len(lines) - 1
        ax.text(x0, 1.0 - lead * 1.10 - i * lead, line, fontsize=MONO_PT,
                family="monospace", color=colour if last else INK,
                fontweight="bold" if last else "normal", va="center",
                transform=ax.transAxes, zorder=4)
    return ax, lead, x0


def render_turn(episode: dict, turn: int, out_dir: str,
                formats=("pdf", "png")) -> List[str]:
    instance = episode["instance"]
    size = instance["grid_size"]
    walls = [tuple(w) for w in instance["walls"]]
    move = episode["moves"][turn - 1]
    side = move["side"]
    colour = OWNER[side]
    before = replay(episode, turn - 1)[-1]
    context, response = exchange_for(episode, turn)

    blocked = blocked_targets(instance, before, side)
    if not blocked:
        raise SystemExit(
            f"turn {turn} has no contested cell for Player {side}; this figure "
            "is about a reference across the masking boundary, so pick a turn "
            "that has one")
    my_obj, holder, cue_cell = blocked[0]

    # Everything from CURRENT BOARD on is quoted verbatim; only the relayed
    # teammate message, which is prose, is cut.
    head, _, board = context.partition("CURRENT BOARD:")
    prompt_lines: List[str] = []
    if head.strip():
        relay = head.strip()
        speaker, _, said = relay.partition(" says: ")
        prompt_lines += wrap_mono(
            f"{speaker} says: {abridge(said, turn - 1, table=curated(episode))}", 45)
        prompt_lines.append("")
    for logical in ("CURRENT BOARD:" + board).strip().split("\n"):
        prompt_lines += wrap_mono(logical, 45, indent="       ") or [""]

    reply_lines: List[str] = []
    for logical in response.strip().split("\n"):
        reply_lines += wrap_mono(logical, 45, indent="  ") or [""]

    FW, FH = 7.2, 3.60
    fig = plt.figure(figsize=(FW, FH))
    fig.text(0.038, 0.945, "One turn, as the player receives and answers it",
             fontsize=11.5, color=INK, fontweight="bold", va="baseline")
    fig.text(0.038, 0.905,
             f"T{turn} of the same episode  ·  the board panel is not the true "
             f"board: it is what the listing to its left says, which is all "
             f"Player {side} has",
             fontsize=6.4, color=MUTED, va="baseline")

    line_f = LEAD_PT / (FH * 72.0)
    PT = 0.790
    h_prompt = (len(prompt_lines) + 1.7) * line_f
    h_reply = (len(reply_lines) + 1.7) * line_f
    PB = PT - h_prompt

    ax_p, lead_p, x0 = _mono_panel(
        fig, [0.038, PB, 0.345, h_prompt], prompt_lines, colour=INK, tint=MUTED,
        title=r"game master $\rightarrow$ Player " + side)
    ax_r, lead_r, _ = _mono_panel(
        fig, [0.655, PT - h_reply, 0.307, h_reply], reply_lines, colour=colour,
        tint=colour, bold_last=True,
        title=(rf"Player {side} $\rightarrow$ game master   "
               f"({', '.join(episode['models'])})"))

    # the board as that listing decodes to, not as it truly is
    bw = 0.185
    ax_b = fig.add_axes([0.425, (PT + PB) / 2 - bw * FW / FH / 2, bw, bw * FW / FH])
    draw_grid(ax_b, size, walls, labels=3.9)
    mine = set(sides(instance)[side])
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    ax_b.add_patch(Rectangle((cue_cell[1], cue_cell[0]), 1, 1,
                             fc=to_rgba(CUE, 0.30), ec=CUE, lw=1.0, zorder=1.6))
    occupant = {cell: o for o, cell in before.items()}
    for obj in sorted(mine):
        draw_target(ax_b, target[obj], obj, colour,
                    nested=occupant.get(target[obj], obj) != obj, letter=5.4,
                    lw=0.85, dashes=(2.0, 1.4))
    for obj, cell in sorted(before.items()):
        draw_disc(ax_b, cell, obj if obj in mine else "X",
                  colour if obj in mine else HIDDEN, radius=0.30, letter=5.0)
    fig.text(0.425, PT + 0.030, "what that listing means", fontsize=7.0,
             color=MUTED, fontweight="bold", va="baseline")

    fig.text(0.655, PT - h_reply - 0.075,
             f"The X it asks about is Player {owner(instance, holder)}'s "
             f"{holder}.\nPlayer {side} cannot know that, and does not need to:\n"
             "the cell address is enough to refer with.",
             fontsize=6.3, color=CUE, va="top")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=colour, mec=PAPER,
               label=f"Player {side}'s own object"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=HIDDEN, mec=PAPER,
               label="a teammate's object, unidentifiable"),
        Line2D([], [], marker="s", ls="none", ms=5.2, mfc="none", mec=colour,
               label=f"Player {side}'s target (teammate targets are omitted)"),
        Line2D([], [], marker="s", ls="none", ms=5.2, mfc=WALL, mec=WALL_HI,
               label="wall"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.038, 0.008),
               ncol=4, frameon=False, fontsize=6.2, handletextpad=0.5,
               columnspacing=1.4)

    # highlights last: they need a renderer to measure the monospace advance
    fig.canvas.draw()
    for ax, lead, lines, needle in (
        (ax_p, lead_p, prompt_lines, f"X@{cellname(cue_cell)}"),
        (ax_r, lead_r, reply_lines, cue_span(response, cue_cell)),
    ):
        row, col_a, col_b = find_span(lines, needle)
        adv = mono_advance(ax, MONO_PT)
        y = 1.0 - lead * 1.10 - row * lead
        ax.add_patch(Rectangle((x0 + col_a * adv, y - lead * 0.40),
                               (col_b - col_a) * adv, lead * 0.80,
                               fc=to_rgba(CUE, 0.34), ec="none",
                               transform=ax.transAxes, zorder=3))

    return save(fig, out_dir, "turn_companion", formats)


# ------------------------------------------------- figure 3: the episode strip


def render_strip(episode: dict, moves: int, out_dir: str,
                 formats=("pdf", "png")) -> List[str]:
    instance = episode["instance"]
    size = instance["grid_size"]
    walls = [tuple(w) for w in instance["walls"]]
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    shown = episode["moves"][:moves]
    states = replay(episode, moves)
    cuts = curated(episode)

    # the cell one player must have vacated for the other, and the panel that
    # still shows it occupied
    first_side = shown[0]["side"]
    other = "B" if first_side == "A" else "A"
    contested = blocked_targets(instance, episode["start"], other)
    cue_cell = contested[0][2] if contested else None
    cue_panel = next((m["index"] - 1 for m in shown if m["from"] == cue_cell), None)

    FW, FH = 7.1, 4.85
    fig = plt.figure(figsize=(FW, FH))
    fig.text(0.038, 0.960, f"{len(shown)} moves of a dyadic episode",
             fontsize=12.5, color=INK, fontweight="bold", va="baseline")
    fig.text(0.038, 0.930,
             f"{' and '.join(episode['models'])} on both sides  ·  "
             f"{episode['level']}  ·  {episode['protocol']} protocol  ·  "
             "each player owns three of the six objects and sees the other "
             "three only as X",
             fontsize=6.6, color=MUTED, va="baseline")

    TOP, ROW = 0.885, 0.083
    for i, move in enumerate(shown):
        colour = OWNER[move["side"]]
        y = TOP - i * ROW
        ax = fig.add_axes([0.038, y - ROW + 0.014, 0.925, ROW - 0.012])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.0, 0.04), 1.0, 0.92,
                                    boxstyle="round,pad=0,rounding_size=0.02",
                                    fc=to_rgba(colour, 0.055),
                                    ec=to_rgba(colour, 0.28), lw=0.6,
                                    transform=ax.transAxes, zorder=1))
        ax.add_patch(Rectangle((0.0, 0.04), 0.0055, 0.92, fc=colour, ec="none",
                               transform=ax.transAxes, zorder=2))
        ax.text(0.014, 0.70, f"T{move['index']}", fontsize=6.4, color=FAINT,
                va="center")
        ax.text(0.038, 0.70, f"Player {move['side']}", fontsize=7.0,
                color=colour, fontweight="bold", va="center")
        said = abridge(move["message"] or "", move["index"], table=cuts)
        for j, line in enumerate(wrap_words(said, 118)[:2]):
            ax.text(0.115, 0.76 - j * 0.28, line, fontsize=6.3, color=INK,
                    va="center")
        ax.text(0.115, 0.155, f"move: {move['object']} {move['direction']}",
                fontsize=6.6, color=colour, va="center", family="monospace",
                fontweight="bold")

    n = len(states)
    L, R, GAP = 0.038, 0.963, 0.021
    pw = (R - L - GAP * (n - 1)) / n
    by, bh = 0.265, pw * FW / FH

    for i, positions in enumerate(states):
        x = L + i * (pw + GAP)
        ax = fig.add_axes([x, by, pw, bh])
        draw_grid(ax, size, walls, labels=3.9 if i == 0 else 0)

        if i:
            prev = shown[i - 1]["from"]
            ax.add_patch(Rectangle((prev[1], prev[0]), 1, 1,
                                   fc=to_rgba(MOVE, 0.16), ec="none", zorder=1.5))
        if i == 0:
            for _o, _h, cell in (contested + blocked_targets(
                    instance, episode["start"], first_side)):
                ax.add_patch(Rectangle((cell[1], cell[0]), 1, 1,
                                       fc=to_rgba(CUE, 0.32), ec="none",
                                       zorder=1.5))
        occupant = {cell: o for o, cell in positions.items()}
        for obj in sorted(target):
            draw_target(ax, target[obj], obj, OWNER[owner(instance, obj)],
                        nested=occupant.get(target[obj], obj) != obj,
                        letter=5.0, lw=0.85, dashes=(2.0, 1.4))
        if i:
            draw_arrow(ax, shown[i - 1]["from"], shown[i - 1]["to"], MOVE,
                       lw=2.0, scale=8.0, zorder=9, shrink=(5.0, 9.5))
        for obj, cell in sorted(positions.items()):
            landed = cell == target[obj]
            draw_disc(ax, cell, obj, OWNER[owner(instance, obj)], radius=0.30,
                      letter=5.0, ring=MOVE if landed else None, ring_lw=1.7)
        if i == cue_panel and cue_cell is not None:
            ax.add_patch(Circle(centre(cue_cell), 0.50, fc="none", ec=OWNER[other],
                                lw=1.1, linestyle=(0, (1.8, 1.4)), zorder=9.5))

        if i == 0:
            tag, sub, tint = "start", (f"A owns {' '.join(sides(instance)['A'])}"
                                       f"  ·  B owns {' '.join(sides(instance)['B'])}"), INK
        else:
            m = shown[i - 1]
            tag = f"T{m['index']}"
            sub = f"{m['side']}:  move {m['object']} {m['direction']}"
            tint = OWNER[m["side"]]
        fig.text(x, by - 0.038, tag, fontsize=7.4, color=tint,
                 fontweight="bold", va="baseline")
        fig.text(x, by - 0.068, sub, fontsize=5.8, color=MUTED, va="baseline",
                 **({} if i == 0 else MONO))
        if i:
            fig.text(x - GAP / 2, by + bh / 2, r"$\blacktriangleright$",
                     fontsize=6.0, color=MOVE, ha="center", va="center")

    caption = chain_caption(instance)
    if caption:
        head, _, rest = caption.partition(":  ")
        fig.text(L, by - 0.100, f"{head}:  " + rest.replace(", ", ",\n"),
                 fontsize=5.6, color=CUE, va="top")
    if cue_panel is not None:
        asker, held = other, contested[0][1]
        fig.text(L + cue_panel * (pw + GAP), by - 0.100,
                 f"the “X at {cellname(cue_cell)}” in {asker}'s request\n"
                 f"is in fact {owner(instance, held)}'s own {held}",
                 fontsize=5.6, color=OWNER[other], va="top")
    landed = on_target(instance, states[-1])
    if landed:
        fig.text(L + (n - 1) * (pw + GAP), by - 0.100,
                 f"{', '.join(landed)} reaches its target: "
                 f"{len(shown) - 1}\nturns of channel, one cell",
                 fontsize=5.6, color=MOVE, va="top")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=OWNER["A"], mec=PAPER,
               label="Player A's object"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=OWNER["B"], mec=PAPER,
               label="Player B's object"),
        Line2D([], [], marker="s", ls="none", ms=5.2, mfc="none", mec=MUTED,
               label="target cell"),
        Line2D([], [], marker="s", ls="none", ms=5.2, mfc=WALL, mec=WALL_HI,
               label="wall"),
        Line2D([], [], color=MOVE, lw=1.4, label="the move made this turn"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc=OWNER["B"], mec=MOVE,
               mew=1.2, label="on target"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(L, 0.012),
               ncol=6, frameon=False, fontsize=6.2, handletextpad=0.5,
               columnspacing=1.5)

    return save(fig, out_dir, "episode_strip", formats)


# ----------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="interactions.json to draw from")
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--figures", nargs="*", default=list(FIGURES),
                        choices=list(FIGURES))
    parser.add_argument("--turn", type=int, default=2,
                        help="which turn the companion figure quotes")
    parser.add_argument("--moves", type=int, default=4,
                        help="how many moves the strip shows")
    parser.add_argument("--list", action="store_true",
                        help="print successful dyadic episodes and exit")
    args = parser.parse_args()

    if args.list:
        for hit in find_episodes("matrixgame_covered_comm/S")[:30]:
            print(f"  {hit['moves']:>3} moves  {hit['experiment']:<24} "
                  f"game_id {hit['game_id']:<3} {hit['path']}")
        return

    episode = load_episode(args.run)
    if episode["game"] != "matrixgame_covered_comm":
        raise SystemExit("these figures assume the two-player game")
    if args.moves >= len(episode["moves"]):
        raise SystemExit(f"--moves must be under {len(episode['moves'])}")

    written: List[str] = []
    if "overview" in args.figures:
        written += render_overview(episode, args.out)
    if "views" in args.figures:
        written += render_views(episode, args.out)
    if "turn" in args.figures:
        written += render_turn(episode, args.turn, args.out)
    if "strip" in args.figures:
        written += render_strip(episode, args.moves, args.out)
    for path in written:
        print("wrote", os.path.relpath(path, os.path.dirname(OUT_DIR)))


if __name__ == "__main__":
    main()
