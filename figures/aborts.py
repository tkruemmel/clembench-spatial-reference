"""Draws the two abort figures for S9.3.

`abort_avoidable` is the pattern 30 of the 32 enclosure aborts follow: the
victim is down to one free cell and a teammate that has other legal moves walks
into it. `abort_forced` is the other 2 (one board, played twice), where the
sealing move was the sealer's only legal move and the avoidable choice sits one
move earlier.

Both use the same instance, S3 board 10, so the geometry is learned once.
Everything drawn is replayed from the recorded move log by
analysis/qualitative_analysis/seal_alt.py; no position here is hand-placed.

    python3 figures/aborts.py
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import matplotlib.pyplot as plt

from style import (CUE, FAINT, INK, MOVE, MUTED, OBJECT, OWNER, draw_arrow,
                   draw_disc, draw_grid, draw_target, save)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis/qualitative_analysis"))
import seal_alt as sa  # noqa: E402

VICTIM = OWNER["B"]  # the object that is left with no legal move
ACTOR = OBJECT       # the player whose choice this panel is about
BOARD = ("qwen38", "S3", "instance_00010")
AVOIDABLE = "freeform"  # the episode of that board drawn in the first figure


def route_home(case, pos, obj):
    """Shortest path for `obj` to its target, other objects and walls blocking."""
    src, dst = pos[obj], case["targets"][obj]
    blocked = set(pos.values()) - {src}
    seen, queue = {src}, collections.deque([[src]])
    while queue:
        path = queue.popleft()
        if path[-1] == dst:
            return path
        for dr, dc in sa.DIRS.values():
            nb = (path[-1][0] + dr, path[-1][1] + dc)
            if (0 <= nb[0] < case["grid"] and 0 <= nb[1] < case["grid"]
                    and nb not in case["walls"] and nb not in blocked
                    and nb not in seen):
                seen.add(nb)
                queue.append(path + [nb])
    return None


def cell_name(cell):
    return f"R{cell[0] + 1}C{cell[1] + 1}"


def square_axes(fig, left_in, bottom_in, side_in):
    w, h = fig.get_size_inches()
    return fig.add_axes([left_in / w, bottom_in / h, side_in / w, side_in / h])


def at(fig, x_in, y_in):
    """Inches from the bottom-left corner as a figure fraction.

    Every block on these figures is stacked against the board, whose size is
    fixed in inches by square_axes, so placing the text in inches too keeps
    the two from drifting apart when a figure is resized.
    """
    w, h = fig.get_size_inches()
    return x_in / w, y_in / h


def board(ax, case, pos, *, actor, victim, played=None, alts=(), free=None,
          warn=(), route=None, labels=5.4):
    """One board: discs, the move played (gold), the ones passed over (green)."""
    draw_grid(ax, case["grid"], case["walls"], labels=labels)
    if free is not None:
        draw_target(ax, free, "", VICTIM, nested=False, lw=1.4)
    if route:
        draw_target(ax, route[-1], victim, VICTIM, nested=False, lw=1.2)
        for src, dst in zip(route, route[1:]):
            draw_arrow(ax, src, dst, VICTIM, lw=1.2, scale=8.0,
                       shrink=(5.0, 1.0), ls=(0, (1.2, 1.4)), zorder=3.5)
    for obj, cell in sorted(pos.items()):
        colour = VICTIM if obj == victim else ACTOR if obj == actor else FAINT
        ring = CUE if obj in warn else (
            MUTED if case["owner"][obj] in case["done"] else None)
        draw_disc(ax, cell, obj, colour, radius=0.30, letter=8.6,
                  ring=ring, ring_lw=1.5)
    for obj, direction in alts:
        draw_arrow(ax, pos[obj], sa.step(pos, obj, direction)[obj], MOVE,
                   lw=1.6, scale=10.0, shrink=(8.0, 2.5))
    if played:
        obj, direction = played
        draw_arrow(ax, pos[obj], sa.step(pos, obj, direction)[obj], CUE,
                   lw=2.6, scale=13.0, shrink=(8.0, -1.0))


def notes(fig, x_in, y_in, lines, *, size=7.5, leading=0.215):
    for i, (colour, text) in enumerate(lines):
        x, y = at(fig, x_in, y_in - i * leading)
        fig.text(x, y, text, fontsize=size, color=colour, va="top")


def pick(cases, proto):
    for c in cases:
        m = c["meta"]
        if (m["model"], m["level"], m["inst"], m["proto"]) == BOARD + (proto,):
            return c
    raise SystemExit(f"no recorded {proto} abort on {BOARD}")


def render_avoidable(case, out_dir):
    free = sa.last_free_cell(case)
    obj, direction = case["played"]
    v, total = case["victim"], case["meta"]["moves"]

    fig = plt.figure(figsize=(6.6, 3.5))
    ax = square_axes(fig, 0.30, 0.26, 2.92)
    board(ax, case, case["pre"], actor=case["sealer"], victim=v,
          played=case["played"], alts=case["harmless"], free=free)

    x = 3.45
    fig.text(*at(fig, x, 3.36), "The seal a teammate did not have to play",
             fontsize=10.0, color=INK, va="top", fontweight="bold")
    fig.text(*at(fig, x, 3.10), f"six players, S3 board 10, move {case['ply']} of {total}",
             fontsize=7.4, color=MUTED, va="top")
    notes(fig, x, 2.82, [
        (VICTIM, f"{v} has one cell left it can move to, {cell_name(free)}."),
        (CUE, f"{obj} plays {direction}, into exactly that cell."),
        (MOVE, f"{obj} could have played "
               + " or ".join(d for _, d in case["harmless"]) + " instead:"),
        (MOVE, "     legal, and leaving every player a move."),
        (INK, f"{v} now has no legal move at all. It goes on proposing"),
        (INK, f"     moves, and {total - case['ply']} moves later the episode is aborted."),
    ])
    fig.text(*at(fig, x, 1.48),
             "30 of the 32 aborts are this picture. The enclosure is\n"
             "a choice the team made, not a board it was dealt.",
             fontsize=8.0, color=INK, va="top", linespacing=1.6)
    notes(fig, x, 0.86, [
        (MUTED, f"dashed cell: the one cell {v} can still move to."),
        (MUTED, "ringed discs (%s) have declared done and will not"
                % ", ".join(sorted(case["done"]))),
        (MUTED, "move again \u2014 %s is the far wall of %s's pocket."
                % (sorted(case["done"])[-1], v)),
    ], size=7.0, leading=0.175)
    return save(fig, out_dir, "abort_avoidable", ("pdf", "png"))


def render_forced(case, protos, out_dir):
    """Four panels: the two moves that sealed the board, and the two that would not.

    Top row is the recorded episode; bottom row replays the same position with
    the other legal choice at each of the two plies. Every board, move and
    route in the bottom row is computed here, not asserted: the alternatives
    come from the rules, and the row is only drawn if they really leave every
    player a move.
    """
    seal = case["ply"]
    fork = sa.replay(case, seal - 2)                  # board the earlier mover saw
    culprit, cdir = case["moves"][seal - 2]           # what it played
    cowner = case["owner"][culprit]
    obj, direction = case["played"]                   # the forced sealing move
    v, free = case["victim"], sa.last_free_cell(case)

    def harmless(pos, player):
        return [(o, d) for o, d in sa.legal_moves(pos, case["mine"][player],
                                                  case["walls"], case["grid"])
                if not [p for p in case["mine"]
                        if not sa.legal_moves(sa.step(pos, o, d), case["mine"][p],
                                              case["walls"], case["grid"])]]

    # the branch: the other legal move at the fork, then a safe reply
    other = [(o, d) for o, d in sa.legal_moves(fork, case["mine"][cowner],
                                               case["walls"], case["grid"])
             if (o, d) != (culprit, cdir)]
    assert len(other) == 1, other
    cf_fork = sa.step(fork, *other[0])                # after the alternative
    cf_safe = harmless(cf_fork, case["owner"][obj])
    cf_risky = [(o, d) for o, d in sa.legal_moves(cf_fork, case["mine"][case["owner"][obj]],
                                                  case["walls"], case["grid"])
                if (o, d) not in cf_safe]
    assert len(cf_safe) == 1 and len(cf_risky) == 1, (cf_safe, cf_risky)
    route = route_home(case, sa.step(cf_fork, *cf_safe[0]), v)
    assert route and route[1] == free, route

    fig = plt.figure(figsize=(6.3, 8.35))
    side, cols = 2.25, (0.45, 3.40)
    panels = [square_axes(fig, x, bottom, side)
              for bottom in (4.97, 1.40) for x in cols]

    board(panels[0], case, fork, actor=cowner, victim=v, free=free,
          played=(culprit, cdir), alts=other, warn=[obj])
    board(panels[1], case, case["pre"], actor=case["owner"][obj], victim=v,
          free=free, played=case["played"])
    board(panels[2], case, fork, actor=cowner, victim=v, free=free, alts=other)
    board(panels[3], case, cf_fork, actor=case["owner"][obj], victim=v,
          alts=cf_safe, route=route)

    fig.text(*at(fig, 3.15, 8.17),
             "The two aborts whose sealing move was forced \u2014 and the line that avoids them",
             fontsize=10.0, color=INK, va="top", ha="center", fontweight="bold")
    fig.text(*at(fig, 3.15, 7.93),
             "one board, S3 board 10, sealed the same way under %s"
             % " and ".join(protos), fontsize=7.4, color=MUTED, va="top",
             ha="center")

    for y, head in ((7.62, "WHAT HAPPENED"),
                    (4.05, "WHAT WAS AVAILABLE, under the same rules")):
        fig.text(*at(fig, 0.45, y), head, fontsize=8.2, color=INK, va="top",
                 fontweight="bold")
    titles = (f"move {seal - 1}: {culprit} plays {cdir}",
              f"move {seal}: {obj} has one legal move",
              f"move {seal - 1}: {culprit} plays {other[0][1]} instead",
              f"move {seal}: {obj} now has two, and one is safe")
    for ax, text in zip(panels, titles):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, box.y1 + 0.10 / fig.get_size_inches()[1],
                 text, fontsize=8.0, color=INK, ha="center", va="bottom")

    notes(fig, cols[0], 4.75, [
        (CUE, f"{culprit} plays {cdir}."),
        (MUTED, f"{obj} is now down to one legal move (gold ring)."),
        (MOVE, f"{culprit} {other[0][1]} was legal and would enclose nobody."),
    ], size=7.2, leading=0.185)
    notes(fig, cols[1], 4.75, [
        (CUE, f"{obj} must move; {direction} is its only legal move."),
        (INK, f"It is the one cell {v} had left, {cell_name(free)}."),
        (VICTIM, f"{v} is enclosed. The episode aborts."),
    ], size=7.2, leading=0.185)
    notes(fig, cols[0], 1.18, [
        (MOVE, f"{culprit} plays {other[0][1]} \u2014 equally legal, same ply."),
        (MUTED, f"{obj} keeps both exits, so it still has a choice."),
        (MUTED, "Nobody is enclosed."),
    ], size=7.2, leading=0.185)
    notes(fig, cols[1], 1.18, [
        (MOVE, f"{obj} plays {cf_safe[0][1]}. Every player still has a move."),
        (MUTED, f"{obj} {cf_risky[0][1]} would still seal {v}: this ply matters too."),
        (VICTIM, f"{v} keeps {cell_name(free)} and its {len(route) - 1}-move route home (dotted)."),
    ], size=7.2, leading=0.185)
    fig.text(*at(fig, 0.45, 0.45),
             "Two ordinary legal moves, one at each ply, and nobody is enclosed \u2014 "
             "the abort was never the board's doing.",
             fontsize=8.0, color=INK, va="top")
    return save(fig, out_dir, "abort_forced", ("pdf", "png"))


FLOATS = r"""%% Figure floats for the two abort figures, written by figures/aborts.py.
%% Every number in a caption is read off the replayed episode, so a changed
%% tree cannot leave a caption asserting something the picture no longer shows.
\begin{figure}[tbp]
    \centering
    \thesisfig{abort_avoidable.pdf}
    \caption{%(cap_av)s}
    \label{fig:abort-avoidable}
\end{figure}

\begin{figure}[p]
    \centering
    \thesisfig{abort_forced.pdf}
    \caption{%(cap_forced)s}
    \label{fig:abort-forced}
\end{figure}
"""

WORDS = {2: "two", 30: "thirty", 32: "thirty-two"}


def write_floats(cases, av, forced, protos, out_dir):
    """The two \\begin{figure} blocks, with their numbers read off the episodes."""
    total, choice = len(cases), sum(1 for c in cases if c["harmless"])
    obj, direction = av["played"]
    seal = forced["ply"]
    route = route_home(forced, sa.step(sa.replay(forced, seal - 2), *[
        x for x in sa.legal_moves(sa.replay(forced, seal - 2),
                                  forced["mine"][forced["owner"][
                                      forced["moves"][seal - 2][0]]],
                                  forced["walls"], forced["grid"])
        if x != forced["moves"][seal - 2]][0]), forced["victim"])
    caps = {
        "cap_av": (
            "The shape of a six-player enclosure abort. %s, S3 board 10, "
            "\\textsc{%s}, move %d of %d: %s's only free neighbour is "
            "\\texttt{%s}, and %s plays into it although \\emph{%s} were "
            "equally legal and would have left every player a move. %s of the "
            "%s enclosure aborts recorded for the strong tier take this form; "
            "Table~\\ref{tab:seal-greedy} gives what the sealing move bought "
            "the player that made it. Discs are objects and dark cells walls; a "
            "ringed disc has declared done and will not move again."
            % ("Qwen3.8" if av["meta"]["model"] == "qwen38" else "DeepSeek",
               av["meta"]["proto"], av["ply"], av["meta"]["moves"], av["victim"],
               cell_name(sa.last_free_cell(av)), obj,
               "} and \\emph{".join(d for _, d in av["harmless"]),
               WORDS[choice].capitalize(), WORDS[total])),
        "cap_forced": (
            "The other %s, and the line that avoids them. The same board as "
            "Figure~\\ref{fig:abort-avoidable}, sealed identically under %s. "
            "The top row is the recorded play: %s's move at %d leaves %s a "
            "single legal move, and at %d that move lands on the one cell %s "
            "had left. The bottom row replays the same two plies with the other "
            "legal choice at each, after which every player still has a move "
            "and %s keeps its %d-move route home, drawn dotted. Each panel "
            "shows the board \\emph{before} the move drawn on it."
            % (WORDS[total - choice],
               " and ".join("\\textsc{%s}" % p for p in protos),
               forced["moves"][seal - 2][0], seal - 1,
               forced["played"][0], seal, forced["victim"],
               forced["victim"], len(route) - 1)),
    }
    path = os.path.join(out_dir, "aborts.tex")
    with open(path, "w") as fh:
        fh.write(FLOATS % caps)
    return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    cases = sa.cases()
    forced = [c for c in cases if not c["harmless"]]
    # the three counts S9.3 quotes, so a changed tree or a broken replay fails
    # loudly instead of drawing a confident lie
    assert len(cases) == 32, len(cases)
    assert len(forced) == 2, [c["meta"] for c in forced]
    assert all(sa.last_free_cell(c) == sa.step(c["pre"], *c["played"])[c["played"][0]]
               for c in cases), "a sealer did not step into the victim's last cell"
    protos = sorted(c["meta"]["proto"] for c in forced)

    avoidable = pick(cases, AVOIDABLE)
    paths = render_avoidable(avoidable, args.out)
    paths += render_forced(forced[0], protos, args.out)
    paths += write_floats(cases, avoidable, forced[0], protos, args.out)
    for path in paths:
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
