"""Draws the matched pair of S9.3.3: one fork, two protocols, opposite choices.

Qwen3.8, six players, S4 board 3, the same six objects, starts and targets in
both episodes. In each, B reaches its own target on the northern way round the
wall block while E's finished object already freezes the southern one, so a
`done` from B would leave D no route to its target. Under `none` B declares
(the cut of figure cut_twocut). Under `hybrid` B had announced it would declare,
then steps off its target instead, says why, and D gets home.

The `none` panel is the recorded cut from analysis/qualitative_analysis/cut.py; the
`hybrid` panel is replayed here from the move log. The boards are asserted
byte-identical, the hazard is asserted live at the drawn instant (B frozen
would cut D, B movable would not), and every quote is checked verbatim.

    python3 figures/giveway.py
"""

from __future__ import annotations

import argparse
import json
import os
import string

import matplotlib.pyplot as plt

import cuts
from cuts import (ACTOR, COLS, OPEN_DASH, SIDE, VICTIM, WIDTH, at, cell_name,
                  square_axes)
from style import INK, MOVE, MUTED, draw_arrow, save
from warning import bubble, height_of
import cut

HYBRID = ("qwen38", "6p", "S4", "hybrid", "instance_00003")
HOLDER, VICTIM_OBJ = "B", "D"

# the only thing a `none` player can say is its action line
NONE_SAID = "done: all my objects are placed"
# (move the message rides on, fragments cut from it, never reworded)
HYBRID_SAID = [
    (44, ["I’ll take (R4,C6) now and stay there",
          "I’m finishing my object this turn and will declare done next."]),
    (49, ["I'll start clearing D's route by moving right to (R4,C7)."]),
    (53, ["keeping (R4,C6) and (R4,C7) open for D’s C8 corridor.",
          "This avoids trapping D behind the C6 wall."]),
]
BOARD_KEYS = ("grid_size", "walls", "start_positions", "target_positions",
              "player_objects", "device")


def run_dir(model, suite, level, proto, inst):
    for rec in json.load(open(os.path.join(cut.HERE, "episodes.json"))):
        p = rec["path"]
        if (rec["model"], rec["suite"]) == (model, suite) and \
                f"/{level}_{proto}_" in p and p.endswith(inst):
            return os.path.join(cut.REPO, p)
    raise SystemExit(f"no episode {model} {suite} {level} {proto} {inst}")


def replay(path):
    """Accepted moves, done declarations and messages, indexed by move."""
    ins = json.load(open(os.path.join(path, "instance.json")))
    it = json.load(open(os.path.join(path, "interactions.json")))
    moves, done_at, said, raw, outcome = [], {}, {}, {}, None
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            kind = a.get("type")
            if kind == "move":
                m = cut.MOVE.match(a.get("content", ""))
                if m:
                    moves.append((m.group(2), m.group(3)))
            elif kind == "player_done":
                done_at[a["content"].split()[1]] = len(moves)
            elif kind == "message":
                said.setdefault(len(moves), a["content"].split(": ", 1)[1])
            elif kind == "get message" and e.get("from", "").startswith("Player"):
                raw.setdefault(len(moves), {})[e["from"]] = str(a.get("content"))
            elif kind in ("success", "lose", "abort"):
                outcome = kind
    return ins, moves, done_at, said, raw, outcome


def board_at(ins, moves, upto):
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    for o, d in moves[:upto]:
        pos = cut.step(pos, o, d)
    return pos


def render(out_dir):
    rows = cut.cases()
    none = cuts.pick(rows, cuts.TWOCUT)
    none_dir = os.path.join(cut.REPO, none["meta"]["path"])
    ins_n, moves_n, done_n, _, raw_n, out_n = replay(none_dir)
    ins, moves, done_h, said, _, out_h = replay(run_dir(*HYBRID))
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    owner = {o: string.ascii_uppercase[i]
             for i, g in enumerate(ins["player_objects"]) for o in g}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}

    # ---- the claims, checked before anything is drawn
    for key in BOARD_KEYS:
        assert ins[key] == ins_n[key], key            # same board, both runs
    assert none["meta"]["model"] == HYBRID[0] and none["meta"]["proto"] == "none"
    assert (none["sealer"], none["victim"]) == (HOLDER, VICTIM_OBJ)
    assert out_n == "lose" and out_h == "success"
    assert NONE_SAID == raw_n[none["ply"]]["Player 2"]  # B is player 2

    arrive = next(p for p in range(1, len(moves) + 1)
                  if board_at(ins, moves, p)[HOLDER] == tgt[HOLDER]
                  and board_at(ins, moves, p - 1)[HOLDER] != tgt[HOLDER])
    fork = arrive + 4                   # the board B sees when it steps off
    step_off = fork + 1
    pos = board_at(ins, moves, fork)
    assert moves[step_off - 1] == (HOLDER, "right"), moves[step_off - 1]
    assert all(pos[HOLDER] == tgt[HOLDER]
               for pos in [board_at(ins, moves, p) for p in range(arrive, fork + 1)])
    frozen = {o: pos[o] for o in pos if done_h.get(owner[o], 1e9) <= fork}
    assert set(frozen) == {"E"}, frozen
    blocked = set(frozen.values())
    live = cut.path(pos[VICTIM_OBJ], tgt[VICTIM_OBJ], n, walls, blocked)
    assert live and cut.path(pos[VICTIM_OBJ], tgt[VICTIM_OBJ], n, walls,
                             blocked | {pos[HOLDER]}) is None  # hazard is live
    assert HOLDER not in done_h or done_h[HOLDER] > fork
    home = next(p for p in range(len(moves) + 1)
                if board_at(ins, moves, p)[VICTIM_OBJ] == tgt[VICTIM_OBJ])
    assert home == len(moves)       # the caption calls D's arrival the last move
    for p, frags in HYBRID_SAID:
        at_ = 0
        for f in frags:
            at_ = said[p].index(f, at_)                  # verbatim, in order

    # ---- the hybrid panel's case, in the shape cuts.board draws
    hyb = dict(grid=n, walls=walls, pre=pos, victim=VICTIM_OBJ, targets=tgt,
               culprits=none["culprits"], frozen_pre=frozen, frozen_post=frozen,
               seals=[])

    texts = [" … ".join(f) for _, f in HYBRID_SAID]
    chars = 42
    right_h = sum(height_of(t, chars) + 0.06 for t in texts)
    H = 1.16 + SIDE + 0.30 + right_h + 0.20 + 0.62
    fig = plt.figure(figsize=(WIDTH, H))
    fig.text(*at(fig, COLS[0], H - 0.20),
             "The same fork, two protocols, the same model", fontsize=10.4,
             color=INK, va="top", fontweight="bold")
    fig.text(*at(fig, COLS[0], H - 0.52),
             "six players, S4 board 3, Qwen3.8 in both; same objects, starts, "
             "targets and victim",
             fontsize=7.5, color=MUTED, va="top")

    bottom = H - 1.16 - SIDE
    ax_a, ax_b = (square_axes(fig, x, bottom, SIDE) for x in COLS)
    before = cuts.reachable(none, set(none["frozen_pre"].values()))
    after = cuts.reachable(none, set(none["frozen_post"].values()))
    cuts.board(ax_a, none, sealed=True,
               lost=before - after - set(none["frozen_post"].values()))
    cuts.board(ax_b, hyb, sealed=False,
               routes=[(live, VICTIM, OPEN_DASH)])
    draw_arrow(ax_b, pos[HOLDER], cut.step(pos, HOLDER, "right")[HOLDER], MOVE,
               lw=2.2, scale=11.0, shrink=(6.0, 0.0), zorder=9)
    for x, tag, text in zip(COLS, "ab", (
            f"none, move {none['ply']}: B declares done",
            f"hybrid, move {step_off}: B steps off its target")):
        fig.text(*at(fig, x, bottom + SIDE + 0.26), f"({tag})", fontsize=8.6,
                 color=INK, va="bottom", fontweight="bold")
        fig.text(*at(fig, x + 0.24, bottom + SIDE + 0.26), text, fontsize=8.2,
                 color=INK, va="bottom")

    y = bottom - 0.24
    ha = bubble(fig, COLS[0], y, SIDE, HOLDER, f"move {none['ply']} · action",
                NONE_SAID, chars=chars, colour=ACTOR)
    fig.text(*at(fig, COLS[0], y - ha - 0.10),
             f"{VICTIM_OBJ} can no longer reach {cell_name(tgt[VICTIM_OBJ])}. The "
             f"team plays {none['tail']} more moves",
             fontsize=7.2, color=INK, va="top")
    fig.text(*at(fig, COLS[0], y - ha - 0.27),
             "and the episode ends on a repeated board: lost.",
             fontsize=7.2, color=INK, va="top")
    yb = y
    for (p, _), t in zip(HYBRID_SAID, texts):
        yb -= bubble(fig, COLS[1], yb, SIDE, HOLDER, f"move {p}", t,
                     chars=chars, colour=ACTOR) + 0.06
    fig.text(*at(fig, COLS[1], yb - 0.04),
             (f"{VICTIM_OBJ} reaches {cell_name(tgt[VICTIM_OBJ])} on move {home}, "
              "the last of the episode: won." if home == len(moves) else
              f"{VICTIM_OBJ} is home at move {home}; all six at move {len(moves)}: won."),
             fontsize=7.2, color=INK, va="top")

    # key, drawn with the same marks as cut_twocut
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba
    from style import FAINT, WALL
    keys = [
        Patch(facecolor=to_rgba(WALL, 0.20), edgecolor=WALL, hatch="xxx",
              linewidth=0.5, label="declared done: object frozen, cell walled"),
        Patch(facecolor=to_rgba(VICTIM, 0.13), edgecolor=VICTIM, linewidth=1.1,
              label="where D can still get to"),
        Patch(facecolor="none", edgecolor=to_rgba(VICTIM, 0.5), hatch="//",
              linewidth=0.3, label="cells D loses to B's done"),
        Patch(facecolor="none", edgecolor=VICTIM, linewidth=1.3,
              linestyle=(0, (2.2, 1.5)), label="D's target"),
        Line2D([], [], color=VICTIM, linestyle=OPEN_DASH, linewidth=1.2,
               label="D's route, open while B can move"),
        Line2D([], [], color=MOVE, linewidth=2.0, marker=">", markersize=4,
               label="B's move off its target"),
    ]
    fig.legend(handles=keys, loc="lower left",
               bbox_to_anchor=at(fig, COLS[0] - 0.06, 0.08), ncol=3,
               frameon=False, fontsize=6.6, handletextpad=0.6,
               columnspacing=1.2, labelspacing=0.45, borderpad=0.0)

    paths = save(fig, out_dir, "cut_vs_giveway", ("pdf", "png"))
    tex = os.path.join(out_dir, "giveway.tex")
    with open(tex, "w") as fh:
        fh.write(FLOAT % dict(ply=none["ply"], tail=none["tail"],
                              arrive=arrive, step=step_off, home=home,
                              total=len(moves),
                              target=cell_name(tgt[VICTIM_OBJ])))
    return paths + [tex]


FLOAT = r"""%% Figure float for the matched cut / give-way pair, written by figures/giveway.py.
\begin{figure}[tbp]
    \centering
    \thesisfig{cut_vs_giveway.pdf}
    \caption{One fork, two protocols. Qwen3.8, six players, S4 board 3, with
    the same six objects, starts, targets and victim in both episodes. In each,
    \texttt{B} stands on its own target on the northern way round the wall
    block while \texttt{E}'s finished object already freezes the southern one,
    so a \texttt{done} from \texttt{B} leaves \texttt{D} no route to
    \texttt{%(target)s}. (a)~Under \textsc{none}, \texttt{B} declares at move
    %(ply)d; the team plays %(tail)d more moves on a board that can no longer
    be solved. (b)~Under \textsc{hybrid}, \texttt{B} reaches its target at
    move %(arrive)d and announces it will declare, then at move %(step)d steps
    off instead and later gives the reason; \texttt{D} reaches its target
    on move %(home)d, the last, and the episode is won. Messages are quoted verbatim;
    \ldots{} marks omitted text.}
    \label{fig:cut-vs-giveway}
\end{figure}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=cuts.OUT_DIR)
    for path in render(ap.parse_args().out):
        print(f"wrote {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
