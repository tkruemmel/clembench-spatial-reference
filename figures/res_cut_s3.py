"""fig:res-cut-s3: the S3 entrance trap on one board, played twice at 3p.

DeepSeek-V4.1, S3 board 13, unshuffled 3p run. B's target is the corridor
entrance (R2,C5) and D's target the dead end (R2,C1); Orange owns B and E,
Green A and D, Blue C and F. In both episodes Green asks for the entrance
while D is still outside, and at action 26 Orange decides. (a) Under freeform
Orange moves B off its target and the team wins. (b) Under structured Orange
declares done, B freezes on the entrance and D can no longer reach its target.

The pair is an illustration chosen after inspection. Positions are replayed
from the logs with analysis/thesis/move_metrics.replay, the quoted fragments
are asserted to occur verbatim, and the facts the caption states are asserted
in main().

    python3 figures/res_cut_s3.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from results_final_figs import FS_LEGEND, FS_SMALL, FS_TICK, FS_TITLE, OUT_DIR, REPO_ROOT, TEXT_W, publish
from style import (INK, MOVE, MUTED, OPEN_DASH, OWNER, PAPER, WALL, at, draw_arrow, draw_grid, draw_target,
                   hatch_cells, marked, marker, outline, route, square_axes, tint, wrap_words)

sys.path[:0] = [os.path.join(REPO_ROOT, "analysis", "thesis"), os.path.join(REPO_ROOT, "analysis", "efficiency_baseline")]
import move_metrics as mm  # noqa: E402
import results_final as rf  # noqa: E402
import stuckfree as SF  # noqa: E402

RUN = "results_dsv41_grid_final_chrono_n15_3p/DeepSeek-V4.1-Flash/matrixgame_covered_comm_3p_grid_final_chrono"
EPISODES = {"freeform": f"{RUN}/S3_freeform_3p_grid/instance_00013",
            "structured": f"{RUN}/S3_structured_3p_grid/instance_00013"}
DECISION = 26          # Orange's action in both episodes
SEAT = {"Green": MOVE, "Orange": OWNER["B"], "Blue": OWNER["A"]}
# (action, speaker, fragments); each fragment must occur verbatim, in order
SAID = {
    "freeform": [
        (14, "Orange", ["when D reaches (R2,C6), I can shift B up temporarily to (R1,C5), then return."]),
        (22, "Green", ["Orange: please do NOT declare done yet; wait for D at (R2,C7), shift B up to (R1,C5) "
                       "so D can pass through (R2,C5), then return B to (R2,C5)."]),
        (26, "Orange", ["Moving B up to (R1,C5) now to clear (R2,C5) for D."]),
    ],
    "structured": [
        (20, "Orange", ["REQUEST: C vacate (R1,C6); B clears (R2,C5) next"]),
        (22, "Green", ["REQUEST: please clear (R2,C5) for D"]),
        (25, "Green", ["REQUEST: B clear (R2,C5) for D"]),
        (26, "Orange", ["done"]),
    ],
}

SIDE, GAP, LEFT = 2.55, 0.62, 0.38
COLS = (LEFT, LEFT + SIDE + GAP)
HEAD = 0.78
FS_MSG = FS_SMALL          # 8 pt, the floor
LEAD = 0.150               # line pitch of 8 pt text, inches
CHARS = 48


def cell(c):
    return f"(R{c[0] + 1},C{c[1] + 1})"


def load(path):
    d = os.path.join(REPO_ROOT, path)
    gi = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    letters, groups, starts, targets, nbr, _, free = SF.board(gi)
    B = type("B", (), dict(letters=letters, groups=groups, starts=starts, targets=targets, nbr=nbr))
    states, acts, _ = mm.replay(B, it)
    said, k = {}, 0                     # message text by the action it was sent with
    for t in it["turns"]:
        for e in t:
            ty, c = e["action"]["type"], e["action"].get("content", "")
            if ty in ("move", "player_done"):
                k += 1
            elif ty == "message":
                said[k] = c.split(": ", 1)
    seat_of = {letters[i]: mm.SEATS[s].split()[1] for s, g in enumerate(groups) for i in g}

    def board(k):
        pos, done, _ = states[k]
        frozen = {letters[i] for s, g in enumerate(groups) if done >> s & 1 for i in g}
        return {letters[i]: tuple(free[p]) for i, p in enumerate(pos)}, frozen

    return dict(gi=gi, it=it, acts=acts, said=said, seat_of=seat_of, board=board, n=gi["grid_size"],
                walls={tuple(w) for w in gi["walls"]}, targets={k: tuple(v) for k, v in gi["target_positions"].items()},
                success=bool(it["Success"]), reason=rf.end_reason(it, bool(it["Success"])))


def region(ep, src, blocked):
    """Cells reachable from src, with walls and `blocked` closed."""
    n, seen, q = ep["n"], {src}, deque([src])
    while q:
        r, c = q.popleft()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in ep["walls"] and nb not in blocked and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return seen


def path(ep, src, dst, blocked):
    """A shortest path for one object with every other object standing still."""
    n, prev, q = ep["n"], {src: None}, deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            out = [u]
            while prev[out[-1]] is not None:
                out.append(prev[out[-1]])
            return out[::-1]
        r, c = u
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in ep["walls"] and nb not in blocked and nb not in prev:
                prev[nb] = u
                q.append(nb)
    return None


def bubble(fig, x, top, w, who, head, text):
    """One message: tinted box, speaker chip, action number, the words (8 pt)."""
    colour, lines = SEAT[who], wrap_words(text, CHARS)
    h = 0.24 + LEAD * len(lines) + 0.05
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, (top - h) / H, w / W, h / H])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc=to_rgba(colour, 0.07), ec=to_rgba(colour, 0.55), lw=0.6))
    ax.add_patch(FancyBboxPatch((0.06, 0.05), 0.56, 0.15, boxstyle="round,pad=0,rounding_size=0.025",
                                fc=colour, ec="none"))
    ax.text(0.34, 0.125, who, fontsize=FS_MSG, color=PAPER, fontweight="bold", ha="center", va="center")
    ax.text(0.70, 0.125, head, fontsize=FS_MSG, color=colour, va="center", fontweight="bold")
    for j, line in enumerate(lines):
        ax.text(0.06, 0.24 + LEAD * (j + 0.5), line, fontsize=FS_MSG, color=INK, va="center")
    return h


def panel(ax, ep, k, *, step_off):
    """The board after action k: objects by owner, B's and D's targets, D's way in."""
    pos, frozen = ep["board"](k)
    draw_grid(ax, ep["n"], ep["walls"], labels=FS_SMALL)
    others = {c for o, c in pos.items() if o != "D"}
    if step_off:
        way = path(ep, pos["D"], ep["targets"]["D"], others)
        route(ax, way[:-1], SEAT["Green"], OPEN_DASH, lw=1.5)
        # the last step and B's move stop short of the letter they point at
        draw_arrow(ax, way[-2], way[-1], SEAT["Green"], lw=1.5, scale=7.5, shrink=(4.0, 9.0), ls=OPEN_DASH, zorder=3.5)
        draw_arrow(ax, ep["board"](k - 1)[0]["B"], pos["B"], INK, lw=2.2, scale=11.0, shrink=(6.0, 8.5), zorder=9)
    else:
        walls_now = {pos[o] for o in frozen}
        before = region(ep, pos["D"], set())
        after = region(ep, pos["D"], walls_now)
        tint(ax, after, SEAT["Green"], 0.13)
        outline(ax, after, ep["walls"], ep["n"], SEAT["Green"])
        hatch_cells(ax, before - after - walls_now, SEAT["Green"], "//", lw=0.3, edge=to_rgba(SEAT["Green"], 0.6))
        hatch_cells(ax, walls_now, WALL, "xxx", alpha=0.20)
    for o in ("B", "D"):
        sits = next((p for p, c in pos.items() if c == ep["targets"][o]), None)
        draw_target(ax, ep["targets"][o], o, SEAT[ep["seat_of"][o]], nested=bool(sits) and sits != o,
                    lw=1.3, letter=8.7)
    for o, c in sorted(pos.items()):
        marker(ax, c, o, SEAT[ep["seat_of"][o]], frozen=o in frozen)


def facts(ep):
    """What the caption and the notes state, read off the replayed episode."""
    acts = ep["acts"]
    kinds = [(mm.SEATS[s].split()[1], lab) for s, _, lab, _ in acts]
    return dict(total=len(acts), kinds=kinds)


def main():
    ff, st = (load(EPISODES[p]) for p in ("freeform", "structured"))
    gi = st["gi"]
    entrance, dead_end = tuple(gi["device"]["entrance"]), tuple(gi["device"]["dead_end"])
    for ep in (ff, st):     # the board the caption describes, in both episodes
        assert ep["targets"]["B"] == entrance and ep["targets"]["D"] == dead_end
        assert ep["seat_of"]["B"] == "Orange" and ep["seat_of"]["D"] == "Green"
        pos, frozen = ep["board"](DECISION - 1)
        assert pos["B"] == entrance and pos["D"] not in {tuple(c) for c in gi["device"]["corridor"]} and not frozen
        # Orange could declare done at action 26 in both episodes: the same choice
        assert all(pos[o] == ep["targets"][o] for o, s in ep["seat_of"].items() if s == "Orange")
        for a, who, frags in SAID["freeform" if ep is ff else "structured"]:
            assert ep["said"][a][0] == f"Player {who}", (a, ep["said"][a])
            marked(ep["said"][a][1], frags)
    F, S = facts(ff), facts(st)
    assert F["kinds"][DECISION - 1] == ("Orange", "B up") and ff["success"]
    assert S["kinds"][DECISION - 1] == ("Orange", "done") and st["reason"] == "lose:cycle"
    pos, frozen = st["board"](DECISION)
    assert dead_end not in region(st, pos["D"], {pos[o] for o in frozen})
    asks = [a for a, (who, txt) in st["said"].items() if a < DECISION and who == "Player Green" and cell(entrance) in txt]
    assert asks == [10, 16, 19, 22, 25], asks
    # in the freeform episode D passes the entrance before B returns to it
    passed = next(k for k in range(DECISION, F["total"] + 1) if ff["board"](k)[0]["D"][1] < entrance[1])
    back = next(k for k in range(passed, F["total"] + 1) if ff["board"](k)[0]["B"] == entrance)
    done = next(k for k, (who, lab) in enumerate(F["kinds"], 1) if who == "Orange" and lab == "done")
    assert passed < back < done

    texts = {p: [(a, who, marked(st_or_ff["said"][a][1], frags)) for a, who, frags in SAID[p]]
             for p, st_or_ff in (("freeform", ff), ("structured", st))}
    texts["structured"][-1] = (DECISION, "Orange", "done")
    col_h = max(sum(0.24 + LEAD * len(wrap_words(t, CHARS)) + 0.05 + 0.06 for _, _, t in texts[p]) for p in texts)
    notes = {
        "freeform": [f"D passes the entrance at action {passed}; B returns",
                     f"at {back}, Orange declares done at {done}; won at {F['total']}."],
        "structured": ["B is frozen on the entrance and D cannot reach",
                       f"its target; lost by the cycle rule after action {S['total']}."],
    }
    H = HEAD + SIDE + 0.30 + col_h + 2 * 0.17 + 0.20 + 0.78
    fig = plt.figure(figsize=(TEXT_W, H))
    bottom = H - HEAD - SIDE
    heads = {"freeform": ("(a) freeform", f"action {DECISION}: Orange moves B off its target"),
             "structured": ("(b) structured", f"action {DECISION}: Orange declares done")}
    for x, p, ep in zip(COLS, ("freeform", "structured"), (ff, st)):
        ax = square_axes(fig, x, bottom, SIDE)
        panel(ax, ep, DECISION, step_off=p == "freeform")
        t1, t2 = heads[p]
        fig.text(*at(fig, x - 0.05, H - 0.10), t1, fontsize=FS_TITLE, color=INK, va="top", fontweight="bold")
        fig.text(*at(fig, x - 0.05, H - 0.33), t2, fontsize=FS_TICK, color=INK, va="top")
        y = bottom - 0.14
        for a, who, t in texts[p]:
            y -= bubble(fig, x, y, SIDE, who, f"action {a}", t) + 0.06
        for i, line in enumerate(notes[p]):
            fig.text(*at(fig, x, y - 0.04 - i * 0.17), line, fontsize=FS_SMALL, color=INK, va="top")

    keys = [
        Line2D([], [], marker="o", linestyle="none", markersize=7, markerfacecolor=SEAT["Green"],
               markeredgecolor=INK, label="object, colour = owner (Green, Orange, Blue)"),
        Patch(facecolor=to_rgba(WALL, 0.20), edgecolor=WALL, hatch="xxx", linewidth=0.5,
              label="declared done: object frozen (square)"),
        Patch(facecolor="none", edgecolor=MUTED, linewidth=1.3, linestyle=(0, (2.2, 1.5)),
              label="target of B (entrance) and D (dead end)"),
        Patch(facecolor=to_rgba(SEAT["Green"], 0.13), edgecolor=SEAT["Green"], linewidth=1.1,
              label="where D can still move"),
        Line2D([], [], color=SEAT["Green"], linestyle=OPEN_DASH, linewidth=1.5, label="D's way to its target"),
        Patch(facecolor="none", edgecolor=to_rgba(SEAT["Green"], 0.6), hatch="//", linewidth=0.3,
              label="cells D loses to the done"),
    ]
    fig.legend(handles=keys, loc="lower left", bbox_to_anchor=at(fig, COLS[0] - 0.10, 0.04), ncol=2,
               frameon=False, fontsize=FS_LEGEND, handletextpad=0.6, columnspacing=1.0, labelspacing=0.5,
               borderpad=0.0)
    print(publish(fig, OUT_DIR, "res_cut_s3"))
    print(dict(passed=passed, back=back, done=done, ff_total=F["total"], st_total=S["total"], asks=asks))


if __name__ == "__main__":
    main()
