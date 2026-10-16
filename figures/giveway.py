"""The give-way panel of fig:cut-corridor (drawn by figures/cuts.py).

DeepSeek, six players, S3 board 14, the same board as the freeform cut of
that figure. Under hybrid, B steps onto its own target, the corridor
entrance, with the same move as under freeform (move 50), promising to step
aside; on its next turn it steps off instead of declaring `done`, and D
later passes. The freeform B declares `done` on its next turn and cuts D off.

Asserted from the move logs: the two episodes share the board; B reaches the
entrance on the same move in both; under hybrid B's next move steps off it,
with no `done` before; at that instant D has a route to its target while B
can move, and none if B were frozen (the fork is live); B declares `done`
only later and the episode is won. Every quote is checked verbatim.

    python3 figures/giveway.py      # prints the facts; the figure is cuts.py's
"""

from __future__ import annotations

import json
import os
import string
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "analysis/qualitative_analysis"))
import cut  # noqa: E402

ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

HYBRID = ("dsv41", "6p", "S3", "hybrid", "instance_00014")
HOLDER, VICTIM_OBJ = "B", "D"
BOARD_KEYS = ("grid_size", "walls", "start_positions", "target_positions",
              "player_objects", "device")
# (move the message rides on, fragments cut from it, never reworded)
SAID = [
    (50, ["D, this blocks (R5,C2) temporarily; I'll move left to (R5,C1) as soon "
          "as you signal you're ready to pass through to (R4,C2)."]),
    (56, ["I’m stepping B left to (R5,C1) to clear (R5,C2) for D’s climb "
          "to (R1,C2)"]),
]


def run_dir(model, suite, level, proto, inst):
    for rec in json.load(open(os.path.join(cut.HERE, f"episodes.{ARM}.json"))):
        p = rec["path"]
        if (rec["model"], rec["suite"]) == (model, suite) and \
                f"/{level}_{proto}_" in p and p.endswith(inst):
            return os.path.join(cut.REPO, p)
    raise SystemExit(f"no episode {model} {suite} {level} {proto} {inst}")


def replay(path):
    """Accepted moves, done declarations (after which move) and the message
    that rides on each move, plus the outcome."""
    ins = json.load(open(os.path.join(path, "instance.json")))
    it = json.load(open(os.path.join(path, "interactions.json")))
    moves, done_at, said, outcome = [], {}, {}, None
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            kind = a.get("type")
            if e.get("from") != "GM":
                continue
            if kind == "move":
                m = cut.MOVE.match(a.get("content", ""))
                if m:
                    moves.append((m.group(2), m.group(3)))
            elif kind == "player_done":
                done_at[a["content"].split()[1]] = len(moves)
            elif kind == "message":
                said.setdefault(len(moves), a["content"].split(": ", 1)[1])
            elif kind in ("success", "lose", "abort"):
                outcome = kind
    return ins, moves, done_at, said, outcome


def board_at(ins, moves, upto):
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    for o, d in moves[:upto]:
        pos = cut.step(pos, o, d)
    return pos


def arrival(ins, moves, obj):
    tgt = tuple(ins["target_positions"][obj])
    return next(p for p in range(1, len(moves) + 1)
                if board_at(ins, moves, p)[obj] == tgt)


def giveway(cut_case):
    """The hybrid panel, checked against the freeform cut it is paired with."""
    ins_f, moves_f, done_f, _, out_f = replay(os.path.join(cut.REPO, cut_case["meta"]["path"]))
    ins, moves, done_h, said, out_h = replay(run_dir(*HYBRID))
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    owner = {o: string.ascii_uppercase[i]
             for i, g in enumerate(ins["player_objects"]) for o in g}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    entrance = cut_case["entrance"]

    for key in BOARD_KEYS:
        assert ins[key] == ins_f[key], key                       # same board
    assert (cut_case["sealer"], cut_case["victim"]) == (HOLDER, VICTIM_OBJ)
    assert tgt[HOLDER] == entrance
    arrive = arrival(ins, moves, HOLDER)
    assert arrive == arrival(ins_f, moves_f, HOLDER)              # same move in both
    # freeform: B's next turn is the done
    assert done_f[HOLDER] == cut_case["ply"]
    assert all(o != HOLDER for o, _ in moves_f[arrive:cut_case["ply"]])
    assert out_f == "lose"
    # hybrid: B's next turn steps off the entrance, before any done of its own
    step_off = next(i + 1 for i in range(arrive, len(moves)) if moves[i][0] == HOLDER)
    assert moves[step_off - 1] == (HOLDER, "left"), moves[step_off - 1]
    assert done_h[HOLDER] > step_off
    fork = step_off - 1                                          # the board B sees
    pos = board_at(ins, moves, fork)
    assert pos[HOLDER] == entrance
    frozen = {o: pos[o] for o in pos if done_h.get(owner[o], 1e9) <= fork}
    blocked = set(frozen.values())
    live = cut.path(pos[VICTIM_OBJ], tgt[VICTIM_OBJ], n, walls, blocked)
    assert live and entrance in live
    assert cut.path(pos[VICTIM_OBJ], tgt[VICTIM_OBJ], n, walls,
                    blocked | {pos[HOLDER]}) is None              # the fork is live
    assert out_h == "success"
    for p, frags in SAID:
        at_ = 0
        for f in frags:
            at_ = said[p].index(f, at_)                          # verbatim, in order
    case = dict(grid=n, walls=walls, pre=pos, victim=VICTIM_OBJ, targets=tgt,
                culprits=cut_case["culprits"], frozen_pre=frozen,
                frozen_post=frozen, seals=[])
    return dict(case=case, live=live, arrive=arrive, step_off=step_off,
                done=done_h[HOLDER], total=len(moves), said=said,
                path=os.path.relpath(run_dir(*HYBRID), REPO_ROOT))


if __name__ == "__main__":
    import cuts
    g = giveway(cuts.pick(cut.cases(), cuts.CORRIDOR))
    print(f"hybrid: B on the entrance after move {g['arrive']}, steps off at move "
          f"{g['step_off']}, done after move {g['done']}, won in {g['total']}")
