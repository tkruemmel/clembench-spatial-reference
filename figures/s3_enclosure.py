"""How B is enclosed on S3 #10, DeepSeek freeform, six players
(fig:s3-10-enclosure).

Left, the board after the sealing move: D has stepped into (R8,C5), the last
free cell next to B, which B had vacated and offered to D two moves earlier.
B's other neighbours are a wall and F, whose owner has declared `done`. D's
two other legal moves are dashed. Right, the channel: the order the team
agreed (D up the corridor first, B after), F's `done`, B's offer, D's
acceptance, and the abort.

Asserted before drawing: seal_alt.py records this abort with B as the victim,
D right as the sealing move and D up / D left as its harmless alternatives;
B's one exit before the move is the cell D takes and it has none after; F was
done before the seal; B's message names the cell and D and passes the written
invitation rule of failures.py; the episode aborts on B's turn. The panel is
cropped to columns C3-C8, and the script asserts that no object or drawn
cell lies outside that window.

    python3 figures/s3_enclosure.py
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from s3_promise import draw_timeline, moves_of, board_at, quote, transcript
from s3_worked import load_instance
from style import (CUE, INK, MUTED, OBJECT, OWNER, centre, draw_arrow, draw_disc,
                   draw_grid, draw_target, save)
import seal_alt as sa          # on sys.path through s3_promise
import seal_averted as sv
import failures

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
RUN = os.path.join(REPO_ROOT, "results_dsv41_grid_updated_prompt_n15_6p", "*", "*",
                   "S3_freeform_6p_grid", "instance_00010")
KEY = ("dsv41", "6p", "S3", "freeform", "instance_00010")
VICTIM, SEALER, DONE = "B", "D", "F"
COLS = range(2, 8)  # C3..C8, zero-based

# (speaker, fragments shown, joined by an ellipsis); each is checked verbatim
QUOTES = [
    ("D", ["B, I need to pass through R5C5; please wait at R5C6/7 until I clear "
           "upward, then take it."]),
    ("B", ["D, take R8C5 and ascend C5 first once A has left R7C5; I’ll follow "
           "from R8C6 and settle at R5C5 after you pass."]),
    ("B", ["B moving right to R8C6, freeing R8C5 for D. D, take R8C5 and ascend "
           "C5; I'll follow once R7C5 clears."]),
    ("D", ["D moving right to R8C5 as planned.",
           "B, hold R8C6 until I leave R8C5 upward."]),
]


def cell_name(c):
    return f"R{c[0] + 1}C{c[1] + 1}"


def facts(inst, run_dir):
    case, = [c for c in sa.cases()
             if (c["meta"]["model"], c["meta"]["suite"], c["meta"]["level"],
                 c["meta"]["proto"], c["meta"]["inst"]) == KEY]
    seal = case["ply"]                                  # the sealing move, 1-based
    assert case["victim"] == VICTIM and case["played"] == (SEALER, "right"), case["played"]
    assert sorted(case["harmless"]) == [(SEALER, "left"), (SEALER, "up")], case["harmless"]
    assert DONE in case["done"], case["done"]           # F walls one side for good
    walls, n = case["walls"], case["grid"]
    moves = moves_of(run_dir)
    assert moves == case["moves"]
    pre, post = board_at(inst, moves, seal - 1), board_at(inst, moves, seal)
    exits = lambda pos: [sa.step(pos, VICTIM, d)[VICTIM]
                         for _, d in sa.legal_moves(pos, [VICTIM], walls, n)]
    cell = post[SEALER]
    assert exits(pre) == [cell] and exits(post) == []
    rows = transcript(run_dir)
    # the offer: B's own move vacated the cell, and its message rides on that move
    offer, _, _ = quote(rows, *QUOTES[2])
    assert moves[offer - 1] == (VICTIM, "right")
    assert board_at(inst, moves, offer - 1)[VICTIM] == cell
    assert seal - offer <= 6                            # within the six-ply window
    text = next(t for m, k, w, t, _ in rows if k == "msg" and w == VICTIM and m == offer)
    rx = re.compile(sv.CELL % (cell[0] + 1, cell[1] + 1), re.I)
    assert failures.invites(text, rx, SEALER)           # the written rule
    done_f = next(m for m, k, w, _, _ in rows if k == "done" and w == DONE)
    end = rows[-1]
    assert end[1] == "abort" and end[0] == len(moves)
    refused = next(r for r in rows if r[1] == "invalid" and r[2] == VICTIM and r[0] > seal)
    harmless = [sa.step(pre, o, d)[o] for o, d in case["harmless"]]
    drawn = list(post.values()) + harmless + [pre[SEALER]]
    assert all(c[1] in COLS for c in drawn), "crop would hide a mark"
    return dict(seal=seal, offer=offer, cell=cell, pre=pre, post=post, rows=rows,
                harmless=harmless, done_f=done_f, end=end, refused=refused)


def render(out_dir: str = OUT_DIR):
    inst = load_instance()
    run_dir, = glob.glob(RUN)
    f = facts(inst, run_dir)
    post, was = f["post"], f["pre"][SEALER]

    timeline, after = [], -1
    for who, frags in QUOTES:
        n, _, text = quote(f["rows"], who, frags, after)
        timeline.append((n, who, text, "quote"))
        after = n
    timeline = sorted(timeline + [(f["done_f"], DONE, "done", "done")],
                      key=lambda t: t[0])
    timeline.append((f["end"][0], "GM",
                     f"{f['refused'][3]} … {f['end'][3]}", "gm"))

    W, aw = 6.3, 2.2
    ah, bottom = aw * 8 / 6, 0.62
    H = bottom + ah + 0.42
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.3 / W, bottom / H, aw / W, ah / H])
    walls = [w for w in inst["walls"] if w[1] in COLS]
    shift = lambda cell: (cell[0], cell[1] - COLS[0])
    draw_grid(ax, 8, [shift(w) for w in walls], labels=0)
    ax.set_xlim(0, 6)
    for c in COLS:
        ax.text(c - COLS[0] + 0.5, -0.3, f"C{c + 1}", ha="center", va="center",
                fontsize=6.0, color=MUTED)
    for r in range(8):
        ax.text(-0.35, r + 0.5, f"R{r + 1}", ha="center", va="center",
                fontsize=6.0, color=MUTED)
    fig.text(0.3 / W, (bottom + ah + 0.27) / H,
             f"after move {f['seal']}: B has no legal move", fontsize=7.4, color=INK)

    draw_target(ax, shift(tuple(inst["target_positions"][SEALER])), SEALER, OBJECT,
                nested=False, letter=9.0)
    for q in f["harmless"]:
        ax.add_patch(Circle(centre(shift(q)), 0.3, facecolor="none",
                            edgecolor=OBJECT, lw=1.0, ls=(0, (1.8, 1.4)), zorder=7))
    ax.add_patch(Circle(centre(shift(was)), 0.32, facecolor=to_rgba(OBJECT, 0.22),
                        edgecolor="none", zorder=7))
    ax.text(*centre(shift(was)), SEALER, ha="center", va="center", fontsize=8.0,
            color=OBJECT, fontweight="bold", zorder=8)
    draw_arrow(ax, shift(was), shift(post[SEALER]), CUE, lw=1.6, scale=9,
               shrink=(10.0, 10.0), zorder=8)
    colour = {k: "#8FAED0" for k in post}
    colour.update({VICTIM: OWNER["B"], SEALER: OBJECT})
    for o, c in sorted(post.items()):
        draw_disc(ax, shift(c), o, colour[o], radius=0.32, letter=7.8,
                  ring=INK if o == DONE else None, ring_lw=1.4)

    handles = [
        Line2D([], [], color=CUE, lw=1.6, marker=">", ms=4,
               label=f"D's move {f['seal']}, into B's last free cell"),
        Line2D([], [], marker="o", ls="none", ms=6.0, mfc="none", mec=OBJECT,
               mew=1.0, label="D's two other legal moves"),
        Line2D([], [], marker="o", ls="none", ms=6.0, mfc="#8FAED0", mec=INK,
               mew=1.2, label="F, declared done"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.2 / W, 0.0),
               ncol=1, frameon=False, fontsize=6.2, handletextpad=0.5,
               labelcolor=INK)
    draw_timeline(fig, aw + 0.55, 0.08, W - aw - 0.6, H - 0.16, timeline, wrap=58)
    print(f"B offers {cell_name(f['cell'])} with move {f['offer']}, D takes it with "
          f"move {f['seal']}; abort after move {f['end'][0]}")
    return save(fig, out_dir, "s3_enclosure", ("pdf", "png"))


if __name__ == "__main__":
    for path in render():
        print(path)
