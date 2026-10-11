"""B's broken promise on S3 #10, DeepSeek hybrid, six players (S9.4).

A timeline of the channel: B's promise to step aside, B's `done` on its next
turn, three of the teammate messages that keep asking it to move, and the
abort. Every quote is cut from the transcript, never reworded, and the script
checks each fragment against the recorded message before drawing it. The move
counts and the tally of requests to B are replayed from the same log.

    python3 figures/s3_promise.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from style import CUE, FAINT, INK, MUTED, OBJECT, OWNER, PAPER, save

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")
RUN = os.path.join(REPO_ROOT, "results_dsv41_grid_n15_6p", "*", "*",
                   "S3_hybrid_6p_grid", "instance_00010", "interactions.json")
PLAYER = dict(zip("123456", "ABCDEF"))

# (speaker, fragment that locates the message, fragment shown)
QUOTES = [
    ("B", "temporarily move right to (R5,C6)",
     "D, I'll temporarily move right to (R5,C6) when you reach (R6,C5) so "
     "you can climb through (R5,C5), then return."),
    ("D", "B, I know you're done",
     "B, I know you're done at (R5,C5), but it's the only gate to my goal. "
     "Please step right to (R5,C6) when I reach (R6,C5)"),
    ("A", "B and F, you are done",
     "B and F, you are done, but your cells (R5,C5) and (R8,C7) block my only "
     "routes to (R6,C3). If B can temporarily step aside from (R5,C5), I can "
     "pass and you can return."),
    ("C", "B, your done at (R5,C5)",
     "B, your done at (R5,C5) still blocks the only upper corridor"),
]


def norm(text: str) -> str:
    return text.replace("’", "'")


def read_log(run: str = RUN):
    """Messages with the number of moves played before each, plus events."""
    d = json.load(open(glob.glob(run)[0]))
    n, msgs, events = 0, [], {}
    for rnd in d["turns"]:
        for e in rnd:
            a = e["action"]
            if e["from"] == "GM":
                if a["type"] == "move":
                    n += 1
                elif a["type"] == "player_done":
                    events.setdefault("done " + a["content"][7], n)
                elif a["type"] == "abort":
                    events["abort"] = n
            elif a["type"] == "get message":
                body = norm(a["content"]).split("PLAN")[0]
                msgs.append((n, PLAYER[e["from"][-1]], body))
    return msgs, events


def locate(msgs, speaker, key, shown):
    for n, who, body in msgs:
        if who == speaker and key in body:
            assert shown.rstrip(".") in body, f"{shown!r} is not in the transcript"
            return n
    raise SystemExit(f"no message from {speaker} containing {key!r}")


def tally(msgs, since, until):
    """Teammate messages after B's done that ask B to move, and all of them.

    One message per turn: the reprompts after the final invalid move are
    retries of the same turn and are not counted again.
    """
    turns = {}
    for n, who, body in msgs:
        if since < n <= until and who != "B":
            turns.setdefault((n, who), body)
    # a request to B names it and later asks it to step or vacate; E's
    # "B at (R5,C5) still blocks the upper corridor; I can't clear it" does not
    asks = sum(bool(re.search(r"\bB\b.*?\b(step|vacate)", b, re.S))
               for b in turns.values())
    return asks, len(turns)


def render(out_dir: str = OUT_DIR):
    msgs, ev = read_log()
    done, abort = ev["done B"], ev["abort"]
    rows = []
    for speaker, key, shown in QUOTES:
        rows.append(("quote", locate(msgs, speaker, key, shown), speaker, shown))
        if speaker == "B":
            rows.append(("done", done, "B", "done"))
    asks, total = tally(msgs, done, abort)
    abort_text = ("Cannot move A left: (R5,C5) is occupied.   \u2026   "
                  "Too many invalid moves")
    rows.append(("abort", abort, "GM", abort_text))

    wrap = 100
    lines = [textwrap.wrap(f"\u201c{r[3]}\u201d", wrap) if r[0] == "quote"
             else [r[3]] for r in rows]
    heights = [0.17 + 0.135 * len(ls) for ls in lines]
    gap, top = 0.09, 0.34
    H = top + sum(heights) + gap * (len(rows) - 1) + 0.06
    W = 6.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    x_move, x_disc, x_text, right = 0.3, 0.72, 0.98, W - 0.06
    ax.text(x_move, top / 2, "moves\nplayed", ha="center", va="center",
            fontsize=6.2, color=MUTED, linespacing=1.0)
    y = top
    for (kind, n, who, _), ls, h in zip(rows, lines, heights):
        mid = y + h / 2
        colour = OWNER["B"] if who == "B" else OBJECT
        ax.text(x_move, mid, str(n), ha="center", va="center", fontsize=7.4,
                color=INK)
        box = dict(boxstyle="round,pad=0,rounding_size=0.06")
        if kind == "abort":
            ax.add_patch(FancyBboxPatch((x_text, y), right - x_text, h, **box,
                                        facecolor="#F3F4F6", edgecolor=FAINT,
                                        lw=0.7))
            ax.text(x_disc, mid, "GM", ha="center", va="center", fontsize=6.6,
                    color=MUTED, fontweight="bold")
            ax.text(x_text + 0.1, mid, ls[0], ha="left", va="center",
                    fontsize=6.6, color=INK, family="monospace")
        else:
            ax.add_patch(Circle((x_disc, mid), 0.12, facecolor=colour,
                                edgecolor=PAPER, lw=0.8))
            ax.text(x_disc, mid, who, ha="center", va="center", fontsize=7.2,
                    color=PAPER, fontweight="bold")
            filled = kind == "done"
            width = 0.6 if filled else right - x_text
            ax.add_patch(FancyBboxPatch((x_text, y), width, h, **box,
                                        facecolor=colour if filled else PAPER,
                                        edgecolor=colour, lw=0.9))
            ax.text(x_text + (width / 2 if filled else 0.1), mid, "\n".join(ls),
                    ha="center" if filled else "left", va="center",
                    fontsize=7.2 if filled else 7.0,
                    color=PAPER if filled else INK,
                    fontweight="bold" if filled else "normal",
                    family="monospace" if filled else None, linespacing=1.3)
        y += h + gap

    print(f"B done after move {done}, abort after move {abort}: "
          f"{asks} of {total} teammate turns ask B to step or vacate")
    return save(fig, out_dir, "s3_promise", ("pdf", "png"))


if __name__ == "__main__":
    for path in render():
        print(path)
