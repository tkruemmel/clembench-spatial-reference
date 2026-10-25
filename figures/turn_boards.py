"""Draws the message board and the move board one player is sent.

    python3 figures/turn_boards.py

Illustrative turns, not a recorded episode: four rounds of a freeform dyad
game on S2 instance 15, the board of the board-model figure, played through
the final arm's own game master, so the two boards are exactly what it sends
Player Green next. Shaded bands mark the rounds; they are not sent.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import List

from spatial_levels import OUT_DIR, REPO_ROOT
from style import INK, MONO, MUTED, plt, save

PKG = os.path.join(REPO_ROOT, "matrixgame_covered_comm_grid_final")
EXPERIMENT, GAME_ID = "S2_freeform_grid", 15
BAND = "#F1F3F6"  # every other round, too light to read as data

# Green owns A, C, D and moves first; Orange owns B, E, F.
TURNS = [
    ("C leaves (R6,C5) and heads for (R6,C2).", "C up"),
    ("Good, (R6,C5) is B's target. B gets there in two moves.", "B up"),
    ("A heads for (R7,C6), going around through row 8.", "A left"),
    ("B is home. I'll keep F off (R7,C6) for your A.", "B left"),
    ("D steps aside to (R7,C2) to let A pass.", "D down"),
    ("F goes up column 8 to (R4,C7).", "F right"),
    ("D waits at (R7,C2) until A is through.", "D left"),
    ("F keeps going up column 8.", "F up"),
]


class _Silent:
    name = "illustration"

    def set_gen_arg(self, *_):
        pass


def sent_boards() -> str:
    """Plays TURNS through the game master and returns Green's two boards."""
    from clemcore.clemgame.registry import GameSpec

    spec = importlib.util.spec_from_file_location(
        "final_dyad_master", os.path.join(PKG, "master.py"))
    master = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(master)

    with open(os.path.join(PKG, "in", "instances.json")) as fh:
        data = json.load(fh)
    instance, = [i for e in data["experiments"] if e["name"] == EXPERIMENT
                 for i in e["game_instances"] if i["game_id"] == GAME_ID]
    gm = master.FinalMatrixGameMaster(
        GameSpec(game_name="g", game_path=PKG, players=2), {"name": "x"},
        [_Silent(), _Silent()])
    gm._on_setup(**instance)
    for k, (message, move) in enumerate(TURNS):
        player = gm.player_a if k % 2 == 0 else gm.player_b
        response = f"message: {message}\nmove: {move}"
        assert gm._validate_player_response(player, response), response
        gm._on_valid_player_response(player, gm._parse_response(player, response))
    context = gm.get_context_for(gm.player_a)["content"]
    return context.split("CURRENT BOARD")[0]


def rounds(lines: List[str], opener: str) -> List[List[str]]:
    """A round opens with the viewer's own line: its turn, then the teammate's."""
    out: List[List[str]] = []
    for line in lines:
        if line.startswith(opener):
            out.append([])
        out[-1].append(line)
    return out


def render(out_dir: str = OUT_DIR, formats=("pdf", "png")) -> List[str]:
    message_part, move_part = sent_boards().strip().split("\n\n")
    blocks = []
    for part, opener in [(message_part, "Player Green (you) said"),
                         (move_part, "You moved")]:
        header, *lines = part.split("\n")
        blocks.append((header, rounds(lines, opener)))
    assert [len(r) for _, r in blocks] == [4, 4]
    assert all(len(rnd) == 2 for _, r in blocks for rnd in r)

    size, line = 8.5, 0.178  # mono pt, inches per line
    gutter, top, gap = 0.66, 0.10, 0.22
    n_lines = sum(1 + 2 * len(r) for _, r in blocks)
    W, H = 6.30, top * 2 + n_lines * line + gap
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # inches from the top
    ax.axis("off")

    y = top + line / 2
    for header, rnds in blocks:
        ax.text(gutter, y, header, fontsize=size, color=INK, va="center", **MONO)
        y += line
        for k, rnd in enumerate(rnds):
            if k % 2 == 0:
                ax.add_patch(plt.Rectangle((0, y - line / 2), W, 2 * line,
                                           facecolor=BAND, edgecolor="none"))
            ax.text(0.06, y + line / 2, f"round {k + 1}", fontsize=9.0,
                    color=MUTED, va="center")
            for i, row in enumerate(rnd):
                ax.text(gutter, y + i * line, row, fontsize=size, color=INK,
                        va="center", **MONO)
            y += 2 * line
        y += gap
    # the widest line must end inside the page
    fig.canvas.draw()
    right = max(t.get_window_extent().x1 for t in ax.texts) / fig.dpi
    assert right < W - 0.02, right
    return save(fig, out_dir, "turn_boards", formats)


if __name__ == "__main__":
    for path in render():
        print(f"wrote {os.path.relpath(path)}")
