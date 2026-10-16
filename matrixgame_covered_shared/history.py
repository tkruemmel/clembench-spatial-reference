"""Recent-play boards for the *_grid_final game arms.

The final arms still rebuild the context every turn, but they no longer drop
all history. Each context renders the last ROUNDS rounds of play, oldest
first, as a MESSAGE BOARD (every message with its sender; absent under
`none` and in solo) and a MOVE BOARD (every move, no player named). It also
re-centres the grid boards' column labels (`aligned`).
"""

import random
import re
from typing import Dict, List

# K: rounds of play kept in view. Every final prompt states this number
# (the packages' tests check it), so change both together.
ROUNDS = 4

# Seat names in turn order (the dyad takes the first two). Neither
# alphabetical, rainbow nor RGB order, so a name does not tell a player
# where its owner sits in the round, and no name is an object letter.
SEATS = tuple(f"Player {c}" for c in ("Green", "Orange", "Blue", "Yellow", "Red", "Purple"))


# The base Board's column header: labels 4 wide after a 5-space indent.
_HEADER = re.compile(r"^ {5}((?: C\d+ )+)$", re.M)


def aligned(text: str) -> str:
    """Put each column label of a grid board over its own cell.

    The base Board renderer sets the labels 4 wide over cells 5 wide (4 plus
    the `|`), so they drift one character per column and C8 lands over the
    seventh cell. The older arms keep that header as they were played; the
    final masters pass every grid through here. Compact boards are unchanged.
    """
    return _HEADER.sub(lambda m: (" " * 6 + " ".join(f"{l:^4}" for l in m.group(1).split())).rstrip(), text)


def _cell(rc) -> str:
    return f"(R{rc[0] + 1},C{rc[1] + 1})"


def capture(record: List[Dict], player: str, parsed: Dict, board, directions) -> None:
    """Append one accepted turn. Call before the master applies the move."""
    entry = {"turn": len(record) + 1, "player": player, "message": parsed.get("message")}
    if parsed.get("is_done"):
        entry["done"] = True
    else:
        obj, direction = parsed["object"], parsed["direction"]
        r, c = board.object_positions[obj]
        dr, dc = directions[direction]
        entry.update(object=obj, direction=direction, src=(r, c), dst=(r + dr, c + dc))
    record.append(entry)


def boards(record: List[Dict], viewer: str, seed: str, messages: bool = False, shuffle: bool = True) -> str:
    """The MESSAGE BOARD and MOVE BOARD over the viewer's last ROUNDS rounds.

    A round is one of the viewer's turns and the teammates' turns that follow it,
    so the window starts at the viewer's ROUNDS-th last own turn (or at the first
    turn before it has taken ROUNDS), however many teammates have finished.

    Messages keep their sender, e.g. `Player Blue said "..."`. Moves name no
    player, so a teammate's move reads `X moved from (R1,C4) to (R2,C4)` and
    only the viewer's own move is shown in full, `You moved A from (R1,C1) to
    (R2,C1)`. Moves stay in the order they were made. The teammates' messages
    between two of the viewer's own are shuffled (seeded, so a replay renders
    the same board), so a message cannot be matched to a move by counting
    lines; `shuffle=False` keeps them in the order sent (the *_final_chrono
    variant). A done declaration is no move. Under a messaging protocol
    (`messages`) it puts `Player Blue declared done` on the MESSAGE BOARD in
    place of its "done" message, and that line is the only finish signal the
    final masters give, so under `none` a finish is not announced. An empty
    board is left out, so "" before the first turn. In solo every turn is the
    viewer's, so a round is one turn.
    """
    rng = random.Random(f"{seed}-{len(record)}-{viewer}")
    own_turns = [i for i, e in enumerate(record) if e["player"] == viewer]
    start = own_turns[-ROUNDS] if len(own_turns) >= ROUNDS else 0
    msgs, moves, block = [], [], []
    for e in record[start:]:
        own = e["player"] == viewer
        msg = " ".join((e["message"] or "").split())
        if own:
            if shuffle:
                rng.shuffle(block)
            msgs += block
            block = []
        lines, who = (msgs if own else block), f'{e["player"]}{" (you)" if own else ""}'
        if msg and not (e.get("done") and msg.lower() == "done"):
            lines.append(f'{who} said "{msg}"')
        if e.get("done"):
            if messages:
                lines.append(f"{who} declared done")
        else:
            cells = f"from {_cell(e['src'])} to {_cell(e['dst'])}"
            moves.append(f"You moved {e['object']} {cells}" if own else f"X moved {cells}")
    if shuffle:
        rng.shuffle(block)
    msgs += block
    return "".join(
        f"{name} (the last {ROUNDS} rounds, oldest first):\n" + "\n".join(lines) + "\n\n"
        for name, lines in (("MESSAGE BOARD", msgs), ("MOVE BOARD", moves)) if lines
    )
