"""Reads one recorded episode and makes it drawable.

The counterpart of spatial_levels.py: that module supplies boards, this one
supplies what happened on a board. Everything here comes out of a real
interactions.json, so a figure built on it cannot drift from the transcript it
claims to show.

Two guards run on load. The instance named by the transcript must reproduce the
board the transcript was played on, and replaying the recorded directions must
land on the recorded cells. Either failing means the instance files have been
regenerated since the run, and every caption derived from that pairing is
suspect.

Usage
    python3 figures/episodes.py                       # summarise the default run
    python3 figures/episodes.py --run PATH            # summarise another
    python3 figures/episodes.py --find S2_hybrid      # successful runs to choose from
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DeepSeek-V4-Flash on both sides, S2, hybrid protocol. Chosen because this
# board's dependency chain straddles the ownership split: A's target is B's
# cell and B's target is C's cell, with A and C on one side and B on the other,
# so the chain cannot be resolved without the channel.
DEFAULT_RUN = os.path.join(
    REPO_ROOT, "results_deepseek_flash_t06", "DeepSeek-V4-Flash",
    "matrixgame_covered_comm", "S2_hybrid_compact", "instance_00002",
    "interactions.json",
)

INSTANCE_FILES = {
    "matrixgame_covered_comm": ("matrixgame_covered_comm", "in", "instances.json"),
    "matrixgame_covered_comm_3p": ("matrixgame_covered_comm_nplayer", "in",
                                   "instances_3p.json"),
    "matrixgame_covered_comm_6p": ("matrixgame_covered_comm_nplayer", "in",
                                   "instances_6p.json"),
    "matrixgame_covered_solo": ("matrixgame_covered_solo", "in", "instances.json"),
}

STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

MOVE_LINE = re.compile(
    r"^\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b", re.M)


def cellname(cell: Cell) -> str:
    return f"R{cell[0] + 1}C{cell[1] + 1}"


# ------------------------------------------------------------------ loading


def _find_instance(game: str, experiment: str, game_id: int) -> dict:
    if game not in INSTANCE_FILES:
        raise SystemExit(f"no instance file registered for game {game!r}")
    path = os.path.join(REPO_ROOT, *INSTANCE_FILES[game])
    with open(path) as fh:
        data = json.load(fh)
    for exp in data["experiments"]:
        if exp["name"] != experiment:
            continue
        for inst in exp["game_instances"]:
            if inst.get("game_id") == game_id:
                return inst
    raise SystemExit(
        f"{os.path.relpath(path, REPO_ROOT)} has no {experiment} instance with "
        f"game_id {game_id}; the instance files have been regenerated since "
        "this run and the episode can no longer be paired with its board"
    )


def _exchanges(turns: Sequence[Sequence[dict]]) -> List[Tuple[str, str, str]]:
    """Every (recipient, context sent, response received) in play order.

    Rounds group a variable number of player turns, so the pairs are collected
    from the flattened event stream rather than assumed one per round.
    """
    pairs: List[Tuple[str, str, str]] = []
    pending: Optional[Tuple[str, str]] = None
    for event in (e for rnd in turns for e in rnd):
        action = event.get("action", {})
        if action.get("type") == "send message" and event["to"].startswith("Player"):
            pending = (event["to"], action["content"])
        elif action.get("type") == "get message" and pending is not None:
            pairs.append((pending[0], pending[1], action["content"]))
            pending = None
    return pairs


def _bind_exchanges(moves: List[dict], pairs: Sequence[Tuple[str, str, str]]) -> int:
    """Attach the prompt and the raw reply to the move each one produced.

    There are more exchanges than moves: a response that fails to parse costs a
    reprompt, and a `done:` declaration is an exchange with no move at all. So
    the k-th exchange is not the k-th move, and pairing them by index quietly
    puts the wrong transcript beside the wrong board. Each move is bound to the
    next exchange whose reply actually parses to it; the replies that cost a
    reprompt are skipped because they do not parse, which is what made them
    reprompts. Consecutive identical moves therefore bind one each, rather than
    the second being mistaken for a retry of the first.
    """
    i, skipped = 0, 0
    for move in moves:
        want = (move["object"], move["direction"])
        found = None
        while i < len(pairs):
            parsed = MOVE_LINE.search(pairs[i][2])
            i += 1
            if parsed and (parsed.group("object"), parsed.group("direction")) == want:
                found = i - 1
                break
            skipped += 1
        if found is None:
            raise SystemExit(
                f"move {move['index']} ({move['object']} {move['direction']}) has "
                "no matching response in the transcript"
            )
        move["recipient"], move["context"], move["response"] = pairs[found]
    return skipped


def _moves(raw: Sequence[dict], start: Dict[str, Cell]) -> List[dict]:
    """Normalise Turn Moves, deriving from/to where an older run omits them."""
    position = dict(start)
    out: List[dict] = []
    for entry in raw:
        obj = entry["object"]
        direction = entry["direction"]
        frm = tuple(entry["from"]) if entry.get("from") else position[obj]
        dr, dc = STEP[direction]
        to = tuple(entry["to"]) if entry.get("to") else (frm[0] + dr, frm[1] + dc)
        if position[obj] != frm:
            raise SystemExit(
                f"move {len(out) + 1} ({obj} {direction}) starts at {cellname(frm)} "
                f"but replay puts {obj} at {cellname(position[obj])}; the "
                "transcript and the instance file disagree"
            )
        if (frm[0] + dr, frm[1] + dc) != to:
            raise SystemExit(
                f"move {len(out) + 1}: {direction} from {cellname(frm)} cannot "
                f"reach {cellname(to)}"
            )
        position[obj] = to
        out.append({
            "index": len(out) + 1,
            "player": entry["player"],
            "side": entry["player"].split()[-1],  # 'Player A' -> 'A'
            "object": obj,
            "direction": direction,
            "from": frm,
            "to": to,
            "message": entry.get("message"),
        })
    return out


def load_episode(path: str = DEFAULT_RUN) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    meta = data["meta"]
    instance = _find_instance(meta["game_name"], meta["experiment_name"],
                              meta["game_id"])
    start = {k: tuple(v) for k, v in instance["start_positions"].items()}
    moves = _moves(data.get("Turn Moves", []), start)

    pairs = _exchanges(data.get("turns", []))
    unbound = _bind_exchanges(moves, pairs)

    outcome = ("SUCCESS" if data.get("Success")
               else "ABORTED" if data.get("Aborted") else "LOSE")
    models = sorted({p["model_name"] for name, p in data["players"].items()
                     if name != "GM"})
    return {
        "path": path,
        "game": meta["game_name"],
        "experiment": meta["experiment_name"],
        "game_id": meta["game_id"],
        "level": instance["spatial_level"],
        "protocol": instance.get("comm_protocol"),
        "models": models,
        "outcome": outcome,
        "instance": instance,
        "start": start,
        "moves": moves,
        "exchanges": pairs,
        "unbound_exchanges": unbound,  # reprompts and done: declarations
    }


# ------------------------------------------------------------------- replay


def replay(episode: dict, upto: Optional[int] = None) -> List[Dict[str, Cell]]:
    """Board positions before move 1, then after each move, in order."""
    moves = episode["moves"] if upto is None else episode["moves"][:upto]
    states = [dict(episode["start"])]
    for move in moves:
        nxt = dict(states[-1])
        nxt[move["object"]] = move["to"]
        states.append(nxt)
    return states


def sides(instance: dict) -> Dict[str, List[str]]:
    return {"A": list(instance["player_a_objects"]),
            "B": list(instance["player_b_objects"])}


def owner(instance: dict, obj: str) -> str:
    return "A" if obj in instance["player_a_objects"] else "B"


def on_target(instance: dict, positions: Dict[str, Cell]) -> List[str]:
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    return sorted(o for o, cell in positions.items() if cell == target[o])


def listing(instance: dict, positions: Dict[str, Cell],
            side: Optional[str] = None) -> Dict[str, str]:
    """The compact listing this player is sent, mirroring Board.render_compact.

    With `side` set, teammate objects collapse to X and teammate targets are
    omitted entirely, which is the masking the dyad and n-player games apply.
    """
    mine = set(sides(instance)[side]) if side else set(positions)
    target = {k: tuple(v) for k, v in instance["target_positions"].items()}
    objects = " ".join(
        f"{obj if obj in mine else 'X'}@{cellname(positions[obj])}"
        for obj in sorted(positions)
    )
    walls = " ".join(cellname(tuple(w))
                     for w in sorted(tuple(w) for w in instance["walls"]))
    targets = " ".join(f"{obj}->{cellname(target[obj])}" for obj in sorted(mine))
    return {"Objects": objects, "Walls": walls, "Targets": targets}


def exchange_for(episode: dict, move_index: int) -> Tuple[str, str]:
    """The context sent and the raw reply given for move `move_index` (1-based)."""
    moves = episode["moves"]
    if move_index < 1 or move_index > len(moves):
        raise SystemExit(
            f"move {move_index} is outside the {len(moves)} recorded moves")
    move = moves[move_index - 1]
    return move["context"], move["response"]


def check_abridged(full: str, shown: str, ellipsis: str = "…") -> str:
    """Verify that an abridged quotation only ever cuts from the original.

    Figures quote long channel messages in shortened form. Each fragment
    between elisions must still occur, in order, in the recorded message, so a
    rerun on different data fails loudly instead of printing words no model
    said.
    """
    cursor = 0
    for fragment in (f for f in shown.split(ellipsis) if f.strip()):
        found = full.find(fragment.strip(), cursor)
        if found < 0:
            raise SystemExit(
                "abridged quotation does not match the transcript.\n"
                f"  missing fragment: {fragment.strip()[:70]!r}\n"
                f"  recorded message: {full[:110]!r}..."
            )
        cursor = found + len(fragment.strip())
    return shown


# --------------------------------------------------------------------- cli


def find_episodes(pattern: str, *, successful: bool = True) -> List[dict]:
    hits = []
    for path in glob.glob(os.path.join(REPO_ROOT, "results*", "*", "matrixgame*",
                                       "*", "instance_*", "interactions.json")):
        if pattern.lower() not in path.lower():
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        ok = bool(data.get("Success"))
        if successful and not ok:
            continue
        hits.append({
            "path": os.path.relpath(path, REPO_ROOT),
            "experiment": data["meta"]["experiment_name"],
            "game_id": data["meta"]["game_id"],
            "moves": len(data.get("Turn Moves", [])),
            "outcome": "SUCCESS" if ok else ("ABORTED" if data.get("Aborted") else "LOSE"),
        })
    return sorted(hits, key=lambda h: h["moves"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--find", metavar="SUBSTRING",
                        help="list successful episodes whose path contains this")
    args = parser.parse_args()

    if args.find:
        hits = find_episodes(args.find)
        print(f"{len(hits)} successful episodes matching {args.find!r}")
        for h in hits[:40]:
            print(f"  {h['moves']:>3} moves  {h['experiment']:<24} "
                  f"game_id {h['game_id']:<3} {h['path']}")
        return

    ep = load_episode(args.run)
    print(f"{ep['experiment']}  game_id {ep['game_id']}  {ep['outcome']}")
    print(f"  level {ep['level']}   protocol {ep['protocol']}   "
          f"models {', '.join(ep['models'])}")
    print(f"  {len(ep['moves'])} moves, {len(ep['exchanges'])} exchanges "
          f"({ep['unbound_exchanges']} reprompt or done:), "
          f"optimal {ep['instance']['optimal_moves']}")
    for move in ep["moves"][:8]:
        landed = on_target(ep["instance"], replay(ep, move["index"])[-1])
        print(f"  T{move['index']:<2} {move['side']}  {move['object']} "
              f"{move['direction']:<5} {cellname(move['from'])}->"
              f"{cellname(move['to'])}   on target: {landed or '-'}")


if __name__ == "__main__":
    main()
