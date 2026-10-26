"""The boards of the four spatial levels, as the level figures draw them.

Boards come from the game's instance file, so the figures regenerate when the
instances do. Dependency chains are derived from the start/target overlap and
checked against each level's documented signature, so a generator change
cannot silently turn a caption into a false claim.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = os.path.join(REPO_ROOT, "matrixgame", "in", "instances_dyad.json")
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")

LEVELS = ("S1", "S2", "S3", "S4")

# Chain signature each level's generator installs, as the sorted lengths (in
# objects) of its dependency chains.
EXPECTED_CHAIN_SIGNATURE = {
    "S1": (),
    "S2": (3,),
    "S3": (4, 2),
    "S4": (4, 2),
}


def load_instances(path: str = INSTANCES) -> Dict[str, List[dict]]:
    """Every board by spatial level. The four protocols share one board per
    index, so boards are de-duplicated on walls, starts and targets."""
    with open(path) as fh:
        data = json.load(fh)
    by_level: Dict[str, List[dict]] = {lvl: [] for lvl in LEVELS}
    seen: Dict[str, set] = {lvl: set() for lvl in LEVELS}
    for exp in data["experiments"]:
        level = exp["name"].split("_")[0]
        for inst in exp["game_instances"]:
            key = json.dumps([inst["walls"], inst["start_positions"], inst["target_positions"]], sort_keys=True)
            if key not in seen[level]:
                seen[level].add(key)
                by_level[level].append(inst)
    return by_level


def dependency_chains(
    start_positions: Dict[str, Sequence[int]],
    target_positions: Dict[str, Sequence[int]],
) -> List[List[str]]:
    """Derive dependency chains from start/target overlap.

    Object v blocks object u when v starts on u's target cell, so v must vacate
    before u can land. Returned as chains in blocked-to-blocker order, i.e.
    ['A', 'B', 'C'] denotes A <- B <- C: C vacates, then B lands, then A lands.
    Longest chain first.
    """
    start = {k: tuple(v) for k, v in start_positions.items()}
    target = {k: tuple(v) for k, v in target_positions.items()}
    occupant = {cell: obj for obj, cell in start.items()}

    blocker = {
        obj: occupant[cell]
        for obj, cell in target.items()
        if cell in occupant and occupant[cell] != obj
    }
    blocked = set(blocker.values())

    chains: List[List[str]] = []
    for head in sorted(obj for obj in blocker if obj not in blocked):
        chain = [head]
        while chain[-1] in blocker:
            nxt = blocker[chain[-1]]
            if nxt in chain:  # defensive: a cycle would be an unsolvable board
                break
            chain.append(nxt)
        chains.append(chain)
    return sorted(chains, key=len, reverse=True)


def chain_signature(chains: Sequence[Sequence[str]]) -> Tuple[int, ...]:
    return tuple(sorted((len(c) for c in chains), reverse=True))


def swap_pair(instance: dict) -> Optional[Tuple[str, str]]:
    """Return (outer, inner) for an S4 board, else None.

    INNER starts at the corridor dead end; OUTER starts on the adjacent
    corridor cell and targets the dead end, so it must retreat clear of the
    corridor before INNER can leave.
    """
    device = instance.get("device") or {}
    if "inner_adjacent" not in device:
        return None
    start = {k: tuple(v) for k, v in instance["start_positions"].items()}
    dead_end = tuple(device["dead_end"])
    adjacent = tuple(device["inner_adjacent"])
    inner = next((o for o, c in start.items() if c == dead_end), None)
    outer = next((o for o, c in start.items() if c == adjacent), None)
    if inner is None or outer is None:
        return None
    return outer, inner


def check_chains(instance: dict) -> None:
    """Raise if the board's chains differ from its level's documented signature."""
    level = instance["spatial_level"]
    actual = chain_signature(dependency_chains(instance["start_positions"], instance["target_positions"]))
    expected = EXPECTED_CHAIN_SIGNATURE[level]
    if actual != expected:
        raise ValueError(
            f"{level} instance {instance.get('game_id')}: chain signature {actual} "
            f"does not match the documented {expected}. "
            "Generators changed; update EXPECTED_CHAIN_SIGNATURE and the captions."
        )


def pick_representative(instances: Sequence[dict]) -> dict:
    """The instance closest to its level's median on optimal_moves and wall count.

    Distance is measured in per-axis median-absolute-deviation units so the two
    axes contribute comparably; ties break on game_id for determinism.
    """
    moves = [i["optimal_moves"] for i in instances]
    walls = [len(i["walls"]) for i in instances]
    med_moves, med_walls = statistics.median(moves), statistics.median(walls)

    def spread(values: Sequence[float], median: float) -> float:
        dev = statistics.median([abs(v - median) for v in values])
        return dev or 1.0

    s_moves, s_walls = spread(moves, med_moves), spread(walls, med_walls)

    def distance(inst: dict) -> Tuple[float, str]:
        d = abs(inst["optimal_moves"] - med_moves) / s_moves
        d += abs(len(inst["walls"]) - med_walls) / s_walls
        return d, str(inst.get("game_id", ""))

    return min(instances, key=distance)
