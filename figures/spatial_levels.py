"""Renders one publication figure per spatial reasoning level (S1-S4).

Each figure pairs the ground-truth board with the compact listing a player
actually reads, so that a reader can see both the structural device the level
installs and the fact that the device must be inferred from a wall list.

Boards come from the released instance files, so the figures regenerate when
instances do. Devices (corridor, entrance, dead end) are read from the
instance's `device` block; dependency chains are derived from the start/target
overlap and cross-checked against the documented per-level signature.

Usage
    python3 figures/spatial_levels.py                 # median instance per level
    python3 figures/spatial_levels.py --list S3       # candidate instances for a level
    python3 figures/spatial_levels.py --pick S3=comm_S3_none_compact_3
"""

from __future__ import annotations

import json
import os
import statistics
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = os.path.join(REPO_ROOT, "matrixgame_covered_comm", "in", "instances.json")
OUT_DIR = os.path.join(REPO_ROOT, "figures", "out")

LEVELS = ("S1", "S2", "S3", "S4")

# Chain signature each level's generator is documented to install, as the
# sorted lengths (in objects) of its dependency chains. Guards against a
# silent generator change turning a figure caption into a false claim.
EXPECTED_CHAIN_SIGNATURE = {
    "S1": (),
    "S2": (3,),
    "S3": (4, 2),
    "S4": (4, 2),
}

# ---------------------------------------------------------------- board logic


def cellname(cell: Cell) -> str:
    """(0,0) -> 'R1C1'. Rows top to bottom, columns left to right."""
    return f"R{cell[0] + 1}C{cell[1] + 1}"


def load_instances(path: str = INSTANCES) -> Dict[str, List[dict]]:
    """Group every released instance by spatial level.

    The four communication protocols share one board per index, so boards are
    de-duplicated on the physical puzzle (walls + starts + targets).
    """
    with open(path) as fh:
        data = json.load(fh)
    by_level: Dict[str, List[dict]] = {lvl: [] for lvl in LEVELS}
    seen: Dict[str, set] = {lvl: set() for lvl in LEVELS}
    for exp in data["experiments"]:
        level = exp["name"].split("_")[0]
        if level not in by_level:
            continue
        for inst in exp["game_instances"]:
            key = json.dumps(
                [inst["walls"], inst["start_positions"], inst["target_positions"]],
                sort_keys=True,
            )
            if key in seen[level]:
                continue
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


def free_path(
    start: Cell, goal: Cell, walls: set, grid_size: int
) -> List[Cell]:
    """Shortest wall-avoiding path ignoring other objects (the optimal_moves basis)."""
    if start == goal:
        return [start]
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < grid_size and 0 <= nxt[1] < grid_size):
                continue
            if nxt in walls or nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] is not None and prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return []


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def worst_detour(instance: dict) -> Optional[Tuple[str, List[Cell], int]]:
    """The object whose wall-avoiding path most exceeds its Manhattan distance.

    Returns (object, path, manhattan_distance). This is the per-object detour
    ratio the S2/S3/S4 generators carve toward, made concrete on one object.
    """
    walls = {tuple(w) for w in instance["walls"]}
    if not walls:
        return None
    grid = instance["grid_size"]
    best = None
    for obj in instance["objects"]:
        start = tuple(instance["start_positions"][obj])
        goal = tuple(instance["target_positions"][obj])
        path = free_path(start, goal, walls, grid)
        base = manhattan(start, goal)
        if not path or base == 0:
            continue
        ratio = (len(path) - 1) / base
        if best is None or ratio > best[0]:
            best = (ratio, obj, path, base)
    if best is None or best[0] <= 1.0:
        return None
    return best[1], best[2], best[3]


def mean_detour_ratio(instance: dict) -> Optional[float]:
    """Mean over objects of wall-avoiding path length / Manhattan distance.

    This is the quantity the generators carve toward (about 1.5 at S2, 2.0 at
    S3 and S4), so a panel that shows one object's detour should report it for
    context rather than let the worst case stand in for the board.
    """
    walls = {tuple(w) for w in instance["walls"]}
    if not walls:
        return None
    ratios = []
    for obj in instance["objects"]:
        start = tuple(instance["start_positions"][obj])
        goal = tuple(instance["target_positions"][obj])
        base = manhattan(start, goal)
        path = free_path(start, goal, walls, instance["grid_size"])
        if base and path:
            ratios.append((len(path) - 1) / base)
    return sum(ratios) / len(ratios) if ratios else None


def retreat_cell(instance: dict) -> Optional[Cell]:
    """Where OUTER must park so that INNER can leave the corridor.

    The recorded `entrance` is the free cell adjacent to the corridor mouth,
    itself outside the corridor, and INNER has to pass through it. OUTER must
    therefore clear the corridor *and* the entrance, which means a free
    neighbour of the entrance that is not itself a corridor cell. Where several
    qualify, the one furthest from INNER's target is chosen so the parked
    object is least likely to sit on INNER's onward route.
    """
    device = instance.get("device") or {}
    pair = swap_pair(instance)
    if not device or pair is None:
        return None
    _outer, inner = pair
    corridor = {tuple(c) for c in device["corridor"]}
    walls = {tuple(w) for w in instance["walls"]}
    entrance = tuple(device["entrance"])
    size = instance["grid_size"]
    inner_target = tuple(instance["target_positions"][inner])

    candidates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cell = (entrance[0] + dr, entrance[1] + dc)
        if not (0 <= cell[0] < size and 0 <= cell[1] < size):
            continue
        if cell in walls or cell in corridor:
            continue
        candidates.append(cell)
    if not candidates:
        return None
    return max(candidates, key=lambda c: (manhattan(c, inner_target), -c[0], -c[1]))


def corridor_extent(device: dict) -> Optional[Tuple[Cell, Cell]]:
    """The corridor's own end cells, mouth first, excluding the entrance."""
    corridor = [tuple(c) for c in device.get("corridor", [])]
    if not corridor:
        return None
    dead_end = tuple(device["dead_end"])
    mouth = max(corridor, key=lambda c: manhattan(c, dead_end))
    return mouth, dead_end


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


def describe(instance: dict) -> dict:
    """Everything the figure needs about one board, derived once."""
    chains = dependency_chains(
        instance["start_positions"], instance["target_positions"]
    )
    level = instance["spatial_level"]
    expected = EXPECTED_CHAIN_SIGNATURE.get(level)
    actual = chain_signature(chains)
    if expected is not None and actual != tuple(sorted(expected, reverse=True)):
        raise ValueError(
            f"{level} instance {instance.get('game_id')}: chain signature {actual} "
            f"does not match the documented {tuple(sorted(expected, reverse=True))}. "
            "Generators changed; update EXPECTED_CHAIN_SIGNATURE and the captions."
        )
    swap = swap_pair(instance)
    return {
        "instance": instance,
        "level": level,
        "chains": chains,
        "swap": swap,
        # The swap pair surfaces as an ordinary 2-chain; label it as the swap.
        "plain_chains": [c for c in chains if not (swap and tuple(c) == swap)],
        "device": instance.get("device") or {},
        "detour": worst_detour(instance),
        "mean_detour": mean_detour_ratio(instance),
        "retreat": retreat_cell(instance),
        "extent": corridor_extent(instance.get("device") or {}),
    }
