"""Crossed-device boards S3x and S4x.

S3 and S4 differ in which chain their corridor device is attached to: S3
threads the deep chain (A<-B<-C<-D) through the corridor and keeps the
shallow pair (E<-F) in open space; S4 puts only the swap pair in the
corridor and keeps the deep chain in open space. The two crossed levels fill
the other cells of that 2x2:

  S3x  S3's threading device on the shallow pair; deep chain in open space.
  S4x  S4's swap device as the head link of the deep chain; shallow pair in
       open space.

Both keep the S3/S4 structure (longest chain 3, 4 edges, 2 chains, one
dead-end corridor of length 4 with one target inside, ~2x Manhattan carving,
free space 2-connected with the corridor treated as blocked).

Pairing: crossed board i uses the corridor of released board i (S3x <- S3,
S4x <- S4), so the two differ only in what sits around the corridor.
Placement and carving draw from their own stream (seed:S3x:idx /
seed:S4x:idx).

Ownership: letters are assigned to roles so that the existing round-robin
n-player partition and the dyad split below both keep consecutive chain
members and the two device objects on different agents, as in S3/S4.

The helpers come from matrixgame_covered_shared.boardgen, which this module
only imports; the released S1-S4 generators are untouched.
"""

from contextlib import contextmanager
from typing import Dict, List, Set, Tuple
from unittest import mock

from matrixgame_covered_shared.boardgen import (
    _carve_for_ratio,
    _is_2connected,
    _pick,
    _verify_reachability,
    generate_board,
    instance_rng,
)

CROSSED_LEVELS = ("S3x", "S4x")
PAIRED_LEVEL = {"S3x": "S3", "S4x": "S4"}
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def generate_crossed_board(
    level: str,
    grid_size: int,
    objects: List[str],
    seed: int,
    idx: int,
    max_restarts: int = 10,
) -> Dict:
    """Same contract as boardgen.generate_board, for S3x / S4x."""
    paired = generate_board(
        PAIRED_LEVEL[level], grid_size, objects, instance_rng(seed, PAIRED_LEVEL[level], idx)
    )
    corridor = _corridor_from_device(grid_size, paired["device"])
    gen_fn = {"S3x": _generate_s3x, "S4x": _generate_s4x}[level]
    rng = instance_rng(seed, level, idx)
    for _ in range(max_restarts):
        result = gen_fn(grid_size, objects, rng, corridor)
        if result is None:
            continue
        starts, targets, walls, optimal, player_a, player_b, device = result
        if any(tuple(starts[o]) == tuple(targets[o]) for o in objects):
            continue
        return {
            "start_positions": starts,
            "target_positions": targets,
            "walls": [list(w) for w in walls],
            "optimal_moves": optimal,
            "dyad_split": (player_a, player_b),
            "device": device,
        }
    raise RuntimeError(f"Could not generate a valid {level} board for idx {idx}")


@contextmanager
def crossed_boards(generator_module):
    """Run an existing instance generator's on_generate with crossed boards.

    The released generators call ``instance_rng(seed, level, idx)`` and then
    ``generate_board(level, grid_size, objects, rng)`` from their module
    namespace. Swapping those two names for the duration of the call lets the
    crossed packages reuse that code unchanged, so every instance field
    (budget, masking, ownership format, prompt) is built exactly as for S3/S4.
    """
    with mock.patch.object(
        generator_module, "instance_rng", lambda seed, level, idx: (seed, idx)
    ), mock.patch.object(
        generator_module,
        "generate_board",
        lambda level, grid_size, objects, key: generate_crossed_board(
            level, grid_size, objects, *key
        ),
    ):
        yield


def _corridor_from_device(grid_size: int, device: Dict):
    """Rebuild (entrance, dead_end, corridor_cells, wall_cells) as
    boardgen._random_dead_end_corridor returned it for the paired board."""
    entrance = tuple(device["entrance"])
    dead_end = tuple(device["dead_end"])
    cells = {tuple(c) for c in device["corridor"]}
    horizontal = len({r for r, _ in cells}) == 1
    perp = ((-1, 0), (1, 0)) if horizontal else ((0, -1), (0, 1))
    walls = {
        (r + dr, c + dc)
        for r, c in cells
        for dr, dc in perp
        if 0 <= r + dr < grid_size and 0 <= c + dc < grid_size
    }
    return entrance, dead_end, cells, walls


def _carve_and_check(grid_size, objects, starts, targets, corridor_cells, corridor_walls, rng):
    """S3/S4's carving and validity checks, verbatim."""
    protected: Set[Tuple[int, int]] = (
        {tuple(p) for p in list(starts.values()) + list(targets.values())}
        | corridor_cells
    )
    walls: Set[Tuple[int, int]] = set(corridor_walls)
    for obj in objects:
        walls = _carve_for_ratio(
            grid_size, walls,
            tuple(starts[obj]), tuple(targets[obj]),
            target_ratio=2.0, rng=rng,
            protected=protected, allow_dead_ends=False,
            extra_walls_for_2conn=corridor_cells,
        )
    if not _is_2connected(grid_size, walls | corridor_cells):
        return None
    ok, optimal = _verify_reachability(grid_size, walls, starts, targets)
    if not ok:
        return None
    return walls, optimal


def _generate_s3x(grid_size, objects, rng, corridor, max_retries=25):
    """S3x: S3's threading device on the shallow pair.

    F starts at the entrance (as C in S3), E targets the dead end (as D) and
    F.target = E.start (as C.target = D.start): F must step aside, E must pass,
    then F lands. Deep chain A<-B<-C<-D (B.start = A.target, C.start =
    B.target, D.start = C.target) sits in open space. Like S3, device and
    deep-chain cells are picked from the interior.

    Dyad: {A, C, E} / {B, D, F}.
    """
    obj_A, obj_B, obj_C, obj_D, obj_E, obj_F = objects[:6]
    entrance, dead_end, corridor_cells, corridor_walls = corridor

    for _ in range(max_retries):
        used: Set[Tuple[int, int]] = corridor_cells | corridor_walls | {entrance}
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        starts[obj_F] = list(entrance)
        targets[obj_E] = list(dead_end)

        cells = []
        for _pick_no in range(5):
            cell = _pick(grid_size, rng, used, interior=True)
            if cell is None:
                break
            used.add(cell)
            cells.append(list(cell))
        if len(cells) < 5:
            continue
        F_target, A_target, B_target, C_target, D_target = cells
        targets[obj_F] = F_target
        starts[obj_E] = F_target         # pair: E.start = F.target
        targets[obj_A] = A_target
        starts[obj_B] = A_target         # chain: B.start = A.target
        targets[obj_B] = B_target
        starts[obj_C] = B_target         # chain: C.start = B.target
        targets[obj_C] = C_target
        starts[obj_D] = C_target         # chain: D.start = C.target
        targets[obj_D] = D_target

        A_start = _pick(grid_size, rng, used, interior=True)
        if A_start is None:
            continue
        used.add(A_start)
        starts[obj_A] = list(A_start)

        checked = _carve_and_check(
            grid_size, objects, starts, targets, corridor_cells, corridor_walls, rng
        )
        if checked is None:
            continue
        walls, optimal = checked
        device = {
            "entrance": list(entrance),
            "dead_end": list(dead_end),
            "corridor": [list(c) for c in sorted(corridor_cells)],
        }
        return (starts, targets, sorted(walls), optimal,
                [obj_A, obj_C, obj_E], [obj_B, obj_D, obj_F], device)

    return None


def _generate_s4x(grid_size, objects, rng, corridor, max_retries=25):
    """S4x: S4's corridor swap as the head link of the deep chain.

    A (OUTER) starts next to the dead end and targets it; B (INNER) starts at
    the dead end. B.target = C.start, C.target = D.start, so the chain
    A<-B<-C<-D resolves D, C, B (exits), A (re-enters). Shallow pair E<-F
    (F.start = E.target) sits in open space. Like S4, no interior preference.

    Dyad: {A, C, E} / {B, D, F}.
    """
    obj_A, obj_B, obj_C, obj_D, obj_E, obj_F = objects[:6]
    entrance, dead_end, corridor_cells, corridor_walls = corridor

    inner_adjacent = next(
        (dead_end[0] + dr, dead_end[1] + dc)
        for dr, dc in DIRS
        if (dead_end[0] + dr, dead_end[1] + dc) in corridor_cells
    )

    for _ in range(max_retries):
        used: Set[Tuple[int, int]] = corridor_cells | corridor_walls | {entrance}
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        starts[obj_B] = list(dead_end)
        starts[obj_A] = list(inner_adjacent)
        targets[obj_A] = list(dead_end)

        cells = []
        for _pick_no in range(6):
            cell = _pick(grid_size, rng, used)
            if cell is None:
                break
            used.add(cell)
            cells.append(list(cell))
        if len(cells) < 6:
            continue
        B_target, C_target, D_target, E_target, F_target, E_start = cells
        targets[obj_B] = B_target
        starts[obj_C] = B_target         # chain: C.start = B.target
        targets[obj_C] = C_target
        starts[obj_D] = C_target         # chain: D.start = C.target
        targets[obj_D] = D_target
        targets[obj_E] = E_target
        starts[obj_F] = E_target         # pair: F.start = E.target
        targets[obj_F] = F_target
        starts[obj_E] = E_start

        checked = _carve_and_check(
            grid_size, objects, starts, targets, corridor_cells, corridor_walls, rng
        )
        if checked is None:
            continue
        walls, optimal = checked
        device = {
            "entrance": list(entrance),
            "dead_end": list(dead_end),
            "inner_adjacent": list(inner_adjacent),
            "corridor": [list(c) for c in sorted(corridor_cells)],
        }
        return (starts, targets, sorted(walls), optimal,
                [obj_A, obj_C, obj_E], [obj_B, obj_D, obj_F], device)

    return None
