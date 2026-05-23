"""Instance generator for matrixgame_covered_solo (full-view only).

Reuses _comm's per-spatial-level board generators (S1–S4) and helpers
unchanged. The single solo player owns all 6 objects; there is no
masked-view variant in this game (the masked variant was dropped during
brainstorming because _comm's S2/S3/S4 levels build cross-agent dependency
chains that become unreachable when half the chain is static).

Per-instance fields:
- objects, start_positions, target_positions, walls (from _generate_s*)
- optimal_moves: per-object BFS sum via _verify_reachability (lower bound;
  matches _comm's optimal_moves formula)
- max_turns = max(4 * optimal_moves, 20)
- thinking, spatial_level, compact_board (from config.json)
- player_prompt: the loaded template string for the (thinking) variant
"""

import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from clemcore.clemgame import GameInstanceGenerator



class SoloMatrixGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        common_config = self.load_json("resources/common_config")
        configs = self.load_json("resources/config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_instances = kwargs.get("num_instances", 10)
        rng = random.Random(seed)

        generators = {
            "S1": _generate_s1,
            "S2": _generate_s2,
            "S3": _generate_s3,
            "S4": _generate_s4,
        }

        for variant_key, variant_config in configs.items():
            thinking = variant_config["thinking"]
            spatial_level = variant_config["spatial_level"]
            compact_board = variant_config.get("compact_board", False)
            num_objects = variant_config["num_objects"]

            objects = all_objects[:num_objects]
            template_name = f"player_prompt_{'thinking' if thinking else 'silent'}"
            prompt_template = self.load_template(
                f"resources/initial_prompts/en/{template_name}"
            )

            experiment = self.add_experiment(variant_config["name"])
            experiment["common_config"] = common_config
            experiment["variant_config"] = variant_config

            gen_fn = generators.get(spatial_level, _generate_s1)

            for idx in range(num_instances):
                game_instance = self.add_game_instance(experiment, idx + 1)
                starts = targets = walls = None
                optimal = 0

                for _attempt in range(80):
                    result = gen_fn(grid_size, objects, rng)
                    starts, targets, walls, _optimal_combined, _player_a, _player_b = result

                    if spatial_level != "S1" and len(walls) == 0:
                        continue

                    wall_set = set(map(tuple, walls))
                    ok, optimal = _verify_reachability(
                        grid_size, wall_set, starts, targets,
                    )
                    if not ok:
                        continue

                    # Invariant: every object's origin differs from its target
                    if any(tuple(starts[o]) == tuple(targets[o]) for o in objects):
                        continue

                    break
                else:
                    raise RuntimeError(
                        f"Could not generate viable instance for {variant_config['name']} idx {idx}"
                    )

                game_instance["grid_size"] = grid_size
                game_instance["walls"] = walls
                game_instance["objects"] = objects
                game_instance["start_positions"] = starts
                game_instance["target_positions"] = targets
                game_instance["optimal_moves"] = optimal
                game_instance["max_turns"] = max(4 * optimal, 20)
                game_instance["max_retries"] = max_retries
                game_instance["strict"] = strict
                game_instance["thinking"] = thinking
                game_instance["spatial_level"] = spatial_level
                game_instance["compact_board"] = compact_board
                game_instance["player_prompt"] = prompt_template


# ── Shared utilities ───────────────────────────────────────────────────

def _bfs_path(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    start: Tuple[int, int],
    target: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """BFS shortest path avoiding walls. Returns cell list (endpoints included) or None."""
    if start == target:
        return [start]
    prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    queue: deque = deque([start])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        r, c = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < grid_size
                and 0 <= nc < grid_size
                and (nr, nc) not in walls
                and (nr, nc) not in prev
            ):
                prev[(nr, nc)] = (r, c)
                if (nr, nc) == target:
                    path: List[Tuple[int, int]] = []
                    cur: Optional[Tuple[int, int]] = target
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    return path[::-1]
                queue.append((nr, nc))
    return None


def _new_wall_creates_dead_end(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    new_wall: Tuple[int, int],
    protected: Set[Tuple[int, int]],
) -> bool:
    """True if adding new_wall gives any non-protected cell exactly 1 free neighbour."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    r, c = new_wall
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < grid_size and 0 <= nc < grid_size):
            continue
        if (nr, nc) in walls:
            continue
        free = sum(
            1
            for dr2, dc2 in dirs
            if (
                0 <= nr + dr2 < grid_size
                and 0 <= nc + dc2 < grid_size
                and (nr + dr2, nc + dc2) not in walls
            )
        )
        if free == 1 and (nr, nc) not in protected:
            return True
    return False


def _carve_for_ratio(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    start: Tuple[int, int],
    target: Tuple[int, int],
    target_ratio: float,
    rng: random.Random,
    protected: Set[Tuple[int, int]],
    allow_dead_ends: bool = False,
    max_attempts: int = 30,
    extra_walls_for_2conn: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    """Iteratively block path cells until actual_steps/manhattan ≥ target_ratio - 0.15."""
    manhattan = abs(start[0] - target[0]) + abs(start[1] - target[1])
    if manhattan == 0:
        return walls

    current = set(walls)
    for _ in range(max_attempts):
        path = _bfs_path(grid_size, current, start, target)
        if path is None:
            return walls

        if (len(path) - 1) / manhattan >= target_ratio - 0.15:
            return current

        candidates = [p for p in path[1:-1] if p not in protected]
        rng.shuffle(candidates)
        placed = False
        for cell in candidates:
            tentative = current | {cell}
            if _bfs_path(grid_size, tentative, start, target) is None:
                continue
            if not allow_dead_ends and _new_wall_creates_dead_end(
                grid_size, tentative, cell, protected
            ):
                continue
            if extra_walls_for_2conn is not None and not _is_2connected(
                grid_size, tentative | extra_walls_for_2conn
            ):
                continue
            current = tentative
            placed = True
            break
        if not placed:
            break

    return current


def _pick(
    grid_size: int,
    rng: random.Random,
    excluded: Set[Tuple[int, int]],
    interior: bool = False,
) -> Optional[Tuple[int, int]]:
    """Random free cell not in excluded; prefer interior when interior=True."""
    lo, hi = (1, grid_size - 1) if interior else (0, grid_size)
    candidates = [
        (r, c)
        for r in range(lo, hi)
        for c in range(lo, hi)
        if (r, c) not in excluded
    ]
    if not candidates and interior:
        candidates = [
            (r, c)
            for r in range(grid_size)
            for c in range(grid_size)
            if (r, c) not in excluded
        ]
    return rng.choice(candidates) if candidates else None


def _verify_reachability(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    starts: Dict[str, List[int]],
    targets: Dict[str, List[int]],
) -> Tuple[bool, int]:
    """Per-object BFS reachability check + sum of individual path lengths.

    This is sufficient when combined with _is_2connected: the pebble-motion
    theorem guarantees joint solvability on a 2-connected graph whenever each
    object can individually reach its target.  The returned cost is the sum of
    individual BFS path lengths — a lower bound on the true joint optimum, but
    accurate enough for max_turns budgeting.
    """
    total = 0
    for obj in starts:
        if obj not in targets:
            continue
        path = _bfs_path(grid_size, walls, tuple(starts[obj]), tuple(targets[obj]))  # type: ignore
        if path is None:
            return False, 0
        total += len(path) - 1
    return True, total


def _is_2connected(grid_size: int, walls: Set[Tuple[int, int]]) -> bool:
    """True iff the free-cell graph is connected with no articulation points."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    free = [(r, c) for r in range(grid_size) for c in range(grid_size) if (r, c) not in walls]
    if not free:
        return True

    free_set = set(free)
    disc: Dict[Tuple[int, int], int] = {}
    low: Dict[Tuple[int, int], int] = {}
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
    ap: Set[Tuple[int, int]] = set()
    timer = [0]

    def dfs(u: Tuple[int, int]) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        children = 0
        for dr, dc in dirs:
            v = (u[0] + dr, u[1] + dc)
            if v not in free_set:
                continue
            if v not in disc:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent.get(u) is None and children > 1:
                    ap.add(u)
                if parent.get(u) is not None and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    parent[free[0]] = None
    dfs(free[0])

    if len(disc) < len(free):
        return False
    return len(ap) == 0


# ── Coordination isolation check (Experiment 3) ────────────────────────

def _coordination_isolation_check(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    starts: Dict[str, List[int]],
    targets: Dict[str, List[int]],
    player_a_objs: List[str],
    player_b_objs: List[str],
) -> bool:
    """Return True if no shared single-cell bottleneck exists between agents.

    Rejects instances where a cell appears in both agents' optimal path sets
    and has corridor width ≤ 1 (only one adjacent path cell), which would
    indicate a coordination demand increase rather than a spatial one.
    """
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def path_cells(obj: str) -> Set[Tuple[int, int]]:
        s = tuple(starts[obj])  # type: ignore
        t = tuple(targets[obj])  # type: ignore
        path = _bfs_path(grid_size, walls, s, t)
        return set(path) if path else set()

    a_cells: Set[Tuple[int, int]] = set()
    for obj in player_a_objs:
        a_cells |= path_cells(obj)

    b_cells: Set[Tuple[int, int]] = set()
    for obj in player_b_objs:
        b_cells |= path_cells(obj)

    shared = a_cells & b_cells
    all_path_cells = a_cells | b_cells

    for cell in shared:
        r, c = cell
        adjacent_path = [
            (r + dr, c + dc)
            for dr, dc in dirs
            if (
                0 <= r + dr < grid_size
                and 0 <= c + dc < grid_size
                and (r + dr, c + dc) not in walls
                and (r + dr, c + dc) in all_path_cells
            )
        ]
        if len(adjacent_path) <= 1:
            return False

    return True


# ── S1 ─────────────────────────────────────────────────────────────────

def _generate_s1(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
) -> Tuple[Dict, Dict, List, int, List[str], List[str]]:
    """S1: region-partitioned placement, no walls, no dependency chains."""
    regions = _partition_regions(grid_size, len(objects))
    starts: Dict[str, List[int]] = {}
    targets: Dict[str, List[int]] = {}
    for obj, region in zip(objects, regions):
        cells = region[:]
        rng.shuffle(cells)
        starts[obj] = list(cells[0])
        targets[obj] = list(cells[1])

    optimal = sum(
        abs(starts[o][0] - targets[o][0]) + abs(starts[o][1] - targets[o][1])
        for o in objects
    )
    half = len(objects) // 2
    player_a = objects[:half]
    player_b = objects[half:]
    return starts, targets, [], optimal, player_a, player_b


def _partition_regions(grid_size: int, num_objects: int) -> List[List[Tuple[int, int]]]:
    num_cols_bands = 2
    num_rows_bands = math.ceil(num_objects / num_cols_bands)
    row_splits = _split_range(grid_size, num_rows_bands)
    col_splits = _split_range(grid_size, num_cols_bands)
    regions: List[List[Tuple[int, int]]] = []
    for r0, r1 in row_splits:
        for c0, c1 in col_splits:
            regions.append([(r, c) for r in range(r0, r1) for c in range(c0, c1)])
            if len(regions) == num_objects:
                return regions
    return regions[:num_objects]


def _split_range(total: int, n: int) -> List[Tuple[int, int]]:
    size, rem = divmod(total, n)
    splits, start = [], 0
    for i in range(n):
        end = start + size + (1 if i < rem else 0)
        splits.append((start, end))
        start = end
    return splits


# ── S2 ─────────────────────────────────────────────────────────────────

def _generate_s2(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
    max_retries: int = 25,
) -> Tuple[Dict, Dict, List, int, List[str], List[str]]:
    """S2: length-2 chain [A←B←C] + corridor carving to ~1.5× Manhattan.

    Cross-agent assignment: A,C → Player A; B → Player B.
    Remaining free objects D,E,F split: D → Player A; E,F → Player B.
    """
    obj_A, obj_B, obj_C = objects[0], objects[1], objects[2]
    free_objs = objects[3:]  # [D, E, F]

    for _ in range(max_retries):
        used: Set[Tuple[int, int]] = set()
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        # Chain: B.start = A.target, C.start = B.target
        A_target = _pick(grid_size, rng, used, interior=True)
        if A_target is None:
            continue
        used.add(A_target)
        targets[obj_A] = list(A_target)
        starts[obj_B] = list(A_target)

        B_target = _pick(grid_size, rng, used, interior=True)
        if B_target is None:
            continue
        used.add(B_target)
        targets[obj_B] = list(B_target)
        starts[obj_C] = list(B_target)

        C_target = _pick(grid_size, rng, used, interior=True)
        if C_target is None:
            continue
        used.add(C_target)
        targets[obj_C] = list(C_target)

        A_start = _pick(grid_size, rng, used, interior=True)
        if A_start is None:
            continue
        used.add(A_start)
        starts[obj_A] = list(A_start)

        ok = True
        for obj in free_objs:
            s = _pick(grid_size, rng, used)
            if s is None:
                ok = False
                break
            used.add(s)
            t = _pick(grid_size, rng, used)
            if t is None:
                ok = False
                break
            used.add(t)
            starts[obj] = list(s)
            targets[obj] = list(t)
        if not ok:
            continue

        protected: Set[Tuple[int, int]] = {
            tuple(p) for p in list(starts.values()) + list(targets.values())  # type: ignore
        }
        walls: Set[Tuple[int, int]] = set()
        for obj in objects:
            walls = _carve_for_ratio(
                grid_size, walls,
                tuple(starts[obj]), tuple(targets[obj]),  # type: ignore
                target_ratio=1.5, rng=rng,
                protected=protected, allow_dead_ends=False,
                extra_walls_for_2conn=set(),  # enforce 2-connectivity at each wall addition
            )

        if not _is_2connected(grid_size, walls):
            continue

        ok, optimal = _verify_reachability(grid_size, walls, starts, targets)
        if not ok:
            continue

        # Cross-agent assignment: A,C,D → Player A; B,E,F → Player B
        player_a = [obj_A, obj_C, free_objs[0]]
        player_b = [obj_B] + list(free_objs[1:])
        return starts, targets, sorted(walls), optimal, player_a, player_b

    starts, targets, walls, optimal, pa, pb = _generate_s1(grid_size, objects, rng)
    return starts, targets, walls, optimal, pa, pb


# ── S3 ─────────────────────────────────────────────────────────────────

def _random_dead_end_corridor(
    grid_size: int,
    rng: random.Random,
    excluded: Set[Tuple[int, int]],
    min_length: int = 2,
    max_length: int = 4,
    max_tries: int = 50,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
    """Random dead-end corridor running from a grid border inward.

    Returns (entrance, dead_end, corridor_cells, wall_cells) or None.
    """
    for _ in range(max_tries):
        length = rng.randint(min_length, max_length)
        border = rng.choice(("right", "left", "down", "up"))

        if border == "right":
            row = rng.randint(1, grid_size - 2)
            dead_end = (row, grid_size - 1)
            corridor_cells = {(row, grid_size - 1 - k) for k in range(length)}
            entrance = (row, grid_size - 1 - length)
            perp = ((-1, 0), (1, 0))
        elif border == "left":
            row = rng.randint(1, grid_size - 2)
            dead_end = (row, 0)
            corridor_cells = {(row, k) for k in range(length)}
            entrance = (row, length)
            perp = ((-1, 0), (1, 0))
        elif border == "down":
            col = rng.randint(1, grid_size - 2)
            dead_end = (grid_size - 1, col)
            corridor_cells = {(grid_size - 1 - k, col) for k in range(length)}
            entrance = (grid_size - 1 - length, col)
            perp = ((0, -1), (0, 1))
        else:  # up
            col = rng.randint(1, grid_size - 2)
            dead_end = (0, col)
            corridor_cells = {(k, col) for k in range(length)}
            entrance = (length, col)
            perp = ((0, -1), (0, 1))

        er, ec = entrance
        if not (0 <= er < grid_size and 0 <= ec < grid_size):
            continue

        wall_cells: Set[Tuple[int, int]] = set()
        for cr, cc in corridor_cells:
            for dr, dc in perp:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    wall_cells.add((nr, nc))

        all_blocked = corridor_cells | wall_cells | {entrance}
        if all_blocked & excluded:
            continue

        dead_end_free = sum(
            1
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (
                0 <= dead_end[0] + dr < grid_size
                and 0 <= dead_end[1] + dc < grid_size
                and (dead_end[0] + dr, dead_end[1] + dc) not in wall_cells
            )
        )
        if dead_end_free != 1:
            continue

        return entrance, dead_end, corridor_cells, wall_cells

    return None


def _generate_s3(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
    max_retries: int = 25,
) -> Tuple[Dict, Dict, List, int, List[str], List[str]]:
    """S3: dead-end corridor + length-3 chain [A←B←C←D] + ~2× Manhattan.

    Cross-agent assignment: D,B,E → Player A; C,A,F → Player B.
    Secondary chain: F.start = E.target.
    """
    if grid_size < 6:
        return _generate_s1(grid_size, objects, rng)

    obj_A, obj_B, obj_C, obj_D = objects[0], objects[1], objects[2], objects[3]
    obj_E, obj_F = objects[4], objects[5]

    for _ in range(max_retries):
        corridor = _random_dead_end_corridor(grid_size, rng, excluded=set())
        if corridor is None:
            continue
        entrance, dead_end, corridor_cells, corridor_walls = corridor

        blocked = corridor_cells | corridor_walls | {entrance}
        used: Set[Tuple[int, int]] = set(blocked)
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        # D starts at entrance, targets dead end; C targets entrance (forcing D out first)
        starts[obj_D] = list(entrance)
        targets[obj_D] = list(dead_end)
        targets[obj_C] = list(entrance)

        A_target = _pick(grid_size, rng, used, interior=True)
        if A_target is None:
            continue
        used.add(A_target)
        targets[obj_A] = list(A_target)
        starts[obj_B] = list(A_target)

        B_target = _pick(grid_size, rng, used, interior=True)
        if B_target is None:
            continue
        used.add(B_target)
        targets[obj_B] = list(B_target)
        starts[obj_C] = list(B_target)

        A_start = _pick(grid_size, rng, used, interior=True)
        if A_start is None:
            continue
        used.add(A_start)
        starts[obj_A] = list(A_start)

        E_target = _pick(grid_size, rng, used)
        if E_target is None:
            continue
        used.add(E_target)
        targets[obj_E] = list(E_target)
        starts[obj_F] = list(E_target)  # F.start = E.target: F must move before E

        F_target = _pick(grid_size, rng, used)
        if F_target is None:
            continue
        used.add(F_target)
        targets[obj_F] = list(F_target)

        E_start = _pick(grid_size, rng, used)
        if E_start is None:
            continue
        used.add(E_start)
        starts[obj_E] = list(E_start)

        protected: Set[Tuple[int, int]] = (
            {tuple(p) for p in list(starts.values()) + list(targets.values())}  # type: ignore
            | corridor_cells
        )
        walls: Set[Tuple[int, int]] = set(corridor_walls)
        for obj in objects:
            walls = _carve_for_ratio(
                grid_size, walls,
                tuple(starts[obj]), tuple(targets[obj]),  # type: ignore
                target_ratio=2.0, rng=rng,
                protected=protected, allow_dead_ends=False,
                extra_walls_for_2conn=corridor_cells,
            )

        if not _is_2connected(grid_size, walls | corridor_cells):
            continue

        ok, optimal = _verify_reachability(grid_size, walls, starts, targets)
        if not ok:
            continue

        # Cross-agent: D,B,E → Player A; C,A,F → Player B
        player_a = [obj_D, obj_B, obj_E]
        player_b = [obj_C, obj_A, obj_F]
        return starts, targets, sorted(walls), optimal, player_a, player_b

    return _generate_s1(grid_size, objects, rng)


# ── S4 ─────────────────────────────────────────────────────────────────

def _generate_s4(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
    max_retries: int = 25,
) -> Tuple[Dict, Dict, List, int, List[str], List[str]]:
    """S4: corridor swap forcing non-monotonic trajectory; same map as S3.

    Dead-end corridor (length ≥ 2) + walls for ~2× Manhattan ratio, 2-connected.
    No wall-line gate — difficulty comes purely from object placement in the corridor.

    INNER (obj_B, Player B): starts at the corridor dead end, targets outside.
      Must exit through the adjacent cell, which OUTER initially occupies.
    OUTER (obj_A, Player A): starts at the corridor cell adjacent to the dead end,
      targets the dead end. Must retreat out of the corridor to let INNER exit,
      then re-enter — forced non-monotonic trajectory.
    Cross-agent: OUTER cannot finish until INNER exits; INNER cannot exit until
      OUTER retreats. Neither player resolves this without coordinating.

    Remaining 4 objects form two length-1 chains:
      D.start = C.target  (D→Player B must move before C→Player A)
      F.start = E.target  (F→Player B must move before E→Player A)

    Player A: [OUTER, C, E]; Player B: [INNER, D, F]
    """
    if grid_size < 6:
        return _generate_s1(grid_size, objects, rng)

    obj_OUTER = objects[0]
    obj_INNER = objects[1]
    obj_C, obj_D, obj_E, obj_F = objects[2], objects[3], objects[4], objects[5]

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(max_retries):
        corridor = _random_dead_end_corridor(
            grid_size, rng, excluded=set(), min_length=3, max_length=4
        )
        if corridor is None:
            continue
        entrance, dead_end, corridor_cells, corridor_walls = corridor

        # Find the corridor cell adjacent to the dead end (OUTER's start).
        # Dead end has exactly one corridor-cell neighbour by construction.
        inner_adjacent: Optional[Tuple[int, int]] = None
        for dr, dc in dirs:
            nr, nc = dead_end[0] + dr, dead_end[1] + dc
            if (nr, nc) in corridor_cells:
                inner_adjacent = (nr, nc)
                break
        if inner_adjacent is None:
            continue

        blocked = corridor_cells | corridor_walls | {entrance}
        used: Set[Tuple[int, int]] = set(blocked)
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        # Corridor swap: INNER at dead end blocks OUTER's target;
        # OUTER at inner_adjacent blocks INNER's only exit.
        starts[obj_INNER] = list(dead_end)
        starts[obj_OUTER] = list(inner_adjacent)
        targets[obj_OUTER] = list(dead_end)

        INNER_target = _pick(grid_size, rng, used)
        if INNER_target is None:
            continue
        used.add(INNER_target)
        targets[obj_INNER] = list(INNER_target)

        # Chain 1: D.start = C.target  (D→Player B must move first)
        C_target = _pick(grid_size, rng, used)
        if C_target is None:
            continue
        used.add(C_target)
        targets[obj_C] = list(C_target)
        starts[obj_D] = list(C_target)

        D_target = _pick(grid_size, rng, used)
        if D_target is None:
            continue
        used.add(D_target)
        targets[obj_D] = list(D_target)

        C_start = _pick(grid_size, rng, used)
        if C_start is None:
            continue
        used.add(C_start)
        starts[obj_C] = list(C_start)

        # Chain 2: F.start = E.target  (F→Player B must move first)
        E_target = _pick(grid_size, rng, used)
        if E_target is None:
            continue
        used.add(E_target)
        targets[obj_E] = list(E_target)
        starts[obj_F] = list(E_target)

        F_target = _pick(grid_size, rng, used)
        if F_target is None:
            continue
        used.add(F_target)
        targets[obj_F] = list(F_target)

        E_start = _pick(grid_size, rng, used)
        if E_start is None:
            continue
        used.add(E_start)
        starts[obj_E] = list(E_start)

        protected: Set[Tuple[int, int]] = (
            {tuple(p) for p in list(starts.values()) + list(targets.values())}  # type: ignore
            | corridor_cells
        )
        walls: Set[Tuple[int, int]] = set(corridor_walls)
        for obj in objects:
            walls = _carve_for_ratio(
                grid_size, walls,
                tuple(starts[obj]), tuple(targets[obj]),  # type: ignore
                target_ratio=2.0, rng=rng,
                protected=protected, allow_dead_ends=False,
                extra_walls_for_2conn=corridor_cells,
            )

        if not _is_2connected(grid_size, walls | corridor_cells):
            continue

        ok, optimal = _verify_reachability(grid_size, walls, starts, targets)
        if not ok:
            continue

        player_a = [obj_OUTER, obj_C, obj_E]
        player_b = [obj_INNER, obj_D, obj_F]
        return starts, targets, sorted(walls), optimal, player_a, player_b

    return _generate_s1(grid_size, objects, rng)


# ── Entry point ────────────────────────────────────────────────────────
#
# Usage:
#   python instancegenerator.py                          # generate everything (Exp2 + Exp3)
#   python instancegenerator.py exp2                     # Exp2 only
#   python instancegenerator.py exp3                     # Exp3 only
#   python instancegenerator.py exp3 --n 10              # override instance count
#   python instancegenerator.py exp3 --levels s1 s2 s3  # only levels S1-S3
#   python instancegenerator.py --levels s2              # only S2 across all experiments

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate solo instances.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--out", default="in/instances.json")
    args = parser.parse_args()

    gen = SoloMatrixGameInstanceGenerator()
    gen.generate(filename=args.out, seed=args.seed, num_instances=args.num_instances)
    print(f"Wrote instances to {args.out}")
