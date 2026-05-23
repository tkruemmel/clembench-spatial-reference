"""Instance generator for matrixgame_covered_base.

Generates game instances on an 8×8 grid with per-object target positions.

Spatial complexity levels
─────────────────────────
S1  No walls, region-partitioned placement — paths never conflict by
    construction.

S2  4–6 wall segments that force detours (~1.5× Manhattan ratio).
    One length-2 dependency chain: objects B and C sit on A's and B's
    targets respectively, so the resolution order is forced (C→B→A).
    No dead ends.

S3  Dead-end corridor in the upper-right area of the grid. Object D's
    target is inside the corridor; it also starts on object C's target,
    so D must move into the dead end before the chain (C→B→A) can
    resolve.  Additional walls push ratios toward ~2× Manhattan.
"""

import os
import math
import random
import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from clemcore.clemgame import GameInstanceGenerator

from utils.board import Board
from utils.astar import astar_solve


class MatrixGameInstanceGenerator(GameInstanceGenerator):
    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed: int, **kwargs):
        variant = kwargs.get("variant", "matrixgame_covered_base_s1")
        variant_config = self.load_json("resources/config")[variant]
        common_config = self.load_json("resources/common_config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_objects = variant_config["num_objects"]
        use_masking = variant_config.get("use_masking", False)
        spatial_level = variant_config.get("spatial_level", "S1")

        objects = all_objects[:num_objects]
        half = num_objects // 2
        player_a_objects = objects[:half]
        player_b_objects = objects[half:]

        prompt_template = self.load_template("resources/initial_prompts/en/player_prompt")
        num_instances = kwargs.get("num_instances", 10)

        experiment_name = f"{spatial_level}_{num_objects}obj"
        if use_masking:
            experiment_name += "_masked"

        experiment = self.add_experiment(experiment_name)
        experiment["common_config"] = common_config
        experiment["variant_config"] = variant_config

        rng = random.Random(seed)

        generators = {"S1": _generate_s1, "S2": _generate_s2, "S3": _generate_s3}
        gen_fn = generators.get(spatial_level, _generate_s1)

        for idx in range(num_instances):
            game_instance = self.add_game_instance(experiment, idx + 1)

            start_positions, target_positions, walls, optimal_moves = gen_fn(
                grid_size, objects, rng
            )

            game_instance["grid_size"] = grid_size
            game_instance["walls"] = walls
            game_instance["objects"] = objects
            game_instance["player_a_objects"] = player_a_objects
            game_instance["player_b_objects"] = player_b_objects
            game_instance["start_positions"] = start_positions
            game_instance["target_positions"] = target_positions
            game_instance["optimal_moves"] = optimal_moves
            game_instance["max_turns"] = max(4 * optimal_moves, 20)
            game_instance["max_retries"] = max_retries
            game_instance["strict"] = strict
            game_instance["use_masking"] = use_masking
            game_instance["spatial_level"] = spatial_level
            game_instance["player_prompt"] = prompt_template


# ── shared utilities ───────────────────────────────────────────────────

def _bfs_path(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    start: Tuple[int, int],
    target: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    """Shortest path avoiding walls. Returns list of cells (incl. endpoints) or None."""
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
    """True if adding new_wall makes any non-protected cell a dead end (1 free neighbour)."""
    r, c = new_wall
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
    """Iteratively block path cells until actual_steps/manhattan ≥ target_ratio - 0.15.

    If extra_walls_for_2conn is given, each candidate wall must leave
    _is_2connected(grid_size, tentative | extra_walls_for_2conn) True.
    This guarantees 2-connectivity of the relevant subgraph at every step.
    """
    manhattan = abs(start[0] - target[0]) + abs(start[1] - target[1])
    if manhattan == 0:
        return walls

    current = set(walls)
    for _ in range(max_attempts):
        path = _bfs_path(grid_size, current, start, target)
        if path is None:
            return walls  # went too far — give up

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
    """Pick a random free cell not in excluded; prefer interior cells when interior=True."""
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


_VERIFY_MAX_NODES = 80_000


def _verify(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    starts: Dict[str, List[int]],
    targets: Dict[str, List[int]],
) -> Tuple[bool, int]:
    """Joint A* solvability check using an integer state representation.

    Encodes each object position as a single integer (row*grid_size + col)
    so the state is a plain int-tuple — much faster to hash/compare than the
    string-keyed dicts used by the gameplay solver.

    Returns (solvable, optimal_joint_moves).  Returns (False, 0) if the node
    budget is exhausted; the caller should treat that as reject-and-retry.
    """
    # Quick per-object connectivity pre-check (catches wall misplacements cheaply)
    for obj in starts:
        if obj not in targets:
            continue
        if _bfs_path(grid_size, walls, tuple(starts[obj]), tuple(targets[obj])) is None:  # type: ignore
            return False, 0

    gs = grid_size
    wall_ints: Set[int] = {r * gs + c for r, c in walls}
    objs = sorted(starts.keys())
    start_state: Tuple[int, ...] = tuple(starts[o][0] * gs + starts[o][1] for o in objs)
    goal_state:  Tuple[int, ...] = tuple(targets[o][0] * gs + targets[o][1] for o in objs)

    # Precompute valid (non-wall) neighbours for every cell — avoids per-node
    # bounds checking and direction arithmetic in the hot loop.
    neighbours: List[List[int]] = [[] for _ in range(gs * gs)]
    for r in range(gs):
        for c in range(gs):
            pos = r * gs + c
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < gs and 0 <= nc < gs:
                    np = nr * gs + nc
                    if np not in wall_ints:
                        neighbours[pos].append(np)

    # Heuristic: sum of per-object Manhattan distances to targets.
    tr = [t // gs for t in goal_state]
    tc = [t % gs  for t in goal_state]

    def heuristic(state: Tuple[int, ...]) -> int:
        return sum(abs(p // gs - tr[i]) + abs(p % gs - tc[i]) for i, p in enumerate(state))

    BIG = 10 ** 9
    counter = 0
    heap: list = [(heuristic(start_state), 0, 0, start_state)]
    g: Dict[Tuple[int, ...], int] = {start_state: 0}
    nodes = 0

    while heap:
        _, _, cost, state = heapq.heappop(heap)

        if cost > g.get(state, BIG):
            continue

        nodes += 1
        if nodes > _VERIFY_MAX_NODES:
            return False, 0

        if state == goal_state:
            return True, cost

        occupied = set(state)
        for i, pos in enumerate(state):
            for np in neighbours[pos]:
                if np in occupied:
                    continue
                new_state = state[:i] + (np,) + state[i + 1:]
                new_cost = cost + 1
                if new_cost < g.get(new_state, BIG):
                    g[new_state] = new_cost
                    counter += 1
                    heapq.heappush(
                        heap,
                        (new_cost + heuristic(new_state), counter, new_cost, new_state),
                    )

    return False, 0


def _verify_groups(
    grid_size: int,
    walls: Set[Tuple[int, int]],
    starts: Dict[str, List[int]],
    targets: Dict[str, List[int]],
    group_a: List[str],
    group_b: List[str],
) -> Tuple[bool, int]:
    """Run _verify independently for each player group and sum optimal moves.

    Splitting by player group reduces the joint state space from O(64^6) to
    O(64^3) per group, making verification ~1000× faster while still catching
    intra-group deadlocks.  Cross-group deadlocks are ruled out separately by
    _is_2connected.
    """
    starts_a = {o: starts[o] for o in group_a}
    targets_a = {o: targets[o] for o in group_a}
    ok_a, opt_a = _verify(grid_size, walls, starts_a, targets_a)
    if not ok_a:
        return False, 0

    starts_b = {o: starts[o] for o in group_b}
    targets_b = {o: targets[o] for o in group_b}
    ok_b, opt_b = _verify(grid_size, walls, starts_b, targets_b)
    if not ok_b:
        return False, 0

    return True, opt_a + opt_b


def _is_2connected(grid_size: int, walls: Set[Tuple[int, int]]) -> bool:
    """True iff the free-cell graph is connected and has no articulation points.

    Uses Tarjan's DFS algorithm.  On an 8×8 grid (≤64 cells) the recursion
    depth is always safe.  A 2-connected free-cell graph guarantees — via the
    pebble-motion theorem — that any individually-reachable multi-object
    configuration is jointly solvable.
    """
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    free = [(r, c) for r in range(grid_size) for c in range(grid_size)
            if (r, c) not in walls]
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

    if len(disc) < len(free):   # disconnected
        return False
    return len(ap) == 0


# ── S1 ─────────────────────────────────────────────────────────────────

def _generate_s1(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List, int]:
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
    return starts, targets, [], optimal


def _partition_regions(
    grid_size: int, num_objects: int
) -> List[List[Tuple[int, int]]]:
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
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List, int]:
    """S2: length-2 dependency chain [A←B←C] + corridor carving to ~1.5× Manhattan."""
    obj_A, obj_B, obj_C = objects[0], objects[1], objects[2]
    free_objs = objects[3:]
    half = len(objects) // 2

    for _ in range(max_retries):
        used: Set[Tuple[int, int]] = set()
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        # Chain setup: B.start = A.target, C.start = B.target
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
            )

        if not _is_2connected(grid_size, walls):
            continue

        ok, optimal = _verify_groups(grid_size, walls, starts, targets,
                                     objects[:half], objects[half:])
        if not ok:
            continue

        return starts, targets, sorted(walls), optimal

    return _generate_s1(grid_size, objects, rng)


# ── S3 ─────────────────────────────────────────────────────────────────

def _random_dead_end_corridor(
    grid_size: int,
    rng: random.Random,
    excluded: Set[Tuple[int, int]],
    min_length: int = 2,
    max_length: int = 4,
    max_tries: int = 50,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int],
                    Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
    """Pick a random dead-end corridor running from a grid border inward.

    The corridor is a straight passage of `length` cells.  The terminal cell
    (dead end) sits against the border; walls line both sides.  The entrance
    is the main-grid cell immediately adjacent to the first corridor cell.

    Returns (entrance, dead_end, corridor_cells, wall_cells) or None.
        entrance       – the main-grid cell bordering the corridor (D's start /
                         C's target)
        dead_end       – the 1-neighbour terminal cell (D's target)
        corridor_cells – all interior cells of the corridor, dead_end included;
                         only reachable via entrance
        wall_cells     – cells walled to isolate the corridor sides
    """
    for _ in range(max_tries):
        length = rng.randint(min_length, max_length)
        border = rng.choice(('right', 'left', 'down', 'up'))

        if border == 'right':
            row = rng.randint(1, grid_size - 2)
            dead_end = (row, grid_size - 1)
            corridor_cells = {(row, grid_size - 1 - k) for k in range(length)}
            entrance = (row, grid_size - 1 - length)
            perp = ((-1, 0), (1, 0))
        elif border == 'left':
            row = rng.randint(1, grid_size - 2)
            dead_end = (row, 0)
            corridor_cells = {(row, k) for k in range(length)}
            entrance = (row, length)
            perp = ((-1, 0), (1, 0))
        elif border == 'down':
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

        # Verify dead end truly has exactly 1 free neighbour.
        dead_end_free = sum(
            1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (0 <= dead_end[0] + dr < grid_size
                and 0 <= dead_end[1] + dc < grid_size
                and (dead_end[0] + dr, dead_end[1] + dc) not in wall_cells)
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
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], List, int]:
    """S3: random dead-end corridor + length-3 chain [A←B←C←D] + ~2× Manhattan.

    A fresh corridor is generated on each attempt: random border, random row/col,
    random length 2-4.  D starts at the corridor entrance and targets the dead end;
    C targets the entrance (so D must clear it first); B and A form the rest of
    the chain.  Secondary chain: F.start = E.target.
    """
    if grid_size < 6:
        return _generate_s1(grid_size, objects, rng)

    obj_A, obj_B, obj_C, obj_D = objects[0], objects[1], objects[2], objects[3]
    obj_E, obj_F = objects[4], objects[5]
    half = len(objects) // 2

    for _ in range(max_retries):
        corridor = _random_dead_end_corridor(grid_size, rng, excluded=set())
        if corridor is None:
            continue
        entrance, dead_end, corridor_cells, corridor_walls = corridor

        blocked = corridor_cells | corridor_walls | {entrance}
        used: Set[Tuple[int, int]] = set(blocked)
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        starts[obj_D] = list(entrance)
        targets[obj_D] = list(dead_end)

        # C.target = entrance: D must vacate it before C can arrive.
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

        # corridor_cells added to protected so carving never walls them off.
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

        ok, optimal = _verify_groups(grid_size, walls, starts, targets,
                                     objects[:half], objects[half:])
        if not ok:
            continue

        return starts, targets, sorted(walls), optimal

    return _generate_s1(grid_size, objects, rng)


if __name__ == "__main__":
    gen = MatrixGameInstanceGenerator()
    gen.generate(filename="instances.json", seed=42,
                 variant="matrixgame_covered_base_s1", num_instances=5)
    gen.generate(filename="instances.json", seed=43,
                 variant="matrixgame_covered_base_s2", num_instances=5)
    gen.generate(filename="instances.json", seed=44,
                 variant="matrixgame_covered_base_s3", num_instances=5)
