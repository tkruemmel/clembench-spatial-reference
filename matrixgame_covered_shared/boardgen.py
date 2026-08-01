"""Single source of truth for S1-S4 board generation.

Shared by the instance generators of matrixgame_covered_solo,
matrixgame_covered_comm, and matrixgame_covered_comm_nplayer, so that the
physical puzzle distribution is identical across the three games by
construction (thesis, comparability across variants). Ownership is layered
on top by the callers; only the dyad's per-level split is returned here.
"""

import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

LEVELS = ("S1", "S2", "S3", "S4")


def instance_rng(base_seed: int, level: str, idx: int) -> random.Random:
    """Deterministic RNG derived per (seed, level, instance index).

    Any caller asking for the same triple gets the identical stream,
    independent of package, protocol, team size, or generation order.
    (random.Random seeds strings via sha512 — stable across runs.)
    """
    return random.Random(f"{base_seed}:{level}:{idx}")


def generate_board(
    level: str,
    grid_size: int,
    objects: List[str],
    rng: random.Random,
    max_restarts: int = 10,
) -> Dict:
    gen_fn = {
        "S1": _generate_s1,
        "S2": _generate_s2,
        "S3": _generate_s3,
        "S4": _generate_s4,
    }[level]
    for _ in range(max_restarts):
        result = gen_fn(grid_size, objects, rng)
        if result is None:
            continue
        starts, targets, walls, optimal, player_a, player_b, device = result
        if level != "S1" and not walls:
            continue
        if any(tuple(starts[o]) == tuple(targets[o]) for o in objects):
            continue
        ok, optimal = _verify_reachability(
            grid_size, set(map(tuple, walls)), starts, targets
        )
        if not ok:
            continue
        return {
            "start_positions": starts,
            "target_positions": targets,
            "walls": [list(w) for w in walls],
            "optimal_moves": optimal,
            "dyad_split": (player_a, player_b),
            "device": device,
        }
    raise RuntimeError(
        f"Could not generate a valid {level} board after {max_restarts} restarts"
    )


# ── S1 ─────────────────────────────────────────────────────────────────

def _generate_s1(grid_size, objects, rng):
    """S1: free placement, no walls, no dependency chains.

    All 12 cells (6 starts + 6 targets) are pairwise distinct so that no
    object's target coincides with another object's start — S1 must contain
    no accidental ordering constraints.
    """
    used: Set[Tuple[int, int]] = set()
    starts: Dict[str, List[int]] = {}
    targets: Dict[str, List[int]] = {}
    for obj in objects:
        s = _pick(grid_size, rng, used)
        if s is None:
            return None
        used.add(s)
        starts[obj] = list(s)
    for obj in objects:
        t = _pick(grid_size, rng, used)
        if t is None:
            return None
        used.add(t)
        targets[obj] = list(t)
    optimal = sum(
        abs(starts[o][0] - targets[o][0]) + abs(starts[o][1] - targets[o][1])
        for o in objects
    )
    half = len(objects) // 2
    return starts, targets, [], optimal, objects[:half], objects[half:], None


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
        return starts, targets, sorted(walls), optimal, player_a, player_b, None

    return None


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


def _generate_s3(grid_size, objects, rng, max_retries=25):
    """S3: dead-end corridor (length 4) threaded by chain A<-B<-C<-D.

    Thesis threading: C starts at the corridor entrance; D's target is the
    dead-end cell. Chain links: B.start = A.target, C.start = B.target
    (= entrance), D.start = C.target. The entrance is the corridor's only
    access, so C must vacate before D can pass, and D must pass before B can
    occupy the entrance. Secondary dependency E<-F (F.start = E.target)
    runs outside the corridor. Walls carved to ~2x Manhattan; free space
    2-connected with the corridor treated as blocked.
    """
    if grid_size < 6:
        return None

    obj_A, obj_B, obj_C, obj_D, obj_E, obj_F = objects[:6]

    for _ in range(max_retries):
        corridor = _random_dead_end_corridor(
            grid_size, rng, excluded=set(), min_length=4, max_length=4
        )
        if corridor is None:
            continue
        entrance, dead_end, corridor_cells, corridor_walls = corridor

        blocked = corridor_cells | corridor_walls | {entrance}
        used: Set[Tuple[int, int]] = set(blocked)
        starts: Dict[str, List[int]] = {}
        targets: Dict[str, List[int]] = {}

        # Corridor device: C blocks the only access; D must pass it.
        starts[obj_C] = list(entrance)
        targets[obj_B] = list(entrance)      # chain: C.start = B.target
        targets[obj_D] = list(dead_end)

        C_target = _pick(grid_size, rng, used, interior=True)
        if C_target is None:
            continue
        used.add(C_target)
        targets[obj_C] = list(C_target)
        starts[obj_D] = list(C_target)       # chain: D.start = C.target

        A_target = _pick(grid_size, rng, used, interior=True)
        if A_target is None:
            continue
        used.add(A_target)
        targets[obj_A] = list(A_target)
        starts[obj_B] = list(A_target)       # chain: B.start = A.target

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
        starts[obj_F] = list(E_target)       # secondary: F.start = E.target

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
            continue
        ok, optimal = _verify_reachability(grid_size, walls, starts, targets)
        if not ok:
            continue

        device = {
            "entrance": list(entrance),
            "dead_end": list(dead_end),
            "corridor": [list(c) for c in sorted(corridor_cells)],
        }
        # Dyad split: chain members alternate owners; corridor pair crosses agents.
        return (starts, targets, sorted(walls), optimal,
                [obj_D, obj_B, obj_E], [obj_C, obj_A, obj_F], device)

    return None


# ── S4 ─────────────────────────────────────────────────────────────────

def _generate_s4(
    grid_size: int,
    objects: List[str],
    rng: random.Random,
    max_retries: int = 25,
) -> Tuple[Dict, Dict, List, int, List[str], List[str]]:
    """S4: corridor swap + length-3 dependency chain; ~2× Manhattan, 2-connected.

    Dead-end corridor (length ≥ 2) + walls for ~2× Manhattan ratio.
    No wall-line gate — difficulty comes purely from object placement.

    INNER (obj_B, Player B): starts at the corridor dead end, targets outside.
      Must exit through the adjacent cell, which OUTER initially occupies.
    OUTER (obj_A, Player A): starts at the corridor cell adjacent to the dead end,
      targets the dead end. Must retreat out of the corridor to let INNER exit,
      then re-enter — forced non-monotonic trajectory.
    Cross-agent: OUTER cannot finish until INNER exits; INNER cannot exit until
      OUTER retreats. Neither player resolves this without coordinating.

    The other four objects form a single length-3 chain  C ← D ← E ← F:
      starts[D] = targets[C]  (D must vacate before C can land)
      starts[E] = targets[D]  (E must vacate before D can land)
      starts[F] = targets[E]  (F must vacate before E can land)
    Forced ordering: F moves first, then E, then D, then C.

    Player A: [OUTER, D, F]; Player B: [INNER, C, E]
    """
    if grid_size < 6:
        return None

    obj_OUTER = objects[0]
    obj_INNER = objects[1]
    obj_C, obj_D, obj_E, obj_F = objects[2], objects[3], objects[4], objects[5]

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(max_retries):
        corridor = _random_dead_end_corridor(
            grid_size, rng, excluded=set(), min_length=4, max_length=4
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

        # Length-3 chain C ← D ← E ← F:
        #   starts[D] = targets[C], starts[E] = targets[D], starts[F] = targets[E].
        # Pick C.target first, then D.target (= E.start), then E.target (= F.start),
        # then F.target. Finally pick C.start (independent).
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
        starts[obj_E] = list(D_target)

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

        C_start = _pick(grid_size, rng, used)
        if C_start is None:
            continue
        used.add(C_start)
        starts[obj_C] = list(C_start)

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

        device = {
            "entrance": list(entrance),
            "dead_end": list(dead_end),
            "inner_adjacent": list(inner_adjacent),
            "corridor": [list(c) for c in sorted(corridor_cells)],
        }
        return (starts, targets, sorted(walls), optimal,
                [obj_OUTER, obj_D, obj_F], [obj_INNER, obj_C, obj_E], device)

    return None
