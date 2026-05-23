"""A* solver for matrixgame_covered_base.

Two modes:
- Full (player_objects=None): moves all objects — used offline for instance
  verification and optimal-move-count computation.
- Partial (player_objects=...): moves only the given objects, treating all
  others as fixed obstacles — used by the programmatic A* gameplay partner.

Wall cells are respected in both modes.
"""

import heapq
import logging
from typing import Dict, List, Optional, Set, Tuple

from utils.board import Board, DIRECTIONS

logger = logging.getLogger(__name__)

MAX_NODES = 2_000_000


def astar_solve(
    board: Board,
    player_objects: Optional[Set[str]] = None,
    max_nodes: int = MAX_NODES,
) -> List[Tuple[str, str]]:
    """Find an optimal move sequence using A*.

    Args:
        board: Current board state (includes target_positions and walls).
        player_objects: If given, only these objects may be moved. All
            other objects are treated as immovable obstacles.

    Returns:
        List of (object_name, direction) tuples, or [] if unsolvable.
    """
    target_positions: Dict[str, Tuple[int, int]] = board.target_positions

    movable: Set[str] = set(board.object_positions.keys())
    if player_objects is not None:
        movable = movable & player_objects

    # Targets that belong to movable objects
    my_targets = {obj: pos for obj, pos in target_positions.items() if obj in movable}

    if not my_targets:
        return []

    n = board.grid_size
    walls = board.walls
    directions_list = list(DIRECTIONS.items())

    # ── helpers ────────────────────────────────────────────────────────

    def make_state(positions: Dict[str, Tuple[int, int]]) -> Tuple:
        return tuple(sorted(positions.items()))

    def heuristic(positions: Dict[str, Tuple[int, int]]) -> int:
        h = 0
        for obj, (tr, tc) in my_targets.items():
            if obj in positions:
                r, c = positions[obj]
                h += abs(r - tr) + abs(c - tc)
        return h

    def is_goal(positions: Dict[str, Tuple[int, int]]) -> bool:
        return all(positions.get(obj) == pos for obj, pos in my_targets.items())

    # ── search ─────────────────────────────────────────────────────────

    start_positions = dict(board.object_positions)
    start_state = make_state(start_positions)

    counter = 0
    open_set: list = [(heuristic(start_positions), counter, 0, start_state)]
    g_scores: Dict[Tuple, int] = {start_state: 0}
    came_from: Dict[Tuple, Tuple] = {}
    nodes_expanded = 0

    while open_set:
        _f, _, g, state = heapq.heappop(open_set)

        if g > g_scores.get(state, float("inf")):
            continue

        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            logger.warning("A* exceeded node limit (%d). Aborting.", max_nodes)
            return []

        state_dict = dict(state)

        if is_goal(state_dict):
            moves: List[Tuple[str, str]] = []
            current = state
            while current in came_from:
                parent, obj, direction = came_from[current]
                moves.append((obj, direction))
                current = parent
            moves.reverse()
            logger.info("A* found solution: %d moves, %d nodes.", len(moves), nodes_expanded)
            return moves

        occupied = set(state_dict.values())

        for obj in movable:
            if obj not in state_dict:
                continue
            row, col = state_dict[obj]
            for direction, (dr, dc) in directions_list:
                new_row, new_col = row + dr, col + dc

                if not (0 <= new_row < n and 0 <= new_col < n):
                    continue
                if (new_row, new_col) in walls:
                    continue
                if (new_row, new_col) in occupied:
                    continue

                new_dict = dict(state_dict)
                new_dict[obj] = (new_row, new_col)
                new_state = make_state(new_dict)
                new_g = g + 1

                if new_g < g_scores.get(new_state, float("inf")):
                    g_scores[new_state] = new_g
                    came_from[new_state] = (state, obj, direction)
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (new_g + heuristic(new_dict), counter, new_g, new_state),
                    )

    logger.warning("A* found no solution (expanded %d nodes).", nodes_expanded)
    return []
