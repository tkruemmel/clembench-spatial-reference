"""A* solver for the matrixgame grid rearrangement puzzle."""

import heapq
import logging
from typing import Dict, List, Tuple

from utils.board import Board, DIRECTIONS

logger = logging.getLogger(__name__)

# Maximum number of nodes to expand before giving up.
MAX_NODES = 2_000_000


def astar_solve(board: Board, goal_row: int, target_row: int) -> List[Tuple[str, str]]:
    """Find an optimal move sequence using A*.

    Args:
        board: Current board state.
        goal_row: The row where objects must match the target pattern.
        target_row: The row containing the target pattern (read-only).

    Returns:
        List of (object_name, direction) tuples, or empty list if unsolvable.
    """
    # Build mapping: object -> desired position in the goal row
    target_positions: Dict[str, Tuple[int, int]] = {}
    for col in range(board.num_columns):
        obj = board.grid[target_row][col]
        if obj is not None:
            target_positions[obj] = (goal_row, col)

    num_rows = board.num_rows
    num_cols = board.num_columns
    directions_list = list(DIRECTIONS.items())

    # ── helpers ────────────────────────────────────────────────────────

    def make_state(positions: Dict[str, Tuple[int, int]]) -> Tuple:
        return tuple(sorted(positions.items()))

    def heuristic(positions: Dict[str, Tuple[int, int]]) -> int:
        """Sum of Manhattan distances to each object's target cell."""
        h = 0
        for obj, (row, col) in positions.items():
            if obj in target_positions:
                tr, tc = target_positions[obj]
                h += abs(row - tr) + abs(col - tc)
        return h

    def is_goal(positions: Dict[str, Tuple[int, int]]) -> bool:
        return all(positions.get(obj) == pos for obj, pos in target_positions.items())

    # ── search ─────────────────────────────────────────────────────────

    start_positions = dict(board.object_positions)
    start_state = make_state(start_positions)

    counter = 0
    open_set: list = [(heuristic(start_positions), counter, 0, start_state)]
    g_scores: Dict[Tuple, int] = {start_state: 0}
    came_from: Dict[Tuple, Tuple] = {}  # state -> (parent_state, obj, direction)
    nodes_expanded = 0

    while open_set:
        _f, _, g, state = heapq.heappop(open_set)

        if g > g_scores.get(state, float("inf")):
            continue

        nodes_expanded += 1
        if nodes_expanded > MAX_NODES:
            logger.warning("A* exceeded node limit (%d). Aborting search.", MAX_NODES)
            return []

        state_dict = dict(state)

        if is_goal(state_dict):
            # Reconstruct path
            moves: List[Tuple[str, str]] = []
            current = state
            while current in came_from:
                parent, obj, direction = came_from[current]
                moves.append((obj, direction))
                current = parent
            moves.reverse()
            logger.info(
                "A* found solution with %d moves (expanded %d nodes).",
                len(moves),
                nodes_expanded,
            )
            return moves

        occupied = set(state_dict.values())

        for obj, (row, col) in state_dict.items():
            for direction, (dr, dc) in directions_list:
                new_row = row + dr
                new_col = col + dc

                if not (0 <= new_row < num_rows and 0 <= new_col < num_cols):
                    continue
                if new_row == target_row:
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
