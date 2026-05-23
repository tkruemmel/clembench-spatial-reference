"""Grid board for matrixgame_covered_comm: 8×8 grid with walls and per-object targets."""

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

WALL_CELL = "##"
BLOCKED_MARKER = "X"

_COL_W = 4  # display width per cell


class Board:
    """8×8 grid with named objects, wall cells, and per-object target positions."""

    def __init__(self, grid_size: int = 8, walls: Optional[Set[Tuple[int, int]]] = None):
        self.grid_size = grid_size
        self.walls: Set[Tuple[int, int]] = set(walls) if walls else set()
        self.grid: List[List[Optional[str]]] = [[None] * grid_size for _ in range(grid_size)]
        self.object_positions: Dict[str, Tuple[int, int]] = {}
        self.target_positions: Dict[str, Tuple[int, int]] = {}

    # ── setup ──────────────────────────────────────────────────────────

    def place_object(self, obj: str, row: int, col: int) -> None:
        if (row, col) in self.walls:
            raise ValueError(f"Cell ({row},{col}) is a wall")
        if self.grid[row][col] is not None:
            raise ValueError(f"Cell ({row},{col}) already occupied by {self.grid[row][col]}")
        self.grid[row][col] = obj
        self.object_positions[obj] = (row, col)

    def set_target(self, obj: str, row: int, col: int) -> None:
        self.target_positions[obj] = (row, col)

    # ── queries ────────────────────────────────────────────────────────

    def get_position(self, obj: str) -> Optional[Tuple[int, int]]:
        return self.object_positions.get(obj)

    def is_cell_empty(self, row: int, col: int) -> bool:
        return self.grid[row][col] is None

    def is_solved(self) -> bool:
        """True when every object is at its target position."""
        return all(
            self.object_positions.get(obj) == pos
            for obj, pos in self.target_positions.items()
        )

    # ── moves ──────────────────────────────────────────────────────────

    def validate_move(
        self,
        obj: str,
        direction: str,
        allowed_objects: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """Return an error string if the move is invalid, else None."""
        if direction not in DIRECTIONS:
            return f"Invalid direction '{direction}'. Must be one of: {', '.join(DIRECTIONS)}."
        if obj not in self.object_positions:
            return f"Unknown object '{obj}'."
        if allowed_objects is not None and obj not in allowed_objects:
            return f"You cannot move '{obj}' — it belongs to the other player."
        row, col = self.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        new_row, new_col = row + dr, col + dc
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return f"Cannot move {obj} {direction}: would leave the grid."
        if (new_row, new_col) in self.walls:
            return f"Cannot move {obj} {direction}: (R{new_row + 1},C{new_col + 1}) is a wall."
        if not self.is_cell_empty(new_row, new_col):
            return f"Cannot move {obj} {direction}: (R{new_row + 1},C{new_col + 1}) is occupied."
        return None

    def apply_move(self, obj: str, direction: str) -> None:
        """Apply a pre-validated move."""
        row, col = self.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        new_row, new_col = row + dr, col + dc
        self.grid[row][col] = None
        self.grid[new_row][new_col] = obj
        self.object_positions[obj] = (new_row, new_col)

    # ── rendering ──────────────────────────────────────────────────────

    def render(self) -> str:
        """Full board: all object labels visible."""
        return self._render_grid(visible_objects=None)

    def render_for_player(self, player_objects: Set[str]) -> str:
        """Board from a player's perspective: opponent objects appear as 'X'."""
        return self._render_grid(visible_objects=player_objects)

    def render_targets(self) -> str:
        """Goal board: shows where each object must end up."""
        cell_to_obj: Dict[Tuple[int, int], str] = {
            pos: obj for obj, pos in self.target_positions.items()
        }
        return self._render_raw(cell_to_obj)

    def render_targets_for_player(self, player_objects: Set[str]) -> str:
        """Goal board from a player's perspective: opponent targets appear as 'X'."""
        cell_to_obj: Dict[Tuple[int, int], str] = {
            pos: (obj if obj in player_objects else BLOCKED_MARKER)
            for obj, pos in self.target_positions.items()
        }
        return self._render_raw(cell_to_obj)

    def _render_grid(self, visible_objects: Optional[Set[str]]) -> str:
        n = self.grid_size
        cell_contents: Dict[Tuple[int, int], str] = {}
        for r in range(n):
            for c in range(n):
                val = self.grid[r][c]
                if val is not None:
                    if visible_objects is None or val in visible_objects:
                        cell_contents[(r, c)] = val
                    else:
                        cell_contents[(r, c)] = BLOCKED_MARKER
        return self._render_raw(cell_contents)

    def render_compact(self, visible_objects: Optional[Set[str]] = None) -> str:
        """Compact representation: objects as Label@RxCy, walls listed on a second line."""
        parts = []
        for obj in sorted(self.object_positions):
            r, c = self.object_positions[obj]
            label = obj if (visible_objects is None or obj in visible_objects) else BLOCKED_MARKER
            parts.append(f"{label}@R{r+1}C{c+1}")
        lines = ["Objects: " + " ".join(parts)]
        if self.walls:
            wall_parts = [f"R{r+1}C{c+1}" for r, c in sorted(self.walls)]
            lines.append("Walls: " + " ".join(wall_parts))
        return "\n".join(lines)

    def render_targets_compact(self, visible_objects: Optional[Set[str]] = None) -> str:
        """Compact representation of target positions: Label->RxCy."""
        parts = []
        for obj in sorted(self.target_positions):
            r, c = self.target_positions[obj]
            label = obj if (visible_objects is None or obj in visible_objects) else BLOCKED_MARKER
            parts.append(f"{label}->R{r+1}C{c+1}")
        return "Targets: " + " ".join(parts)

    def _render_raw(self, cell_contents: Dict[Tuple[int, int], str]) -> str:
        n = self.grid_size
        sep = "     +" + "+".join(["----"] * n) + "+"
        header = "     " + "".join(f"{'C' + str(c + 1):^{_COL_W}}" for c in range(n))
        lines = [header, sep]
        for r in range(n):
            cells = []
            for c in range(n):
                if (r, c) in self.walls:
                    display = WALL_CELL
                else:
                    display = cell_contents.get((r, c), ".")
                cells.append(f" {display:^2} ")
            lines.append(f"R{r + 1:>1}   |" + "|".join(cells) + "|")
            lines.append(sep)
        return "\n".join(lines)

    # ── state ──────────────────────────────────────────────────────────

    def state_key(self) -> str:
        return str(sorted(self.object_positions.items()))

    def copy(self) -> "Board":
        b = Board(self.grid_size, self.walls)
        b.grid = deepcopy(self.grid)
        b.object_positions = dict(self.object_positions)
        b.target_positions = dict(self.target_positions)
        return b
