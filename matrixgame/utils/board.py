"""Grid board representation and manipulation for matrixgame."""

from typing import Dict, List, Optional, Tuple
from copy import deepcopy

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class Board:
    """4-row × N-column grid with named objects."""

    def __init__(self, num_columns: int, num_rows: int = 4, target_row: int = 3):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.target_row = target_row
        # grid[row][col] -> object label or None
        self.grid: List[List[Optional[str]]] = [
            [None] * num_columns for _ in range(num_rows)
        ]
        # object_name -> (row, col)
        self.object_positions: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def place_object(self, obj: str, row: int, col: int):
        if self.grid[row][col] is not None:
            raise ValueError(f"Cell ({row},{col}) already occupied by {self.grid[row][col]}")
        self.grid[row][col] = obj
        self.object_positions[obj] = (row, col)

    def set_target_row(self, pattern: List[Optional[str]]):
        """Fill the target row (read-only reference) with the given pattern."""
        for col, obj in enumerate(pattern):
            self.grid[self.target_row][col] = obj

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, obj: str) -> Optional[Tuple[int, int]]:
        return self.object_positions.get(obj)

    def is_cell_empty(self, row: int, col: int) -> bool:
        return self.grid[row][col] is None

    def goal_reached(self, goal_row: int) -> bool:
        """Check if goal_row matches the target_row pattern."""
        return self.grid[goal_row] == self.grid[self.target_row]

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def validate_move(self, obj: str, direction: str) -> Optional[str]:
        """Return an error message if the move is invalid, else None."""
        if direction not in DIRECTIONS:
            return f"Invalid direction '{direction}'. Must be one of: {', '.join(DIRECTIONS)}."
        if obj not in self.object_positions:
            return f"Unknown object '{obj}'."
        row, col = self.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        new_row, new_col = row + dr, col + dc
        if new_row < 0 or new_row >= self.num_rows:
            return f"Cannot move {obj} {direction}: would leave the grid."
        if new_row == self.target_row:
            return f"Cannot move {obj} {direction}: row {self.target_row + 1} is the target row and is read-only."
        if new_col < 0 or new_col >= self.num_columns:
            return f"Cannot move {obj} {direction}: would leave the grid."
        if not self.is_cell_empty(new_row, new_col):
            return f"Cannot move {obj} {direction}: cell ({new_row + 1},{new_col + 1}) is occupied by {self.grid[new_row][new_col]}."
        return None

    def apply_move(self, obj: str, direction: str):
        """Apply a validated move. Call validate_move first."""
        row, col = self.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        new_row, new_col = row + dr, col + dc
        self.grid[row][col] = None
        self.grid[new_row][new_col] = obj
        self.object_positions[obj] = (new_row, new_col)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, show_target: bool = True) -> str:
        """Render the board as a human-readable text grid."""
        col_width = 4
        header = "     " + "".join(f"{'C' + str(c + 1):^{col_width}}" for c in range(self.num_columns))
        separator = "     " + "+".join(["----"] * self.num_columns)
        separator = "     +" + separator[5:] + "+"
        lines = [header]
        row_labels = {0: "R1", 1: "R2", 2: "R3", 3: "R4"}
        for r in range(self.num_rows):
            if not show_target and r == self.target_row:
                continue
            label = row_labels.get(r, f"R{r + 1}")
            tag = " (target)" if r == self.target_row else ""
            cells = []
            for c in range(self.num_columns):
                val = self.grid[r][c] if self.grid[r][c] else "."
                cells.append(f" {val:^2} ")
            row_str = f"{label:>3}{tag} |" + "|".join(cells) + "|"
            lines.append(separator)
            lines.append(row_str)
        lines.append(separator)
        return "\n".join(lines)

    def state_key(self) -> str:
        """Return a hashable fingerprint of the current movable object positions."""
        # Exclude target row since it never changes
        items = sorted(self.object_positions.items())
        return str(items)

    def copy(self) -> "Board":
        new_board = Board(self.num_columns, self.num_rows, self.target_row)
        new_board.grid = deepcopy(self.grid)
        new_board.object_positions = dict(self.object_positions)
        return new_board
