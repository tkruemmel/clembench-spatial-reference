"""8x8 board with walls, named objects and one target cell per object."""

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


class Board:

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
    ) -> Optional[Tuple[str, str]]:
        """Return (code, message) if the move is invalid, else None.

        Codes 'bad_direction', 'unknown_object', 'not_owner', 'off_grid',
        'wall' are ILLEGAL moves (static rule violations that indict the
        mover alone). Code 'occupied' is a BLOCKED move (locally
        well-formed, fails only against the current joint state). The
        scorer builds the illegal-move and blocked-move rates from this
        distinction.
        """
        if direction not in DIRECTIONS:
            return ("bad_direction",
                    f"Invalid direction '{direction}'. Must be one of: {', '.join(DIRECTIONS)}.")
        if obj not in self.object_positions:
            return ("unknown_object", f"Unknown object '{obj}'.")
        if allowed_objects is not None and obj not in allowed_objects:
            own = ", ".join(sorted(allowed_objects))
            return ("not_owner",
                    f"You cannot move '{obj}' — you only control: {own}.")
        row, col = self.object_positions[obj]
        dr, dc = DIRECTIONS[direction]
        new_row, new_col = row + dr, col + dc
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return ("off_grid", f"Cannot move {obj} {direction}: would leave the grid.")
        if (new_row, new_col) in self.walls:
            return ("wall", f"Cannot move {obj} {direction}: (R{new_row + 1},C{new_col + 1}) is a wall.")
        if not self.is_cell_empty(new_row, new_col):
            return ("occupied", f"Cannot move {obj} {direction}: (R{new_row + 1},C{new_col + 1}) is occupied.")
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

    def render(self, player_objects: Optional[Set[str]] = None) -> str:
        """Current board. Objects outside `player_objects` (a teammate's) show
        as 'X'; without it every object shows its letter."""
        cells: Dict[Tuple[int, int], str] = {}
        for (r, c), obj in ((pos, obj) for obj, pos in self.object_positions.items()):
            visible = player_objects is None or obj in player_objects
            cells[(r, c)] = obj if visible else BLOCKED_MARKER
        return self._render_raw(cells)

    def render_targets(self, player_objects: Optional[Set[str]] = None) -> str:
        """Goal board. Only the targets of `player_objects` are drawn, since a
        player cannot move a teammate's objects; without it every target is."""
        return self._render_raw({
            pos: obj
            for obj, pos in self.target_positions.items()
            if player_objects is None or obj in player_objects
        })

    def _render_raw(self, cell_contents: Dict[Tuple[int, int], str]) -> str:
        n = self.grid_size
        sep = "     +" + "+".join(["----"] * n) + "+"
        header = (" " * 6 + " ".join(f"{'C' + str(c + 1):^4}" for c in range(n))).rstrip()
        lines = [header, sep]
        for r in range(n):
            cells = []
            for c in range(n):
                display = WALL_CELL if (r, c) in self.walls else cell_contents.get((r, c), ".")
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
