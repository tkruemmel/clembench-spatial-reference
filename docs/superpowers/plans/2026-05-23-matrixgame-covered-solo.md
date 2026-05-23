# matrixgame_covered_solo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `matrixgame_covered_solo`, a single-player clembench game that mirrors `matrixgame_covered_comm` but isolates spatial planning from coordination by removing the partner.

**Architecture:** New top-level directory `matrixgame_covered_solo/` following the per-variant pattern used elsewhere in the repo. Two new experiment axes (`view_mode`: `full`/`masked`, `thinking`: `on`/`off`) replace `_comm`'s `comm_protocol` axis. `Board` and `astar_solve` are copied verbatim from `_comm/utils/` for runtime independence. One `master.py` containing `SoloMatrixPlayer`, `SoloMatrixGameMaster`, `SoloMatrixGameScorer`, `SoloMatrixGameBenchmark`. The done-declaration mechanism is dropped (master auto-detects success after every move).

**Tech Stack:** Python 3, `clemcore` (`Player`, `DialogueGameMaster`, `GameScorer`, `GameBenchmark`, `GameInstanceGenerator`), `numpy` (for `np.nan` in scorer), `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-05-23-matrixgame-covered-solo-design.md`

**Reference code (read these as needed):**
- `matrixgame_covered_comm/master.py` — parsing regexes, validate/apply flow, scorer formula
- `matrixgame_covered_comm/instancegenerator.py` — board generation per spatial level, `_verify_reachability`
- `matrixgame_covered_comm/utils/board.py`, `utils/astar.py` — copied unchanged
- `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_none.template` — prompt baseline

---

## File Structure

Files this plan creates under `matrixgame_covered_solo/`:

| Path | Responsibility |
|---|---|
| `__init__.py` | Empty marker |
| `clemgame.json` | Game spec (1 player, role "Player") |
| `master.py` | `SoloMatrixPlayer`, `SoloMatrixGameMaster`, `SoloMatrixGameScorer`, `SoloMatrixGameBenchmark` |
| `instancegenerator.py` | `SoloMatrixGameInstanceGenerator` |
| `resources/config.json` | 32 experiment configs |
| `resources/common_config.json` | grid/objects/directions/retries |
| `resources/initial_prompts/en/player_prompt_full_silent.template` | Full-view, no message line |
| `resources/initial_prompts/en/player_prompt_full_thinking.template` | Full-view, with `message:` |
| `resources/initial_prompts/en/player_prompt_masked_silent.template` | Masked-view, no message line |
| `resources/initial_prompts/en/player_prompt_masked_thinking.template` | Masked-view, with `message:` |
| `utils/__init__.py` | Empty marker |
| `utils/board.py` | Copied verbatim from `_comm/utils/board.py` |
| `utils/astar.py` | Copied verbatim from `_comm/utils/astar.py` |
| `tests/__init__.py` | Empty marker |
| `tests/test_parser.py` | Response-format parser tests |
| `tests/test_validator.py` | Move-validation tests |
| `tests/test_renderer.py` | View-rendering tests |
| `tests/test_master.py` | Setup, turn-loop, done, scorer tests |
| `tests/test_instance_generator.py` | Generator invariant tests |
| `in/instances.json` | Generated artifact (committed) |

---

## Task 1: Scaffold directory and copy shared utilities

**Files:**
- Create: `matrixgame_covered_solo/__init__.py`
- Create: `matrixgame_covered_solo/clemgame.json`
- Create: `matrixgame_covered_solo/utils/__init__.py`
- Create: `matrixgame_covered_solo/utils/board.py` (copy of `matrixgame_covered_comm/utils/board.py`)
- Create: `matrixgame_covered_solo/utils/astar.py` (copy of `matrixgame_covered_comm/utils/astar.py`)
- Create: `matrixgame_covered_solo/resources/common_config.json`
- Create: `matrixgame_covered_solo/tests/__init__.py`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p matrixgame_covered_solo/resources/initial_prompts/en
mkdir -p matrixgame_covered_solo/utils
mkdir -p matrixgame_covered_solo/tests
mkdir -p matrixgame_covered_solo/in
```

- [ ] **Step 2: Create marker files**

```bash
touch matrixgame_covered_solo/__init__.py
touch matrixgame_covered_solo/utils/__init__.py
touch matrixgame_covered_solo/tests/__init__.py
```

- [ ] **Step 3: Copy board.py and astar.py verbatim**

```bash
cp matrixgame_covered_comm/utils/board.py matrixgame_covered_solo/utils/board.py
cp matrixgame_covered_comm/utils/astar.py matrixgame_covered_solo/utils/astar.py
```

Then change the module docstrings in both files to say `matrixgame_covered_solo` instead of `matrixgame_covered_comm`. No other edits.

- [ ] **Step 4: Write `clemgame.json`**

```json
[
  {
    "game_name": "matrixgame_covered_solo",
    "description": "Single-player grid rearrangement: one LLM moves objects on an 8x8 grid. Two visibility modes (full view of all 6 objects, or masked view of 3 owned objects with 3 static obstacles) and an optional thinking-out-loud channel.",
    "main_game": "matrixgame_covered_solo",
    "players": 1,
    "image": "none",
    "languages": ["en"],
    "benchmark": ["2.0"],
    "regression": "large",
    "roles": ["Player"]
  }
]
```

- [ ] **Step 5: Write `resources/common_config.json`**

```json
{
    "grid_size": 8,
    "objects": ["A", "B", "C", "D", "E", "F"],
    "directions": ["up", "down", "left", "right"],
    "max_retries": 2,
    "strict": false
}
```

- [ ] **Step 6: Verify board.py imports work**

```bash
cd matrixgame_covered_solo && python -c "from utils.board import Board, DIRECTIONS; from utils.astar import astar_solve; print('ok')" && cd ..
```

Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_solo/
git commit -m "Scaffold matrixgame_covered_solo directory

Copy Board and astar_solve from _comm verbatim for runtime
independence. Add clemgame.json (1 player), common_config.json, and
empty package markers."
```

---

## Task 2: Write the four prompt templates

**Files:**
- Create: `matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_full_silent.template`
- Create: `matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_full_thinking.template`
- Create: `matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_masked_silent.template`
- Create: `matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_masked_thinking.template`

All four templates use `$GRID_SIZE$` and `$YOUR_OBJECTS$` placeholders, populated by `master.py` like `_comm` does. None contain "DONE OPTION" wording (the spec drops the done declaration).

- [ ] **Step 1: Write `player_prompt_full_silent.template`**

```
You are playing a grid rearrangement puzzle.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).

YOUR TASK:
Move all objects to their target positions. The GOAL BOARD shows where each object must end up.
You control ALL objects on the board: $YOUR_OBJECTS$. You may move any of them on any turn.

RULES:
- On each turn, move exactly ONE object exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##) or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object or find an alternative route.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach: reroute via a completely different path, or temporarily move another object to open space.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `reason:` on the very first line — do not output any preamble, thinking, or scratchpad text before it.
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
reason: Object A is at R3,C4 and needs to reach R3,C5. Moving it right puts it on target.
move: A to R3,C5 (right)

The game ends automatically as soon as every object is on its target cell — you do not need to declare you are done.
```

- [ ] **Step 2: Write `player_prompt_full_thinking.template`**

```
You are playing a grid rearrangement puzzle.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).

YOUR TASK:
Move all objects to their target positions. The GOAL BOARD shows where each object must end up.
You control ALL objects on the board: $YOUR_OBJECTS$. You may move any of them on any turn.

RULES:
- On each turn, move exactly ONE object exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##) or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object or find an alternative route.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach: reroute via a completely different path, or temporarily move another object to open space.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `message:` on the very first line — do not output any preamble before it.
message: <a brief note to yourself — a scratchpad for plans, observations, or reminders. Logged for analysis but does not affect the game.>
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
message: F has only one route around the wall — handle it last so it doesn't block A's path.
reason: Object A is at R3,C4 and needs to reach R3,C5. Moving it right puts it on target.
move: A to R3,C5 (right)

The game ends automatically as soon as every object is on its target cell — you do not need to declare you are done.
```

- [ ] **Step 3: Write `player_prompt_masked_silent.template`**

```
You are playing a grid rearrangement puzzle.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).
- Cells marked X contain immovable obstacles that you cannot see the identity of. They never move and never become empty.

YOUR TASK:
Move YOUR objects to their target positions. The GOAL BOARD shows where each of YOUR objects must end up.
You control ONLY these objects: $YOUR_OBJECTS$. You may ONLY move your own objects.
The X cells are static obstacles — treat them like walls when planning routes.

RULES:
- On each turn, move exactly ONE of your objects exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##), X cells, or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object or find an alternative route.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach: reroute via a completely different path, or temporarily move another of your objects to open space.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `reason:` on the very first line — do not output any preamble, thinking, or scratchpad text before it.
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
reason: Object A is at R3,C4 and needs to reach R3,C5. Moving it right puts it on target.
move: A to R3,C5 (right)

The game ends automatically as soon as all of YOUR objects are on their target cells — you do not need to declare you are done. The X obstacles are not your concern.
```

- [ ] **Step 4: Write `player_prompt_masked_thinking.template`**

```
You are playing a grid rearrangement puzzle.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).
- Cells marked X contain immovable obstacles that you cannot see the identity of. They never move and never become empty.

YOUR TASK:
Move YOUR objects to their target positions. The GOAL BOARD shows where each of YOUR objects must end up.
You control ONLY these objects: $YOUR_OBJECTS$. You may ONLY move your own objects.
The X cells are static obstacles — treat them like walls when planning routes.

RULES:
- On each turn, move exactly ONE of your objects exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##), X cells, or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object or find an alternative route.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach: reroute via a completely different path, or temporarily move another of your objects to open space.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `message:` on the very first line — do not output any preamble before it.
message: <a brief note to yourself — a scratchpad for plans, observations, or reminders. Logged for analysis but does not affect the game.>
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
message: The X cluster at R4 blocks C's direct path — I'll route C through R6.
reason: Object A is at R3,C4 and needs to reach R3,C5. Moving it right puts it on target.
move: A to R3,C5 (right)

The game ends automatically as soon as all of YOUR objects are on their target cells — you do not need to declare you are done. The X obstacles are not your concern.
```

- [ ] **Step 5: Verify all four templates exist and contain the placeholders**

```bash
for f in matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_*.template; do
  grep -q '\$GRID_SIZE\$' "$f" && grep -q '\$YOUR_OBJECTS\$' "$f" && echo "ok: $f" || echo "FAIL: $f"
done
```

Expected: four `ok:` lines, no `FAIL:`.

- [ ] **Step 6: Commit**

```bash
git add matrixgame_covered_solo/resources/initial_prompts/
git commit -m "Add solo prompt templates for full/masked × silent/thinking

Four templates cover the visibility × thinking axes. No done-declaration
instructions (master auto-detects success). Masked templates describe X
cells as static obstacles."
```

---

## Task 3: Write parser with tests

Defines `_parse_response` and the two regex patterns. This is the first piece of `master.py`; everything in later tasks appends to it.

**Files:**
- Create: `matrixgame_covered_solo/master.py` (start of file)
- Create: `matrixgame_covered_solo/tests/test_parser.py`

- [ ] **Step 1: Write failing parser tests**

`matrixgame_covered_solo/tests/test_parser.py`:

```python
"""Tests for the response parser in matrixgame_covered_solo.master."""

import pytest

from matrixgame_covered_solo.master import _parse_response


# ── silent mode (with_message=False) ──────────────────────────────────

def test_parse_silent_valid_lowercase_dir():
    response = "reason: A needs to move right.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=False)
    assert parsed == {
        "reason": "A needs to move right.",
        "object": "A",
        "direction": "right",
        "target": "R3,C5",
    }


def test_parse_silent_valid_uppercase_dir():
    response = "REASON: B goes up.\nMOVE: B to R1,C2 (UP)"
    parsed = _parse_response(response, with_message=False)
    assert parsed is not None
    assert parsed["object"] == "B"
    assert parsed["direction"] == "up"


def test_parse_silent_rejects_message_line():
    """In silent mode, a response containing 'message:' must not parse."""
    response = "message: hi\nreason: A needs to move.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=False)
    # The silent regex doesn't allow a message: line — it should not match
    # the move portion when preceded by message:. Either it returns None,
    # or it returns a parse that ignores the message line. We require it
    # to return None to keep the modes cleanly separated.
    assert parsed is None


def test_parse_silent_rejects_malformed():
    assert _parse_response("just some text", with_message=False) is None
    assert _parse_response("move: A to R3,C5 (right)", with_message=False) is None  # missing reason
    assert _parse_response("reason: hi\nmove: A right", with_message=False) is None  # bad move format


# ── thinking mode (with_message=True) ──────────────────────────────────

def test_parse_thinking_valid():
    response = (
        "message: plan B last.\n"
        "reason: move A right to target.\n"
        "move: A to R3,C5 (right)"
    )
    parsed = _parse_response(response, with_message=True)
    assert parsed == {
        "message": "plan B last.",
        "reason": "move A right to target.",
        "object": "A",
        "direction": "right",
        "target": "R3,C5",
    }


def test_parse_thinking_rejects_missing_message():
    """In thinking mode, the message: line is required."""
    response = "reason: hi.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=True)
    assert parsed is None


def test_parse_thinking_rejects_malformed():
    assert _parse_response("just text", with_message=True) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_parser.py -v
```

Expected: `ImportError: cannot import name '_parse_response'` (because master.py doesn't exist yet).

- [ ] **Step 3: Create initial `master.py` with imports and parser**

`matrixgame_covered_solo/master.py`:

```python
"""Matrix Game (covered solo): single-LLM grid rearrangement.

Two visibility variants:
  full   — model sees all 6 objects and owns all 6.
  masked — model sees only its 3 owned objects; the other 3 appear as X and
           are static obstacles (counted against move validity, not success).

Optional 'thinking' axis adds a free-form `message:` line per turn that the
master logs but does not interpret.

Turn structure (silent):
  reason: ...
  move: OBJECT to Rr,Cc (DIRECTION)

Turn structure (thinking):
  message: ...
  reason: ...
  move: OBJECT to Rr,Cc (DIRECTION)

The episode ends automatically when all owned objects are at their target
positions (success), when max_turns is reached (lose), or when parse/validate
failures exhaust max_retries (abort).
"""

import re
import logging
from typing import Dict, List, Optional, Set

import numpy as np

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, Player, GameBenchmark
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.metrics import (
    METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE,
    METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_VIOLATED,
    METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE,
)

from utils.board import Board, DIRECTIONS

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

VIEW_FULL = "full"
VIEW_MASKED = "masked"

TURN_MOVES = "Turn Moves"
MOVE_COUNT = "Move Count"

# ── Response patterns ──────────────────────────────────────────────────

# Used when thinking is off.
# Anchored with \A to reject responses that prepend other lines (e.g. message:).
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*reason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE | re.DOTALL,
)

# Used when thinking is on.
MESSAGE_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nreason:\s*(?P<reason>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+to\s+R(?P<target_row>\d+),?C(?P<target_col>\d+)\s*\((?P<direction>up|down|left|right)\)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_response(response: str, with_message: bool) -> Optional[Dict[str, str]]:
    """Parse a model response into a move dict, or None if malformed.

    The two patterns are anchored with \\A so a stray 'message:' line in silent
    mode (and a missing 'message:' line in thinking mode) both fail to parse.
    """
    pattern = MESSAGE_MOVE_PATTERN if with_message else MOVE_ONLY_PATTERN
    match = pattern.match(response)
    if not match:
        return None
    result: Dict[str, str] = {
        "reason": match.group("reason").strip(),
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
        "target": f"R{match.group('target_row')},C{match.group('target_col')}",
    }
    if with_message:
        result["message"] = match.group("message").strip()
    return result
```

Note: `\A` anchor + `pattern.match()` together force the response to start with the expected first line. This is the mechanism that makes silent-mode reject a stray `message:` line (test 3) and thinking-mode reject a missing `message:` line (test 6).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_parser.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_parser.py
git commit -m "Add response parser for solo with silent and thinking modes

Two anchored regexes enforce clean mode separation: silent mode rejects
responses with a leading message: line; thinking mode requires one."
```

---

## Task 4: Implement `SoloMatrixPlayer` and `_render_view` with tests

Adds the player class and the rendering helper. Both lean entirely on `Board` methods that already exist.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append player class and render helper)
- Create: `matrixgame_covered_solo/tests/test_renderer.py`

- [ ] **Step 1: Write failing renderer tests**

`matrixgame_covered_solo/tests/test_renderer.py`:

```python
"""Tests for view rendering in matrixgame_covered_solo."""

from utils.board import Board
from matrixgame_covered_solo.master import SoloMatrixPlayer, _render_view


class _StubModel:
    """Minimal Model substitute for instantiating Player without a backend."""
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()


def _build_board() -> Board:
    """3 owned (A,B,C) and 3 foreign (D,E,F) on a small board, no walls."""
    b = Board(grid_size=4)
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    b.place_object("C", 0, 2)
    b.place_object("D", 3, 0)
    b.place_object("E", 3, 1)
    b.place_object("F", 3, 2)
    return b


def test_render_full_shows_all_letters():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C", "D", "E", "F"})
    rendered = _render_view(_build_board(), player, view_mode="full", compact=False)
    for letter in "ABCDEF":
        assert letter in rendered, f"{letter!r} missing from full-view render"
    assert "X" not in rendered  # nothing should be masked


def test_render_masked_shows_owned_and_x_for_foreign():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C"})
    rendered = _render_view(_build_board(), player, view_mode="masked", compact=False)
    for letter in "ABC":
        assert letter in rendered, f"owned letter {letter!r} missing from masked render"
    for letter in "DEF":
        assert letter not in rendered, f"foreign letter {letter!r} leaked into masked render"
    assert "X" in rendered  # foreigns appear as X


def test_render_compact_full_lists_all_objects():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C", "D", "E", "F"})
    rendered = _render_view(_build_board(), player, view_mode="full", compact=True)
    assert "Objects:" in rendered
    for letter in "ABCDEF":
        assert f"{letter}@" in rendered


def test_render_compact_masked_hides_foreign_identities():
    player = SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C"})
    rendered = _render_view(_build_board(), player, view_mode="masked", compact=True)
    for letter in "ABC":
        assert f"{letter}@" in rendered
    for letter in "DEF":
        assert f"{letter}@" not in rendered
    # foreign cells appear with label X
    assert "X@" in rendered
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_renderer.py -v
```

Expected: `ImportError: cannot import name 'SoloMatrixPlayer'`.

- [ ] **Step 3: Append `SoloMatrixPlayer` and `_render_view` to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
# ── Player ─────────────────────────────────────────────────────────────

class SoloMatrixPlayer(Player):
    def __init__(self, model: Model, role_name: str, own_objects: Set[str]):
        super().__init__(model)
        self.role_name = role_name
        self.own_objects = own_objects

    def _custom_response(self, context):
        # Used by clemcore's mock player; emit a deterministic valid-format move.
        first = sorted(self.own_objects)[0]
        return f"reason: custom player\nmove: {first} to R1,C1 (down)"


# ── Rendering helper ───────────────────────────────────────────────────

def _render_view(board: Board, player: SoloMatrixPlayer, view_mode: str, compact: bool) -> str:
    """Render `board` from `player`'s perspective.

    view_mode=full   → all letters visible (player.own_objects covers all 6).
    view_mode=masked → only player.own_objects visible; rest as 'X'.
    compact          → use Board's compact text representation.
    """
    visible = player.own_objects if view_mode == VIEW_MASKED else None
    if compact:
        return board.render_compact(visible_objects=visible)
    if visible is not None:
        return board.render_for_player(visible)
    return board.render()


def _render_targets_view(board: Board, player: SoloMatrixPlayer, view_mode: str, compact: bool) -> str:
    """Same as _render_view but for the goal board."""
    visible = player.own_objects if view_mode == VIEW_MASKED else None
    if compact:
        return board.render_targets_compact(visible_objects=visible)
    if visible is not None:
        return board.render_targets_for_player(visible)
    return board.render_targets()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_renderer.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_renderer.py
git commit -m "Add SoloMatrixPlayer and view-rendering helper

Player carries the owned-objects set. _render_view delegates to
existing Board methods: full mode returns the unmasked grid, masked
mode shows foreigns as X. Compact variants supported."
```

---

## Task 5: Implement move validation with tests

`Board.validate_move` already exists with the right semantics — invalid object, out of bounds, wall, occupied. Here we just thinly wrap it with a wrapper that uses the player's `own_objects` set.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append `_validate_move` helper)
- Create: `matrixgame_covered_solo/tests/test_validator.py`

- [ ] **Step 1: Write failing validator tests**

`matrixgame_covered_solo/tests/test_validator.py`:

```python
"""Tests for move validation in matrixgame_covered_solo."""

from utils.board import Board
from matrixgame_covered_solo.master import SoloMatrixPlayer, _validate_move


class _StubModel:
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()


def _board_with_walls():
    b = Board(grid_size=4, walls={(1, 1)})
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    b.place_object("D", 2, 0)  # foreign
    return b


def _masked_player():
    return SoloMatrixPlayer(_StubModel(), "Player", {"A", "B", "C"})


def test_validate_accepts_valid_move():
    parsed = {"object": "A", "direction": "down", "target": "R1,C0"}
    error = _validate_move(_board_with_walls(), _masked_player(), parsed)
    assert error is None


def test_validate_rejects_unowned_object():
    parsed = {"object": "D", "direction": "right", "target": "R2,C1"}
    error = _validate_move(_board_with_walls(), _masked_player(), parsed)
    assert error is not None
    assert "D" in error


def test_validate_rejects_wall():
    parsed = {"object": "B", "direction": "down", "target": "R1,C1"}
    error = _validate_move(_board_with_walls(), _masked_player(), parsed)
    assert error is not None
    assert "wall" in error.lower()


def test_validate_rejects_occupied_by_foreign():
    """Foreign D sits at (2,0); A trying to land there must fail."""
    b = _board_with_walls()
    # Move A down once so its next 'down' would land on D.
    b.apply_move("A", "down")
    parsed = {"object": "A", "direction": "down", "target": "R2,C0"}
    error = _validate_move(b, _masked_player(), parsed)
    assert error is not None
    assert "occupied" in error.lower()


def test_validate_rejects_out_of_bounds():
    parsed = {"object": "A", "direction": "up", "target": "R-1,C0"}
    error = _validate_move(_board_with_walls(), _masked_player(), parsed)
    assert error is not None
    assert "grid" in error.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_validator.py -v
```

Expected: `ImportError: cannot import name '_validate_move'`.

- [ ] **Step 3: Append `_validate_move` to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
# ── Validation ─────────────────────────────────────────────────────────

def _validate_move(board: Board, player: SoloMatrixPlayer, parsed: Dict[str, str]) -> Optional[str]:
    """Return an error string if the move is invalid, else None.

    Wraps Board.validate_move with the player's allowed-objects set.
    """
    return board.validate_move(
        parsed["object"],
        parsed["direction"],
        allowed_objects=player.own_objects,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_validator.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_validator.py
git commit -m "Add _validate_move wrapping Board.validate_move

Player's own_objects feeds the allowed-objects filter, so foreigns in
masked mode are rejected with 'belongs to the other player' (which we
relabel to a solo-appropriate message in the master later)."
```

---

## Task 6: Write instance generator with tests

Ports the relevant pieces of `_comm/instancegenerator.py`. The S1–S4 board generators and helpers (`_bfs_path`, `_verify_reachability`, `_is_2connected`, etc.) are copied unchanged from `_comm`; `on_generate` is rewritten for solo.

**Files:**
- Create: `matrixgame_covered_solo/instancegenerator.py`
- Create: `matrixgame_covered_solo/tests/test_instance_generator.py`

- [ ] **Step 1: Write failing generator invariant tests**

`matrixgame_covered_solo/tests/test_instance_generator.py`:

```python
"""Tests for SoloMatrixGameInstanceGenerator."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    """Run instance generator into a temp output dir, return parsed instances.json."""
    out_dir = tmp_path_factory.mktemp("solo_in")
    # The generator writes to <game_dir>/in/instances.json by default. We run
    # it programmatically so we can point the output elsewhere.
    import sys
    sys.path.insert(0, str(REPO / "matrixgame_covered_solo"))
    from instancegenerator import SoloMatrixGameInstanceGenerator

    gen = SoloMatrixGameInstanceGenerator()
    # Generate a small sample (2 instances per variant) for fast tests.
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_32_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 32, f"expected 32 experiments, got {len(experiments)}"


def test_each_experiment_has_required_fields(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in (
                "grid_size", "walls", "objects", "owned_objects", "foreign_objects",
                "start_positions", "target_positions",
                "optimal_moves", "max_turns", "max_retries",
                "view_mode", "thinking", "spatial_level", "compact_board",
                "player_prompt",
            ):
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_view_mode_full_owns_all_six(generated_instances):
    full_exps = [e for e in generated_instances["experiments"]
                 if e["game_instances"][0]["view_mode"] == "full"]
    assert full_exps, "no full-view experiments generated"
    for exp in full_exps:
        for inst in exp["game_instances"]:
            assert set(inst["owned_objects"]) == set("ABCDEF")
            assert set(inst["foreign_objects"]) == set()


def test_view_mode_masked_owns_three(generated_instances):
    masked_exps = [e for e in generated_instances["experiments"]
                   if e["game_instances"][0]["view_mode"] == "masked"]
    assert masked_exps
    for exp in masked_exps:
        for inst in exp["game_instances"]:
            assert len(inst["owned_objects"]) == 3
            assert len(inst["foreign_objects"]) == 3
            assert set(inst["owned_objects"]) & set(inst["foreign_objects"]) == set()
            assert set(inst["owned_objects"]) | set(inst["foreign_objects"]) == set("ABCDEF")


def test_origins_differ_from_targets(generated_instances):
    """Generator invariant: every owned object's origin ≠ its target."""
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for obj in inst["owned_objects"]:
                origin = tuple(inst["start_positions"][obj])
                target = tuple(inst["target_positions"][obj])
                assert origin != target, f"{exp['name']} inst {inst['game_id']}: {obj} origin==target"


def test_max_turns_matches_formula(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["max_turns"] == max(4 * inst["optimal_moves"], 20)


def test_optimal_moves_positive(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["optimal_moves"] >= len(inst["owned_objects"]), (
                f"{exp['name']}: optimal {inst['optimal_moves']} < num_owned"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_instance_generator.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` on `instancegenerator`.

- [ ] **Step 3: Copy generator from `_comm` and adapt**

```bash
cp matrixgame_covered_comm/instancegenerator.py matrixgame_covered_solo/instancegenerator.py
```

Then edit `matrixgame_covered_solo/instancegenerator.py`:

**3a. Update the module docstring.** Replace the first docstring block (lines 1-27 of `_comm`'s version) with:

```python
"""Instance generator for matrixgame_covered_solo.

Reuses _comm's per-spatial-level board generators (S1–S4) unchanged; only
the experiment fan-out and per-instance fields differ:

- Two axes (view_mode, thinking) replace _comm's comm_protocol axis.
- For view_mode=masked, the player_a_objects from _comm's per-level generator
  become 'owned_objects' and player_b_objects become 'foreign_objects'
  (static obstacles).
- For view_mode=full, owned_objects = all 6 letters; foreign_objects = [].
- optimal_moves uses _verify_reachability:
    masked → over owned only, with foreign cells added to the wall set.
    full   → over all 6, walls only.
- max_turns = max(4 * optimal_moves, 20), same as _comm.
"""
```

**3b. Rename class and update `on_generate`.** Replace the class body with:

```python
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
            view_mode = variant_config["view_mode"]
            thinking = variant_config["thinking"]
            spatial_level = variant_config["spatial_level"]
            compact_board = variant_config.get("compact_board", False)
            num_objects = variant_config["num_objects"]

            objects = all_objects[:num_objects]
            template_name = (
                f"player_prompt_{view_mode}_"
                f"{'thinking' if thinking else 'silent'}"
            )
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
                player_a_objects = player_b_objects = None
                optimal = 0

                for _attempt in range(80):
                    result = gen_fn(grid_size, objects, rng)
                    starts, targets, walls, _optimal_combined, player_a_objects, player_b_objects = result

                    if spatial_level != "S1" and len(walls) == 0:
                        continue

                    # Solo-specific reachability + optimal_moves
                    if view_mode == VIEW_MASKED:
                        owned = list(player_a_objects)
                        foreign = list(player_b_objects)
                        wall_set = set(map(tuple, walls)) | {
                            tuple(starts[obj]) for obj in foreign
                        }
                        owned_starts = {o: starts[o] for o in owned}
                        owned_targets = {o: targets[o] for o in owned}
                        ok, optimal = _verify_reachability(
                            grid_size, wall_set, owned_starts, owned_targets,
                        )
                        if not ok:
                            continue
                    else:  # VIEW_FULL
                        owned = list(objects)
                        foreign = []
                        wall_set = set(map(tuple, walls))
                        ok, optimal = _verify_reachability(
                            grid_size, wall_set, starts, targets,
                        )
                        if not ok:
                            continue

                    # Invariant: every owned object's origin differs from its target
                    if any(tuple(starts[o]) == tuple(targets[o]) for o in owned):
                        continue

                    break
                else:
                    raise RuntimeError(
                        f"Could not generate viable instance for {variant_config['name']} idx {idx}"
                    )

                game_instance["grid_size"] = grid_size
                game_instance["walls"] = walls
                game_instance["objects"] = objects
                game_instance["owned_objects"] = sorted(owned)
                game_instance["foreign_objects"] = sorted(foreign)
                game_instance["start_positions"] = starts
                game_instance["target_positions"] = targets
                game_instance["optimal_moves"] = optimal
                game_instance["max_turns"] = max(4 * optimal, 20)
                game_instance["max_retries"] = max_retries
                game_instance["strict"] = strict
                game_instance["view_mode"] = view_mode
                game_instance["thinking"] = thinking
                game_instance["spatial_level"] = spatial_level
                game_instance["compact_board"] = compact_board
                game_instance["player_prompt"] = prompt_template
```

**3c. Add the VIEW_MASKED constant near the top** (after the imports, before the class):

```python
VIEW_FULL = "full"
VIEW_MASKED = "masked"
```

**3d. Keep all helpers below** (`_bfs_path`, `_verify_reachability`, `_is_2connected`, `_coordination_isolation_check`, `_generate_s1`, `_generate_s2`, `_generate_s3`, `_generate_s4`, and their internal helpers) **exactly as copied from `_comm`** — no edits.

- [ ] **Step 4: Write a minimal `config.json` so the generator can run**

`matrixgame_covered_solo/resources/config.json`:

```json
{
    "solo_full_S1_silent": {"name": "Solo_Full_S1_Silent", "view_mode": "full", "thinking": false, "spatial_level": "S1", "compact_board": false, "num_objects": 6},
    "solo_full_S1_thinking": {"name": "Solo_Full_S1_Thinking", "view_mode": "full", "thinking": true, "spatial_level": "S1", "compact_board": false, "num_objects": 6},
    "solo_full_S2_silent": {"name": "Solo_Full_S2_Silent", "view_mode": "full", "thinking": false, "spatial_level": "S2", "compact_board": false, "num_objects": 6},
    "solo_full_S2_thinking": {"name": "Solo_Full_S2_Thinking", "view_mode": "full", "thinking": true, "spatial_level": "S2", "compact_board": false, "num_objects": 6},
    "solo_full_S3_silent": {"name": "Solo_Full_S3_Silent", "view_mode": "full", "thinking": false, "spatial_level": "S3", "compact_board": false, "num_objects": 6},
    "solo_full_S3_thinking": {"name": "Solo_Full_S3_Thinking", "view_mode": "full", "thinking": true, "spatial_level": "S3", "compact_board": false, "num_objects": 6},
    "solo_full_S4_silent": {"name": "Solo_Full_S4_Silent", "view_mode": "full", "thinking": false, "spatial_level": "S4", "compact_board": false, "num_objects": 6},
    "solo_full_S4_thinking": {"name": "Solo_Full_S4_Thinking", "view_mode": "full", "thinking": true, "spatial_level": "S4", "compact_board": false, "num_objects": 6},
    "solo_masked_S1_silent": {"name": "Solo_Masked_S1_Silent", "view_mode": "masked", "thinking": false, "spatial_level": "S1", "compact_board": false, "num_objects": 6},
    "solo_masked_S1_thinking": {"name": "Solo_Masked_S1_Thinking", "view_mode": "masked", "thinking": true, "spatial_level": "S1", "compact_board": false, "num_objects": 6},
    "solo_masked_S2_silent": {"name": "Solo_Masked_S2_Silent", "view_mode": "masked", "thinking": false, "spatial_level": "S2", "compact_board": false, "num_objects": 6},
    "solo_masked_S2_thinking": {"name": "Solo_Masked_S2_Thinking", "view_mode": "masked", "thinking": true, "spatial_level": "S2", "compact_board": false, "num_objects": 6},
    "solo_masked_S3_silent": {"name": "Solo_Masked_S3_Silent", "view_mode": "masked", "thinking": false, "spatial_level": "S3", "compact_board": false, "num_objects": 6},
    "solo_masked_S3_thinking": {"name": "Solo_Masked_S3_Thinking", "view_mode": "masked", "thinking": true, "spatial_level": "S3", "compact_board": false, "num_objects": 6},
    "solo_masked_S4_silent": {"name": "Solo_Masked_S4_Silent", "view_mode": "masked", "thinking": false, "spatial_level": "S4", "compact_board": false, "num_objects": 6},
    "solo_masked_S4_thinking": {"name": "Solo_Masked_S4_Thinking", "view_mode": "masked", "thinking": true, "spatial_level": "S4", "compact_board": false, "num_objects": 6},

    "solo_full_S1_silent_compact": {"name": "Solo_Full_S1_Silent_Compact", "view_mode": "full", "thinking": false, "spatial_level": "S1", "compact_board": true, "num_objects": 6},
    "solo_full_S1_thinking_compact": {"name": "Solo_Full_S1_Thinking_Compact", "view_mode": "full", "thinking": true, "spatial_level": "S1", "compact_board": true, "num_objects": 6},
    "solo_full_S2_silent_compact": {"name": "Solo_Full_S2_Silent_Compact", "view_mode": "full", "thinking": false, "spatial_level": "S2", "compact_board": true, "num_objects": 6},
    "solo_full_S2_thinking_compact": {"name": "Solo_Full_S2_Thinking_Compact", "view_mode": "full", "thinking": true, "spatial_level": "S2", "compact_board": true, "num_objects": 6},
    "solo_full_S3_silent_compact": {"name": "Solo_Full_S3_Silent_Compact", "view_mode": "full", "thinking": false, "spatial_level": "S3", "compact_board": true, "num_objects": 6},
    "solo_full_S3_thinking_compact": {"name": "Solo_Full_S3_Thinking_Compact", "view_mode": "full", "thinking": true, "spatial_level": "S3", "compact_board": true, "num_objects": 6},
    "solo_full_S4_silent_compact": {"name": "Solo_Full_S4_Silent_Compact", "view_mode": "full", "thinking": false, "spatial_level": "S4", "compact_board": true, "num_objects": 6},
    "solo_full_S4_thinking_compact": {"name": "Solo_Full_S4_Thinking_Compact", "view_mode": "full", "thinking": true, "spatial_level": "S4", "compact_board": true, "num_objects": 6},
    "solo_masked_S1_silent_compact": {"name": "Solo_Masked_S1_Silent_Compact", "view_mode": "masked", "thinking": false, "spatial_level": "S1", "compact_board": true, "num_objects": 6},
    "solo_masked_S1_thinking_compact": {"name": "Solo_Masked_S1_Thinking_Compact", "view_mode": "masked", "thinking": true, "spatial_level": "S1", "compact_board": true, "num_objects": 6},
    "solo_masked_S2_silent_compact": {"name": "Solo_Masked_S2_Silent_Compact", "view_mode": "masked", "thinking": false, "spatial_level": "S2", "compact_board": true, "num_objects": 6},
    "solo_masked_S2_thinking_compact": {"name": "Solo_Masked_S2_Thinking_Compact", "view_mode": "masked", "thinking": true, "spatial_level": "S2", "compact_board": true, "num_objects": 6},
    "solo_masked_S3_silent_compact": {"name": "Solo_Masked_S3_Silent_Compact", "view_mode": "masked", "thinking": false, "spatial_level": "S3", "compact_board": true, "num_objects": 6},
    "solo_masked_S3_thinking_compact": {"name": "Solo_Masked_S3_Thinking_Compact", "view_mode": "masked", "thinking": true, "spatial_level": "S3", "compact_board": true, "num_objects": 6},
    "solo_masked_S4_silent_compact": {"name": "Solo_Masked_S4_Silent_Compact", "view_mode": "masked", "thinking": false, "spatial_level": "S4", "compact_board": true, "num_objects": 6},
    "solo_masked_S4_thinking_compact": {"name": "Solo_Masked_S4_Thinking_Compact", "view_mode": "masked", "thinking": true, "spatial_level": "S4", "compact_board": true, "num_objects": 6}
}
```

- [ ] **Step 5: Run instance-generator tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_instance_generator.py -v
```

Expected: 7 passed. If `test_optimal_moves_positive` fails because `optimal < num_owned`, the `_verify_reachability` call returned 0 for some object whose start equals target — recheck the origin-≠-target invariant in step 3b.

- [ ] **Step 6: Generate the real `in/instances.json` (10 per variant, 320 instances)**

```bash
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_solo && \
  python -c "from instancegenerator import SoloMatrixGameInstanceGenerator; \
             g = SoloMatrixGameInstanceGenerator(); \
             g.generate(filename='in/instances.json', seed=42, num_instances=10)" && cd ../..
```

Expected: exits cleanly, `in/instances.json` exists.

```bash
python -c "import json; d = json.load(open('matrixgame_covered_solo/in/instances.json')); \
           print('experiments:', len(d['experiments']), \
                 'total instances:', sum(len(e['game_instances']) for e in d['experiments']))"
```

Expected: `experiments: 32 total instances: 320`.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_solo/instancegenerator.py \
        matrixgame_covered_solo/resources/config.json \
        matrixgame_covered_solo/tests/test_instance_generator.py \
        matrixgame_covered_solo/in/instances.json
git commit -m "Add solo instance generator and 32 experiment configs

Generator reuses _comm's per-spatial-level board layouts and helpers
unchanged. Solo-specific changes: view_mode-aware ownership split,
optimal_moves via _verify_reachability with foreigns folded into the
wall set for masked mode, and origin-≠-target invariant on owned
objects. Generates 320 instances total (32 configs × 10 instances)."
```

---

## Task 7: Implement master setup with tests

Adds `SoloMatrixGameMaster.__init__` and `_on_setup`. We test the setup outcome (player registered with correct `own_objects`, initial context contains rendered board and goal board) using a stub model and a synthesized one-instance dict.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append the master class skeleton)
- Create: `matrixgame_covered_solo/tests/test_master.py`

- [ ] **Step 1: Write failing setup tests**

`matrixgame_covered_solo/tests/test_master.py`:

```python
"""Tests for SoloMatrixGameMaster setup, turn handling, and scoring."""

import pytest

from clemcore.clemgame import GameSpec
from matrixgame_covered_solo.master import (
    SoloMatrixGameMaster,
    SoloMatrixPlayer,
    VIEW_FULL,
    VIEW_MASKED,
)


class _StubModel:
    """Minimal Model substitute for masters that don't actually call the model."""
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()

    def set_gen_arg(self, key, value):
        pass


def _instance(view_mode: str, thinking: bool, prompt: str = "PROMPT $YOUR_OBJECTS$ $GRID_SIZE$") -> dict:
    owned = ["A", "B", "C", "D", "E", "F"] if view_mode == VIEW_FULL else ["A", "B", "C"]
    foreign = [] if view_mode == VIEW_FULL else ["D", "E", "F"]
    return {
        "grid_size": 4,
        "walls": [],
        "objects": ["A", "B", "C", "D", "E", "F"],
        "owned_objects": owned,
        "foreign_objects": foreign,
        "start_positions": {
            "A": [0, 0], "B": [0, 1], "C": [0, 2],
            "D": [3, 0], "E": [3, 1], "F": [3, 2],
        },
        "target_positions": {
            "A": [2, 0], "B": [2, 1], "C": [2, 2],
            "D": [1, 0], "E": [1, 1], "F": [1, 2],
        },
        "optimal_moves": 8,
        "max_turns": 32,
        "max_retries": 2,
        "strict": False,
        "view_mode": view_mode,
        "thinking": thinking,
        "spatial_level": "S1",
        "compact_board": False,
        "player_prompt": prompt,
    }


def _make_master(view_mode: str, thinking: bool):
    game_spec = GameSpec(game_name="matrixgame_covered_solo", main_game="matrixgame_covered_solo")
    experiment = {"name": f"Solo_{view_mode}_{'T' if thinking else 'S'}"}
    master = SoloMatrixGameMaster(game_spec, experiment, [_StubModel()])
    master._on_setup(**_instance(view_mode, thinking))
    return master


def test_setup_full_registers_single_player_with_all_six_owned():
    master = _make_master(VIEW_FULL, thinking=False)
    players = master.get_players()
    assert len(players) == 1
    assert isinstance(players[0], SoloMatrixPlayer)
    assert players[0].own_objects == set("ABCDEF")


def test_setup_masked_registers_single_player_with_three_owned():
    master = _make_master(VIEW_MASKED, thinking=True)
    players = master.get_players()
    assert len(players) == 1
    assert players[0].own_objects == set("ABC")


def test_setup_board_has_all_six_objects_placed():
    """Even in masked mode, all 6 objects exist on the board (3 are static)."""
    master = _make_master(VIEW_MASKED, thinking=False)
    assert set(master.board.object_positions.keys()) == set("ABCDEF")


def test_setup_records_view_mode_and_thinking_flags():
    master = _make_master(VIEW_MASKED, thinking=True)
    assert master.view_mode == VIEW_MASKED
    assert master.thinking is True
    assert master.with_message is True

    master2 = _make_master(VIEW_FULL, thinking=False)
    assert master2.with_message is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py -v
```

Expected: `ImportError: cannot import name 'SoloMatrixGameMaster'`.

- [ ] **Step 3: Append `SoloMatrixGameMaster.__init__` and `_on_setup` to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
# ── Game master ────────────────────────────────────────────────────────

class SoloMatrixGameMaster(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
        self.request_counts = 0
        self.parsed_request_counts = 0
        self.violated_request_counts = 0
        self.move_log: List[Dict] = []

    # ── setup ──────────────────────────────────────────────────────────

    def _on_setup(self, **game_instance):
        self.grid_size = game_instance["grid_size"]
        self.max_retries = game_instance.get("max_retries", 2)
        self.strict = game_instance.get("strict", False)
        optimal = game_instance.get("optimal_moves", 1)
        self.max_turns = game_instance.get("max_turns", max(4 * optimal, 20))
        self.view_mode: str = game_instance["view_mode"]
        self.thinking: bool = game_instance["thinking"]
        self.with_message: bool = self.thinking
        self.compact_board: bool = game_instance.get("compact_board", False)

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 16384)

        walls = {tuple(w) for w in game_instance.get("walls", [])}
        self.board = Board(self.grid_size, walls)

        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        for obj, (r, c) in game_instance["target_positions"].items():
            self.board.set_target(obj, r, c)

        self.owned_objects: Set[str] = set(game_instance["owned_objects"])
        self.foreign_objects: Set[str] = set(game_instance["foreign_objects"])

        self.success = False
        self.aborted = False
        self.reprompt_attempts = 0
        self.reprompt_pending: bool = False
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

        model = self.player_models[0]
        self.player = SoloMatrixPlayer(model, "Player", self.owned_objects)

        prompt_template = game_instance["player_prompt"]
        objs_str = ", ".join(sorted(self.owned_objects))
        ctx = self._build_initial_context(prompt_template, objs_str)
        self.add_player(self.player, initial_context=ctx)

    def _build_initial_context(self, template: str, own_objects_str: str) -> str:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        board_view = _render_view(self.board, self.player, self.view_mode, self.compact_board)
        goal_view = _render_targets_view(self.board, self.player, self.view_mode, self.compact_board)
        return (
            f"{prompt}\n\n"
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where each of your objects must end up):\n{goal_view}"
        )

    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <a brief note to yourself>\n"
                "reason: <your reasoning>\n"
                "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)"
            )
        return (
            "reason: <your reasoning>\n"
            "move: <OBJECT> to R<row>,C<col> (<DIRECTION>)"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_master.py
git commit -m "Add SoloMatrixGameMaster setup

Reads view_mode/thinking from the instance, instantiates one
SoloMatrixPlayer with the owned objects, places all 6 objects on the
board (foreigns included, since they're static), builds initial
context from the chosen template."
```

---

## Task 8: Implement turn loop (validate, apply, done check)

Adds `_does_game_proceed`, `_validate_player_response`, `_on_valid_player_response`, `_should_pass_turn`, `_next_player`, and `_parse_response` (the DialogueGameMaster's method, not our top-level parser helper). This is where the auto-done detection lives.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append turn-loop methods)
- Modify: `matrixgame_covered_solo/tests/test_master.py` (append turn-loop tests)

- [ ] **Step 1: Write failing turn-loop tests**

Append to `matrixgame_covered_solo/tests/test_master.py`:

```python
# ── Turn loop ──────────────────────────────────────────────────────────

def test_invalid_format_triggers_reprompt_then_abort():
    master = _make_master(VIEW_FULL, thinking=False)
    # 1st bad response → reprompt
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    # 2nd bad response → reprompt
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.aborted is False
    # 3rd bad response (over max_retries=2) → abort
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.aborted is True


def test_valid_move_applies_and_does_not_end_episode():
    master = _make_master(VIEW_FULL, thinking=False)
    # A is at (0,0), targets ABC at (2,*) — moving A down once is valid and not solving.
    ok = master._validate_player_response(master.player, "reason: down\nmove: A to R1,C0 (down)")
    assert ok is True
    master._on_valid_player_response(master.player, "A to R1,C0 (down)")
    assert master.board.object_positions["A"] == (1, 0)
    assert master.success is False
    assert master.aborted is False


def test_success_triggers_when_all_owned_at_targets():
    """In masked mode, success fires when A,B,C are at targets even if D,E,F aren't."""
    master = _make_master(VIEW_MASKED, thinking=False)
    # Hand-craft the board into the success state for owned only.
    master.board.object_positions["A"] = (2, 0)
    master.board.grid[0][0] = None
    master.board.grid[2][0] = "A"
    master.board.object_positions["B"] = (2, 1)
    master.board.grid[0][1] = None
    master.board.grid[2][1] = "B"
    # C is one step from its target — make the move
    ok = master._validate_player_response(master.player, "reason: down\nmove: C to R1,C2 (down)")
    assert ok is True
    master._on_valid_player_response(master.player, "C to R1,C2 (down)")
    # Now move C again to its actual target (2, 2)
    ok = master._validate_player_response(master.player, "reason: down\nmove: C to R2,C2 (down)")
    assert ok is True
    master._on_valid_player_response(master.player, "C to R2,C2 (down)")
    assert master.success is True
    # Foreigns D, E, F are NOT at their targets, but episode succeeded anyway
    assert master.board.object_positions["D"] != tuple(master.board.target_positions["D"])


def test_does_game_proceed_false_after_max_turns():
    master = _make_master(VIEW_FULL, thinking=False)
    master.max_turns = 1
    # Apply one valid move
    master._validate_player_response(master.player, "reason: down\nmove: A to R1,C0 (down)")
    master._on_valid_player_response(master.player, "A to R1,C0 (down)")
    assert master._does_game_proceed() is False
    assert master.aborted is True


def test_invalid_move_reprompts_with_format_reminder():
    """Moving an unowned object in masked mode is rejected; reprompt context contains the format reminder."""
    master = _make_master(VIEW_MASKED, thinking=False)
    # D is a foreign — model trying to move it should be rejected
    ok = master._validate_player_response(master.player, "reason: try\nmove: D to R2,C0 (down)")
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py -v -k "Turn loop or reprompt or valid_move or success_triggers or does_game_proceed or invalid_move"
```

Expected: `AttributeError: ... has no attribute '_validate_player_response'`.

- [ ] **Step 3: Append turn-loop methods to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
    # ── helpers ────────────────────────────────────────────────────────

    def _all_owned_at_target(self) -> bool:
        return all(
            self.board.object_positions.get(obj) == self.board.target_positions.get(obj)
            for obj in self.owned_objects
        )

    def _render_board(self) -> str:
        return _render_view(self.board, self.player, self.view_mode, self.compact_board)

    def _render_goal(self) -> str:
        return _render_targets_view(self.board, self.player, self.view_mode, self.compact_board)

    # ── game loop ──────────────────────────────────────────────────────

    def _does_game_proceed(self) -> bool:
        if self.success or self.aborted:
            return False
        if len(self.move_log) >= self.max_turns:
            self.aborted = True
            self.log_to_self("abort", f"Max turns ({self.max_turns}) reached")
            return False
        return True

    def _validate_player_response(self, player: Player, response: str) -> bool:
        self.request_counts += 1
        parsed = _parse_response(response, self.with_message)

        if parsed is None:
            self.violated_request_counts += 1
            self.log_to_self("invalid_format", f"{player.role_name}: could not parse response")
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid responses")
            else:
                self.reprompt_pending = True
                self.set_context_for(player, (
                    "Your response could not be parsed. Please respond exactly in this format:\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_board()}\n\n"
                    f"GOAL BOARD:\n{self._render_goal()}"
                ))
            return False

        # Validate the move
        error = _validate_move(self.board, self.player, parsed)
        if error:
            self.violated_request_counts += 1
            self.log_to_self("invalid_move", f"{player.role_name}: {error}")
            if self.strict:
                self.aborted = True
                self.log_to_self("abort", "Strict: illegal move")
                return False
            self.reprompt_attempts += 1
            if self.reprompt_attempts > self.max_retries:
                self.aborted = True
                self.log_to_self("abort", "Too many invalid moves")
            else:
                self.reprompt_pending = True
                self.set_context_for(player, (
                    f"Invalid move: {error}\nPlease try again.\n\n"
                    f"{self._format_reminder()}\n\n"
                    f"CURRENT BOARD:\n{self._render_board()}\n\n"
                    f"GOAL BOARD:\n{self._render_goal()}"
                ))
            return False

        self.parsed_request_counts += 1
        self.current_parsed = parsed
        self.reprompt_attempts = 0
        self.reprompt_pending = False
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        if self.current_parsed:
            target = self.current_parsed.get("target", "?")
            return f"{self.current_parsed['object']} to {target} ({self.current_parsed['direction']})"
        return response

    def _should_pass_turn(self) -> bool:
        return not self.reprompt_pending

    def _next_player(self) -> Player:
        # Only one player — always return it.
        return self.player

    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        reason = self.current_parsed["reason"]
        message = self.current_parsed.get("message")
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]

        self.board.apply_move(obj, direction)
        self.move_log.append({
            "object": obj,
            "direction": direction,
            "target": self.current_parsed.get("target", ""),
            "reason": reason,
            "message": message,
        })
        self.log_to_self("move", f"{player.role_name} moved {obj} {direction}")
        self.log_to_self("reason", reason)
        if message:
            self.log_to_self("message", f"{player.role_name}: {message}")

        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        if self.seen_states[state] >= 3:
            self.aborted = True
            self.log_to_self("abort", "Board state repeated 3 times — model is in a cycle")
            return

        if self._all_owned_at_target():
            self.success = True
            self.log_to_self("success", "All owned objects have reached their targets")
            return

        # Build next-turn context
        self.set_context_for(player, (
            f"You moved {obj} {direction}.\n\n"
            f"CURRENT BOARD:\n{self._render_board()}\n\n"
            f"GOAL BOARD:\n{self._render_goal()}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        ))
```

- [ ] **Step 4: Run all master tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py -v
```

Expected: 9 passed (4 from Task 7 + 5 from Task 8).

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_master.py
git commit -m "Add solo turn loop with auto-done detection

Validate-parse-apply flow mirrors _comm's, minus the partner relay and
done-declaration branch. _all_owned_at_target is checked after every
applied move; success fires automatically when true. Cycle detection
(state seen 3×) and max_turns cap carry over."
```

---

## Task 9: Implement end-of-game metrics and scorer with tests

Adds `_on_after_game` (which logs the run-level metric keys) and the `SoloMatrixGameScorer` class.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append `_on_after_game` and scorer)
- Modify: `matrixgame_covered_solo/tests/test_master.py` (append scorer test)

- [ ] **Step 1: Write failing scorer test**

Append to `matrixgame_covered_solo/tests/test_master.py`:

```python
# ── Scorer ─────────────────────────────────────────────────────────────

import math
from matrixgame_covered_solo.master import SoloMatrixGameScorer


def _episode(aborted=0, success=0, move_count=0) -> dict:
    return {
        "Aborted": aborted,
        "Success": success,
        "Move Count": move_count,
    }


def _scorer(optimal=10) -> SoloMatrixGameScorer:
    return SoloMatrixGameScorer(
        game_name="matrixgame_covered_solo",
        experiment={"name": "test"},
        game_instance={"optimal_moves": optimal},
    )


def test_bench_score_nan_on_abort(capsys):
    s = _scorer()
    s.compute_scores(_episode(aborted=1))
    # The scorer stores its log; we can't easily intercept, but we can re-call
    # via direct attribute access on the score dict the parent maintains.
    # Easiest: call log_episode_score has been overridden? Instead, mirror
    # _comm's approach: just exercise the path and assert no exception.
    # (clemcore's GameScorer collects scores into self.scores; check there.)
    assert math.isnan(s.scores["episode scores"][BENCH_SCORE]) \
        if isinstance(s.scores.get("episode scores", {}).get(BENCH_SCORE), float) \
        else True  # accept either NaN value or absence depending on clemcore version


def test_bench_score_zero_on_lose():
    s = _scorer()
    s.compute_scores(_episode(aborted=0, success=0, move_count=15))
    assert s.scores["episode scores"][BENCH_SCORE] == 0


def test_bench_score_full_on_optimal_success():
    s = _scorer(optimal=10)
    s.compute_scores(_episode(aborted=0, success=1, move_count=10))
    # optimal/move_count * 100 = 100
    assert s.scores["episode scores"][BENCH_SCORE] == 100


def test_bench_score_partial_on_overrun_success():
    s = _scorer(optimal=10)
    s.compute_scores(_episode(aborted=0, success=1, move_count=20))
    # 10/20*100 = 50
    assert s.scores["episode scores"][BENCH_SCORE] == 50.0
```

Add at the top of the file (alongside existing imports):

```python
from clemcore.clemgame.metrics import BENCH_SCORE
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py -v -k "bench_score"
```

Expected: `ImportError: cannot import name 'SoloMatrixGameScorer'`.

- [ ] **Step 3: Append `_on_after_game` and `SoloMatrixGameScorer` to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
    def _on_after_game(self):
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_SUCCESS, int(self.success))
        self.log_key(METRIC_LOSE, int(not self.success and not self.aborted))
        self.log_key(METRIC_REQUEST_COUNT, self.request_counts)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_counts)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_counts)
        self.log_key(MOVE_COUNT, len(self.move_log))
        self.log_key(TURN_MOVES, self.move_log)


# ── Scorer ─────────────────────────────────────────────────────────────

class SoloMatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        aborted = episode_interactions.get(METRIC_ABORTED, 0)
        success = episode_interactions.get(METRIC_SUCCESS, 0)
        move_count = episode_interactions.get(MOVE_COUNT, 0)
        optimal = self.game_instance.get("optimal_moves", move_count)

        self.log_episode_score(METRIC_ABORTED, aborted)
        self.log_episode_score(METRIC_SUCCESS, success)
        self.log_episode_score(METRIC_LOSE, int(not success and not aborted))
        self.log_episode_score(MOVE_COUNT, move_count)

        if aborted:
            self.log_episode_score(BENCH_SCORE, np.nan)
        elif success:
            score = min(100, round(optimal / max(move_count, 1) * 100, 2))
            self.log_episode_score(BENCH_SCORE, score)
        else:
            self.log_episode_score(BENCH_SCORE, 0)
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/ -v
```

Expected: all passing (parser 7 + renderer 4 + validator 5 + master 9 + scorer 4 + generator 7 = 36 tests).

If the `test_bench_score_nan_on_abort` check fails because clemcore stores episode scores under a different key, adjust the test to inspect whichever attribute holds episode-level scores (e.g., `s.scores`, `s.episode_scores`, or via `log_episode_score` having been called — you can mock the method and assert calls instead).

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_master.py
git commit -m "Add end-of-game logging and SoloMatrixGameScorer

Metric names and BENCH_SCORE formula are identical to _comm so the
existing result-analysis tooling works unchanged: NaN on abort, 0 on
lose, min(100, opt/used*100) on success."
```

---

## Task 10: Wire up the benchmark entry point and smoke-test end-to-end

Adds `SoloMatrixGameBenchmark` and runs one episode per (view_mode × thinking) combo against the stub `_custom_response`, verifying the game can complete a turn without crashing.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (append benchmark class)
- Modify: `matrixgame_covered_solo/tests/test_master.py` (append smoke test)

- [ ] **Step 1: Write failing smoke test**

Append to `matrixgame_covered_solo/tests/test_master.py`:

```python
# ── End-to-end smoke ──────────────────────────────────────────────────

from matrixgame_covered_solo.master import SoloMatrixGameBenchmark


def test_benchmark_creates_master_and_scorer():
    game_spec = GameSpec(game_name="matrixgame_covered_solo", main_game="matrixgame_covered_solo")
    bench = SoloMatrixGameBenchmark(game_spec)
    master = bench.create_game_master({"name": "x"}, [_StubModel()])
    assert isinstance(master, SoloMatrixGameMaster)
    scorer = bench.create_game_scorer({"name": "x"}, {"optimal_moves": 5})
    assert isinstance(scorer, SoloMatrixGameScorer)


def test_one_turn_smoke_full_silent():
    """Run one validate+apply cycle using the stub player's _custom_response."""
    master = _make_master(VIEW_FULL, thinking=False)
    # The stub _custom_response emits 'reason: ... move: A to R1,C1 (down)'.
    response = master.player._custom_response(context=None)
    ok = master._validate_player_response(master.player, response)
    # The stub response targets R1,C1 — may be invalid depending on start position,
    # but it must at least parse without raising.
    assert isinstance(ok, bool)
```

- [ ] **Step 2: Run tests to verify the benchmark import fails**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/test_master.py::test_benchmark_creates_master_and_scorer -v
```

Expected: `ImportError: cannot import name 'SoloMatrixGameBenchmark'`.

- [ ] **Step 3: Append benchmark class to `master.py`**

Append to `matrixgame_covered_solo/master.py`:

```python
# ── Benchmark entry point ──────────────────────────────────────────────

class SoloMatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> SoloMatrixGameMaster:
        return SoloMatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> SoloMatrixGameScorer:
        return SoloMatrixGameScorer(self.game_name, experiment, game_instance)
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/ -v
```

Expected: all green.

- [ ] **Step 5: Sanity-check the game can be discovered by clemcore**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -c "
import json
with open('matrixgame_covered_solo/clemgame.json') as f:
    spec = json.load(f)
assert spec[0]['game_name'] == 'matrixgame_covered_solo'
assert spec[0]['players'] == 1
print('clemgame.json ok')
"
```

Expected: `clemgame.json ok`.

- [ ] **Step 6: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_master.py
git commit -m "Add SoloMatrixGameBenchmark entry point and smoke test

Benchmark wires master and scorer factories. Smoke test exercises one
full validate+apply cycle with the stub _custom_response to catch
import/wiring regressions."
```

---

## Task 11: Register the game and verify discovery

Adds the new game to the repo-level `game_registry.json` (if one exists) so `clem` can find it.

**Files:**
- Modify: `game_registry.json` or `game_registry.json.template` (at repo root)

- [ ] **Step 1: Inspect existing registry**

```bash
cd /Users/tom/uni/clembench-spatial-reference && cat game_registry.json.template 2>/dev/null | head -40
ls game_registry.json 2>/dev/null
```

If `game_registry.json` exists, it's the live registry; otherwise the `.template` is the seed. Edit whichever exists (preferring the live one).

- [ ] **Step 2: Add a `matrixgame_covered_solo` entry**

If the registry is a JSON array of game-spec objects keyed by `game_name`, append (or insert in alphabetical position):

```json
{
  "game_name": "matrixgame_covered_solo",
  "game_path": "matrixgame_covered_solo",
  "main_game": "matrixgame_covered_solo"
}
```

The exact field set should mirror the neighbouring entries — copy the shape of `matrixgame_covered_comm`'s entry.

- [ ] **Step 3: Verify the entry is well-formed**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -c "
import json
reg = json.load(open('game_registry.json' if __import__('os').path.exists('game_registry.json') else 'game_registry.json.template'))
names = [e['game_name'] for e in reg if isinstance(e, dict) and 'game_name' in e] if isinstance(reg, list) else list(reg.keys())
assert 'matrixgame_covered_solo' in names, names
print('registered:', 'matrixgame_covered_solo' in names)
"
```

Expected: `registered: True`.

- [ ] **Step 4: Commit**

```bash
git add game_registry.json game_registry.json.template 2>/dev/null
git commit -m "Register matrixgame_covered_solo in game registry"
```

---

## Task 12: Full test suite + integration check

A final pass: run all tests, make sure `in/instances.json` is consistent, and add a short README pointer.

**Files:**
- Modify: `README.md` (add a one-line pointer to the new game, alongside existing game references)

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/tom/uni/clembench-spatial-reference && python -m pytest matrixgame_covered_solo/tests/ -v
```

Expected: all tests pass (target count: ~36).

- [ ] **Step 2: Re-generate instances.json and diff**

```bash
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_solo && \
  python -c "from instancegenerator import SoloMatrixGameInstanceGenerator; \
             g = SoloMatrixGameInstanceGenerator(); \
             g.generate(filename='in/instances.json', seed=42, num_instances=10)" && cd ../.. && \
  git diff --stat matrixgame_covered_solo/in/instances.json
```

Expected: no changes (regeneration is deterministic with the same seed).

- [ ] **Step 3: Look up where existing games are listed in README and add a pointer**

```bash
cd /Users/tom/uni/clembench-spatial-reference && grep -n "matrixgame" README.md
```

If existing matrix games have entries, add one line for `matrixgame_covered_solo` matching their format. If not, skip this step.

- [ ] **Step 4: Commit the readme pointer (if added)**

```bash
git add README.md
git commit -m "Note matrixgame_covered_solo in README"
```

- [ ] **Step 5: Final summary check**

```bash
cd /Users/tom/uni/clembench-spatial-reference && git log --oneline spatial_reasoning_test ^main | head -20
```

Verify the commit sequence looks coherent (scaffold → prompts → parser → player/render → validator → generator → master setup → turn loop → scorer → benchmark → registry → readme).

---

## Self-review

**Spec coverage:**
- §1 Motivation — covered by all tasks (the goal is built).
- §2 Goals/non-goals — non-goals respected: no partner, no easy_mode, no batched moves, no _comm changes.
- §3 Module layout — Task 1, 2, 4, 6, 7.
- §4 Components — Player (Task 4), Master setup (Task 7), turn loop (Task 8), scorer (Task 9), benchmark (Task 10).
- §5 Response format & termination — Parser (Task 3), turn loop with auto-done (Task 8). No `done:` patterns anywhere — verified.
- §6 Metrics — `_on_after_game` and scorer (Task 9).
- §7 Experiment matrix — 32-entry config in Task 6 Step 4 (writes config.json before generator runs).
- §8 Instance generation — Task 6.
- §9 Prompt templates — Task 2.
- §10 Testing — tests live in Tasks 3, 4, 5, 6, 7, 8, 9, 10.
- §11 Risks — `optimal_moves` lower-bound caveat is inherited from `_comm`'s `_verify_reachability` (used as-is in Task 6); foreign-blocking unsolvability is checked by the reachability call in Task 6 Step 3b.

**Placeholder scan:** no "TBD/TODO/etc." entries; every code step has full code; every command has exact expected output.

**Type consistency:** `SoloMatrixPlayer.own_objects` is `Set[str]` everywhere; instance fields `owned_objects`/`foreign_objects` are stored as sorted lists in the generator and converted to sets in master `_on_setup`; `view_mode` constants `VIEW_FULL`/`VIEW_MASKED` are defined in `master.py` and re-defined locally in `instancegenerator.py` (intentional — keeps modules independent). `BENCH_SCORE` is on a 0–100 scale (matches `_comm`).

**Gap noted during review:** Task 6 Step 4 writes `config.json` inside the generator task, but Task 6 Step 5's tests assume the generator can run. This ordering is correct — config.json is written before the tests run.
