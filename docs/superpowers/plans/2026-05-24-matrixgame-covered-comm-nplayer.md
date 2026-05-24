# matrixgame_covered_comm_nplayer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new clembench game directory `matrixgame_covered_comm_nplayer/` that hosts two game entries — `matrixgame_covered_comm_3p` (3 players × 2 objects each) and `matrixgame_covered_comm_6p` (6 players × 1 object each) — sharing one master and one instance generator. Compact rendering is the only mode. All four comm protocols are supported with broadcast messaging. 32 variants total (S1–S4 × {none, structured, freeform, hybrid} × {3p, 6p}).

**Architecture:** Copy-and-adapt from `matrixgame_covered_comm/`. The spatial-level generator functions (`_generate_s1`–`_generate_s4`) and `utils/board.py` / `utils/astar.py` are reused verbatim. The instance generator discards comm/'s 2-player split and re-partitions the 6 objects into N groups via index. The master generalizes `player_a` / `player_b` aliases into indexed lists (`self.player_objects: List[Set[str]]`, `self.player_done: List[bool]`, `self.last_message: Dict[int, Optional[str]]`) and replaces the "deliver to partner" code path with a broadcast loop over every other not-yet-done player.

**Tech Stack:** Python 3.10+, `clemcore.clemgame` (`GameInstanceGenerator`, `DialogueGameMaster`, `Player`, `GameBenchmark`, `GameScorer`), pytest. No new dependencies beyond what `matrixgame_covered_comm/` already uses.

**Spec:** `docs/superpowers/specs/2026-05-24-matrixgame-covered-comm-nplayer-design.md`

**Source of truth to copy/adapt from:** `matrixgame_covered_comm/`

---

## File Structure

To be created:

```
matrixgame_covered_comm_nplayer/
├── __init__.py                              # empty
├── clemgame.json                            # two entries: _3p and _6p
├── instancegenerator.py                     # copy comm's helpers + S1-S4 verbatim; rewrite on_generate
├── master.py                                # copy from comm; surgical N-player edits
├── resources/
│   ├── common_config.json                   # same as comm
│   ├── config.json                          # 32 variants
│   └── initial_prompts/en/
│       ├── player_prompt_none.template
│       ├── player_prompt_structured.template
│       ├── player_prompt_freeform.template
│       └── player_prompt_hybrid.template
├── utils/
│   ├── __init__.py                          # empty
│   ├── astar.py                             # cp from comm/utils/astar.py
│   └── board.py                             # cp from comm/utils/board.py
├── tests/
│   ├── __init__.py                          # empty
│   ├── conftest.py                          # sys.path shim, mirrors solo/tests/conftest.py
│   └── test_smoke.py                        # instance generator + master setup smoke tests
└── in/                                       # generated; not committed in repo skeleton
```

To be modified:

```
game_registry.json                            # add two new entries
```

---

## Task 1: Scaffold directory + copy `utils/` verbatim from `comm/`

**Files:**
- Create: `matrixgame_covered_comm_nplayer/__init__.py` (empty)
- Create: `matrixgame_covered_comm_nplayer/utils/__init__.py` (empty)
- Create: `matrixgame_covered_comm_nplayer/utils/board.py` (verbatim copy)
- Create: `matrixgame_covered_comm_nplayer/utils/astar.py` (verbatim copy)
- Create: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/` (directory)
- Create: `matrixgame_covered_comm_nplayer/tests/__init__.py` (empty)
- Create: `matrixgame_covered_comm_nplayer/tests/conftest.py` (sys.path shim)
- Create: `matrixgame_covered_comm_nplayer/in/.gitkeep` (so directory exists)

- [ ] **Step 1: Create the directory tree**

Run from the repo root (`/Users/tom/uni/clembench-spatial-reference`):

```bash
mkdir -p matrixgame_covered_comm_nplayer/{utils,tests,resources/initial_prompts/en,in}
touch matrixgame_covered_comm_nplayer/__init__.py
touch matrixgame_covered_comm_nplayer/utils/__init__.py
touch matrixgame_covered_comm_nplayer/tests/__init__.py
touch matrixgame_covered_comm_nplayer/in/.gitkeep
```

- [ ] **Step 2: Copy `board.py` and `astar.py` verbatim**

```bash
cp matrixgame_covered_comm/utils/board.py matrixgame_covered_comm_nplayer/utils/board.py
cp matrixgame_covered_comm/utils/astar.py matrixgame_covered_comm_nplayer/utils/astar.py
```

Do not edit these files. They are reused as-is.

- [ ] **Step 3: Write `tests/conftest.py`**

Create `matrixgame_covered_comm_nplayer/tests/conftest.py`:

```python
"""pytest configuration for matrixgame_covered_comm_nplayer tests.

Adds the game package root to sys.path so that 'from utils.board import ...'
resolves to matrixgame_covered_comm_nplayer/utils/board.py without requiring
PYTHONPATH on the command line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

- [ ] **Step 4: Smoke-test the copy**

Create `matrixgame_covered_comm_nplayer/tests/test_smoke.py` with this initial content (will grow in later tasks):

```python
"""Smoke tests for matrixgame_covered_comm_nplayer."""


def test_board_module_imports():
    from utils.board import Board, DIRECTIONS
    assert DIRECTIONS  # non-empty dict/list
    board = Board(grid_size=8, walls=set())
    assert board is not None


def test_astar_module_imports():
    from utils import astar
    assert astar is not None
```

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/
git commit -m "Scaffold matrixgame_covered_comm_nplayer with verbatim utils

Sets up the directory skeleton and copies Board / A* helpers
unchanged from matrixgame_covered_comm. Adds conftest.py and a
smoke test verifying both utility modules import cleanly."
```

---

## Task 2: `common_config.json`

**Files:**
- Create: `matrixgame_covered_comm_nplayer/resources/common_config.json`
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
import json
from pathlib import Path

PKG = Path(__file__).parent.parent


def test_common_config_shape():
    with open(PKG / "resources" / "common_config.json") as f:
        cfg = json.load(f)
    assert cfg["grid_size"] == 8
    assert cfg["objects"] == ["A", "B", "C", "D", "E", "F"]
    assert cfg["directions"] == ["up", "down", "left", "right"]
    assert cfg["max_retries"] == 2
    assert cfg["strict"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py::test_common_config_shape -v`
Expected: FAIL with FileNotFoundError on `common_config.json`.

- [ ] **Step 3: Write `common_config.json`**

Create `matrixgame_covered_comm_nplayer/resources/common_config.json`:

```json
{
    "grid_size": 8,
    "objects": ["A", "B", "C", "D", "E", "F"],
    "directions": ["up", "down", "left", "right"],
    "max_retries": 2,
    "strict": false
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py::test_common_config_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/resources/common_config.json matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Add common_config.json for matrixgame_covered_comm_nplayer"
```

---

## Task 3: `config.json` with 32 variants

**Files:**
- Create: `matrixgame_covered_comm_nplayer/resources/config.json`
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_config_has_32_variants():
    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    assert len(cfg) == 32, f"expected 32 variants, got {len(cfg)}"


def test_every_variant_has_required_fields():
    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    for key, variant in cfg.items():
        for field in ("name", "num_players", "spatial_level", "comm_protocol", "use_masking", "compact_board"):
            assert field in variant, f"variant {key} missing {field}"
        assert variant["num_players"] in (3, 6), f"{key}: num_players must be 3 or 6"
        assert variant["spatial_level"] in ("S1", "S2", "S3", "S4"), f"{key}: invalid spatial_level"
        assert variant["comm_protocol"] in ("none", "structured", "freeform", "hybrid"), f"{key}: invalid comm_protocol"
        assert variant["use_masking"] is True, f"{key}: use_masking must be true"
        assert variant["compact_board"] is True, f"{key}: compact_board must be true"


def test_variants_cover_full_matrix():
    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    seen = set()
    for variant in cfg.values():
        seen.add((variant["spatial_level"], variant["comm_protocol"], variant["num_players"]))
    expected = {
        (s, p, n)
        for s in ("S1", "S2", "S3", "S4")
        for p in ("none", "structured", "freeform", "hybrid")
        for n in (3, 6)
    }
    assert seen == expected, f"missing combinations: {expected - seen}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k variant`
Expected: 3 FAIL with FileNotFoundError on `config.json`.

- [ ] **Step 3: Write `config.json` with all 32 variants**

Create `matrixgame_covered_comm_nplayer/resources/config.json`:

```json
{
  "s1_none_3p": {"name": "S1_None_3p", "num_players": 3, "spatial_level": "S1", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s1_structured_3p": {"name": "S1_Structured_3p", "num_players": 3, "spatial_level": "S1", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s1_freeform_3p": {"name": "S1_Freeform_3p", "num_players": 3, "spatial_level": "S1", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s1_hybrid_3p": {"name": "S1_Hybrid_3p", "num_players": 3, "spatial_level": "S1", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s2_none_3p": {"name": "S2_None_3p", "num_players": 3, "spatial_level": "S2", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s2_structured_3p": {"name": "S2_Structured_3p", "num_players": 3, "spatial_level": "S2", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s2_freeform_3p": {"name": "S2_Freeform_3p", "num_players": 3, "spatial_level": "S2", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s2_hybrid_3p": {"name": "S2_Hybrid_3p", "num_players": 3, "spatial_level": "S2", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s3_none_3p": {"name": "S3_None_3p", "num_players": 3, "spatial_level": "S3", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s3_structured_3p": {"name": "S3_Structured_3p", "num_players": 3, "spatial_level": "S3", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s3_freeform_3p": {"name": "S3_Freeform_3p", "num_players": 3, "spatial_level": "S3", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s3_hybrid_3p": {"name": "S3_Hybrid_3p", "num_players": 3, "spatial_level": "S3", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s4_none_3p": {"name": "S4_None_3p", "num_players": 3, "spatial_level": "S4", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s4_structured_3p": {"name": "S4_Structured_3p", "num_players": 3, "spatial_level": "S4", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s4_freeform_3p": {"name": "S4_Freeform_3p", "num_players": 3, "spatial_level": "S4", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s4_hybrid_3p": {"name": "S4_Hybrid_3p", "num_players": 3, "spatial_level": "S4", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s1_none_6p": {"name": "S1_None_6p", "num_players": 6, "spatial_level": "S1", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s1_structured_6p": {"name": "S1_Structured_6p", "num_players": 6, "spatial_level": "S1", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s1_freeform_6p": {"name": "S1_Freeform_6p", "num_players": 6, "spatial_level": "S1", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s1_hybrid_6p": {"name": "S1_Hybrid_6p", "num_players": 6, "spatial_level": "S1", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s2_none_6p": {"name": "S2_None_6p", "num_players": 6, "spatial_level": "S2", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s2_structured_6p": {"name": "S2_Structured_6p", "num_players": 6, "spatial_level": "S2", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s2_freeform_6p": {"name": "S2_Freeform_6p", "num_players": 6, "spatial_level": "S2", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s2_hybrid_6p": {"name": "S2_Hybrid_6p", "num_players": 6, "spatial_level": "S2", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s3_none_6p": {"name": "S3_None_6p", "num_players": 6, "spatial_level": "S3", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s3_structured_6p": {"name": "S3_Structured_6p", "num_players": 6, "spatial_level": "S3", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s3_freeform_6p": {"name": "S3_Freeform_6p", "num_players": 6, "spatial_level": "S3", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s3_hybrid_6p": {"name": "S3_Hybrid_6p", "num_players": 6, "spatial_level": "S3", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true},

  "s4_none_6p": {"name": "S4_None_6p", "num_players": 6, "spatial_level": "S4", "comm_protocol": "none", "use_masking": true, "compact_board": true},
  "s4_structured_6p": {"name": "S4_Structured_6p", "num_players": 6, "spatial_level": "S4", "comm_protocol": "structured", "use_masking": true, "compact_board": true},
  "s4_freeform_6p": {"name": "S4_Freeform_6p", "num_players": 6, "spatial_level": "S4", "comm_protocol": "freeform", "use_masking": true, "compact_board": true},
  "s4_hybrid_6p": {"name": "S4_Hybrid_6p", "num_players": 6, "spatial_level": "S4", "comm_protocol": "hybrid", "use_masking": true, "compact_board": true}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k variant`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/resources/config.json matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Add config.json with 32 variants (S1-S4 x 4 protocols x {3p,6p})"
```

---

## Task 4: Four prompt templates

**Files:**
- Create: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_none.template`
- Create: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_structured.template`
- Create: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_freeform.template`
- Create: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_hybrid.template`
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
import pytest


@pytest.mark.parametrize("protocol", ["none", "structured", "freeform", "hybrid"])
def test_prompt_template_exists(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    assert path.exists(), f"missing template for {protocol}"
    text = path.read_text()
    assert "$YOUR_OBJECTS$" in text
    assert "$GRID_SIZE$" in text
    assert "$YOUR_ROLE$" in text
    assert "$N_OTHER_PLAYERS$" in text


@pytest.mark.parametrize("protocol", ["structured", "freeform", "hybrid"])
def test_message_protocols_mention_broadcast(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    text = path.read_text().lower()
    assert "broadcast" in text or "teammates" in text, \
        f"{protocol} template should mention teammates/broadcast"


def test_none_template_has_no_message_line():
    path = PKG / "resources" / "initial_prompts" / "en" / "player_prompt_none.template"
    text = path.read_text()
    assert "message:" not in text.lower(), "none template must not include 'message:'"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k template`
Expected: all FAIL with missing-file errors.

- [ ] **Step 3: Create `player_prompt_freeform.template`**

Copy the file from `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_freeform.template` to the new location, then apply these edits:

1. Add a line near the top: `You are $YOUR_ROLE$.`
2. Replace `"with a partner"` → `"with your teammates"`.
3. Replace `"Your partner controls the remaining objects (shown as X on your board)."` → `"Your $N_OTHER_PLAYERS$ teammates control the remaining objects (shown as X on your board)."`.
4. Replace `"ask your partner to clear the way in your message"` → `"ask your teammates to clear the way in your message"`.
5. In the `COMMUNICATION:` section, replace the first sentence with: `"Before every move you must send your teammates a short broadcast message (keep it under 100 words). It will be shown to every teammate on their next turn. Use it to share your plan AND to reflect on your teammates' situations: where do you think their objects are heading? Are you about to create an obstacle for them, or free up space they need?"`
6. In the example, replace `"your partner"` → `"your teammates"`.

The final file should look like:

```
You are $YOUR_ROLE$.

You are playing a collaborative grid rearrangement game with your teammates.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).

YOUR TASK:
Move all objects to their target positions. The GOAL BOARD shows where each object must end up.
You control ONLY these objects: $YOUR_OBJECTS$. You may ONLY move your own objects.
Your $N_OTHER_PLAYERS$ teammates control the remaining objects (shown as X on your board).

RULES:
- On each turn, move exactly ONE object exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##) or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.
- You may only move objects you control.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object, find an alternative route, or ask your teammates to clear the way in your message.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach: reroute via a completely different path, temporarily move another of your objects to open space, or explicitly ask your teammates to yield.
- "I have no good move" is never a reason to oscillate. A deliberate detour — even several steps out of the way — is always better than bouncing back and forth until the game aborts.

COMMUNICATION:
Before every move you must send your teammates a short broadcast message (keep it under 100 words). It will be shown to every teammate on their next turn. Use it to share your plan AND to reflect on your teammates' situations: where do you think their objects are heading? Are you about to create an obstacle for them, or free up space they need? Acknowledge what they might need from you, even when you can't always provide it right now.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `message:` on the very first line — do not output any preamble, thinking, or scratchpad text before it.
message: <your message to your teammates>
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
message: I'm moving A down toward row 4. Could someone get an object out of R3,C3 if possible?
reason: A is at R3,C3 and needs to reach its target at R4,C3. Moving it down is the first step.
move: A to R4,C3 (down)

Think carefully before each move. Before choosing your move, reason about both yourself and your teammates: what do you need, and what do they likely need? The best move often serves multiple goals — and when it can't, be honest about that in your message.
```

- [ ] **Step 4: Create `player_prompt_hybrid.template`**

Copy `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_hybrid.template` to the new location and apply the same six edits as for `freeform` (Step 3). The hybrid template differs from freeform mainly in its system-prompt mention of a shared coordinate system; the partner→teammates edits are identical.

- [ ] **Step 5: Create `player_prompt_structured.template`**

Copy `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_structured.template` to the new location and apply the same six edits. The structured template additionally specifies the `PLAN: ... REQUEST: ...` message format; that format is unchanged. The REQUEST is interpreted as a broadcast request — add a sentence in the message section: `"Your REQUEST will be read by all $N_OTHER_PLAYERS$ teammates."`

- [ ] **Step 6: Create `player_prompt_none.template`**

Copy `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_none.template` to the new location and apply edits 1–4 from Step 3 (no message line in this template, so skip the COMMUNICATION / message edits). The final file is:

```
You are $YOUR_ROLE$.

You are playing a collaborative grid rearrangement game with your teammates.

THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- Cells marked ## are walls — objects cannot enter them.
- Empty cells are shown as a dot (.).

YOUR TASK:
Move all objects to their target positions. The GOAL BOARD shows where each object must end up.
You control ONLY these objects: $YOUR_OBJECTS$. You may ONLY move your own objects.
Your $N_OTHER_PLAYERS$ teammates control the remaining objects (shown as X on your board).

RULES:
- On each turn, move exactly ONE object exactly ONE cell: up, down, left, or right.
- Objects CANNOT move into wall cells (##) or outside the grid boundaries.
- Two objects CANNOT occupy the same cell.
- You may only move objects you control.

AVOIDING DEADLOCKS:
- If an object's path is blocked, do NOT keep attempting the same move. Switch to a different object or find an alternative route.
- If you notice you are moving an object back to a cell it just left, you are cycling. Stop immediately and commit to a different approach.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
reason: <your reasoning about what move to make and why>
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
reason: A is at R3,C3 and needs to reach its target at R4,C3. Moving it down is the first step.
move: A to R4,C3 (down)
```

Note: even though the `none` template doesn't include `teammates` in the COMMUNICATION section (it has no comm section at all), the YOUR TASK section already mentions teammates, so the `test_message_protocols_mention_broadcast` only applies to structured/freeform/hybrid.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k template`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add matrixgame_covered_comm_nplayer/resources/initial_prompts/ matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Add 4 prompt templates adapted for N-player broadcast comm"
```

---

## Task 5: `instancegenerator.py` — copy spatial helpers + S1–S4 from `comm/`

**Files:**
- Create: `matrixgame_covered_comm_nplayer/instancegenerator.py` (initial copy)

This task copies the full file from `comm/` and renames the class. The new `on_generate` body is written in Task 6.

- [ ] **Step 1: Copy `comm/instancegenerator.py` verbatim**

```bash
cp matrixgame_covered_comm/instancegenerator.py matrixgame_covered_comm_nplayer/instancegenerator.py
```

- [ ] **Step 2: Replace the module docstring**

Open `matrixgame_covered_comm_nplayer/instancegenerator.py` and replace the top-of-file docstring (lines 1–27) with:

```python
"""Instance generator for matrixgame_covered_comm_nplayer.

Two modes:
  3p — 3 players, each owning 2 of 6 objects (A,B / C,D / E,F).
  6p — 6 players, each owning 1 of 6 objects.

Spatial levels (S1–S4), wall layouts, optimal-path computation, and chain
construction are reused verbatim from matrixgame_covered_comm. Only the
player partition step differs: the 2-player split returned by the spatial
generators is discarded and we re-partition by index based on `num_players`.

The compact board representation is the only supported rendering mode.
"""
```

- [ ] **Step 3: Rename the class**

In the same file, change:

```python
class MatrixGameInstanceGenerator(GameInstanceGenerator):
```

to:

```python
class MatrixGameNPlayerInstanceGenerator(GameInstanceGenerator):
```

- [ ] **Step 4: Empty out `on_generate` (to be rewritten in Task 6)**

Locate the `on_generate` method (starts around line 44 in the copied file). Replace its entire body with:

```python
    def on_generate(self, seed: int, **kwargs):
        raise NotImplementedError("on_generate is rewritten in Task 6")
```

Leave every other function in the file (`_bfs_path`, `_pick`, `_partition_regions`, `_split_range`, `_carve_for_ratio`, `_is_2connected`, `_new_wall_creates_dead_end`, `_verify_reachability`, `_generate_s1`, `_generate_s2`, `_generate_s3`, `_generate_s4`, and any S2/S3/S4-local helpers) untouched.

- [ ] **Step 5: Update the `if __name__ == "__main__"` block**

Locate it (near the bottom of the file). Replace its body with a placeholder that will be filled in in Task 6:

```python
if __name__ == "__main__":
    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename="instances.json", seed=42)
```

- [ ] **Step 6: Smoke-test the import**

Add a new test in `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_instance_generator_module_imports():
    from instancegenerator import MatrixGameNPlayerInstanceGenerator
    gen = MatrixGameNPlayerInstanceGenerator()
    assert gen is not None
```

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py::test_instance_generator_module_imports -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_comm_nplayer/instancegenerator.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Copy spatial-level generators from comm/ (on_generate stubbed)

Re-uses _generate_s1..s4 and their helpers verbatim. on_generate
is left as NotImplementedError; the N-player partition and instance
schema are written in the next task."
```

---

## Task 6: `instancegenerator.py` — `on_generate` with N-way partition

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/instancegenerator.py` (`on_generate` + `__main__`)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("nplayer_in")
    import sys
    sys.path.insert(0, str(PKG))
    from instancegenerator import MatrixGameNPlayerInstanceGenerator

    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_32_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 32, f"expected 32 experiments, got {len(experiments)}"


def test_instance_schema(generated_instances):
    required = (
        "grid_size", "walls", "objects",
        "num_players", "player_objects",
        "start_positions", "target_positions",
        "optimal_moves", "max_turns", "max_retries", "strict",
        "use_masking", "spatial_level", "comm_protocol",
        "compact_board", "player_prompt",
    )
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in required:
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_player_partition_3p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_3p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 3
            po = inst["player_objects"]
            assert len(po) == 3, f"3p should have 3 groups, got {len(po)}"
            assert all(len(g) == 2 for g in po), f"3p: every group must own 2 objects, got {po}"
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            # Static-by-index partition
            assert po == [["A", "B"], ["C", "D"], ["E", "F"]]


def test_player_partition_6p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_6p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 6
            po = inst["player_objects"]
            assert len(po) == 6
            assert all(len(g) == 1 for g in po)
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            assert po == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]


def test_compact_board_always_true(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["compact_board"] is True


def test_experiment_naming(generated_instances):
    names = {exp["name"] for exp in generated_instances["experiments"]}
    # 32 unique names of form "S<level>_<protocol>_<n>p"
    assert len(names) == 32
    for name in names:
        parts = name.split("_")
        assert len(parts) == 3, f"unexpected experiment name shape: {name}"
        level, protocol, mode = parts
        assert level in ("S1", "S2", "S3", "S4")
        assert protocol.lower() in ("none", "structured", "freeform", "hybrid")
        assert mode in ("3p", "6p")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k "partition or schema or experiment or compact_board_always"`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Write the real `on_generate`**

In `matrixgame_covered_comm_nplayer/instancegenerator.py`, replace the stubbed `on_generate` body with this implementation:

```python
    def on_generate(self, seed: int, **kwargs):
        all_variant_configs = self.load_json("resources/config")
        common_config = self.load_json("resources/common_config")

        grid_size = common_config["grid_size"]
        max_retries = common_config["max_retries"]
        strict = common_config.get("strict", False)
        all_objects = common_config["objects"]

        num_instances = kwargs.get("num_instances", 10)

        generators = {
            "S1": _generate_s1,
            "S2": _generate_s2,
            "S3": _generate_s3,
            "S4": _generate_s4,
        }

        rng = random.Random(seed)

        for variant_key, variant_config in all_variant_configs.items():
            num_players = variant_config["num_players"]
            spatial_level = variant_config["spatial_level"]
            comm_protocol = variant_config["comm_protocol"]
            use_masking = variant_config.get("use_masking", True)
            compact_board = variant_config.get("compact_board", True)

            objects = all_objects[:]                       # always 6 objects
            assert len(objects) % num_players == 0, (
                f"{variant_key}: {len(objects)} objects not divisible by {num_players} players"
            )
            objs_per_player = len(objects) // num_players
            player_objects = [
                objects[i * objs_per_player:(i + 1) * objs_per_player]
                for i in range(num_players)
            ]

            prompt_template = self.load_template(
                f"resources/initial_prompts/en/player_prompt_{comm_protocol}"
            )

            experiment_name = f"{spatial_level}_{comm_protocol}_{num_players}p"
            experiment = self.add_experiment(experiment_name)
            experiment["common_config"] = common_config
            experiment["variant_config"] = variant_config

            gen_fn = generators[spatial_level]

            for idx in range(num_instances):
                game_instance = self.add_game_instance(experiment, idx + 1)

                starts = targets = walls = None
                optimal = 0
                for _outer in range(40):
                    result = gen_fn(grid_size, objects, rng)
                    starts, targets, walls, optimal, _pa, _pb = result
                    if spatial_level != "S1" and len(walls) == 0:
                        continue
                    break

                game_instance["grid_size"] = grid_size
                game_instance["walls"] = walls
                game_instance["objects"] = objects
                game_instance["num_players"] = num_players
                game_instance["player_objects"] = player_objects
                game_instance["start_positions"] = starts
                game_instance["target_positions"] = targets
                game_instance["optimal_moves"] = optimal
                game_instance["max_turns"] = max(4 * optimal, 20)
                game_instance["max_retries"] = max_retries
                game_instance["strict"] = strict
                game_instance["use_masking"] = use_masking
                game_instance["spatial_level"] = spatial_level
                game_instance["comm_protocol"] = comm_protocol
                game_instance["compact_board"] = compact_board
                game_instance["player_prompt"] = prompt_template
```

Also confirm these imports are present near the top of the file (they should already be, copied from comm/):

```python
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from clemcore.clemgame import GameInstanceGenerator

from utils.board import Board
```

- [ ] **Step 4: Update the `__main__` block**

Replace the bottom-of-file block with:

```python
if __name__ == "__main__":
    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename="instances.json", seed=42, num_instances=10)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v`
Expected: all PASS (existing + new tests).

- [ ] **Step 6: Commit**

```bash
git add matrixgame_covered_comm_nplayer/instancegenerator.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Implement on_generate with N-way player partition

Iterates over all variants in config.json. For each variant, partitions
the six objects into num_players equal-size groups by index, then runs
the spatial-level generator and writes instance fields. Discards
comm's 2-player split."
```

---

## Task 7: `master.py` — wholesale copy + parser regex / `MatrixPlayer` rename check

**Files:**
- Create: `matrixgame_covered_comm_nplayer/master.py` (initial copy from comm/)

This task copies the entire file, renames the docstring and the class for clarity, and leaves all 2-player logic in place — to be surgically edited in subsequent tasks.

- [ ] **Step 1: Copy `comm/master.py` verbatim**

```bash
cp matrixgame_covered_comm/master.py matrixgame_covered_comm_nplayer/master.py
```

- [ ] **Step 2: Replace the module docstring**

Open `matrixgame_covered_comm_nplayer/master.py` and replace the top-of-file docstring (lines 1–20) with:

```python
"""Matrix Game (covered comm, N-player): collaborative grid rearrangement with broadcast communication.

Supports both 3-player and 6-player modes (selected via num_players in the
instance JSON). Each turn one player moves a single object and may emit
one broadcast message that is shown to every other not-yet-done player on
their next turn.

Communication protocols: none, structured, freeform, hybrid.
Same response formats as matrixgame_covered_comm. The "partner" wording is
replaced by "teammates" throughout.
"""
```

- [ ] **Step 3: Rename the class**

Find:

```python
class MatrixGameMaster(DialogueGameMaster):
```

Change to:

```python
class NPlayerMatrixGameMaster(DialogueGameMaster):
```

Also rename `MatrixGameScorer` to `NPlayerMatrixGameScorer` and `MatrixGameBenchmark` to `NPlayerMatrixGameBenchmark`. Update the two references in `NPlayerMatrixGameBenchmark.create_game_master` and `create_game_scorer` accordingly.

- [ ] **Step 4: Smoke-test the import**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_master_module_imports():
    from master import (
        NPlayerMatrixGameMaster,
        NPlayerMatrixGameScorer,
        NPlayerMatrixGameBenchmark,
        MatrixPlayer,
        _parse_response,
    )
    assert NPlayerMatrixGameMaster is not None
    assert MatrixPlayer is not None


def test_parse_response_move_only():
    from master import _parse_response
    text = "reason: heading down\nmove: A to R3,C5 (down)"
    parsed = _parse_response(text, with_message=False)
    assert parsed["object"] == "A"
    assert parsed["direction"] == "down"
    assert parsed["target"] == "R3,C5"


def test_parse_response_with_message():
    from master import _parse_response
    text = "message: moving A\nreason: heading down\nmove: A to R3,C5 (down)"
    parsed = _parse_response(text, with_message=True)
    assert parsed["message"] == "moving A"
    assert parsed["object"] == "A"
```

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k master`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Copy master.py from comm/ with renamed classes (no behavior change yet)

The 2-player code path is intact; N-player generalization happens in
subsequent commits."
```

---

## Task 8: `master.py` — generalize `_on_setup` to N players

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`_on_setup` and surrounding state)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def _make_game_instance(num_players=3, comm_protocol="freeform"):
    """Build a minimal game_instance dict suitable for setup tests."""
    objs = ["A", "B", "C", "D", "E", "F"]
    objs_per = len(objs) // num_players
    player_objects = [objs[i*objs_per:(i+1)*objs_per] for i in range(num_players)]
    return {
        "grid_size": 8,
        "walls": [],
        "objects": objs,
        "num_players": num_players,
        "player_objects": player_objects,
        "start_positions": {o: [0, i] for i, o in enumerate(objs)},
        "target_positions": {o: [7, i] for i, o in enumerate(objs)},
        "optimal_moves": 42,
        "max_turns": 100,
        "max_retries": 2,
        "strict": False,
        "use_masking": True,
        "spatial_level": "S1",
        "comm_protocol": comm_protocol,
        "compact_board": True,
        "player_prompt": (
            "You are $YOUR_ROLE$. You control $YOUR_OBJECTS$ on a $GRID_SIZE$x$GRID_SIZE$ "
            "grid with $N_OTHER_PLAYERS$ teammates."
        ),
    }


class _DummyModel:
    """Stand-in for clemcore.backends.Model with the methods setup touches."""
    def __init__(self, name="dummy"):
        self.name = name
        self._gen_args = {}

    def set_gen_arg(self, key, value):
        self._gen_args[key] = value

    def generate_response(self, *_args, **_kwargs):
        return "reason: stub\nmove: A to R1,C1 (down)"


@pytest.mark.parametrize("num_players", [3, 6])
def test_master_setup_builds_n_players(num_players):
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=num_players)
    models = [_DummyModel(f"m{i}") for i in range(num_players)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=models,
    )
    master._on_setup(**instance)
    assert master.num_players == num_players
    assert len(master.player_objects) == num_players
    assert len(master.player_done) == num_players
    assert all(d is False for d in master.player_done)
    assert len(master.last_message) == num_players
    assert all(v is None for v in master.last_message.values())


def test_master_setup_single_model_self_play():
    """If only one model is supplied for N>1, every player shares it."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    one_model = _DummyModel("solo")
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=[one_model],
    )
    master._on_setup(**instance)
    # All MatrixPlayer.model fields should point at the same object
    players = master.get_players()
    assert len(players) == 6
    assert all(p.model is one_model for p in players)


def test_master_setup_initial_context_substitutions():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3)
    models = [_DummyModel(f"m{i}") for i in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # In 3p mode, $N_OTHER_PLAYERS$ should equal 2
    for p in players:
        ctx = master.get_context_for(p)
        assert "$YOUR_ROLE$" not in ctx, "role placeholder not substituted"
        assert "$N_OTHER_PLAYERS$" not in ctx, "n_other placeholder not substituted"
        assert "$YOUR_OBJECTS$" not in ctx
        assert "$GRID_SIZE$" not in ctx
        # And the substituted role/objects should appear
        assert p.role_name in ctx
```

Note on `get_context_for`: this is provided by `DialogueGameMaster`. If it's not available, the equivalent is reading `self._contexts[player]` — adjust the test if your version of clemcore exposes this differently. Inspect `clemcore.clemgame.legacy.master.DialogueGameMaster` if needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k "master_setup or self_play or substitution"`
Expected: FAIL — current `_on_setup` reads `player_a_objects` / `player_b_objects` which no longer exist in the new instance schema.

- [ ] **Step 3: Replace `_on_setup` and add a `_player_index` helper**

In `matrixgame_covered_comm_nplayer/master.py`, locate the `_on_setup` method (around line 139–191 in the copied file). Replace its full body with:

```python
    def _on_setup(self, **game_instance):
        self.grid_size = game_instance["grid_size"]
        self.max_retries = game_instance.get("max_retries", 2)
        self.strict = game_instance.get("strict", False)
        optimal = game_instance.get("optimal_moves", 1)
        self.max_turns = game_instance.get("max_turns", max(50, 5 * optimal))
        self.use_masking: bool = game_instance.get("use_masking", True)
        self.comm_protocol: str = game_instance.get("comm_protocol", COMM_NONE)
        self.with_message: bool = self.comm_protocol != COMM_NONE

        self.num_players: int = game_instance["num_players"]
        self.player_objects: List[Set[str]] = [
            set(group) for group in game_instance["player_objects"]
        ]
        self.player_done: List[bool] = [False] * self.num_players
        self.last_message: Dict[int, Optional[str]] = {i: None for i in range(self.num_players)}

        for model in self.player_models:
            model.set_gen_arg("max_tokens", 16384)

        walls = {tuple(w) for w in game_instance.get("walls", [])}
        self.board = Board(self.grid_size, walls)

        for obj, (r, c) in game_instance["start_positions"].items():
            self.board.place_object(obj, r, c)
        for obj, (r, c) in game_instance["target_positions"].items():
            self.board.set_target(obj, r, c)

        self.success = False
        self.aborted = False
        self.reprompt_attempts = 0
        self.reprompt_pending: bool = False
        self.current_parsed: Optional[Dict[str, str]] = None
        self.seen_states: Dict[str, int] = {self.board.state_key(): 1}

        prompt_template = game_instance["player_prompt"]
        self._prompt_template = prompt_template

        # Build all N players upfront with their initial contexts.
        self._player_to_index: Dict["MatrixPlayer", int] = {}
        n_other = self.num_players - 1
        for i in range(self.num_players):
            role = f"Player {chr(ord('A') + i)}"
            model = (
                self.player_models[i]
                if i < len(self.player_models)
                else self.player_models[0]
            )
            player = MatrixPlayer(model, role, self.player_objects[i])
            self._player_to_index[player] = i
            own_objs_str = ", ".join(sorted(self.player_objects[i]))
            ctx = self._build_initial_context(prompt_template, own_objs_str, player, role, n_other)
            self.add_player(player, initial_context=ctx)
```

- [ ] **Step 4: Update `_build_initial_context` signature**

Locate `_build_initial_context` (around line 193). Replace it with:

```python
    def _build_initial_context(
        self,
        template: str,
        own_objects_str: str,
        player: Player,
        role_name: str,
        n_other_players: int,
    ) -> str:
        prompt = template.replace("$YOUR_OBJECTS$", own_objects_str)
        prompt = prompt.replace("$GRID_SIZE$", str(self.grid_size))
        prompt = prompt.replace("$YOUR_ROLE$", role_name)
        prompt = prompt.replace("$N_OTHER_PLAYERS$", str(n_other_players))
        if "DONE OPTION" not in prompt:
            prompt += self._done_instructions_text()
        board_view = self._render_for(player)
        goal_view = self._render_targets_for(player)
        return (
            f"{prompt}\n\n"
            f"CURRENT BOARD:\n{board_view}\n\n"
            f"GOAL BOARD (where each object must end up):\n{goal_view}"
        )
```

Note that the `_easy_mode_context` suffix is removed (no easy mode in this game).

- [ ] **Step 5: Replace `_objects_for` to use the player-index map**

Locate `_objects_for` (around line 231):

```python
    def _objects_for(self, player: Player) -> Set[str]:
        return self.player_objects[self._player_to_index[player]]
```

- [ ] **Step 6: Delete easy-mode helpers**

Delete `_render_object_status`, `_render_adjacency`, and `_easy_mode_context` (around lines 250–298). These are unused in this game.

Also delete the lines in `_on_setup` that read `easy_mode` (already done in Step 3, but double-check).

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k "master_setup or self_play or substitution"`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Generalize master._on_setup to N players with upfront contexts

Replaces player_a/player_b state with indexed lists. Builds all N
initial contexts upfront with the new \$YOUR_ROLE\$ and
\$N_OTHER_PLAYERS\$ placeholders. Removes easy-mode helpers."
```

---

## Task 9: `master.py` — rendering helpers (compact-only)

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`_render_for`, `_render_targets_for`, `_done_instructions_text`, `_format_reminder`)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_render_for_uses_compact():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    rendered = master._render_for(players[0])
    # Compact rendering produces a single-line-per-row format; just check it's a string
    assert isinstance(rendered, str)
    assert len(rendered) > 0


def test_render_for_masks_other_players_objects():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A (index 0) owns only "A". B-F's objects should be rendered as X.
    rendered_a = master._render_for(players[0])
    assert "B" not in rendered_a, "Player A should not see B"
    assert "X" in rendered_a, "other players' objects should be masked as X"
```

- [ ] **Step 2: Run tests to verify they fail or pass spuriously**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k render`

The existing `_render_for` body branches on `self.compact_board`. After Task 8 we no longer set `self.compact_board`, so the test may fail with AttributeError. Expected: FAIL.

- [ ] **Step 3: Simplify `_render_for` and `_render_targets_for` to always compact**

Locate `_render_for` and `_render_targets_for` (around line 234–248). Replace both with:

```python
    def _render_for(self, player: Player) -> str:
        visible = self._objects_for(player) if self.use_masking else None
        return self.board.render_compact(visible_objects=visible)

    def _render_targets_for(self, player: Player) -> str:
        visible = self._objects_for(player) if self.use_masking else None
        return self.board.render_targets_compact(visible_objects=visible)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k render`
Expected: PASS.

- [ ] **Step 5: Keep `_format_reminder` and `_done_instructions_text` as-is**

Verify these two methods are unchanged from the comm/ copy (around lines 207–227 and 310–327). They reference `self.with_message` only, which is still set. No change needed.

- [ ] **Step 6: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Simplify renderers to compact-only (drop compact_board branching)"
```

---

## Task 10: `master.py` — validation logic (small surgical edit)

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`_validate_player_response`)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

The `_validate_player_response` from comm/ already uses `self._objects_for(player)` to get the player's owned objects, and that helper is now generic. The only edits needed are: remove `_easy_mode_context(...)` calls in the reprompt strings.

- [ ] **Step 1: Write the failing test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_validate_response_accepts_valid_move():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A starts at R0,C0 with no walls; can move down to R1,C0
    resp = "reason: heading down\nmove: A to R2,C1 (down)"
    assert master._validate_player_response(players[0], resp)


def test_validate_response_rejects_unowned_object():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A tries to move B (owned by Player B)
    resp = "reason: trying B\nmove: B to R2,C2 (down)"
    assert not master._validate_player_response(players[0], resp)


def test_validate_response_rejects_invalid_format():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="none")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    assert not master._validate_player_response(players[0], "no move here")
```

- [ ] **Step 2: Run tests to verify behavior**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k validate`
Expected: some FAIL because the existing reprompt code in `_validate_player_response` calls `self._easy_mode_context(...)`, which we deleted in Task 8.

- [ ] **Step 3: Remove easy-mode calls from `_validate_player_response`**

In `matrixgame_covered_comm_nplayer/master.py`, locate `_validate_player_response` (around line 340–420). Find the three reprompt-context strings and remove the `f"{self._easy_mode_context(player)}"` suffix from each. The cleaned reprompt strings should look like:

```python
self.set_context_for(player, (
    "Your response could not be parsed. Please respond exactly in this format:\n"
    f"{self._format_reminder()}\n\n"
    f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
    f"GOAL BOARD:\n{self._render_targets_for(player)}"
))
```

And similarly for the "invalid move" reprompt.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k validate`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Drop easy-mode context from validation reprompts"
```

---

## Task 11: `master.py` — game loop and broadcast comm

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`_next_player`, `_on_valid_player_response`, helpers)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

This is the largest single edit. We replace the 2-player "deliver to the other player" code with a broadcast loop.

- [ ] **Step 1: Write the failing tests**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_next_player_skips_done():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Mark players B (index 1) and C (index 2) done
    master.player_done[1] = True
    master.player_done[2] = True
    # Current player is at index 0 → next should skip 1, 2 and land on 3
    master._current_player_idx = 0
    nxt = master._next_player()
    assert nxt is players[3]


def test_broadcast_after_valid_move_updates_other_contexts():
    """After Player A makes a valid move with a message, all other active
    players should see Player A's message in their next context."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="freeform")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()

    # Player A produces a valid response with a message
    resp = (
        "message: hello teammates\n"
        "reason: heading down\n"
        "move: A to R2,C1 (down)"
    )
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A to R2,C1 (down)")

    # last_message[0] should now hold the message
    assert master.last_message[0] == "hello teammates"

    # Players B and C should have new contexts mentioning the message and the move
    for idx in (1, 2):
        ctx = master.get_context_for(players[idx])
        assert "Player A says: hello teammates" in ctx
        assert "Player A moved A down" in ctx or "Player A moved a blocked object down" in ctx


def test_broadcast_with_masking_hides_unseen_object():
    """In 6p mode (every object owned by a different player), Player B should
    see Player A's move as 'a blocked object' because B can't see A."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    resp = "reason: heading down\nmove: A to R2,C1 (down)"
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A to R2,C1 (down)")

    # Player B (index 1) should see masked move
    ctx_b = master.get_context_for(players[1])
    assert "Player A moved a blocked object down" in ctx_b
    assert "Player A moved A down" not in ctx_b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k "next_player or broadcast"`
Expected: FAIL — the current `_on_valid_player_response` looks at `self.player_a` and `self.player_b` which no longer exist.

- [ ] **Step 3: Replace `_is_done_player` with index-based version**

Locate `_is_done_player` (around line 300). Replace with:

```python
    def _is_done_player(self, player: Player) -> bool:
        return self.player_done[self._player_to_index[player]]
```

- [ ] **Step 4: Replace `_own_objects_at_target` to use indexed objects**

Locate `_own_objects_at_target` (around line 304). It already uses `self._objects_for(player)` — verify and leave unchanged.

- [ ] **Step 5: Update `_next_player` (minor)**

Locate `_next_player` (around line 433). It already uses `self.get_players()` and `self._is_done_player`. After the changes in Step 3, it works for N players. No edits needed.

- [ ] **Step 6: Replace `_on_valid_player_response` with broadcast version**

Locate `_on_valid_player_response` (around line 445–545). Replace its entire body with:

```python
    def _on_valid_player_response(self, player: Player, parsed_response: str) -> None:
        reason = self.current_parsed["reason"]
        message = self.current_parsed.get("message")
        player_idx = self._player_to_index[player]

        # Cache this player's most recent message for broadcast.
        # For protocols without a message, this stays None.
        self.last_message[player_idx] = message if (message and self.with_message) else None

        if self.current_parsed.get("is_done"):
            self.player_done[player_idx] = True
            self.log_to_self("player_done", f"{player.role_name} declared done")
            if message:
                self.log_to_self("message", f"{player.role_name}: {message}")

            if self.board.is_solved():
                self.success = True
                self.log_to_self("success", "All objects have reached their target positions!")
                return

            # Broadcast the "X has finished" notice to every other active player.
            for other_idx, other in enumerate(self.get_players()):
                if other_idx == player_idx or self.player_done[other_idx]:
                    continue
                self.set_context_for(other, self._build_broadcast_context(
                    other,
                    move_desc=f"{player.role_name} has finished — all their objects are at their target positions. Continue without them.",
                ))
            return

        # A regular move.
        obj = self.current_parsed["object"]
        direction = self.current_parsed["direction"]
        self.board.apply_move(obj, direction)
        self.move_log.append({
            "player": player.role_name,
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

        # Cycle detection (3 identical board states → abort).
        state = self.board.state_key()
        self.seen_states[state] = self.seen_states.get(state, 0) + 1
        if self.seen_states[state] >= 3:
            self.aborted = True
            self.log_to_self("abort", "Board state repeated 3 times — players stuck in cycle")
            return

        if self.board.is_solved():
            self.success = True
            self.log_to_self("success", "All objects have reached their target positions!")
            return

        # Broadcast the move (with optional message) to every other active player.
        for other_idx, other in enumerate(self.get_players()):
            if other_idx == player_idx or self.player_done[other_idx]:
                continue
            # Mask the moved object's identity if the recipient doesn't own it.
            if self.use_masking and obj not in self._objects_for(other):
                move_desc = f"{player.role_name} moved a blocked object {direction}."
            else:
                move_desc = f"{player.role_name} moved {obj} {direction}."
            self.set_context_for(other, self._build_broadcast_context(other, move_desc))
```

- [ ] **Step 7: Add `_build_broadcast_context` helper**

Immediately below `_on_valid_player_response`, add:

```python
    def _build_broadcast_context(self, recipient: Player, move_desc: str) -> str:
        """Build the next-turn context for a recipient: all other players' last
        messages, the just-happened move description, current and goal boards,
        and the response format reminder."""
        recipient_idx = self._player_to_index[recipient]
        message_lines: List[str] = []
        for other_idx, other in enumerate(self.get_players()):
            if other_idx == recipient_idx:
                continue
            msg = self.last_message[other_idx]
            if msg:
                message_lines.append(f"{other.role_name} says: {msg}")
        comm_block = ("\n".join(message_lines) + "\n\n") if message_lines else ""

        return (
            f"{comm_block}"
            f"{move_desc}\n\n"
            f"CURRENT BOARD:\n{self._render_for(recipient)}\n\n"
            f"GOAL BOARD:\n{self._render_targets_for(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        )
```

- [ ] **Step 8: Verify `_should_pass_turn` and `_does_game_proceed` are unchanged**

These two methods (around lines 331–338 and 430–431 in the copy) use only `self.success`, `self.aborted`, `len(self.move_log)`, `self.max_turns`, and `self.reprompt_pending`. They work for N players without modification. No edits needed.

- [ ] **Step 9: Update `_on_after_game` log keys**

Locate `_on_after_game` (around line 547). Replace the lines that log `"Player A Done"` and `"Player B Done"` with a generic per-player loop:

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
        for i, done in enumerate(self.player_done):
            role = f"Player {chr(ord('A') + i)}"
            self.log_key(f"{role} Done", int(done))
```

- [ ] **Step 10: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v`
Expected: all PASS (existing + new broadcast tests).

- [ ] **Step 11: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Replace 2-player turn delivery with broadcast loop

After each valid move (or done declaration), every other not-yet-done
player receives a context containing all other players' cached last
messages plus the move description. Masking hides the moved object's
identity from players who do not own it. Per-author message cache;
recipients do not consume cached messages."
```

---

## Task 12: `master.py` — Scorer and Benchmark entry

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (scorer/benchmark — already renamed in Task 7; just verify)
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append tests)

- [ ] **Step 1: Verify class signatures are correct**

In `matrixgame_covered_comm_nplayer/master.py`, scroll to the bottom and confirm:

```python
class NPlayerMatrixGameScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        ...

class NPlayerMatrixGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> NPlayerMatrixGameMaster:
        return NPlayerMatrixGameMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> NPlayerMatrixGameScorer:
        return NPlayerMatrixGameScorer(self.game_name, experiment, game_instance)
```

If any class body still references the old names or the comm/-specific `MatrixGameMaster`, fix it now.

- [ ] **Step 2: Append a scorer sanity test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
import numpy as np
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE


def test_scorer_success_score_capped():
    from master import NPlayerMatrixGameScorer, MOVE_COUNT
    scorer = NPlayerMatrixGameScorer(
        game_name="matrixgame_covered_comm_6p",
        experiment={},
        game_instance={"optimal_moves": 20},
    )
    captured = {}
    scorer.log_episode_score = lambda k, v: captured.__setitem__(k, v)
    scorer.compute_scores({
        METRIC_ABORTED: 0,
        METRIC_SUCCESS: 1,
        MOVE_COUNT: 30,
    })
    # 20 / 30 * 100 = 66.67
    assert captured[BENCH_SCORE] == pytest.approx(66.67, rel=0.01)


def test_scorer_abort_returns_nan():
    from master import NPlayerMatrixGameScorer, MOVE_COUNT
    scorer = NPlayerMatrixGameScorer(
        game_name="matrixgame_covered_comm_6p",
        experiment={},
        game_instance={"optimal_moves": 20},
    )
    captured = {}
    scorer.log_episode_score = lambda k, v: captured.__setitem__(k, v)
    scorer.compute_scores({
        METRIC_ABORTED: 1,
        METRIC_SUCCESS: 0,
        MOVE_COUNT: 5,
    })
    assert np.isnan(captured[BENCH_SCORE])
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k scorer`
Expected: 2 PASS.

- [ ] **Step 4: Commit**

```bash
git add matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Verify scorer / benchmark entry points compute correct BENCH_SCORE"
```

---

## Task 13: `clemgame.json` with both entries

**Files:**
- Create: `matrixgame_covered_comm_nplayer/clemgame.json`
- Modify: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_clemgame_json_has_both_entries():
    with open(PKG / "clemgame.json") as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    assert len(entries) == 2
    names = {e["game_name"] for e in entries}
    assert names == {"matrixgame_covered_comm_3p", "matrixgame_covered_comm_6p"}
    by_name = {e["game_name"]: e for e in entries}
    assert by_name["matrixgame_covered_comm_3p"]["players"] == 3
    assert by_name["matrixgame_covered_comm_6p"]["players"] == 6
    assert by_name["matrixgame_covered_comm_3p"]["roles"] == ["Player A", "Player B", "Player C"]
    assert by_name["matrixgame_covered_comm_6p"]["roles"] == [
        "Player A", "Player B", "Player C", "Player D", "Player E", "Player F"
    ]
    for e in entries:
        assert e["main_game"] == "matrixgame_covered_comm_nplayer"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k clemgame_json`
Expected: FAIL with FileNotFoundError.

- [ ] **Step 3: Write `clemgame.json`**

Create `matrixgame_covered_comm_nplayer/clemgame.json`:

```json
[
  {
    "game_name": "matrixgame_covered_comm_3p",
    "description": "3-player collaborative 8x8 grid rearrangement; each player owns 2 of 6 objects. Compact board, broadcast communication across four protocols (none, structured, free-form, hybrid).",
    "main_game": "matrixgame_covered_comm_nplayer",
    "players": 3,
    "image": "none",
    "languages": ["en"],
    "benchmark": ["2.0"],
    "regression": "large",
    "roles": ["Player A", "Player B", "Player C"]
  },
  {
    "game_name": "matrixgame_covered_comm_6p",
    "description": "6-player collaborative 8x8 grid rearrangement; each player owns 1 of 6 objects. Compact board, broadcast communication across four protocols (none, structured, free-form, hybrid).",
    "main_game": "matrixgame_covered_comm_nplayer",
    "players": 6,
    "image": "none",
    "languages": ["en"],
    "benchmark": ["2.0"],
    "regression": "large",
    "roles": ["Player A", "Player B", "Player C", "Player D", "Player E", "Player F"]
  }
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k clemgame_json`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm_nplayer/clemgame.json matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "Add clemgame.json with 3p and 6p entries"
```

---

## Task 14: Register in top-level `game_registry.json`

**Files:**
- Modify: `/Users/tom/uni/clembench-spatial-reference/game_registry.json` (add two entries)

- [ ] **Step 1: Open the registry**

Open `/Users/tom/uni/clembench-spatial-reference/game_registry.json`. Find the existing entry for `matrixgame_covered_comm` (search for `"game_name": "matrixgame_covered_comm"`).

- [ ] **Step 2: Add the two new entries**

Immediately after the existing `matrixgame_covered_comm` entry's closing brace (or anywhere within the top-level array — the position is cosmetic), insert two new entries. They are identical to the `clemgame.json` entries but with the additional `game_path` key:

```json
,
  {
    "game_name": "matrixgame_covered_comm_3p",
    "game_path": "matrixgame_covered_comm_nplayer",
    "description": "3-player collaborative 8x8 grid rearrangement; each player owns 2 of 6 objects. Compact board, broadcast communication across four protocols (none, structured, free-form, hybrid).",
    "main_game": "matrixgame_covered_comm_nplayer",
    "players": 3,
    "image": "none",
    "languages": ["en"],
    "benchmark": ["2.0"],
    "regression": "large",
    "roles": ["Player A", "Player B", "Player C"]
  },
  {
    "game_name": "matrixgame_covered_comm_6p",
    "game_path": "matrixgame_covered_comm_nplayer",
    "description": "6-player collaborative 8x8 grid rearrangement; each player owns 1 of 6 objects. Compact board, broadcast communication across four protocols (none, structured, free-form, hybrid).",
    "main_game": "matrixgame_covered_comm_nplayer",
    "players": 6,
    "image": "none",
    "languages": ["en"],
    "benchmark": ["2.0"],
    "regression": "large",
    "roles": ["Player A", "Player B", "Player C", "Player D", "Player E", "Player F"]
  }
```

Pay attention to the comma placement so the resulting file is still valid JSON.

- [ ] **Step 3: Validate the JSON**

Run: `python -c "import json; json.load(open('game_registry.json'))"`
Expected: no output, no error.

- [ ] **Step 4: Verify both entries are present**

Run: `python -c "import json; r=json.load(open('game_registry.json')); names=[e['game_name'] for e in r]; assert 'matrixgame_covered_comm_3p' in names; assert 'matrixgame_covered_comm_6p' in names; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add game_registry.json
git commit -m "Register matrixgame_covered_comm_3p and _6p in game registry"
```

---

## Task 15: End-to-end generation + final smoke run

**Files:**
- Create: `matrixgame_covered_comm_nplayer/in/instances.json` (generated artifact)

- [ ] **Step 1: Generate instances**

```bash
cd matrixgame_covered_comm_nplayer && python instancegenerator.py && cd ..
```

Expected: a new file `matrixgame_covered_comm_nplayer/in/instances.json` exists.

- [ ] **Step 2: Sanity-check the artifact**

```bash
python -c "
import json
with open('matrixgame_covered_comm_nplayer/in/instances.json') as f:
    data = json.load(f)
exps = data['experiments']
assert len(exps) == 32, f'expected 32 experiments, got {len(exps)}'
for exp in exps:
    assert exp['game_instances'], f'no instances in {exp[\"name\"]}'
print(f'OK: 32 experiments, total instances = {sum(len(e[\"game_instances\"]) for e in exps)}')
"
```

Expected: `OK: 32 experiments, total instances = 320`.

- [ ] **Step 3: Run the full smoke test suite**

Run: `pytest matrixgame_covered_comm_nplayer/tests/ -v`
Expected: all PASS.

- [ ] **Step 4: Verify the game is discoverable by clemcore (if possible without running an LLM)**

Run (from repo root):

```bash
python -c "
import json
with open('game_registry.json') as f:
    reg = json.load(f)
matches = [e for e in reg if e['game_name'].startswith('matrixgame_covered_comm_')]
print('Registered:', [e['game_name'] for e in matches])
"
```

Expected output includes `matrixgame_covered_comm_3p` and `matrixgame_covered_comm_6p` alongside the existing `matrixgame_covered_comm`.

- [ ] **Step 5: Commit the generated instances**

```bash
git add matrixgame_covered_comm_nplayer/in/instances.json
git commit -m "Generate instances.json for matrixgame_covered_comm_nplayer

32 experiments, 10 instances per experiment (320 total)."
```

---

## Self-Review Notes

After all 15 tasks are complete, the spec's coverage:

- ✓ Directory layout: Task 1 (scaffolding), Tasks 5–13 (file creation/copy)
- ✓ `clemgame.json` with two entries: Task 13
- ✓ `common_config.json`: Task 2
- ✓ 32 variants in `config.json`: Task 3
- ✓ Spatial-level generators reused verbatim: Task 5
- ✓ N-way player partition: Task 6
- ✓ Instance schema (with `num_players`, `player_objects`): Task 6
- ✓ Master `_on_setup` for N players with upfront contexts: Task 8
- ✓ Per-player rendering, compact-only: Task 9
- ✓ Turn rotation skipping done players: Task 11 (verified)
- ✓ Broadcast comm with per-author message cache: Task 11
- ✓ Done declaration handling and broadcast notice: Task 11
- ✓ Validation, reprompting, cycle detection, max_turns: Task 10 (only easy-mode removal needed; rest is inherited)
- ✓ Scorer and benchmark entry: Task 12
- ✓ Four prompt templates with new placeholders: Task 4
- ✓ Registry update: Task 14
- ✓ Smoke test covering schema validation and master setup: Tasks 2–15 (incremental)
- ✓ End-to-end instance generation: Task 15

Edge cases:
- All-done-but-unsolved → lose: handled by the default fall-through in `_does_game_proceed` when no more turns can be taken; verified implicitly by `success = False` and `aborted = False` at game end.
- Single-model self-play: Task 8 falls back to `self.player_models[0]` when fewer models are supplied than players.
- Done players are skipped in the broadcast loop: Task 11 (`if other_idx == player_idx or self.player_done[other_idx]: continue`).

No placeholders, no TBDs, no "similar to Task N" references. Every code snippet contains actual code.
