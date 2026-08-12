# Direction-only Move Grammar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the matrix-game move command from `move: A to R3,C5 (right)` to a coordinate-free `move: A right` across all three variants (solo, comm, comm_nplayer).

**Architecture:** The three variants each keep a private copy of the move regexes, `_parse_response`, `_declared_cell`, `_format_reminder`, `validate_move`, and `utils/board.py`; there is no shared parsing module. Every change below is therefore applied three times, once per variant. Board *generation* is shared (`matrixgame_covered_shared/boardgen.py`) and is untouched. The declared-coordinate cross-check (`coordinate_mismatch`) is removed entirely; all other semantic checks (off-grid, wall, occupied, ownership, unknown-object, bad-direction) stay.

**Tech Stack:** Python 3, `re`, pytest, clemcore `GameInstanceGenerator` / `DialogueGameMaster`.

## Global Constraints

- New move grammar (all variants, both channel-off and channel-on): `move: <OBJECT> <DIRECTION>` where `<OBJECT>` is a single `[A-Z]` and `<DIRECTION>` ∈ `{up, down, left, right}`, case-insensitive. No parentheses, no coordinate.
- Clean break: the old coordinate form (`move: A to R3,C5 (right)`) must NOT parse — it falls through to the standard reprompt path.
- PLAN clause in structured/hybrid comm KEEPS coordinates. Its regex is unchanged; only wording/examples are reframed so the second `(Rr,Cc)` is the object's **goal** cell (the plan may span several moves; the `move:` line is a single step toward it).
- No commit contains a `Co-Authored-By: Claude …` trailer (repo convention).
- Instances are regenerated from the generators (seed 42), never hand-edited.
- Metrics consume only `object`, `direction`, `to` from each move-log entry — the `target` field is informational and is being dropped.

## File Structure

Per variant `V ∈ {matrixgame_covered_solo, matrixgame_covered_comm, matrixgame_covered_comm_nplayer}`:

- `V/utils/board.py` — `validate_move` loses the `declared_target` param and the `coordinate_mismatch` branch.
- `V/master.py` — move regexes, `_parse_response` (module fn), `_declared_cell` (deleted), `validate_move` call sites, `_format_reminder`, the master's `_parse_response` log override, `_custom_response` mock, module docstring; comm/nplayer additionally: `MASKED_OBJECT_MOVE_PATTERN`, the `coordinate_mismatch` reprompt branch (deleted), the move-log `target` field (deleted), and `_STRUCTURED_REMINDER`/`_HYBRID_REMINDER` (goal reframe).
- `V/resources/initial_prompts/en/*.template` — RESPONSE-FORMAT move line, examples, and (structured/hybrid) PLAN framing.
- `V/in/instances*.json` — regenerated.
- Tests: solo `tests/{test_validator,test_parser,test_master,test_instance_generator}.py`; comm/nplayer `tests/test_smoke.py`.

**Test command (all tasks):** run pytest from the repo root against the variant's `tests/` dir, e.g. `python3 -m pytest matrixgame_covered_solo/tests/ -q`. Each variant's `conftest.py` puts its own package root on `sys.path`, so `from utils.board import Board` resolves.

---

### Task 1: Solo — remove the coordinate cross-check from `Board.validate_move`

**Files:**
- Modify: `matrixgame_covered_solo/utils/board.py:59-98`
- Test: `matrixgame_covered_solo/tests/test_validator.py`

**Interfaces:**
- Produces: `Board.validate_move(obj: str, direction: str, allowed_objects: Optional[Set[str]] = None) -> Optional[Tuple[str,str]]` — the `declared_target` parameter is removed. Codes returned: `bad_direction`, `unknown_object`, `not_owner`, `off_grid`, `wall`, `occupied`. `coordinate_mismatch` no longer exists.

- [ ] **Step 1: Update the failing tests**

In `matrixgame_covered_solo/tests/test_validator.py`: delete every test that asserts the `coordinate_mismatch` code or passes `declared_target=`. Run `grep -n "coordinate_mismatch\|declared_target" matrixgame_covered_solo/tests/test_validator.py` first to find them. Add this self-contained test that pins the new signature and the surviving codes:

```python
def test_validate_move_has_no_declared_target_param():
    import inspect
    from utils.board import Board
    sig = inspect.signature(Board.validate_move)
    assert "declared_target" not in sig.parameters

def test_valid_direction_move_ok():
    from utils.board import Board
    b = Board(8, set())
    b.place_object("A", 3, 3)
    assert b.validate_move("A", "right", allowed_objects={"A"}) is None

def test_off_grid_and_wall_still_illegal():
    from utils.board import Board
    b = Board(8, {(2, 3)})
    b.place_object("A", 0, 0)
    assert b.validate_move("A", "up")[0] == "off_grid"
    b.place_object("B", 1, 3)
    assert b.validate_move("B", "up")[0] == "wall"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest matrixgame_covered_solo/tests/test_validator.py -q`
Expected: FAIL — `test_validate_move_has_no_declared_target_param` fails (param still present), and any remaining `coordinate_mismatch` test errors.

- [ ] **Step 3: Edit `validate_move`**

In `matrixgame_covered_solo/utils/board.py`, replace the signature + docstring + coordinate branch. Old:

```python
    def validate_move(
        self,
        obj: str,
        direction: str,
        allowed_objects: Optional[Set[str]] = None,
        declared_target: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Return (code, message) if the move is invalid, else None.

        Codes 'bad_direction', 'unknown_object', 'not_owner', 'off_grid',
        'wall', 'coordinate_mismatch' are ILLEGAL moves (static rule
        violations that indict the mover alone). Code 'occupied' is a
        BLOCKED move (locally well-formed, fails only against the current
        joint state). The scorer builds the illegal-move and blocked-move
        rates from this distinction.
        """
```

New:

```python
    def validate_move(
        self,
        obj: str,
        direction: str,
        allowed_objects: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Return (code, message) if the move is invalid, else None.

        Codes 'bad_direction', 'unknown_object', 'not_owner', 'off_grid',
        'wall' are ILLEGAL moves (static rule violations that indict the
        mover alone). Code 'occupied' is a BLOCKED move (locally well-formed,
        fails only against the current joint state). The scorer builds the
        illegal-move and blocked-move rates from this distinction.
        """
```

Then delete the coordinate-mismatch branch (old lines 89-93):

```python
        if declared_target is not None and declared_target != (new_row, new_col):
            return ("coordinate_mismatch",
                    f"Cannot move {obj} {direction}: you named "
                    f"R{declared_target[0] + 1}C{declared_target[1] + 1} but that move lands on "
                    f"R{new_row + 1}C{new_col + 1}.")
```

Leave the `off_grid`, `wall`, `occupied` checks intact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest matrixgame_covered_solo/tests/test_validator.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/utils/board.py matrixgame_covered_solo/tests/test_validator.py
git commit -m "solo: drop coordinate_mismatch check from validate_move"
```

---

### Task 2: Solo — direction-only move grammar in `master.py`

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (docstring 8-13; `MOVE_ONLY_PATTERN` 52-55; `MESSAGE_MOVE_PATTERN` 61-64; `_parse_response` 77-84; `_custom_response` 98; `_declared_cell` 117-122 delete; `_validate_move` 127-137; `_format_reminder` 206-212; log override 296-300)
- Test: `matrixgame_covered_solo/tests/test_parser.py`, `matrixgame_covered_solo/tests/test_master.py`

**Interfaces:**
- Consumes: `Board.validate_move(...)` without `declared_target` (Task 1).
- Produces: `_parse_response(response, with_message)` returns `{"object", "direction"[, "message"]}` (no `"target"`). Move regex `move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)`.

- [ ] **Step 1: Update the failing tests**

In `matrixgame_covered_solo/tests/test_parser.py`: replace every fixture move string of the form `move: X to Rr,Cc (dir)` with `move: X dir`, and drop any assertion on a parsed `"target"` key. Add:

```python
def test_parses_direction_only_move():
    from master import _parse_response
    r = _parse_response("move: A right", with_message=False)
    assert r == {"object": "A", "direction": "right"}

def test_old_coordinate_form_no_longer_parses():
    from master import _parse_response
    assert _parse_response("move: A to R3,C5 (right)", with_message=False) is None

def test_thinking_message_then_direction_move():
    from master import _parse_response
    r = _parse_response("message: go\nmove: B down", with_message=True)
    assert r["object"] == "B" and r["direction"] == "down" and r["message"] == "go"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest matrixgame_covered_solo/tests/test_parser.py -q`
Expected: FAIL — new grammar not yet implemented; `_parse_response("move: A right", ...)` returns `None`.

- [ ] **Step 3: Edit the regexes and `_parse_response`**

`MOVE_ONLY_PATTERN` (52-55) →

```python
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE,
)
```

`MESSAGE_MOVE_PATTERN` (61-64) — keep the NOTE comment above it and the no-DOTALL choice →

```python
MESSAGE_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE,
)
```

In `_parse_response` (77-81), drop the `"target"` entry:

```python
    result: Dict[str, str] = {
        "object": match.group("object").strip().upper(),
        "direction": match.group("direction").strip().lower(),
    }
```

- [ ] **Step 4: Delete `_declared_cell` and update `_validate_move`**

Delete the whole `_declared_cell` function (117-122). Change `_validate_move` (132-137) to:

```python
    return board.validate_move(
        parsed["object"],
        parsed["direction"],
        allowed_objects=player.own_objects,
    )
```

- [ ] **Step 5: Update reminder, mock, log override, and docstring**

`_format_reminder` (206-212):

```python
    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <a brief note to yourself>\n"
                "move: <OBJECT> <DIRECTION>"
            )
        return "move: <OBJECT> <DIRECTION>"
```

`_custom_response` (98): `return f"move: {first} to R1,C1 (down)"` → `return f"move: {first} down"`

Master log override `_parse_response` (296-300):

```python
    def _parse_response(self, player: Player, response: str) -> str:
        if self.current_parsed:
            return f"{self.current_parsed['object']} {self.current_parsed['direction']}"
        return response
```

Module docstring (lines 9 and 13): change both `  move: OBJECT to Rr,Cc (DIRECTION)` lines to `  move: OBJECT DIRECTION`.

- [ ] **Step 6: Update `test_master.py` fixtures and run**

In `matrixgame_covered_solo/tests/test_master.py`, replace any `move: X to Rr,Cc (dir)` fixtures with `move: X dir`. Then run:

Run: `python3 -m pytest matrixgame_covered_solo/tests/test_parser.py matrixgame_covered_solo/tests/test_master.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_solo/tests/test_parser.py matrixgame_covered_solo/tests/test_master.py
git commit -m "solo: direction-only move grammar in master"
```

---

### Task 3: Solo — prompt templates + regenerate instances

**Files:**
- Modify: `matrixgame_covered_solo/resources/initial_prompts/en/player_prompt_silent.template`, `player_prompt_thinking.template`
- Regenerate: `matrixgame_covered_solo/in/instances.json`
- Test: `matrixgame_covered_solo/tests/` (full suite)

- [ ] **Step 1: Edit `player_prompt_silent.template`**

Replace the response-format line and example:

- `move: <OBJECT> to R<row>,C<col> (<DIRECTION>)` → `move: <OBJECT> <DIRECTION>`
- `move: A to R3,C5 (right)` → `move: A right`

- [ ] **Step 2: Edit `player_prompt_thinking.template`**

Same two replacements as Step 1 (identical strings appear once each).

- [ ] **Step 3: Regenerate solo instances**

Run: `python3 matrixgame_covered_solo/instancegenerator.py`
Expected output ends with `Wrote instances to instances.json`.

- [ ] **Step 4: Verify only prompt text changed**

Run: `git diff --stat matrixgame_covered_solo/in/instances.json` then `git diff matrixgame_covered_solo/in/instances.json | grep -c '"start_positions"\|"target_positions"'`
Expected: the diff touches `player_prompt` strings only; board fields (`start_positions`, `target_positions`, `walls`) are unchanged (grep count on removed/added board lines is 0 in the `-`/`+` sense — confirm visually the changed lines are all inside `player_prompt`).

- [ ] **Step 5: Run the full solo suite**

Run: `python3 -m pytest matrixgame_covered_solo/tests/ -q`
Expected: PASS (includes `test_instance_generator.py`, `test_renderer.py`).

- [ ] **Step 6: Commit**

```bash
git add matrixgame_covered_solo/resources/initial_prompts/en/ matrixgame_covered_solo/in/instances.json
git commit -m "solo: coordinate-free move in prompts; regenerate instances"
```

---

### Task 4: Comm — remove the coordinate cross-check from `Board.validate_move`

**Files:**
- Modify: `matrixgame_covered_comm/utils/board.py` (`validate_move`)
- Test: `matrixgame_covered_comm/tests/test_smoke.py`

**Interfaces:**
- Produces: same new `validate_move` signature as Task 1 (no `declared_target`).

- [ ] **Step 1: Apply the identical edit from Task 1 Step 3**

In `matrixgame_covered_comm/utils/board.py`, make the exact same two edits as Task 1 Step 3: drop the `declared_target` parameter, update the docstring to remove `'coordinate_mismatch'` from the ILLEGAL list, and delete the branch:

```python
        if declared_target is not None and declared_target != (new_row, new_col):
            return ("coordinate_mismatch",
                    f"Cannot move {obj} {direction}: you named "
                    f"R{declared_target[0] + 1}C{declared_target[1] + 1} but that move lands on "
                    f"R{new_row + 1}C{new_col + 1}.")
```

(The comm `board.py` differs from solo only in render-masking; `validate_move` is byte-identical.)

- [ ] **Step 2: Run the smoke test to confirm no import/signature break**

Run: `python3 -m pytest matrixgame_covered_comm/tests/test_smoke.py -q`
Expected: this will still FAIL until Task 5 (master still passes `declared_target=`). That is expected — proceed to Task 5 before committing, OR verify in isolation:

Run: `python3 -c "import inspect,sys; sys.path.insert(0,'matrixgame_covered_comm'); from utils.board import Board; assert 'declared_target' not in inspect.signature(Board.validate_move).parameters; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit (together with Task 5)**

Defer the commit; commit board.py + master.py together at the end of Task 5 so the tree never has a broken intermediate state.

---

### Task 5: Comm — direction-only grammar + reprompt/PLAN cleanup in `master.py`

**Files:**
- Modify: `matrixgame_covered_comm/master.py` (docstring 10,14; `MASKED_OBJECT_MOVE_PATTERN` 60-63; `MOVE_ONLY_PATTERN` 66-69; `COMM_MOVE_PATTERN` 78-81; `_STRUCTURED_REMINDER`/`_HYBRID_REMINDER` 120-127; module `_parse_response` 187; `_declared_cell` 207-212 delete; `_custom_response` 236; `_format_reminder` 410-423; `validate_move` call 527-529; `coordinate_mismatch` branch 555-565; log override 586-587; move-log `target` field 645)
- Test: `matrixgame_covered_comm/tests/test_smoke.py`

**Interfaces:**
- Consumes: `Board.validate_move(...)` without `declared_target` (Task 4).
- Produces: same parse dict shape as solo (no `"target"`).

- [ ] **Step 1: Update the smoke test fixtures**

In `matrixgame_covered_comm/tests/test_smoke.py`: replace every scripted move string `move: X to Rr,Cc (dir)` with `move: X dir` (find via `grep -n "move: .* to R"`). If the smoke test scripts structured/hybrid messages, keep the PLAN clause `PLAN: move A from (Rr,Cc) to (Rr,Cc)` intact (its grammar is unchanged) and only change the following `move:` line.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest matrixgame_covered_comm/tests/test_smoke.py -q`
Expected: FAIL (old grammar removed from fixtures; masters still expect coordinates).

- [ ] **Step 3: Edit regexes**

`MOVE_ONLY_PATTERN` (66-69):

```python
MOVE_ONLY_PATTERN = re.compile(
    r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE,
)
```

`COMM_MOVE_PATTERN` (78-81) — keep `re.DOTALL` and the explanatory comment:

```python
COMM_MOVE_PATTERN = re.compile(
    r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
    re.IGNORECASE | re.DOTALL,
)
```

`MASKED_OBJECT_MOVE_PATTERN` (60-63) — simplify to the bare form and trim the comment's `@R3C4` reference:

```python
# Matches a move line whose object is the masked marker (the partner's object).
# Used to give a targeted semantic redirect instead of a generic parse error.
MASKED_OBJECT_MOVE_PATTERN = re.compile(
    rf"move:\s*{re.escape(BLOCKED_MARKER)}\b",
    re.IGNORECASE,
)
```

- [ ] **Step 4: Edit parse dict, delete `_declared_cell`, fix validate call**

Module `_parse_response` (187): delete the line `"target": f"R{match.group('target_row')},C{match.group('target_col')}",`.

Delete the whole `_declared_cell` function (207-212).

`validate_move` call (527-529):

```python
        error = self.board.validate_move(
            obj, direction, allowed_objects=allowed
        )
```

- [ ] **Step 5: Remove the `coordinate_mismatch` reprompt branch**

Replace the guidance block (551-567) — drop the now-unused `pos` line and the `coordinate_mismatch` branch:

```python
                masked_attempt = self.use_masking and obj == BLOCKED_MARKER
                if masked_attempt and self.masking_hints:
                    guidance = self._masked_move_message(player)
                else:
                    guidance = f"Invalid move: {err_msg}\nPlease try again."
```

- [ ] **Step 6: Reminder, mock, log override, move-log, docstring, PLAN reminders**

`_format_reminder` (410-423):

```python
    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <your message to your partner>\n"
                "move: <OBJECT> <DIRECTION>\n\n"
                "OR, if ALL your objects are at their target positions:\n"
                "message: done\n"
                "done: all my objects are placed"
            )
        return (
            "move: <OBJECT> <DIRECTION>\n\n"
            "OR, if ALL your objects are at their target positions:\n"
            "done: all my objects are placed"
        )
```

`_custom_response` (236): `return f"move: {first} to R1,C1 (down)"` → `return f"move: {first} down"`

Log override (586-587): replace the `target`/parenthesized line:

```python
            return f"{self.current_parsed['object']} {self.current_parsed['direction']}"
```
(delete the preceding `target = self.current_parsed.get("target", "?")` line; keep the `is_done` → `"done"` branch above it).

Move-log entry (645): delete the line `"target": self.current_parsed.get("target", ""),`.

Docstring (10 and 14): change both `  move: OBJECT to Rr,Cc (DIRECTION)   (or: done: ...)` lines' move part to `  move: OBJECT DIRECTION   (or: done: <explanation> if all own objects are at targets)`.

`_STRUCTURED_REMINDER` / `_HYBRID_REMINDER` (120-127) — add a goal-clarifying sentence, keep the literal form (it is shown, not matched):

```python
_STRUCTURED_REMINDER = (
    'Use exactly:\n'
    'message: PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]\n'
    "The first (Rr,Cc) is the object's current cell; the second is its goal cell."
)
_HYBRID_REMINDER = (
    'Use exactly:\n'
    'message: <natural language> PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]\n'
    "The first (Rr,Cc) is the object's current cell; the second is its goal cell."
)
```

- [ ] **Step 7: Run smoke test**

Run: `python3 -m pytest matrixgame_covered_comm/tests/ -q`
Expected: PASS.

- [ ] **Step 8: Commit (board.py + master.py + test together)**

```bash
git add matrixgame_covered_comm/utils/board.py matrixgame_covered_comm/master.py matrixgame_covered_comm/tests/test_smoke.py
git commit -m "comm: direction-only move grammar; drop coordinate_mismatch; reframe PLAN goal"
```

---

### Task 6: Comm — prompt templates (move + PLAN goal reframe) + regenerate instances

**Files:**
- Modify: `matrixgame_covered_comm/resources/initial_prompts/en/{player_prompt_none,player_prompt_freeform,player_prompt_structured,player_prompt_hybrid}.template`
- Regenerate: `matrixgame_covered_comm/in/instances.json`
- Test: `matrixgame_covered_comm/tests/`

- [ ] **Step 1: Replace the move-format line in all four templates**

In each of the four templates, replace the single occurrence:
`move: <OBJECT> to R<row>,C<col> (<DIRECTION>)` → `move: <OBJECT> <DIRECTION>`

- [ ] **Step 2: Replace the example `move:` lines**

- `player_prompt_none.template`: `move: A to R3,C3 (down)` → `move: A down`
- `player_prompt_freeform.template`: `move: A to R4,C3 (up)` → `move: A up`
- `player_prompt_structured.template`: `move: A to R3,C3 (down)` → `move: A down`
- `player_prompt_hybrid.template`: `move: A to R3,C3 (down)` → `move: A down`

- [ ] **Step 3: Reframe the PLAN goal wording (structured + hybrid only)**

In `player_prompt_structured.template`, in the COMMUNICATION section, append one sentence after the line describing the PLAN format (`PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: ...`):

```
The first (Rr,Cc) is where the object is now; the second is its GOAL cell (from the GOAL BOARD). Your PLAN states where the object needs to end up — it may take several moves — while your `move:` line makes just one step toward it, and your REQUEST asks a partner for help serving that goal.
```

In `player_prompt_hybrid.template`, add the same clarifying sentence directly under item 2 of the COMMUNICATION list (the `PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc)` line). Leave the freeform and none templates' communication sections unchanged. Do NOT change any literal `PLAN: move ... from (Rr,Cc) to (Rr,Cc)` example text — the schema regex still requires that exact form.

- [ ] **Step 4: Regenerate comm instances**

Run: `python3 matrixgame_covered_comm/instancegenerator.py`
Expected: prints `Generated: ...` lines for each experiment/level/protocol condition with no error.

- [ ] **Step 5: Verify only prompt text changed**

Run: `git diff matrixgame_covered_comm/in/instances.json | grep -E '^\+|^-' | grep -v player_prompt | grep -vE '^\+\+\+|^---'`
Expected: no board-field lines (`start_positions`/`target_positions`/`walls`) appear — every changed line is inside a `player_prompt` string. (Changed `move:`/PLAN text lives inside `player_prompt`.)

- [ ] **Step 6: Run the comm suite**

Run: `python3 -m pytest matrixgame_covered_comm/tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_comm/resources/initial_prompts/en/ matrixgame_covered_comm/in/instances.json
git commit -m "comm: coordinate-free move + goal-framed PLAN in prompts; regenerate instances"
```

---

### Task 7: Nplayer — remove the coordinate cross-check from `Board.validate_move`

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/utils/board.py` (`validate_move`)

- [ ] **Step 1: Apply the identical edit from Task 1 Step 3**

Same two edits as Task 1 Step 3 (drop `declared_target` param + docstring mention, delete the `coordinate_mismatch` branch). `validate_move` is byte-identical to comm/solo.

- [ ] **Step 2: Verify signature in isolation**

Run: `python3 -c "import inspect,sys; sys.path.insert(0,'matrixgame_covered_comm_nplayer'); from utils.board import Board; assert 'declared_target' not in inspect.signature(Board.validate_move).parameters; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Defer commit to Task 8** (avoid a broken intermediate tree).

---

### Task 8: Nplayer — direction-only grammar + reprompt/PLAN cleanup in `master.py`

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`MASKED_OBJECT_MOVE_PATTERN` 55-58; `MOVE_ONLY_PATTERN` 65; `COMM_MOVE_PATTERN` 77; `_STRUCTURED_REMINDER`/`_HYBRID_REMINDER` ~120-127; module `_parse_response` 185; `_declared_cell` 205-... delete; `_custom_response` 234; `_format_reminder` 387-400; `validate_move` call 504-506; `coordinate_mismatch` branch 528-544; log override 563-564; move-log `target` field 623)
- Test: `matrixgame_covered_comm_nplayer/tests/test_smoke.py`

**Interfaces:**
- Consumes: `Board.validate_move(...)` without `declared_target` (Task 7).

- [ ] **Step 1: Update the smoke-test fixtures**

In `matrixgame_covered_comm_nplayer/tests/test_smoke.py`: replace every `move: X to Rr,Cc (dir)` with `move: X dir` (find via `grep -n "move: .* to R"`). Keep any PLAN clauses intact.

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -q`
Expected: FAIL.

- [ ] **Step 3: Edit regexes** (identical replacements to Task 5 Step 3)

`MOVE_ONLY_PATTERN` (65):
```python
    r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
```
`COMM_MOVE_PATTERN` (77):
```python
    r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)",
```
`MASKED_OBJECT_MOVE_PATTERN` (55-58): simplify to `rf"move:\s*{re.escape(BLOCKED_MARKER)}\b"` and trim the `@R3C4` comment reference (comm/nplayer say "teammate" — keep that wording).

- [ ] **Step 4: Parse dict, delete `_declared_cell`, fix validate call** (identical to Task 5 Step 4)

- Module `_parse_response` (185): delete the `"target": f"R{...},C{...}",` line.
- Delete `_declared_cell` (205-...).
- `validate_move` call (504-506): `error = self.board.validate_move(obj, direction, allowed_objects=allowed)`.

- [ ] **Step 5: Remove the `coordinate_mismatch` reprompt branch** (identical to Task 5 Step 5)

Replace the guidance block (528-544) with:
```python
                masked_attempt = self.use_masking and obj == BLOCKED_MARKER
                if masked_attempt and self.masking_hints:
                    guidance = self._masked_move_message(player)
                else:
                    guidance = f"Invalid move: {err_msg}\nPlease try again."
```

- [ ] **Step 6: Reminder, mock, log override, move-log, PLAN reminders**

`_format_reminder` (387-400): same as Task 5 Step 6 but keep the "teammates" wording:
```python
    def _format_reminder(self) -> str:
        if self.with_message:
            return (
                "message: <your message to your teammates>\n"
                "move: <OBJECT> <DIRECTION>\n\n"
                "OR, if ALL your objects are at their target positions:\n"
                "message: done\n"
                "done: all my objects are placed"
            )
        return (
            "move: <OBJECT> <DIRECTION>\n\n"
            "OR, if ALL your objects are at their target positions:\n"
            "done: all my objects are placed"
        )
```

`_custom_response` (234): `return f"move: {first} down"`

Log override (563-564): delete the `target = ...` line; replace the return with `return f"{self.current_parsed['object']} {self.current_parsed['direction']}"` (keep the `is_done` branch).

Move-log entry (623): delete the `"target": self.current_parsed.get("target", ""),` line.

`_STRUCTURED_REMINDER` / `_HYBRID_REMINDER`: same goal-clarifying sentence as Task 5 Step 6.

(The nplayer module docstring has no `move: OBJECT to Rr,Cc` turn-structure line to change, but its `structured`/`hybrid` protocol summary lines mention `PLAN: move X from (Rr,Cc) to (Rr,Cc)` — leave those, the PLAN form is unchanged.)

- [ ] **Step 7: Run the nplayer suite**

Run: `python3 -m pytest matrixgame_covered_comm_nplayer/tests/ -q`
Expected: PASS (includes `test_partition.py`).

- [ ] **Step 8: Commit (board.py + master.py + test together)**

```bash
git add matrixgame_covered_comm_nplayer/utils/board.py matrixgame_covered_comm_nplayer/master.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "nplayer: direction-only move grammar; drop coordinate_mismatch; reframe PLAN goal"
```

---

### Task 9: Nplayer — prompt templates + regenerate instances

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/{player_prompt_none,player_prompt_freeform,player_prompt_structured,player_prompt_hybrid}.template`
- Regenerate: `matrixgame_covered_comm_nplayer/in/instances.json`
- Test: `matrixgame_covered_comm_nplayer/tests/`

- [ ] **Step 1: Replace the move-format line in all four templates**

`move: <OBJECT> to R<row>,C<col> (<DIRECTION>)` → `move: <OBJECT> <DIRECTION>` (one occurrence each: none:29, freeform:33, structured:36, hybrid:37).

- [ ] **Step 2: Replace the example `move:` lines**

- `player_prompt_none.template:32`: `move: A to R3,C3 (down)` → `move: A down`
- `player_prompt_freeform.template:37`: `move: A to R4,C3 (up)` → `move: A up`
- `player_prompt_structured.template:40`: `move: A to R3,C3 (down)` → `move: A down`
- `player_prompt_hybrid.template:41`: `move: A to R3,C3 (down)` → `move: A down`

- [ ] **Step 3: Reframe the PLAN goal wording (structured + hybrid only)**

Add the same clarifying sentence as Task 6 Step 3 (teammates wording), after the PLAN-format line in the COMMUNICATION section of `player_prompt_structured.template` (near line 28) and under item 2 of `player_prompt_hybrid.template` (near line 30):

```
The first (Rr,Cc) is where the object is now; the second is its GOAL cell (from the GOAL BOARD). Your PLAN states where the object needs to end up — it may take several moves — while your `move:` line makes just one step toward it, and your REQUEST asks a teammate for help serving that goal.
```

Leave the literal `PLAN: move ... from (Rr,Cc) to (Rr,Cc)` example strings unchanged.

- [ ] **Step 4: Regenerate nplayer instances**

Run: `python3 matrixgame_covered_comm_nplayer/instancegenerator.py`
Expected: no error; instances written.

- [ ] **Step 5: Verify only prompt text changed**

Run: `git diff matrixgame_covered_comm_nplayer/in/instances.json | grep -E '^\+|^-' | grep -v player_prompt | grep -vE '^\+\+\+|^---'`
Expected: no board-field lines appear.

- [ ] **Step 6: Run the nplayer suite**

Run: `python3 -m pytest matrixgame_covered_comm_nplayer/tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_comm_nplayer/resources/initial_prompts/en/ matrixgame_covered_comm_nplayer/in/instances.json
git commit -m "nplayer: coordinate-free move + goal-framed PLAN in prompts; regenerate instances"
```

---

### Task 10: Full-repo verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Confirm no lingering old grammar or dead code**

Run:
```bash
grep -rn "coordinate_mismatch\|declared_target\|_declared_cell" matrixgame_covered_solo matrixgame_covered_comm matrixgame_covered_comm_nplayer
grep -rn "to R<row>,C<col>\|to Rr,Cc (DIRECTION)\| to R[0-9]" matrixgame_covered_*/master.py matrixgame_covered_*/resources
```
Expected: first grep returns nothing; second returns nothing in `master.py`/templates (matches inside PLAN `from (Rr,Cc) to (Rr,Cc)` are acceptable and expected — inspect any hit to confirm it is a PLAN clause, not a `move:` line).

- [ ] **Step 2: Run every test suite**

Run: `python3 -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ -q`
Expected: all PASS.

- [ ] **Step 3: Drive one live episode per variant (smoke)**

Use each variant's mock player path (already exercised by the smoke tests) or run a single instance through the master with the mock model to confirm a full episode reaches a terminal state with the new grammar. If the repo has a `clem run`-style entry, run one instance of solo + one comm condition + one nplayer condition and confirm the transcript shows `move: <LETTER> <dir>` parsed moves and no `invalid_format` loops.

- [ ] **Step 4: Final status**

Run: `git log --oneline -9 | cat` and confirm the nine implementation commits are present and the working tree is clean (`git status`).
```

