# Open-Findings Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three matrixgame packages match the thesis on outcome scoring (reachable LOSE), coordinate validation, masking scaffolding, prompt-template truthfulness, and repo hygiene — per `docs/superpowers/specs/2026-08-05-open-findings-design.md`.

**Architecture:** Small, parallel edits repeated across the three standalone packages (`matrixgame_covered_solo`, `matrixgame_covered_comm`, `matrixgame_covered_comm_nplayer`). Each master + its duplicated `utils/board.py` + prompt templates change independently; instances are regenerated at the end because prompt text is baked into `instances.json`. TDD throughout, one commit per task.

**Tech Stack:** Python 3, stdlib (`re`, `random`, `collections`) + `numpy`, `pytest` with each package's `tests/conftest.py` sys.path pattern, clemcore/clemgame framework (unchanged).

## Global Constraints

- Commit messages: short imperative style matching repo history; **never add a `Co-Authored-By: Claude` trailer** (repo rule).
- No new third-party dependencies; stdlib + numpy only.
- `master.py` files must NOT import across packages at runtime.
- `utils/board.py` is duplicated per package by design — edits to it must be applied identically to all three copies.
- Base seed **42**, **8 instances per experiment** at regeneration.
- Existing recorded results become incomparable after these changes (grammar feedback + prompts change). Do not mix old and new result directories.
- Run package tests from the package's `tests/` dir (each `conftest.py` inserts the package root into `sys.path`).

---

### Task 1: Reachable LOSE outcome (spec Change-area A, Items 1 & 2)

Add a real `lost` state so a timeout is scored LOSE (BENCH_SCORE=0, played=1) instead of ABORTED. Cycle-abort (`seen_states >= 3`) is left untouched as ABORTED.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (`_on_setup`, `_does_game_proceed`, `_on_after_game`, `SoloMatrixGameScorer.compute_scores`)
- Modify: `matrixgame_covered_comm/master.py` (same four sites)
- Modify: `matrixgame_covered_comm_nplayer/master.py` (same four sites)
- Test: `matrixgame_covered_solo/tests/test_master.py`
- Test: `matrixgame_covered_comm/tests/test_smoke.py`
- Test: `matrixgame_covered_comm_nplayer/tests/test_smoke.py`

**Interfaces:**
- Produces: each master exposes `self.lost: bool`; `METRIC_LOSE` is logged as `int(self.lost)`; scorers read `episode_interactions.get(METRIC_LOSE, int(not success and not aborted))`.

- [ ] **Step 1: Write the failing test (solo)**

Add to `matrixgame_covered_solo/tests/test_master.py`. The master test near the top of the file uses the existing `_make_master` helper; the scorer tests use the existing `_scorer`, `_episode`, and `_episode_score` helpers plus the `BENCH_SCORE` import already present:

```python
def test_timeout_is_lose_not_abort():
    m = _make_master(thinking=False)
    m.move_log = [{}] * m.max_turns
    assert m._does_game_proceed() is False
    assert m.lost is True
    assert m.aborted is False


def test_lost_episode_scores_zero_and_played():
    from clemcore.clemgame.metrics import METRIC_LOSE, METRIC_PLAYED
    s = _scorer()
    ep = _episode(aborted=0, success=0, move_count=20)
    ep[METRIC_LOSE] = 1  # METRIC_LOSE == "Lose"
    s.compute_scores(ep)
    assert _episode_score(s, BENCH_SCORE) == 0
    assert _episode_score(s, METRIC_PLAYED) == 1
    assert _episode_score(s, METRIC_LOSE) == 1
```

(`_scorer()` builds a `SoloMatrixGameScorer` with `optimal_moves=10` and empty start/target dicts — see its definition around line 215. `_episode(...)` returns a dict keyed by the metric label strings; adding `METRIC_LOSE` sets the `"Lose"` key that the updated scorer now reads.)

- [ ] **Step 2: Run to verify it fails**

Run (from `matrixgame_covered_solo/tests/`): `python -m pytest test_master.py -v -k "timeout_is_lose or lost_episode"`
Expected: FAIL — `AttributeError: 'SoloMatrixGameMaster' object has no attribute 'lost'`.

- [ ] **Step 3: Implement in `matrixgame_covered_solo/master.py`**

In `_on_setup`, after `self.aborted = False` (currently line 171) add:

```python
        self.lost = False
```

In `_does_game_proceed`, replace the guard and timeout branch:

```python
    def _does_game_proceed(self) -> bool:
        if self.success or self.aborted or self.lost:
            return False
        if len(self.move_log) >= self.max_turns:
            self.lost = True
            self.log_to_self("lose", f"Max turns ({self.max_turns}) reached without success")
            return False
        return True
```

In `_on_after_game`, replace the derived LOSE line:

```python
        self.log_key(METRIC_LOSE, int(self.lost))
```

In `SoloMatrixGameScorer.compute_scores`, replace the derived LOSE line (currently `self.log_episode_score(METRIC_LOSE, int(not success and not aborted))`):

```python
        lost = episode_interactions.get(METRIC_LOSE, int(not success and not aborted))
        self.log_episode_score(METRIC_LOSE, lost)
```

(The `BENCH_SCORE` block is unchanged: `aborted → nan`, `success → score`, else → `0`. A lost episode has `aborted=0, success=0` → falls into the `0` branch; `PLAYED = 1 - aborted = 1`.)

- [ ] **Step 4: Run to verify solo passes**

Run: `python -m pytest test_master.py -v -k "timeout_is_lose or lost_episode"`
Expected: PASS.

- [ ] **Step 5: Mirror the change into comm and nplayer masters**

Apply the identical four edits to `matrixgame_covered_comm/master.py` and `matrixgame_covered_comm_nplayer/master.py`:
1. `_on_setup`: add `self.lost = False` after `self.aborted = False`.
2. `_does_game_proceed`: guard `if self.success or self.aborted or self.lost:` and timeout branch sets `self.lost = True` + `self.log_to_self("lose", f"Max turns ({self.max_turns}) reached without success")`.
3. `_on_after_game`: `self.log_key(METRIC_LOSE, int(self.lost))`.
4. Scorer `compute_scores`: `lost = episode_interactions.get(METRIC_LOSE, int(not success and not aborted))` then `self.log_episode_score(METRIC_LOSE, lost)`.

- [ ] **Step 6: Write and run the mirrored tests (comm, nplayer)**

Add to `matrixgame_covered_comm/tests/test_smoke.py`:

```python
def test_timeout_is_lose_not_abort():
    master = _setup_master("none")
    master.move_log = [{}] * master.max_turns
    assert master._does_game_proceed() is False
    assert master.lost is True
    assert master.aborted is False
```

Add to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_timeout_is_lose_not_abort():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3)
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    master.move_log = [{}] * master.max_turns
    assert master._does_game_proceed() is False
    assert master.lost is True
    assert master.aborted is False
```

Run:
`python -m pytest matrixgame_covered_comm/tests/test_smoke.py -v -k timeout_is_lose`
`python -m pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k timeout_is_lose`
Expected: PASS. Then run each full package suite to confirm no regression:
`python -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ -q`

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_comm/master.py matrixgame_covered_comm_nplayer/master.py \
        matrixgame_covered_solo/tests/test_master.py matrixgame_covered_comm/tests/test_smoke.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "reachable LOSE: timeout scored as lose (played, score 0) not abort"
```

---

### Task 2: Coordinate consistency validation (spec Change-area B, Item 5a)

The declared `R<row>,C<col>` must equal the cell the move actually lands on; mismatch is a new illegal code `coordinate_mismatch`.

**Files:**
- Modify: `matrixgame_covered_solo/utils/board.py`, `matrixgame_covered_comm/utils/board.py`, `matrixgame_covered_comm_nplayer/utils/board.py` (`validate_move`)
- Modify: `matrixgame_covered_solo/master.py` (`_validate_move` helper), `matrixgame_covered_comm/master.py` and `matrixgame_covered_comm_nplayer/master.py` (`_validate_player_response`, add `_declared_cell` helper)
- Test: `matrixgame_covered_solo/tests/test_validator.py`, `matrixgame_covered_comm/tests/test_smoke.py`, `matrixgame_covered_comm_nplayer/tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `Board.validate_move(obj, direction, allowed_objects=None, declared_target: Optional[Tuple[int, int]] = None)` — returns `("coordinate_mismatch", msg)` when `declared_target` (0-based) differs from the computed destination. Classified illegal (not `occupied`).
  - module-level `_declared_cell(parsed: Dict[str, str]) -> Optional[Tuple[int, int]]` in comm and nplayer masters, converting the parsed 1-based `"RrowCcol"` target to a 0-based `(row, col)`.

- [ ] **Step 1: Write the failing test (solo board level)**

Add to `matrixgame_covered_solo/tests/test_validator.py`:

```python
def test_validate_move_coordinate_mismatch():
    from utils.board import Board
    b = Board(8, set())
    b.place_object("A", 2, 2)  # 0-based -> displayed R3C3
    # moving A down lands on (3,2) == R4C3; declaring R9C9 (8,8 0-based) is a mismatch
    err = b.validate_move("A", "down", allowed_objects={"A"}, declared_target=(8, 8))
    assert err is not None and err[0] == "coordinate_mismatch"
    # correct declaration passes
    assert b.validate_move("A", "down", allowed_objects={"A"}, declared_target=(3, 2)) is None
    # omitting declared_target keeps old behaviour
    assert b.validate_move("A", "down", allowed_objects={"A"}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run (from `matrixgame_covered_solo/tests/`): `python -m pytest test_validator.py -v -k coordinate_mismatch`
Expected: FAIL — `validate_move() got an unexpected keyword argument 'declared_target'`.

- [ ] **Step 3: Implement in all three `utils/board.py` copies**

Change the signature:

```python
    def validate_move(
        self,
        obj: str,
        direction: str,
        allowed_objects: Optional[Set[str]] = None,
        declared_target: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[str, str]]:
```

Insert the mismatch check immediately after the `off_grid` return and before the `wall` check:

```python
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return ("off_grid", f"Cannot move {obj} {direction}: would leave the grid.")
        if declared_target is not None and declared_target != (new_row, new_col):
            return ("coordinate_mismatch",
                    f"Cannot move {obj} {direction}: you named "
                    f"R{declared_target[0] + 1}C{declared_target[1] + 1} but that move lands on "
                    f"R{new_row + 1}C{new_col + 1}.")
        if (new_row, new_col) in self.walls:
            return ("wall", f"Cannot move {obj} {direction}: (R{new_row + 1},C{new_col + 1}) is a wall.")
```

- [ ] **Step 4: Wire the callers**

In `matrixgame_covered_solo/master.py`, add near the other module-level helpers (after `_render_targets_view`):

```python
def _declared_cell(parsed: Dict[str, str]) -> Optional[Tuple[int, int]]:
    """0-based (row, col) of the declared 'RrowCcol' target, or None if absent."""
    m = re.match(r"R(\d+),?C(\d+)", parsed.get("target", ""))
    if not m:
        return None
    return int(m.group(1)) - 1, int(m.group(2)) - 1
```

Update the `_validate_move` wrapper to pass it:

```python
def _validate_move(board: Board, player: SoloMatrixPlayer, parsed: Dict[str, str]) -> Optional[Tuple[str, str]]:
    return board.validate_move(
        parsed["object"],
        parsed["direction"],
        allowed_objects=player.own_objects,
        declared_target=_declared_cell(parsed),
    )
```

In `matrixgame_covered_comm/master.py` and `matrixgame_covered_comm_nplayer/master.py`, add the same `_declared_cell` helper at module level (both already `import re` and import `Dict`, `Optional`, `Tuple`), and change the `validate_move` call inside `_validate_player_response`:

```python
        error = self.board.validate_move(
            obj, direction, allowed_objects=allowed, declared_target=_declared_cell(parsed)
        )
```

(`coordinate_mismatch != "occupied"`, so the existing error handler counts it under `illegal_attempts` and routes it through the reprompt/abort path unchanged.)

- [ ] **Step 5: Master-level test (comm)**

Add to `matrixgame_covered_comm/tests/test_smoke.py`:

```python
def test_coordinate_mismatch_is_illegal_and_reprompts():
    master = _setup_master("none")
    a0 = master.player_a
    # A legal one-cell move but with a wrong declared destination cell.
    obj = sorted(master._objects_for(a0))[0]
    r, c = master.board.object_positions[obj]
    # choose a legal direction into an empty neighbour, then lie about the cell
    for direction, (dr, dc) in DIRECTIONS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < master.grid_size and 0 <= nc < master.grid_size \
           and (nr, nc) not in master.board.walls and master.board.is_cell_empty(nr, nc):
            break
    resp = f"move: {obj} to R9,C9 ({direction})"
    ok = master._validate_player_response(a0, resp)
    assert ok is False
    assert master.illegal_attempts == 1
    assert master.reprompt_pending is True
```

`DIRECTIONS` is importable from the master (`from master import ..., DIRECTIONS`); add it to the test's imports if absent.

- [ ] **Step 6: Run all coordinate tests**

Run:
`python -m pytest matrixgame_covered_solo/tests/test_validator.py -v -k coordinate_mismatch`
`python -m pytest matrixgame_covered_comm/tests/test_smoke.py -v -k coordinate_mismatch`
Expected: PASS. Then full suites (solo, comm, nplayer) to catch fallout — some existing move fixtures declare arbitrary coordinates and may now be rejected. Fix any fixture whose declared `R..,C..` no longer matches its direction by giving it the correct destination cell (do NOT relax the check).

`python -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ -q`

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_*/utils/board.py matrixgame_covered_*/master.py matrixgame_covered_*/tests/
git commit -m "validate declared destination coordinate; mismatch is illegal (coordinate_mismatch)"
```

---

### Task 3: `masking_hints` flag gates the redirect text (spec Change-area C, Item 12a)

A config flag controls only the *hint text* shown when a player tries to move a masked partner object. Counting is untouched.

**Files:**
- Modify: `matrixgame_covered_comm/resources/common_config.json`, `matrixgame_covered_comm_nplayer/resources/common_config.json` (add `"masking_hints": true`)
- Modify: `matrixgame_covered_comm/instancegenerator.py`, `matrixgame_covered_comm_nplayer/instancegenerator.py` (copy flag onto instances)
- Modify: `matrixgame_covered_comm/master.py`, `matrixgame_covered_comm_nplayer/master.py` (`_on_setup` + two redirect sites)
- Test: `matrixgame_covered_comm/tests/test_smoke.py`, `matrixgame_covered_comm_nplayer/tests/test_smoke.py`

**Interfaces:**
- Produces: each cooperative master exposes `self.masking_hints: bool` (default `True`); redirect hint text is shown only when `masked_attempt and self.masking_hints`; the `invalid_move` / `illegal_attempts` counting is identical in both flag states.

- [ ] **Step 1: Write the failing tests (comm)**

Add to `matrixgame_covered_comm/tests/test_smoke.py`:

```python
def _setup_master_hints(comm_protocol, masking_hints):
    from master import MatrixGameMaster
    inst = _make_game_instance(comm_protocol)
    inst["masking_hints"] = masking_hints
    master = MatrixGameMaster(
        game_spec=_make_game_spec(),
        experiment={"variant_config": {"comm_protocol": comm_protocol}},
        player_models=[_DummyModel("a"), _DummyModel("b")],
    )
    master._on_setup(**inst)
    return master


def test_masking_hint_shown_when_flag_on():
    master = _setup_master_hints("none", masking_hints=True)
    ok = master._validate_player_response(master.player_a, "move: X to R1,C1 (down)")
    assert ok is False
    assert "marks your partner's" in _reprompt_text(master, master.player_a)
    assert master.illegal_attempts == 1  # counting unaffected


def test_masking_hint_suppressed_when_flag_off():
    master = _setup_master_hints("none", masking_hints=False)
    ok = master._validate_player_response(master.player_a, "move: X to R1,C1 (down)")
    assert ok is False
    assert "marks your partner's" not in _reprompt_text(master, master.player_a)
    assert master.illegal_attempts == 1  # counting still identical
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest matrixgame_covered_comm/tests/test_smoke.py -v -k masking_hint`
Expected: `test_masking_hint_suppressed_when_flag_off` FAILS (flag not read yet; hint always shown).

- [ ] **Step 3: Implement — config, generators, masters**

Add to both `resources/common_config.json` files a new key (comm shown; nplayer identical):

```json
{
    "grid_size": 8,
    "objects": ["A", "B", "C", "D", "E", "F"],
    "directions": ["up", "down", "left", "right"],
    "max_retries": 2,
    "strict": false,
    "masking_hints": true
}
```

In both instance generators, where other per-instance keys are set, add:

```python
            game_instance["masking_hints"] = common_config.get("masking_hints", True)
```

In both masters' `_on_setup`, after `self.use_masking = game_instance.get("use_masking", True)` add:

```python
        self.masking_hints: bool = game_instance.get("masking_hints", True)
```

At the **first** redirect site in `_validate_player_response` (the unparseable branch, where `masked_attempt` is computed and `guidance` chosen), gate only the text:

```python
                guidance = (
                    self._masked_move_message(player)
                    if (masked_attempt and self.masking_hints)
                    else "Your response could not be parsed. Please respond exactly in this format:"
                )
```

At the **second** redirect site (the invalid-move branch, `masked_attempt = self.use_masking and obj == BLOCKED_MARKER`):

```python
                guidance = (
                    self._masked_move_message(player)
                    if (masked_attempt and self.masking_hints)
                    else f"Invalid move: {err_msg}\nPlease try again."
                )
```

Leave every `self.log_to_self("invalid_move", ...)` / `log_to_self("invalid_format", ...)` and every counter increment exactly as they are — only the `guidance` string is now flag-dependent.

- [ ] **Step 4: Run to verify comm passes; mirror + test nplayer**

Run: `python -m pytest matrixgame_covered_comm/tests/test_smoke.py -v -k masking_hint`
Expected: PASS. Then copy the two tests + `_setup_master_hints` helper into `matrixgame_covered_comm_nplayer/tests/test_smoke.py`, adapting the constructor to the nplayer pattern (build via `NPlayerMatrixGameMaster` + `_make_game_instance(num_players=3)`, set `inst["masking_hints"]`, target `master.get_players()[0]`). Run:
`python -m pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k masking_hint`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_comm/resources/common_config.json matrixgame_covered_comm_nplayer/resources/common_config.json \
        matrixgame_covered_comm/instancegenerator.py matrixgame_covered_comm_nplayer/instancegenerator.py \
        matrixgame_covered_comm/master.py matrixgame_covered_comm_nplayer/master.py \
        matrixgame_covered_comm/tests/test_smoke.py matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "masking_hints flag gates redirect hint text; counting unchanged"
```

---

### Task 4: Remove master narration of moves (spec Change-area C, Item 12b)

Delete every master-authored move narration (partner-facing and self-confirmation). Done-notifications and message-bulletin lines stay.

**Files:**
- Modify: `matrixgame_covered_solo/master.py` (`_on_valid_player_response` self-confirmation)
- Modify: `matrixgame_covered_comm/master.py` (`_on_valid_player_response` move_desc + both use sites)
- Modify: `matrixgame_covered_comm_nplayer/master.py` (`_on_valid_player_response` move_desc, `_build_broadcast_context` signature)
- Test: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (update assertions at lines ~646, ~669), add narration-absence tests

**Interfaces:**
- Produces: `NPlayerMatrixGameMaster._build_broadcast_context(recipient, move_desc: str = "")` — includes the move-description line only when `move_desc` is truthy (done-notifications still pass their text; move turns pass nothing).

- [ ] **Step 1: Write the failing test (nplayer narration absent)**

Add to `matrixgame_covered_comm_nplayer/tests/test_smoke.py`:

```python
def test_move_is_not_narrated_to_recipient():
    """After a normal move, the recipient's context carries no master
    narration of the move — only messages + boards."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="freeform")
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    ctx = master._build_broadcast_context(master.get_players()[1])
    assert "moved" not in ctx
    assert "CURRENT BOARD:" in ctx
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest matrixgame_covered_comm_nplayer/tests/test_smoke.py -v -k not_narrated`
Expected: FAIL — `_build_broadcast_context` still requires a `move_desc` positional argument.

- [ ] **Step 3: Implement — nplayer**

Change `_build_broadcast_context` to make `move_desc` optional and conditional:

```python
    def _build_broadcast_context(self, recipient: Player, move_desc: str = "") -> str:
        recipient_idx = self._player_to_index[recipient]
        message_lines: List[str] = []
        for other_idx, other in enumerate(self.get_players()):
            if other_idx == recipient_idx:
                continue
            msg = self.last_message[other_idx]
            if msg:
                message_lines.append(f"{other.role_name} says: {msg}")
        comm_block = ("\n".join(message_lines) + "\n\n") if message_lines else ""
        move_block = (f"{move_desc}\n\n") if move_desc else ""

        return (
            f"{comm_block}"
            f"{move_block}"
            f"CURRENT BOARD:\n{self._render_for(recipient)}\n\n"
            f"GOAL BOARD:\n{self._render_targets_for(recipient)}\n\n"
            f"Your turn. Respond with:\n{self._format_reminder()}"
        )
```

In `_on_valid_player_response`, delete the `move_desc` construction block (the `if self.use_masking and obj not in self._objects_for(other): move_desc = ... else: move_desc = ...`) and change the broadcast call to omit it:

```python
            self.set_context_for(other, self._build_broadcast_context(other))
```

Leave the done-branch call that passes `move_desc=f"{player.role_name} has finished …"` unchanged (that is a done-notification, not move narration).

- [ ] **Step 4: Implement — comm and solo**

In `matrixgame_covered_comm/master.py` `_on_valid_player_response`:
- Delete the `move_desc` construction block (`if self.use_masking and obj not in self._objects_for(other): move_desc = ... else: ...`).
- In the `other_done` branch, delete the line `f"You moved {obj} {direction}.\n\n"`.
- In the final `else` branch, delete the line `f"{move_desc}\n\n"`.
- Keep the `comm_line` (`{player.role_name} says: {message}`) lines — those are the message channel, not move narration.

In `matrixgame_covered_solo/master.py` `_on_valid_player_response`, delete the self-confirmation line `f"You moved {obj} {direction}.\n\n"` from the next-turn context (the context becomes CURRENT BOARD / GOAL BOARD / reminder only).

- [ ] **Step 5: Update existing smoke-test assertions**

In `matrixgame_covered_comm_nplayer/tests/test_smoke.py`, the assertions at ~line 646 (`"Player A moved A down" in ctx or "…moved a blocked object down" in ctx`) and ~line 669 (`"Player A moved a blocked object down" in ctx_b`) now contradict the design. Replace each with an assertion that the move is NOT narrated while the message bulletin is still present, e.g.:

```python
    assert "moved" not in ctx
```

Keep the `"Player A says: hello teammates" in ctx` assertion (bulletin, unchanged). Grep both cooperative smoke tests for leftover narration expectations and fix them:
`grep -rn "moved a blocked\|moved A down\|You moved" matrixgame_covered_comm*/tests/`

- [ ] **Step 6: Run suites**

Run: `python -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add matrixgame_covered_solo/master.py matrixgame_covered_comm/master.py matrixgame_covered_comm_nplayer/master.py \
        matrixgame_covered_comm_nplayer/tests/test_smoke.py matrixgame_covered_comm/tests/test_smoke.py
git commit -m "remove master narration of moves; players read moves off the board"
```

---

### Task 5: Truthful prompt templates (spec Change-area D, Items 14.1 & 14.3)

Rewrite the board description to the actual compact listing, and delete the "AVOIDING DEADLOCKS" coaching from every template.

**Files:**
- Modify: every `*.template` under `matrixgame_covered_solo/resources/initial_prompts/en/`, `matrixgame_covered_comm/resources/initial_prompts/en/`, `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/`

**Interfaces:** none (prompt text only).

- [ ] **Step 1: Enumerate the templates**

Run: `ls matrixgame_covered_solo/resources/initial_prompts/en/ matrixgame_covered_comm/resources/initial_prompts/en/ matrixgame_covered_comm_nplayer/resources/initial_prompts/en/`
(Expected: solo `player_prompt_silent.template`, `player_prompt_thinking.template`; comm + nplayer `player_prompt_{none,structured,freeform,hybrid}.template`.)

- [ ] **Step 2: Rewrite the board-description block in each template**

Replace the `THE GRID:` bullet list (the lines describing `##` walls and the dot-for-empty ASCII form) with the compact description. Canonical block (adjust the `$GRID_SIZE$` line only if a template phrases it differently):

```
THE GRID:
- The board is an $GRID_SIZE$×$GRID_SIZE$ grid. Rows are labelled R1–R$GRID_SIZE$ (top to bottom), columns C1–C$GRID_SIZE$ (left to right).
- The board is given to you as compact listings, refreshed every turn:
  - CURRENT BOARD lists each object as `Label@RrowCcol` (e.g. `A@R1C1`), and any walls as a second line `Walls: R4C4 R5C6 …`.
  - GOAL BOARD lists your objects' targets as `Label->RrowCcol` (e.g. `A->R3C5`).
- Any cell not listed as an object or a wall is empty. Objects cannot enter wall cells or leave the grid.
```

In the `RULES:` section of each template, remove the parenthetical wall-glyph references `(##)` (e.g. change `Objects CANNOT move into wall cells (##) or outside the grid boundaries.` to `Objects CANNOT move into wall cells or outside the grid boundaries.`).

- [ ] **Step 3: Delete the AVOIDING DEADLOCKS section in each template**

Remove the entire block that begins with the line `AVOIDING DEADLOCKS:` through its last bullet (in the reference `none` template these are the lines about not repeating blocked moves, cycling, and oscillation), plus the surrounding blank line so no double blank remains.

- [ ] **Step 4: Verify no leftovers**

Run:
```bash
grep -rn "AVOIDING DEADLOCK\|## are walls\|shown as a dot\|(##)\|only given to the players at the start" \
  matrixgame_covered_solo/resources matrixgame_covered_comm/resources matrixgame_covered_comm_nplayer/resources
```
Expected: no output. Then confirm the compact description landed:
```bash
grep -rln "Label@RrowCcol\|@R1C1" matrixgame_covered_*/resources/initial_prompts/en/
```
Expected: every template listed.

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/resources matrixgame_covered_comm/resources matrixgame_covered_comm_nplayer/resources
git commit -m "templates describe compact board listing; drop ASCII/deadlock coaching text"
```

---

### Task 6: Mechanical cleanups (spec Change-area E, Items 14.2 & 14.6)

Unify the dormant `max_turns` default and remove the two excluded packages.

**Files:**
- Modify: `matrixgame_covered_comm/master.py:166`, `matrixgame_covered_comm_nplayer/master.py:164` (`max_turns` default)
- Delete: `matrixgame_covered_base/` (tracked), `matrixgame_covered_multi/` (untracked)

**Interfaces:** none.

- [ ] **Step 1: Unify the `max_turns` default**

In `matrixgame_covered_comm/master.py` and `matrixgame_covered_comm_nplayer/master.py`, change the `_on_setup` line
`self.max_turns = game_instance.get("max_turns", max(50, 5 * optimal))`
to
`self.max_turns = game_instance.get("max_turns", max(4 * optimal, 20))`.

- [ ] **Step 2: Confirm nothing external references the excluded packages**

Run:
```bash
grep -rln --include="*.py" --include="*.json" "matrixgame_covered_base\|matrixgame_covered_multi" . \
  | grep -vE "matrixgame_covered_base/|matrixgame_covered_multi/"
```
Expected: no output (all references are self-contained). If anything appears, stop and resolve before deleting.

- [ ] **Step 3: Remove the packages**

```bash
git rm -r matrixgame_covered_base
rm -rf matrixgame_covered_multi
```

- [ ] **Step 4: Sanity-run the remaining suites**

Run: `python -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ matrixgame_covered_shared/tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "unify max_turns default; remove excluded base/multi packages"
```

---

### Task 7: Regenerate instances and verify (spec Regeneration + Item 14.8)

Prompt text, the `masking_hints` flag, and grammar feedback all changed, so all instances must be regenerated; then verify the metric plumbing.

**Files:**
- Regenerate: `matrixgame_covered_solo/in/instances.json`, `matrixgame_covered_comm/in/instances.json`, `matrixgame_covered_comm_nplayer/in/instances.json`

**Interfaces:** none.

- [ ] **Step 1: Regenerate all instances (seed 42, n=8)**

From each package directory, run its generator with the repo's standard invocation:
```bash
cd matrixgame_covered_solo && python instancegenerator.py --seed 42 --num-instances 8 && cd ..
cd matrixgame_covered_comm && python instancegenerator.py --n 8 && cd ..
cd matrixgame_covered_comm_nplayer && python instancegenerator.py --n 8 && cd ..
```
(If a generator's flags differ, use `python instancegenerator.py --help` to confirm; the seed is 42 and instance count is 8 everywhere.)

- [ ] **Step 2: Verify the flag and shared boards landed**

```bash
python -c "
import json
comm = json.load(open('matrixgame_covered_comm/in/instances.json'))
gi = comm['experiments'][0]['game_instances'][0]
assert 'masking_hints' in gi, 'masking_hints not written onto instances'
print('masking_hints present:', gi['masking_hints'])
"
```
Expected: prints `masking_hints present: True`. Spot-check that instance prompts no longer contain `##`/`AVOIDING DEADLOCK`:
```bash
python -c "
import json
d = json.load(open('matrixgame_covered_comm/in/instances.json'))
p = d['experiments'][0]['game_instances'][0]['player_prompt']
assert 'AVOIDING DEADLOCK' not in p and '## are walls' not in p
assert '@R1C1' in p or 'RrowCcol' in p
print('prompt updated in instances: OK')
"
```

- [ ] **Step 3: Full test sweep**

Run: `python -m pytest matrixgame_covered_solo/tests/ matrixgame_covered_comm/tests/ matrixgame_covered_comm_nplayer/tests/ matrixgame_covered_shared/tests/ -q`
Expected: PASS.

- [ ] **Step 4: Verify no METRIC_PLAYED double-logging (Item 14.8)**

Confirm the scorer is the only place logging `Played`:
```bash
grep -rn "METRIC_PLAYED\|\"Played\"\|log_key(METRIC_PLAYED\|log_episode_score(METRIC_PLAYED" matrixgame_covered_*/master.py
```
Expected: `METRIC_PLAYED` appears only in each scorer's `log_episode_score`, never in a master's `log_key`. Record a note in the spec appendix that after the first real `clem run`, the results table must be checked for a single `Played` column (clemcore's standard eval must not also derive one).

- [ ] **Step 5: Commit**

```bash
git add matrixgame_covered_solo/in/instances.json matrixgame_covered_comm/in/instances.json matrixgame_covered_comm_nplayer/in/instances.json
git commit -m "regenerate instances: updated prompts + masking_hints flag, seed 42 n=8"
```

---

## Thesis-text edits (no code — track separately in the LaTeX)

These are not part of the code plan; collected here so they aren't lost:
- **5b:** add the channel-off `done:` production to `fig:task-grammar`.
- **5c:** state the accepted PLAN/REQUEST separator (`[—\-]+`) consistently with the figure.
- **14.4:** note strict-mode is disabled in the shipped experiments.
- Document: the LOSE outcome (Item 1); cycle-abort as a documented ABORTED cause, and that oscillation is *terminated at k=3* rather than measured to completion (Items 2 / 14.3); coordinate now validated (5a); the `masking_hints` flag and removal of move narration (12); the persistent latest-message bulletin as the n-player generalization plus the dyad's one-shot relay difference (13); the compact-listing board description (14.1).

## Self-Review notes

- Spec coverage: A→Task 1; B→Task 2; C(12a)→Task 3; C(12b)→Task 4; C(13)→no code (documented); D(14.1,14.3)→Task 5; E(14.2,14.6)→Task 6; regeneration + 14.8→Task 7; thesis-only items listed above.
- Item 2 (cycle-abort) is intentionally left unchanged — no task modifies the `seen_states >= 3` block; the accepted metric-censoring tradeoff is recorded in the spec.
