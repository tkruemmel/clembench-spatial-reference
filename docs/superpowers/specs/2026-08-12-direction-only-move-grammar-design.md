# Direction-only move grammar — design

Date: 2026-08-12
Status: approved (design)

## Goal

Change the matrix game move grammar so a player names only the **object and a
direction**, not an absolute target coordinate. The current grammar carries
both a declared coordinate and a direction and cross-checks them; we drop the
coordinate from the move command.

- **Old:** `move: A to R3,C5 (right)`
- **New:** `move: A right`

The object letter is retained (needed to disambiguate objects A–F). The
direction becomes a bare word (no parentheses).

## Scope decisions (agreed)

1. **New syntax:** `move: A right` (bare direction, no parentheses).
2. **Clean break:** the old coordinate-bearing form no longer parses; it falls
   through to the standard reprompt path. No backward-compatibility tolerance.
3. **PLAN clause (structured/hybrid comm) keeps coordinates**, but its meaning
   broadens from "the next cell" to **current position → goal position**
   (possibly several moves away). The REQUEST serves that goal. The PLAN regex
   is unchanged; only wording/interpretation changes.
4. **Instances regenerated** from the instance generators (fixed seeds →
   identical boards; only the embedded `player_prompt` text changes).

## What this changes about the measurement

Removing the declared coordinate removes the `coordinate_mismatch` check — the
game no longer probes "can the model compute the resulting cell." It still
probes "can the model choose a correct direction," and every other spatial
constraint remains enforced (off-grid, wall, occupied, ownership,
unknown-object, bad-direction). The win condition (objects reaching their
targets) is unchanged, so path planning is still required — the model simply
does not have to *state* the landing cell. This is an intended narrowing of
the probe, not an accident.

## Architecture note

The three variants — `matrixgame_covered_solo`, `matrixgame_covered_comm`,
`matrixgame_covered_comm_nplayer` — each keep their **own private copy** of the
move regexes, `_parse_response`, `_declared_cell`, `_format_reminder`,
`validate_move`, and `utils/board.py`. There is no shared parsing module (only
board *generation* is shared via `matrixgame_covered_shared/boardgen.py`).
Therefore every grammar change must be applied in all three copies. The move
grammar string is currently byte-identical across the three, so the three edits
are parallel.

## Changes by layer

### 1. Move grammar (each `master.py`)

- `MOVE_ONLY_PATTERN` (channel off / solo silent):
  `r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)"`,
  `re.IGNORECASE`.
- `COMM_MOVE_PATTERN` (comm/nplayer, channel on) and solo `MESSAGE_MOVE_PATTERN`
  (thinking): keep the `message:` prefix and `re.DOTALL` where present; replace
  the `to R\d+,?C\d+ \(dir\)` tail with the bare `[A-Z]\s+(dir)` tail.
- `DONE_ONLY_PATTERN` / `COMM_DONE_PATTERN`: unchanged.
- `MASKED_OBJECT_MOVE_PATTERN` (comm/nplayer): simplify to match the bare
  `move:\s*X\b` form (drop the `@R\d+,?C\d+` position-qualified variant, which
  models emitted only because coordinates were part of the grammar).

### 2. Parsing (each `master.py`)

- `_parse_response`: drop the `"target"` key from the result dict. Keep
  `object`, `direction`, and (when applicable) `message` / `is_done`.
- Delete `_declared_cell` and remove all its call sites.

### 3. Semantic validation (each `utils/board.py`)

- `validate_move`: remove the `declared_target` parameter and the
  `coordinate_mismatch` branch. Update all callers in the masters.
- Remove `coordinate_mismatch` from the ILLEGAL classification set.
- Keep every other code: `bad_direction`, `unknown_object`, `not_owner`,
  `off_grid`, `wall`, `occupied`.

### 4. Reminders & diagnostics (each `master.py`)

- `_format_reminder`: return `move: <OBJECT> <DIRECTION>` (and the
  `message:`-prefixed variant for the with-message case).
- Remove the coordinate-mismatch-specific targeted reprompt / redirect text in
  comm and nplayer.

### 5. PLAN clause reframe (comm + nplayer `master.py`)

- `STRUCTURED_MSG_PATTERN` / `HYBRID_MSG_PATTERN`: **unchanged** — the form
  `move OBJECT from (Rr,Cc) to (Rr,Cc)` already accepts any two cells; only the
  interpretation (goal vs. next cell) changes.
- `_STRUCTURED_REMINDER`, `_HYBRID_REMINDER`, and the inline reminder strings in
  `_schema_violation`: reword so the "to (Rr,Cc)" cell is described as the
  object's **goal** position, and note the PLAN may span several moves.
- `CLAUSE_MAX_WORDS` stays 10 (a goal-framed PLAN is still ~6 words).
- Consequence: the PLAN's from/to and the single-step `move:` line legitimately
  decouple. No cross-check exists today and none is added.

### 6. Prompt templates (`resources/initial_prompts/en/*.template`)

Update the RESPONSE-FORMAT move line and every example to the new grammar:

- Solo: `player_prompt_silent`, `player_prompt_thinking`.
- Comm: `player_prompt_none`, `_structured`, `_freeform`, `_hybrid`.
- Nplayer: `player_prompt_none`, `_structured`, `_freeform`, `_hybrid`
  (preserving the `$YOUR_ROLE$` / `$N_OTHER_PLAYERS$` placeholders).

For structured/hybrid, also update the PLAN example to goal framing, e.g.
`message: PLAN: move A from (R2,C3) to (R4,C6) — REQUEST: please clear (R3,C6)`
paired with a single-step `move: A down`.

### 7. Regenerate instances

Re-run each variant's `instancegenerator.py`. Fixed seeds reproduce identical
boards; only the embedded `player_prompt` string changes. This overwrites the
already-modified `in/instances*.json` files currently in the working tree.

### 8. Tests

- `matrixgame_covered_solo/tests/`: `test_parser.py`, `test_validator.py`,
  `test_master.py` — switch fixtures to the new grammar, drop
  `coordinate_mismatch` cases, add a case asserting the old coordinate form now
  fails to parse (and reprompts).
- `matrixgame_covered_comm/tests/test_smoke.py` and
  `matrixgame_covered_comm_nplayer/tests/test_smoke.py` — update move fixtures to
  the new grammar; verify PLAN goal-framing still validates.

## Not affected

- Scoring, metrics, and `optimal_moves` — position/target based, not
  declared-coordinate based.
- Board rendering — players still *see* object and target coordinates
  (`Label@RrCc`, `Label->RrCc`); they simply no longer type them in the move
  line. This keeps the PLAN clause's coordinate references grounded in what the
  player observes.

## Files touched (summary)

- `matrixgame_covered_{solo,comm,comm_nplayer}/master.py`
- `matrixgame_covered_{solo,comm,comm_nplayer}/utils/board.py`
- `matrixgame_covered_{solo,comm,comm_nplayer}/resources/initial_prompts/en/*.template`
- `matrixgame_covered_{solo,comm,comm_nplayer}/in/instances*.json` (regenerated)
- `matrixgame_covered_solo/tests/{test_parser,test_validator,test_master}.py`
- `matrixgame_covered_{comm,comm_nplayer}/tests/test_smoke.py`
