# matrixgame_covered_solo — Design

**Status:** draft for review
**Date:** 2026-05-23
**Author:** Tom (with Claude)
**Parent:** `matrixgame_covered_comm`

## 1. Motivation

`matrixgame_covered_comm` is a two-LLM collaborative grid-rearrangement benchmark. Each model sees a masked half of an 8×8 board, owns 3 of 6 objects, and coordinates moves via four communication protocols (none/structured/freeform/hybrid). The communication axis is the experimental focus there.

This spec defines a **single-player** sibling, `matrixgame_covered_solo`, that isolates the spatial-planning component from the coordination component. With one model and no partner, the comm-protocol axis drops out and is replaced by two new axes:

- **`view_mode`** — `full` (model sees all 6 objects and owns all 6) vs `masked` (model sees only its 3 owned objects; the other 3 appear as `X` and are static obstacles).
- **`thinking`** — `off` (move-only response) vs `on` (optional `message:` scratchpad line preserved for analysis but ignored by game mechanics).

The spatial-level axis (`S1`–`S4`) and the `compact_board` rendering axis carry over from `_comm`. `easy_mode` and `enforce_isolation` do not.

## 2. Goals & non-goals

**Goals**
- Provide a clembench game that measures LLM spatial-planning ability on the same boards as `_comm`, with no partner-coordination confound.
- Emit metrics with the same names and formulas as `_comm` so the existing result-analysis tooling works unchanged.
- Stay independent of `_comm` at runtime — changes to `_comm` should not break `_solo` and vice versa.

**Non-goals**
- No partner, scripted or otherwise.
- No new metrics beyond what `_comm` already emits.
- No `easy_mode` axis, no `enforce_isolation` axis.
- No batched (multi-move-per-turn) response format; one move per turn, matching `_comm`.
- No changes to `_comm`.

## 3. Module layout

```
matrixgame_covered_solo/
  __init__.py
  clemgame.json                  # game spec, players: 1, roles: ["Player"]
  master.py                      # SoloMatrixGameMaster + SoloMatrixPlayer + SoloMatrixGameScorer
  instancegenerator.py           # builds in/instances.json from resources/config.json
  resources/
    config.json                  # 32 experiment configs (Section 7)
    common_config.json           # grid_size, objects, directions, max_retries, strict, move_multiplier
    initial_prompts/en/
      player_prompt_full_silent.template
      player_prompt_full_thinking.template
      player_prompt_masked_silent.template
      player_prompt_masked_thinking.template
  utils/
    board.py                     # copied verbatim from _comm/utils/board.py
    astar.py                     # copied verbatim from _comm/utils/astar.py
  in/
    instances.json               # generated artifact (git-tracked)
```

`board.py` and `astar.py` are **copied** rather than imported from `_comm` so the new game has zero runtime dependency on `_comm`.

Four prompt templates (one per `view_mode × thinking` combination) rather than a single template with conditionals — keeps each template short.

## 4. Components

### `SoloMatrixPlayer(Player)`
Single field beyond the base class: `own_objects: Set[str]`.
- `view_mode=full`  → `{"A","B","C","D","E","F"}`
- `view_mode=masked` → `{"A","B","C"}` (foreigns `{"D","E","F"}` render as `X` and are immutable)

This set drives both the rendering masker and the done check.

### `SoloMatrixGameMaster(DialogueGameMaster)`

- **`_on_setup(game_instance, **kwargs)`** — reads `view_mode`, `thinking`, `compact_board` from the experiment; instantiates one `Board` from the instance; selects one of four prompt templates by `(view_mode, thinking)`; registers one `SoloMatrixPlayer` with the correct `own_objects`.
- **`_on_before_turn(...)`** — calls `_render_view(board, player)` and appends the rendered grid to the user-turn message.
- **`_render_view(board, player)`** —
  ```python
  if self.compact_board:
      return board.render_compact_for(player, masked=(self.view_mode == "masked"))
  return board.render_for(player, masked=(self.view_mode == "masked"))
  ```
  In `view_mode=full`, `own_objects` covers all letters, so the existing masker returns the full grid unchanged.
- **`_parse_response(response)`** — selects `COMM_MOVE_PATTERN` if `thinking=on` else `MOVE_ONLY_PATTERN`. Both regexes are copied verbatim from `matrixgame_covered_comm/master.py:55-64`. No `done:` patterns (Section 5).
- **`_validate_move(parsed)`** — object must be in `player.own_objects`; target cell must be in-bounds, not a wall, and not currently occupied by any other object (static `X` foreigns count as occupied). Invalid → retry up to `max_retries`, then abort.
- **`_apply_move(parsed)`** — mutates `Board` via its existing single-object move API.
- **`_check_done()`** — `True` when every object in `own_objects` sits on its target cell. The 3 static `X` foreigns in `masked` mode are not in `own_objects`, so their targets are never checked — this is intentional.
- **`_advance_turn()`** — single player, no rotation; increment turn counter; if `_check_done()` → end episode (success); if `moves_used >= move_cap` → end episode (lose).

### `SoloMatrixGameScorer(GameScorer)`
Emits the metrics listed in Section 6.

Expected master size: ~250–300 lines (vs `_comm`'s 596), because message-relay, two-player rotation, isolation, structured-message validation, and `done:` declaration handling all drop out.

## 5. Response format & episode termination

**Per turn** the model emits exactly one of:

- `thinking=off`:
  ```
  reason: <free text>
  move: <OBJECT> to R<row>,C<col> (<direction>)
  ```
- `thinking=on`:
  ```
  message: <free text — scratchpad, logged but not interpreted>
  reason: <free text>
  move: <OBJECT> to R<row>,C<col> (<direction>)
  ```

`<OBJECT>` ∈ `own_objects`; `<direction>` ∈ `{up, down, left, right}`; coordinates are 0-indexed.

**No `done:` declaration.** The master has full ground truth of `own_objects` positions and targets, so it auto-detects completion after every successful `_apply_move`. This differs from `_comm`, where `done:` exists for two-player reasons (asymmetric info between players, turn scheduling around a finished player, and "premature done" as a coordination-failure mode) that don't apply in solo.

**Episode ends in exactly three ways:**
1. **success** — `_check_done()` returns true after a move
2. **lose** — `moves_used >= move_cap` without success
3. **abort** — parse or validate failures exhaust `max_retries` on a single turn

## 6. Metrics

Same names and formulas as `_comm`, so existing analysis tooling works without changes.

| metric | definition |
|---|---|
| `METRIC_SUCCESS` | 1 if `_check_done()` true at episode end, else 0 |
| `METRIC_LOSE` | 1 if move-cap exhausted without success, else 0 |
| `METRIC_ABORTED` | 1 if parse/validate aborted, else 0 |
| `MOVE_COUNT` | number of moves applied |
| `TURN_MOVES` | per-turn move log (object, from, to, direction, plus `message:` content if `thinking=on`) |
| `METRIC_REQUEST_COUNT` | total model requests |
| `METRIC_REQUEST_COUNT_PARSED` | requests that parsed successfully |
| `METRIC_REQUEST_COUNT_VIOLATED` | requests that failed parse or validate |
| `BENCH_SCORE` | `min(100, round(optimal_moves / max(move_count, 1) * 100, 2))` on success; `0` on lose; `NaN` on abort. Identical to `_comm`'s `MatrixGameScorer.compute_scores`. |

## 7. Experiment matrix

32 experiment configs in `resources/config.json` = 2 (`view_mode`) × 2 (`thinking`) × 4 (`spatial_level`) × 2 (`compact_board`).

**Naming:** `solo_<view>_<level>_<thinking>[_compact]`
e.g. `solo_full_S2_silent`, `solo_masked_S3_thinking_compact`.

**Per-config fields:**
```json
{
  "name": "Solo_Masked_S2_Thinking",
  "view_mode": "masked",
  "thinking": true,
  "spatial_level": "S2",
  "compact_board": false,
  "num_objects": 6
}
```

The per-instance `max_turns` is computed by the instance generator as `max(4 * optimal_moves, 20)`, matching `_comm` exactly — no `move_multiplier` field is needed in the config.

`thinking` is a bool (not a string), matching `_comm`'s convention for binary flags like `use_masking`, `enforce_isolation`, `compact_board`. The naming-convention suffix `_thinking`/`_silent` is just a human-readable label in `name`.

`comm_protocol`, `enforce_isolation`, `use_masking`, `easy_mode` are intentionally absent.

## 8. Instance generation

`instancegenerator.py` is a port of `_comm/instancegenerator.py` with message-related and isolation-related branches removed. For each experiment config:

1. Generate **10 boards** per spatial level using `_comm`'s wall-generation logic (copied), seeded for reproducibility.
2. Place 6 objects (`A`–`F`) and 6 target cells uniformly at random subject to:
   - Targets reachable from origins (A* path exists ignoring other objects).
   - No two objects or two targets coincident.
   - No overlap with walls.
3. Designate owned set:
   - `view_mode=full` → all 6 owned.
   - `view_mode=masked` → `{A,B,C}` owned, `{D,E,F}` static foreigns.
4. Pre-compute `optimal_moves` using `_comm`'s `_verify_reachability` helper (per-object BFS, sums shortest-path lengths). This is a lower bound on the true joint optimum, same as `_comm`.
   - `view_mode=masked` → BFS over `{A,B,C}` starts/targets, treating the 3 foreign cells as additional walls. Foreigns are static at runtime so this matches their actual blocking behaviour.
   - `view_mode=full` → BFS over all 6 starts/targets, walls only. Inter-object blocking is ignored (matches `_comm`'s per-player approach).
5. Store per-instance: `grid_size`, `walls`, `objects` (list of letters), `owned_objects` (list), `foreign_objects` (list, empty in full mode), `start_positions` (dict obj→[r,c]), `target_positions` (dict obj→[r,c]), `optimal_moves` (int), `max_turns = max(4 * optimal_moves, 20)` (matches `_comm`), `max_retries`, `strict`, `view_mode`, `thinking`, `spatial_level`, `compact_board`, `player_prompt` (loaded template string).

**Generator invariant:** every owned object's origin must differ from its target, so `optimal_moves >= num_owned` and the `BENCH_SCORE` denominator is non-zero (it's `max(move_count, 1)` anyway, but we also want a non-trivial task). Instances violating this are rejected and regenerated.

Total instances generated: 32 × 10 = **320**.

## 9. Prompt templates

Four templates under `resources/initial_prompts/en/`, one per `(view_mode, thinking)` combination. Each template describes:

- The grid representation (8×8, 0-indexed, coordinate format `R{row},C{col}`).
- Wall character `#`, empty `.`, owned objects as their letters, foreigns as `X` (masked only).
- The objects the model owns and their targets (listed inline).
- The response format expected this turn (one of the two formats in Section 5).
- The movement rules (one cell per move, blocked by walls, by other objects, and by `X` foreigns).

No "declare done" instruction (Section 5). No mention of a partner.

The four templates are kept as separate files rather than one with conditionals because each is short (~30 lines) and they're easier to iterate on independently.

## 10. Testing strategy

Unit tests under `matrixgame_covered_solo/tests/`:

- **Parser tests** — `_parse_response` accepts valid silent and thinking-mode outputs; rejects malformed; rejects wrong-mode outputs (silent format submitted under `thinking=on` and vice versa).
- **Validator tests** — rejects moves of unowned objects (e.g., `X` foreign in masked mode); rejects out-of-bounds, into-wall, into-occupied-cell.
- **Done-check tests** — `_check_done` returns true exactly when all `own_objects` are on their targets; ignores foreigns in masked mode; requires all 6 in full mode.
- **Renderer tests** — `_render_view` returns full grid in `view_mode=full`; returns `X`-masked grid in `view_mode=masked`; honours `compact_board`.
- **Instance-generator tests** — generates the expected count of instances per config; all generated instances satisfy invariants (reachability, no overlap, walls don't cover origins/targets); `astar_optimal` is consistent with the stored board.
- **Scorer tests** — `BENCH_SCORE` formula matches `_comm`'s on shared inputs; success/lose/abort branches set the expected metric flags.

Integration smoke test: run one instance per experiment config end-to-end with a stub model that always emits a valid move toward a target object's destination; assert success on `view_mode=full + S1` and on `view_mode=masked + S1`.

## 11. Risks & open questions

- **`optimal_moves` is a lower bound, not the true joint optimum** — same caveat as `_comm`. Per-object BFS ignores inter-object blocking; the actual minimum can be higher when objects must clear each other's paths. We accept this for cross-game comparability rather than computing a true joint optimum. `BENCH_SCORE = optimal_moves / move_count * 100` can therefore exceed 100 in theory, which is why `_comm` clamps with `min(100, …)`; we keep the same clamp.
- **Prompt drift** — keeping four templates means a wording fix in one mode can fall out of sync with the others. Mitigation: a small test that asserts shared phrasing snippets (movement rules, coordinate format) appear in all four.
- **Foreign `X` blocking in masked mode can make some instances unsolvable for `{A,B,C}`** — the instance generator must verify that an A* path exists for each owned object on the board *with foreigns treated as static obstacles*, not just on an empty board. Captured as a generator invariant in Section 10.

## 12. Out of scope (revisit later if useful)

- A `scripted partner` variant of solo (model + oracle B).
- An `easy_mode` axis for solo.
- A `batched moves per turn` response format.
- Refactoring shared utilities (`board.py`, `astar.py`) into a cross-game package.
