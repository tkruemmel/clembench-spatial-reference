# matrixgame_covered_solo — Design

**Status:** draft for review
**Date:** 2026-05-23
**Author:** Tom (with Claude)
**Parent:** `matrixgame_covered_comm`

## 1. Motivation

`matrixgame_covered_comm` is a two-LLM collaborative grid-rearrangement benchmark. Each model sees a masked half of an 8×8 board, owns 3 of 6 objects, and coordinates moves via four communication protocols (none/structured/freeform/hybrid). The communication axis is the experimental focus there.

This spec defines a **single-player** sibling, `matrixgame_covered_solo`, that isolates the spatial-planning component from the coordination component. With one model and no partner, the comm-protocol axis drops out and is replaced by one new axis:

- **`thinking`** — `off` (move-only response) vs `on` (optional `message:` scratchpad line preserved for analysis but ignored by game mechanics).

The single model has **full view** of the 8×8 board and **owns all 6 objects** — there is no partial-observability variant in this game. (A masked variant was considered but dropped: in `_comm`'s S2/S3/S4 the spatial levels build cross-agent dependency chains by setting one object's start cell equal to another object's target cell, so freezing half the chain as static obstacles makes the remaining half unreachable. Adding a masked solo variant would require new spatial-level generators rather than reusing `_comm`'s — out of scope here.)

The spatial-level axis (`S1`–`S4`) and the `compact_board` rendering axis carry over from `_comm`. `easy_mode` and `enforce_isolation` do not.

## 2. Goals & non-goals

**Goals**
- Provide a clembench game that measures LLM spatial-planning ability on the same boards as `_comm`, with no partner-coordination confound.
- Emit metrics with the same names and formulas as `_comm` so the existing result-analysis tooling works unchanged.
- Stay independent of `_comm` at runtime — changes to `_comm` should not break `_solo` and vice versa.

**Non-goals**
- No partner, scripted or otherwise.
- No partial observability / masked view. (See Motivation; revisit only with redesigned spatial generators.)
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
    config.json                  # 16 experiment configs (Section 7)
    common_config.json           # grid_size, objects, directions, max_retries, strict
    initial_prompts/en/
      player_prompt_silent.template
      player_prompt_thinking.template
  utils/
    board.py                     # copied verbatim from _comm/utils/board.py
    astar.py                     # copied verbatim from _comm/utils/astar.py
  in/
    instances.json               # generated artifact (git-tracked)
```

`board.py` and `astar.py` are **copied** rather than imported from `_comm` so the new game has zero runtime dependency on `_comm`.

Two prompt templates (one per `thinking` value) rather than one with conditionals — keeps each template short.

## 4. Components

### `SoloMatrixPlayer(Player)`
Carries `own_objects: Set[str]`, which is always `{"A","B","C","D","E","F"}`. The attribute exists as a cheap audit hook (it's what `_validate_move`'s allowed-objects filter checks against), but in this game it never varies.

### `SoloMatrixGameMaster(DialogueGameMaster)`

- **`_on_setup(game_instance, **kwargs)`** — reads `thinking` and `compact_board` from the experiment; instantiates one `Board` from the instance; selects one of two prompt templates by `thinking`; registers one `SoloMatrixPlayer`.
- **`_on_before_turn(...)`** — renders the full board (optionally compact) and appends it to the user-turn message.
- **`_render_view(board, compact)`** — returns `board.render_compact()` if compact else `board.render()`. No masking.
- **`_parse_response(response)`** — selects `MESSAGE_MOVE_PATTERN` if `thinking=on` else `MOVE_ONLY_PATTERN`. Both regexes are anchored with `\A` so silent mode rejects a leading `message:` line and thinking mode requires one. No `done:` patterns (Section 5).
- **`_validate_move(parsed)`** — wraps `Board.validate_move` with `allowed_objects=player.own_objects`. Invalid → retry up to `max_retries`, then abort.
- **`_apply_move(parsed)`** — mutates `Board` via its existing single-object move API.
- **`_check_done()`** — `True` when every owned object sits on its target cell — i.e., when `Board.is_solved()` returns True, since `own_objects` covers everything.
- **`_advance_turn()`** — single player, no rotation; increment turn counter; if `_check_done()` → end episode (success); if `moves_used >= max_turns` → end episode (lose).

### `SoloMatrixGameScorer(GameScorer)`
Emits the metrics listed in Section 6.

Expected master size: ~250 lines (vs `_comm`'s 596), because message-relay, two-player rotation, isolation, structured-message validation, and `done:` declaration handling all drop out.

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

`<OBJECT>` ∈ `{A,B,C,D,E,F}`; `<direction>` ∈ `{up, down, left, right}`; coordinates are 0-indexed.

**No `done:` declaration.** The master has full ground truth and auto-detects completion after every successful `_apply_move`. This differs from `_comm`, where `done:` exists for two-player reasons that don't apply in solo.

**Episode ends in exactly three ways:**
1. **success** — `_check_done()` returns true after a move
2. **lose** — `moves_used >= max_turns` without success
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

16 experiment configs in `resources/config.json` = 2 (`thinking`) × 4 (`spatial_level`) × 2 (`compact_board`).

**Naming:** `solo_<level>_<thinking>[_compact]`
e.g. `solo_S2_silent`, `solo_S3_thinking_compact`.

**Per-config fields:**
```json
{
  "name": "Solo_S2_Thinking",
  "thinking": true,
  "spatial_level": "S2",
  "compact_board": false,
  "num_objects": 6
}
```

The per-instance `max_turns` is computed by the instance generator as `max(4 * optimal_moves, 20)`, matching `_comm` exactly — no `move_multiplier` field is needed in the config.

`thinking` is a bool (not a string), matching `_comm`'s convention for binary flags. The naming-convention suffix `_thinking`/`_silent` is just a human-readable label in `name`.

`comm_protocol`, `enforce_isolation`, `use_masking`, `easy_mode`, `view_mode` are intentionally absent.

## 8. Instance generation

`instancegenerator.py` is a port of `_comm/instancegenerator.py` with message-related and isolation-related branches removed. For each experiment config:

1. Generate **10 boards** per spatial level using `_comm`'s wall-generation logic (copied), seeded for reproducibility.
2. Place 6 objects (`A`–`F`) and 6 target cells subject to the spatial level's invariants (the `_generate_s*` functions handle reachability and chain construction).
3. The model owns all 6 objects; there are no foreigns.
4. Pre-compute `optimal_moves` using `_comm`'s `_verify_reachability` helper (per-object BFS over all 6 starts/targets, walls only). Inter-object blocking is ignored — this is a lower bound on the true joint optimum, same as `_comm`.
5. Store per-instance: `grid_size`, `walls`, `objects` (list of letters), `start_positions` (dict obj→[r,c]), `target_positions` (dict obj→[r,c]), `optimal_moves` (int), `max_turns = max(4 * optimal_moves, 20)`, `max_retries`, `strict`, `thinking`, `spatial_level`, `compact_board`, `player_prompt` (loaded template string).

**Generator invariant:** every object's origin must differ from its target. Instances violating this are rejected and regenerated.

Total instances generated: 16 × 10 = **160**.

## 9. Prompt templates

Two templates under `resources/initial_prompts/en/`, one per `thinking` value:
- `player_prompt_silent.template`
- `player_prompt_thinking.template`

Each describes:
- The grid representation (8×8, 0-indexed, coordinate format `R{row},C{col}`).
- Wall character `#`, empty `.`, objects as their letters (`A`–`F`).
- The objects the model controls (all 6) and their targets (listed inline).
- The response format expected this turn (one of the two formats in Section 5).
- The movement rules (one cell per move, blocked by walls and by other objects).

No "declare done" instruction (Section 5). No mention of a partner.

The two templates are separate files rather than one with conditionals because each is short (~30 lines) and they're easier to iterate on independently.

## 10. Testing strategy

Unit tests under `matrixgame_covered_solo/tests/`:

- **Parser tests** — `_parse_response` accepts valid silent and thinking-mode outputs; rejects malformed; rejects wrong-mode outputs (silent format submitted under `thinking=on` and vice versa).
- **Validator tests** — rejects out-of-bounds, into-wall, into-occupied-cell. (No "unowned object" test — the player owns everything.)
- **Done-check tests** — `_check_done` returns true exactly when all 6 objects are on their targets.
- **Renderer tests** — `_render_view` returns the full grid; honours `compact_board`.
- **Instance-generator tests** — generates the expected count of instances per config; all instances satisfy invariants (reachability, no overlap, walls don't cover origins/targets); `optimal_moves` is consistent with the stored board.
- **Scorer tests** — `BENCH_SCORE` formula matches `_comm`'s on shared inputs; success/lose/abort branches set the expected metric flags.

Integration smoke test: instantiate the master with a stub model and exercise one validate+apply cycle to catch wiring regressions.

## 11. Risks & open questions

- **`optimal_moves` is a lower bound, not the true joint optimum** — same caveat as `_comm`. Per-object BFS ignores inter-object blocking; the actual minimum can be higher when objects must clear each other's paths. We accept this for cross-game comparability rather than computing a true joint optimum. `BENCH_SCORE = optimal_moves / move_count * 100` can therefore exceed 100 in theory, which is why `_comm` clamps with `min(100, …)`; we keep the same clamp.
- **Prompt drift** — keeping two templates means a wording fix in one mode can fall out of sync with the other. Mitigation: a small test that asserts shared phrasing snippets (movement rules, coordinate format) appear in both.

## 12. Out of scope (revisit later if useful)

- A **masked-view variant** (requires new spatial-level generators that don't build cross-agent chains).
- A `scripted partner` variant of solo (model + oracle B).
- An `easy_mode` axis for solo.
- A `batched moves per turn` response format.
- Refactoring shared utilities (`board.py`, `astar.py`) into a cross-game package.
