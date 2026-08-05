# Open-Findings Alignment — Design Spec

**Date:** 2026-08-05
**Branch:** `spatial_reasoning_test`
**Scope:** The six deferred items from `thesis_code_open_findings.md` (Items 1, 2, 5, 12, 13, 14). The companion `thesis_code_alignment_plan.md` (items 3, 4, 6, 7, 8, 9, 10, 11) is already merged; `utils/metrics.py`, anchored grammar, shared `boardgen.py`, and n=8 instances are in place.

## Context

Three thesis game packages, each a standalone clembench game:
- `matrixgame_covered_solo` — 1 player (silent / thinking variants)
- `matrixgame_covered_comm` — dyad (protocols none / structured / freeform / hybrid)
- `matrixgame_covered_comm_nplayer` — registered as `_3p` and `_6p`

Each package: `master.py` (game master + scorer), `instancegenerator.py`, `utils/board.py`, `utils/metrics.py`, `resources/initial_prompts/en/*.template`, `in/instances.json`, `tests/`. `master.py` must stay runtime-standalone (no cross-package imports); `utils/board.py` is duplicated per package by design.

## Decisions (locked)

| Item | Decision |
|------|----------|
| 1 (timeout) | Timeout → new `LOSE` outcome (`BENCH_SCORE = 0`, counts as played). |
| 2 (cycle-abort) | Keep `seen_states >= 3` abort as **ABORTED**, unchanged; document in thesis. |
| 5a (coordinate) | **Validate consistency**; mismatch = illegal move, new code `coordinate_mismatch`. |
| 5b, 5c | Thesis-text only. |
| 12a (redirect) | Config flag `masking_hints` gates **hint text only**; counting untouched. |
| 12b (narration) | **Remove all** master narration of moves (partner-facing and self-confirmation). |
| 13 (broadcast) | Keep **persistent latest-message bulletin** (no code); document in thesis. |
| 14.1 | Rewrite template GRID sections to describe compact listing. |
| 14.2 | Unify `max_turns` default to `max(4*optimal, 20)`. |
| 14.3 | **Remove** the "AVOIDING DEADLOCKS" coaching paragraph from all templates. |
| 14.4 | Thesis note: strict-mode disabled in shipped experiments. |
| 14.6 | **Remove** `matrixgame_covered_base` and `matrixgame_covered_multi`. |
| 14.8 | Verify no `METRIC_PLAYED` double-logging. |

## Change-area A — Outcome semantics: reachable LOSE (Items 1, 2)

All three masters:
- Add `self.lost = False` in `_on_setup`.
- `_does_game_proceed`: guard `if self.success or self.aborted or self.lost: return False`; the `len(self.move_log) >= self.max_turns` branch sets `self.lost = True` + `log_to_self("lose", …)` (was `self.aborted = True`).
- `_on_after_game`: `log_key(METRIC_LOSE, int(self.lost))` (was derived `int(not success and not aborted)`).

Scorers (all three):
- Read `METRIC_LOSE` from the logged episode value instead of recomputing it.
- `BENCH_SCORE` logic is already correct: `aborted → nan`, `success → efficiency score`, else → `0`. Timeouts now fall into the `0` branch; `PLAYED = 1 - aborted` counts them.

Terminal-state audit: after this change an episode ends only via `success`, `aborted` (bad format / illegal / false done / cycle-repeat), or `lost` (timeout) — mutually exclusive.

**Item 2:** cycle-abort left as-is (ABORTED). **Accepted tradeoff:** Oscillation Flag (k=4) and Deadlock Flag (k=6) rarely fire because the k=3 abort pre-empts them. The thesis text for the oscillation/deadlock family and for Item 14.3 must state that oscillation is *terminated at k=3*, not measured to completion.

## Change-area B — Coordinate validation (Item 5a)

`Board.validate_move` (all three `utils/board.py` copies):
- New optional parameter `declared_target: Optional[Tuple[int, int]] = None` (0-based, caller-converted).
- After computing the destination cell and passing the `off_grid` check, and **before** the `wall`/`occupied` checks: if `declared_target is not None and declared_target != (new_row, new_col)` → return `("coordinate_mismatch", <message>)`.
- `coordinate_mismatch` is an **illegal** code (indicts the mover), not `occupied`.

All three masters, `_validate_player_response`:
- Parse `target_row` / `target_col` from the grammar (1-based), convert to 0-based, pass into `validate_move`.
- The existing error handler already splits `occupied → blocked_attempts` vs. else `→ illegal_attempts`; `coordinate_mismatch` lands in `illegal_attempts` and flows through the existing reprompt / abort-on-exhaustion path.

Convention note: compact rendering and the grammar use 1-based `R<row>,C<col>`; the board stores 0-based; wall messages print `R{row+1},C{col+1}`. Caller converts declared 1-based → 0-based before comparison.

## Change-area C — Masking: hint flag + narration removal (Items 12a, 12b, 13)

**12a — `masking_hints` flag** (both cooperative masters):
- Added to `common_config.json`; copied onto each instance by the generators; read in `_on_setup` as `self.masking_hints = game_instance.get("masking_hints", True)`. Default **ON** (preserves current behavior).
- Gates only the *hint text* at the two redirect sites: the unparseable `move: X@R3C4` case and the parseable-but-not-owned `move: X` case (`obj == BLOCKED_MARKER`).
  - ON → targeted semantic message ("'X' marks your partner's objects — you can only move: …").
  - OFF → generic invalid-move / invalid-format message.
- **Counting untouched:** a detected masked attempt still logs `invalid_move` exactly as today, in both flag states. The flag never changes what is counted.

**12b — remove all move narration** (both cooperative masters; solo if present):
- Delete the partner-facing narration ("`Player A moved a blocked object down`" / "`… moved A down`") injected into the other player's next-turn context.
- Delete the self-confirmation ("`You moved A down`") shown back to the mover.
- `nplayer`: `_build_broadcast_context` drops its `move_desc` parameter and the `f"{move_desc}\n\n"` line.
- Recipients and movers read moves off the board (CURRENT BOARD / GOAL BOARD) only.
- Done-notifications ("Player A has finished — all their objects are at their target positions") are **kept** — they are not move narration.

**13 — persistent bulletin kept** (no code): each player's most recent message stays posted in teammates' contexts until replaced (the `last_message` digest in `_build_broadcast_context` stays). Documented in thesis as the n-player generalization; thesis also notes the dyad uses one-shot relay.

## Change-area D — Prompt templates (Items 14.1, 14.3)

All templates across the three packages:
- **14.1:** rewrite the GRID / board-description section to describe the actual compact listing (`A@R1C1`, `Walls: R4C4 …`, `Targets: A->R3C5`). Delete ASCII `##`-wall and dot-for-empty descriptions and the "board only shown at start" claim (compact CURRENT/GOAL boards appear every turn and every reprompt).
- **14.3:** delete the "AVOIDING DEADLOCKS" coaching paragraph. Intended effect: cleaner separation between apparatus and measured behavior; expect higher abort/loop rates.

## Change-area E — Mechanical cleanups (Items 14.2, 14.6, 14.8)

- **14.2:** unify `max_turns` default to `max(4 * optimal, 20)` in the comm and nplayer masters (solo already uses it). Dormant (all shipped instances set `max_turns`) but removes the divergence trap.
- **14.6:** remove `matrixgame_covered_base` (`git rm -r`) and `matrixgame_covered_multi` (untracked; delete). Precondition: grep confirms nothing imports them or registers them via a `clemgame.json`.
- **14.8:** after regeneration, run one episode through the eval pipeline and confirm no duplicated / conflicting `METRIC_PLAYED` key in the scores output.

## Regeneration

Grammar feedback (5a), the `masking_hints` flag, and template text (14.1, 14.3) all change instance content (prompts are baked into `instances.json`). Regenerate all instances at the end: **seed 42, n=8, all packages**. Old result directories become incomparable and must not be mixed with new ones.

## Testing strategy

Per-package `pytest`, using each package's existing `tests/conftest.py` sys.path pattern.

- **Outcome (each master):** tiny-`max_turns` instance + legal moves to exhaustion → assert `lost is True`, `aborted is False`; scorer emits `BENCH_SCORE == 0` and `PLAYED == 1`.
- **Coordinate:** `board.validate_move` returns `coordinate_mismatch` on a wrong declared cell and `None` on the correct one; master routes a wrong-coordinate move through the illegal + reprompt path.
- **Masking flag:** ON → hint text present in the reprompt context; OFF → generic text; **both** still log `invalid_move` (counting unchanged).
- **Narration:** after a move, the recipient's next-turn context contains no move-narration line; update existing smoke tests that asserted the old narration strings.

## Sequencing

A → B → C → D → E → regenerate instances → full test suite + 14.8 check.

## Thesis-text edits (no code — appendix for the LaTeX)

- 5b: add the channel-off `done:` production to `fig:task-grammar`.
- 5c: state the actual accepted PLAN/REQUEST separator (`[—\-]+`) consistently with the figure.
- 14.4: note strict-mode is disabled in the shipped experiments (or drop it from the abort list).
- Document decisions: LOSE outcome (Item 1); cycle-abort as a documented ABORTED cause and that oscillation is terminated at k=3 (Item 2 / 14.3); coordinate now validated (5a); `masking_hints` flag and removal of move narration (12); persistent-bulletin n-player comm + dyad one-shot relay difference (13); compact-listing board description (14.1).

## Constraints

- Commit messages: short imperative style; **no `Co-Authored-By: Claude` trailer** (repo rule).
- No new third-party dependencies; stdlib + numpy only.
- `master.py` files must not import across packages at runtime.
