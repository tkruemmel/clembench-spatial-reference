# matrixgame_covered_comm_nplayer — Design

**Date:** 2026-05-24
**Status:** Approved (pending user spec review)
**Parent game:** `matrixgame_covered_comm` (2-player collaborative grid rearrangement with comm protocols)

## Goal

Add a new game directory `matrixgame_covered_comm_nplayer/` that hosts two clemgame entries:

- `matrixgame_covered_comm_3p` — 3 players, each owning 2 of 6 objects
- `matrixgame_covered_comm_6p` — 6 players, each owning 1 of 6 objects

Both modes share the same `master.py` and `instancegenerator.py`. The compact board representation is the default and only rendering mode. All four communication protocols from `comm/` (`none`, `structured`, `freeform`, `hybrid`) are supported, with messages broadcast to all other active players. All four spatial levels (S1–S4) are supported, giving 32 variants total.

## Non-goals

- No `easy_mode` (object-status / adjacency hints).
- No `enforce_isolation` generator check.
- No non-compact (verbose) rendering mode.
- No modification of the existing `matrixgame_covered_comm/`, `_solo/`, or `_multi/` directories.
- No new spatial levels.

## Directory layout

```
matrixgame_covered_comm_nplayer/
├── __init__.py
├── clemgame.json
├── instancegenerator.py
├── master.py
├── resources/
│   ├── common_config.json
│   ├── config.json
│   └── initial_prompts/en/
│       ├── player_prompt_none.template
│       ├── player_prompt_structured.template
│       ├── player_prompt_freeform.template
│       └── player_prompt_hybrid.template
├── utils/
│   ├── __init__.py
│   ├── astar.py            # copied verbatim from matrixgame_covered_comm/
│   └── board.py            # copied verbatim from matrixgame_covered_comm/
├── tests/
│   └── test_smoke.py
└── in/
    └── instances.json      # generated
```

Every file is self-contained. No imports from `matrixgame_covered_comm`. This matches the existing repo convention where each game directory is independent.

## `clemgame.json`

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

Both entries are mirrored in `/Users/tom/uni/clembench-spatial-reference/game_registry.json` with `"game_path": "matrixgame_covered_comm_nplayer"`.

## Configuration

### `resources/common_config.json`

```json
{
  "grid_size": 8,
  "objects": ["A", "B", "C", "D", "E", "F"],
  "directions": ["up", "down", "left", "right"],
  "max_retries": 2,
  "strict": false
}
```

Identical to `comm/common_config.json`.

### `resources/config.json` — 32 variants

Cartesian product of `{S1, S2, S3, S4} × {none, structured, freeform, hybrid} × {3p, 6p}`.

Each entry has this shape (example):

```json
"s2_structured_6p": {
  "name": "S2_Structured_6p",
  "num_players": 6,
  "spatial_level": "S2",
  "comm_protocol": "structured",
  "use_masking": true,
  "compact_board": true
}
```

Keys:

- `name` — human-readable label, used as the experiment name suffix.
- `num_players` — `3` or `6`. Sole knob differentiating the two modes.
- `spatial_level` — `S1`, `S2`, `S3`, or `S4`. Same semantics as `comm/`.
- `comm_protocol` — `none`, `structured`, `freeform`, or `hybrid`. Same semantics as `comm/`.
- `use_masking` — always `true` (own objects shown by letter; others as `X`).
- `compact_board` — always `true`. Set for traceability; the master always renders compact regardless.

No `easy_mode`, no `enforce_isolation`. No `_compact` suffix on variant keys — compact is the only mode.

## Instance generation

### Reused spatial-level functions

`comm/instancegenerator.py` contains four spatial-level generators returning `(starts, targets, walls, optimal, player_a_objects, player_b_objects)`:

- `_generate_s1` — region-partitioned placement, no walls
- `_generate_s2` — length-2 dependency chain + corridor carving to ~1.5× Manhattan
- `_generate_s3` — dead-end corridor + length-3 chain + length-1 chain (~2× Manhattan)
- `_generate_s4` — S3 walls + INNER/OUTER corridor non-monotonic trajectory

These functions, plus their helpers (`_bfs_path`, `_pick`, `_partition_regions`, `_split_range`, `_carve_for_ratio`, `_is_2connected`, `_new_wall_creates_dead_end`, `_verify_reachability`, and any S2/S3/S4-local helpers), are **copied verbatim** into `matrixgame_covered_comm_nplayer/instancegenerator.py`. No behavior change.

The 2-player split (`player_a_objects`, `player_b_objects`) returned by these functions is **discarded**. We re-partition externally.

### N-way player partition (new)

```python
num_players = variant_config["num_players"]              # 3 or 6
objects = common_config["objects"]                        # ["A","B","C","D","E","F"]
assert len(objects) % num_players == 0
objs_per_player = len(objects) // num_players             # 2 or 1
player_objects = [
    objects[i*objs_per_player : (i+1)*objs_per_player]
    for i in range(num_players)
]
# 3p: [["A","B"], ["C","D"], ["E","F"]]
# 6p: [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]
```

This static-by-index assignment is deterministic and chain-agnostic. Because S2/S3/S4 build dependency chains via the alphabetic ordering (A←B in S2; A←B←C←D and E←F in S3), every chain edge crosses a player boundary in both modes. No special routing is required.

### Instance schema

Each `game_instance` carries:

```python
game_instance["grid_size"] = 8
game_instance["walls"] = walls                       # List[Tuple[int,int]]
game_instance["objects"] = ["A","B","C","D","E","F"]
game_instance["num_players"] = num_players           # 3 or 6
game_instance["player_objects"] = player_objects     # List[List[str]]
game_instance["start_positions"] = starts            # Dict[str, [row, col]]
game_instance["target_positions"] = targets          # Dict[str, [row, col]]
game_instance["optimal_moves"] = optimal             # int (sum of per-object BFS lengths)
game_instance["max_turns"] = max(4 * optimal, 20)
game_instance["max_retries"] = 2
game_instance["strict"] = False
game_instance["use_masking"] = True
game_instance["spatial_level"] = "S2"
game_instance["comm_protocol"] = "structured"
game_instance["compact_board"] = True
game_instance["player_prompt"] = prompt_template     # filled in below
```

The old `player_a_objects` and `player_b_objects` keys are replaced by the single `player_objects` list-of-lists.

### Experiment naming

`f"{spatial_level}_{comm_protocol}_{num_players}p"`, e.g.:

- `S1_none_3p`
- `S2_structured_6p`
- `S4_hybrid_6p`

## Master loop

`master.py` is a generalization of `comm/master.py`. Two-player-specific structures (`player_a`, `player_b`, `player_a_objects`, `player_b_objects`, `player_a_done`, `player_b_done`) are replaced by indexed lists.

### Setup (`_on_setup`)

```python
self.num_players: int = game_instance["num_players"]
self.player_objects: List[Set[str]] = [set(o) for o in game_instance["player_objects"]]
self.player_done: List[bool] = [False] * self.num_players
self.last_message: Dict[int, Optional[str]] = {i: None for i in range(self.num_players)}
```

Roles are `["Player A", "Player B", ...]` sliced to `num_players` letters.

Model assignment: clembench supplies a list of `Model` objects whose length matches the `players:` field of `clemgame.json`. If only one model is supplied (single-model self-play), all `num_players` players use it — mirrors the comm/ fallback at `comm/master.py:175`.

For each player `i`:

1. Construct a `MatrixPlayer(model_i, role_name=f"Player {chr(65+i)}", own_objects=self.player_objects[i])`.
2. Build the initial context via `_build_initial_context(prompt_template, own_objects_str, player_i)`.
3. Add to the master with `self.add_player(player, initial_context=ctx)`.

All N initial contexts are built and delivered upfront, in `_on_setup`. (The 2-player game's lazy Player-B initial-context delivery in `comm/master.py:533-536` is removed — it does not generalize cleanly and the upfront delivery is simpler.)

The `Board` instance is built once with the wall set and seeded with `start_positions` and `target_positions`, identical to comm/.

### Per-player rendering

`_render_for(player)` and `_render_targets_for(player)` look up the player's owned objects from `self.player_objects[i]` (resolving player → index via a `Dict[Player, int]` built in setup, or via the `MatrixPlayer.own_objects` attribute already on the player). Both always invoke `board.render_compact(visible_objects=...)` and `board.render_targets_compact(visible_objects=...)`. Masking rule unchanged: own objects show their letter, others show `X`.

### Turn rotation

`_next_player` already iterates `self.get_players()` and skips done players. It is N-player-correct without modification. The only change is removing the `player_a` / `player_b` aliases and looking up done status via the `player_done[i]` list keyed by player index.

### Broadcast comm flow

After a player makes a valid move (or declares done), the master constructs each other not-yet-done player's next context as:

```
[Player X says: <message X's last turn>]
[Player Y says: <message Y's last turn>]
...

<move_desc>

CURRENT BOARD:
<masked compact render>

GOAL BOARD:
<masked compact target render>

Your turn. Respond with:
<format_reminder>
```

Where:

- The bracketed `[Player … says:]` lines are drawn from `self.last_message`. We include the latest cached message from every other player whose `last_message` slot is non-None. The cache is per-author, **not** per-(author, recipient): when an author speaks, their slot is overwritten; the message stays visible to every other player until the author speaks again. A recipient who takes two turns in a row without the author speaking will see the same message both times — acceptable, mirrors "you remember what your teammate last said". For the `none` protocol, all slots stay `None` for the entire game (no `message:` line is ever produced or shown).
- `<move_desc>` is one line: `"Player <Z> moved <OBJ> <DIRECTION>."` If `use_masking` is on and the moved object is not in the recipient's owned set, the description is masked to `"Player <Z> moved a blocked object <DIRECTION>."`
- The board view and goal view are rendered for the recipient's masking.
- `<format_reminder>` mirrors `comm/`'s `_format_reminder()`: includes `message:` only when `comm_protocol != "none"`.

For the `none` protocol, the bracketed message lines are simply absent (all `last_message` slots stay `None`).

### Done declarations

When a player declares done:

1. Validate that all their owned objects are at their targets. If not, abort the game (same rule as `comm/`).
2. Mark `player_done[i] = True`.
3. Append the player's done message (if any, for non-`none` protocols) to their `last_message` slot so other players see the farewell line on their next turn.
4. If the board is fully solved, set `self.success = True` and end.
5. Otherwise, broadcast: each remaining active player's next context includes `"Player X has finished — all their objects are at their target positions. Continue without them."` alongside the board view.

If all active players declare done before the board is fully solved, the game ends in a lose state (success = False, aborted = False).

### Validation, reprompting, cycle detection

Unchanged from `comm/`:

- Parse format: `_parse_response` accepts the same `MOVE_ONLY_PATTERN`, `COMM_MOVE_PATTERN`, `DONE_ONLY_PATTERN`, `COMM_DONE_PATTERN`.
- `STRUCTURED_MSG_PATTERN` warning (not abort) on malformed structured messages.
- `max_retries = 2` on invalid format or invalid move.
- Board-state-repeated-3-times cycle abort.
- `max_turns = max(4 * optimal_moves, 20)` cap, baked into each instance.

### Scorer

`MatrixGameScorer` is copied verbatim from `comm/`. Computes:

- `BENCH_SCORE = min(100, round(optimal_moves / move_count * 100, 2))` on success.
- `BENCH_SCORE = 0` on lose.
- `BENCH_SCORE = NaN` on abort.

Per-mode comparison (3p vs 6p) is performed externally by aggregating over the experiment names.

### Benchmark entry

`MatrixGameBenchmark(GameBenchmark)` defines `create_game_master` and `create_game_scorer` identically to `comm/`. Both clemgame entries (`_3p` and `_6p`) dispatch to this single class via the shared `main_game: matrixgame_covered_comm_nplayer` setting.

## Prompt templates

Four templates, one per protocol, in `resources/initial_prompts/en/`. Each is copied from the corresponding `comm/` template and edited for the N-player broadcast setting.

### Substitutions filled in by the master

| Variable | Value |
|----------|-------|
| `$YOUR_OBJECTS$` | comma-joined sorted list of the player's owned objects (existing) |
| `$GRID_SIZE$` | `"8"` (existing) |
| `$YOUR_ROLE$` | `"Player A"` … `"Player F"` — **new** |
| `$N_OTHER_PLAYERS$` | `"2"` (in 3p mode) or `"5"` (in 6p mode) — **new** |

### Text changes vs. `comm/` templates

- `"with a partner"` → `"with your teammates"`.
- `"Your partner controls the remaining objects (shown as X on your board)."` → `"Your $N_OTHER_PLAYERS$ teammates control the remaining objects (shown as X on your board)."`.
- The `message:` guidance line (in `freeform`, `hybrid`, `structured`) gains: `"Your message is broadcast to all your teammates."`
- `"ask your partner to clear the way"` → `"ask your teammates to clear the way"`.
- The `structured` template's PLAN/REQUEST format is unchanged — REQUEST is now naturally interpreted as a broadcast request to whoever can help.
- A new line near the top of each template states: `"You are $YOUR_ROLE$."`

The response-format block (the `reason:` / `move:` / `done:` lines and their `message:` variants) is unchanged in shape.

## Smoke test

`tests/test_smoke.py` runs the instance generator on a small subset of variants and asserts:

1. Every `game_instance` has `num_players in {3, 6}`.
2. `len(game_instance["player_objects"]) == num_players`.
3. `sum(len(p) for p in game_instance["player_objects"]) == 6`.
4. Every object in `objects` appears in exactly one player's set (no overlaps, no gaps).
5. `compact_board == True` on every instance.
6. `start_positions` and `target_positions` each have entries for all 6 objects.

The test does not invoke the master loop; it only validates the generator's output shape. (Following the precedent of `matrixgame_covered_solo/tests/`.)

## Migration / coexistence

- Existing `matrixgame_covered_comm/` is unmodified.
- Existing `matrixgame_covered_multi/` (a stale uncommitted prototype) is unmodified.
- Existing `game_registry.json` gains two new entries; no existing entries change.
- No shared state, no shared imports between the new game and existing games.

## Open questions

None. All edge-case behaviors (all-done-but-unsolved → lose; single-model self-play for N=6; not pushing context to done players) are specified in the body above.
