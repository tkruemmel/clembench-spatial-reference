# Human play via web — design

Date: 2026-09-01
Status: approved (design)

## Goal

Let humans play the covered matrix games (solo, dyad, 3p, 6p) through a web
interface, and log the results so they enter the existing analysis as just
another model. The primary study is **six friends across multiple afternoons**;
a crowd platform (Prolific) is a preserved fallback, not the plan.

The design constraint that shapes everything: **no existing machinery is
modified.** Nothing under `matrixgame_covered_*`, `matrixgame_covered_shared/`,
or `clemcore` is edited. The whole build is two new things — `humanweb_api.py`
at the repo root and a `humanweb/` service package.

## Scope decisions (agreed)

1. **All four suites**: solo, dyad, 3p, 6p.
2. **All-human seats.** No mixed human/model episodes in the primary design.
   (Model substitution survives only as a Prolific-mode fallback; see §6.)
3. **Verbatim text stimulus.** Participants read exactly the string the model
   receives — `Objects: A@R7C3 …` plus the flat `Walls: …` line — and type
   `move: A down`. No rendered grid.
4. **Turn budget untouched.** Human episodes run the canonical `max_turns`
   (4× optimal), against byte-identical instance files.
5. **Multiple afternoons**, resumed via the existing episode-level resume
   signal.
6. **Lab-first, Prolific-capable.** Recruitment lives behind a swappable
   `lobby/` module; the core never learns how a participant arrived.

## Why a backend, not a browser reimplementation

`clemcore.backends.BackendRegistry.from_packaged_and_cwd_files()` discovers any
`*_api.py` in the cwd. This repo already ships six such custom backends
(`deepseek_api.py`, `amazon_bedrock_api.py`, …), so `humanweb_api.py` at the
root becomes the `humanweb` backend with no framework changes.

The pattern is already proven inside clemcore: `backends/slurk_api.py:127`
defines `SlurkModel`, a `Model` whose `generate_response()` blocks on a socket
until a human types. Its own comment reads *"todo: make this HumanModel when
HumanModel is fully integrated as a backend."*

Consequence: `master.py`, board rendering, move validation, `interactions.json`,
`clem score`, `clem transcribe`, and every existing analysis script work
byte-identically for humans and models.

The rejected alternative — reimplementing the game in TypeScript and
replay-scoring offline — would duplicate `Board.validate_move` and the
reprompt/abort state machine in a second language. Any drift becomes a confound
in precisely the human-vs-model comparison the study exists to make. The UX
argument for a native client also evaporated with decision 3: there is no rich
grid to justify it.

`clem run -g matrixgame_covered_solo -m human` already works today via terminal
input (`clemcore/clemgame/player.py:194`). The website's job is multi-seat play
and participant-facing usability, not capability.

## 1. Components

Five units, each independently testable.

| Unit | Responsibility | Depends on |
|---|---|---|
| `humanweb_api.py` (repo root) | A `Backend` + `Model` pair. `generate_response()` blocks on the session service and returns the participant's raw text. Nothing else. | HTTP client only |
| `humanweb/service` | Turn exchange: holds one pending context per seat, accepts one submission per seat. | Redis or SQLite for seat state |
| `humanweb/lobby` | `LabLobby` (named seats, no gate) today; `ProlificLobby` later. | participant store |
| `humanweb/console` | Outstanding-episode view, cell selection, seat rotation, stop-time guard; spawns and reaps one `clem run` per episode block. | filesystem, `clem` CLI |
| `humanweb/web` | Monospace context block, text input, turn indicator. **No game logic.** | the service's HTTP API |

The load-bearing property: the web tier never knows the rules. It renders a
string the Python game master produced and posts back a string the game master
parses. Reprompts, illegal-move messages and aborts flow through the same
channel, so the participant sees exactly what a model would — including
`"Your response could not be parsed…"` (`matrixgame_covered_solo/master.py:256`).

## 2. Turn protocol and termination

### Seat binding

`load_models` (`clemcore/backends/__init__.py:83-90`) constructs a **fresh
Backend and a fresh Model per selector**, so `-m humanweb humanweb humanweb
humanweb humanweb humanweb` yields six distinct seat objects, and
`matrixgame_covered_comm_nplayer/master.py:289` hands seat *i* to
`player_models[i]`.

Because the Backend instance is not reused, seat assignment **must live in
module-level state**, not on the Backend. A module counter hands out
`seat_id = 0,1,2…` in `get_model_for()` call order, matching `Player A…F`. The
broker passes `HUMANWEB_SERVICE_URL` and `HUMANWEB_SESSION_ID` in the subprocess
environment.

All seats register under the model name `human-text`. This matters: `Model
.to_identifier` (`backends/model_registry.py:421`) builds the results run
directory as `"--".join(names)` for ≤2 seats and `group-{n}p-{sha1[:8]}` above
that. Per-participant model names would give **every session its own top-level
results root**, fragmenting the tree into hundreds of one-episode folders that
`clem score` cannot group. One shared name keeps a single stable tree per suite.

### Wire protocol

- `POST /internal/turn` — `{session, seat, context, pinned}`; parks the turn.
- `GET /internal/turn/{session}/{seat}` — long-poll; returns submitted text, or
  `410 Gone` on abandonment.
- `POST /internal/episode/{session}/end` — clears parked state between episodes.

The participant-facing side uses a websocket for liveness; only the internal
side blocks.

### What the participant sees

`perceive_context` returns `get_perspective() + [context]`
(`clemcore/clemgame/player.py:134`), so with the Markovian preamble a seat
receives exactly `[rules, current_context]`. `messages[0]` renders once in a
collapsible rules panel; `messages[-1]["content"]` is the monospace block that
refreshes each turn. Nothing is summarised or reformatted.

### Termination

| Situation | Mechanism | Result |
|---|---|---|
| Slow but present | UI countdown only (soft, advisory) | plays on |
| Turn budget exhausted | existing `max_turns` from the canonical instance | **Lose** |
| Stop for the day / seat abandoned | exception from `generate_response`, caught by `clemcore/clemgame/runners/sequential.py:42` | no `interactions.json` → episode replayed |
| Parse / illegal-move exhaustion | existing reprompt → abort path | **Aborted** |

In lab mode there is **no hard deadline**: friends who step away just make
everyone wait, and the long-poll waits indefinitely.

**Anti-pattern, stated explicitly.** Do not implement a timeout by returning a
garbage string and letting the game master's reprompt path absorb it. That
inflates `Violated Request Count` and yields `Aborted=1`, which in the existing
analyses means "the agent could not follow the format". Human idleness would
masquerade as human incompetence and contaminate every abort-rate comparison.

## 3. Multiple afternoons

### Resume is already solved

`campaign/remaining.py` states the property: *"an episode killed mid-flight
(network drop, laptop sleep, SIGKILL) leaves nothing behind and is replayed,
while an episode the game legitimately aborted keeps its result."* The resume
predicate is one line:

```python
(game_root / experiment_name / f"instance_{game_id:05d}" / "interactions.json").exists()
```

**"Stop for the day" is the same mechanism as abandonment**: SIGTERM the `clem`
subprocess, and the in-flight episode reappears in tomorrow's outstanding set.
Nothing persists between afternoons except the results tree.

Because the turn budget is untouched, each afternoon runs against a filtered
copy of the **canonical** `instances.json`. Every human episode's
`instance.json` is byte-identical to the model episode's for the same board —
pairing is a `diff`, not an argument.

### Stop-time guard

At 4× budget a 6p S4 episode reaches 232 turns, roughly 90 minutes. Killing one
wastes an afternoon's final block. The console therefore **refuses to start an
episode that cannot finish before the declared stop time**, estimating from
`max_turns` × the per-turn time it has measured from prior timestamps.

### Level interleaving (deviation from `remaining.py`)

`remaining.py` sorts outstanding episodes S1→S4, which is right for model
campaigns and **wrong here**: the friends would play S1 untrained and S4 after
twenty afternoons, perfectly confounding spatial level with expertise. The
console does its own selection using the same one-line predicate, but
**interleaves levels** so practice spreads evenly across S1–S4.
`remaining.py` stays untouched and remains usable for a status count.

### Counterbalancing

- **Seat rotation** by Latin square across afternoons. Who sits at Player A vs
  Player F matters — see `project_3p_below_6p` (per-agent object load).
- **Session index and cumulative episode count** per participant, so the
  learning curve is modelled rather than assumed away.
- **A seventh trained participant** rotating in from the first afternoon. One
  permanent departure otherwise strands the 6p suite, which is 52% of the work.

## 4. Data contract

### Layout

```
results_human/
  human-text/                matrixgame_covered_solo/     Solo_S1_Silent_Compact/instance_00005/
  human-text--human-text/    matrixgame_covered_comm/     S1_freeform/instance_00005/
  group-3p-<hash>/           matrixgame_covered_comm_3p/  S1_freeform_3p/instance_00005/
  group-6p-<hash>/           matrixgame_covered_comm_6p/  S1_freeform_6p/instance_00005/
  _sessions/<sid>/{manifest.json, turns.jsonl}
```

Each episode directory holds the usual `instance.json`, `interactions.json`,
`scores.json`, `transcript.html`. `_sessions/` mirrors the `_campaign/`
convention the existing result trees already use, so `clem score` walks past it.

### One field differs

`players["Player 1"].model_name` reads `human-text` instead of a model name.
Everything else is produced by the same untouched code. `clem score -r
results_human` emits `scores.json` with identical keys, so the human tree enters
the existing analysis as another model with no adapter.

### Pairing

`(game_name, experiment_name, game_id)` pairs a human episode to its model
counterpart. `(spatial_level, game_id)` pairs **across suites**: all 60 boards
(4 levels × 15) are byte-identical between solo, dyad, 3p and 6p — verified
60/60 by comparing `start_positions`, `target_positions` and `walls` across the
four instance files.

### Latency

Within a turn, all events are stamped at flush time (~0.1 ms apart), so
within-turn deltas are meaningless. The delta between **consecutive turn blocks**
is real deliberation time. Measured on a DeepSeek solo episode: median 6.5 s,
mean 18.1 s, max 106 s. The same computation on human episodes yields a directly
comparable per-decision time on the same boards.

Caveat: a human inter-block delta includes idle time — coffee is
indistinguishable from thought. `turns.jsonl` records page visibility and first
keystroke to separate them. Without that, report medians, not means: one
twenty-minute break destroys a mean.

### What cannot ride in `interactions.json`

Because the machinery stays untouched:

| datum | location |
|---|---|
| seat → participant pseudonym | `_sessions/<sid>/manifest.json` |
| afternoon index, episode order, seat rotation | manifest |
| keystroke timing, page-visible vs idle | `turns.jsonl` (append-only) |
| real identity / Prolific ID | outside the repo entirely |

The service never writes into an episode directory mid-run. It appends to
`turns.jsonl`; the join happens afterwards on `(session, seat, timestamp)`
against the turn blocks in `interactions.json`. Nothing to coordinate, and it
survives a SIGTERM stop.

### Participant free text

Under freeform/hybrid/structured protocols, whatever participants type lands in
`interactions.json` verbatim. Pseudonyms only in the tree. Agree before the
first afternoon that these transcripts may appear in the thesis.

## 5. Cost

Full suite at n=3 (168 episodes = 56 cells × 3 boards), six people running as
many concurrent sessions as each suite allows (6 solo / 3 dyads / 2 triads /
1 hexad), S1 exiting early on success while S2–S4 burn the full budget, plus
20% for setup and breaks:

| turn budget | @15 s/turn | @25 s/turn | @35 s/turn |
|---|---|---|---|
| 1.5× optimal | 28 h | 46 h | 65 h |
| 2× optimal | 36 h | 59 h | 83 h |
| **4× optimal (chosen)** | **67 h** | **112 h** | **157 h** |

At the chosen budget and a central 25 s/turn: **112 hours ≈ 28 afternoons.**
Expect less in practice — per-turn time should fall substantially with
repetition, and that learning curve is itself measurable from the timestamps.

Distribution at the chosen 4× budget and 25 s/turn, showing where the time goes
(hours before the 20% overhead):

| suite | episodes | hours | share | note |
|---|---|---|---|---|
| solo | 24 | 4.1 | 4% | six play at once — 6× speedup |
| dyad | 48 | 16.2 | 17% | three pairs at once |
| 3p | 48 | 24.3 | 26% | two triads at once |
| **6p** | 48 | **48.6** | **52%** | one group, **zero parallelism** |

Two structural drivers, both from existing code: six friends *are* one 6p group,
so that suite gets no speedup and accounts for **52%** of the total; and per
`campaign/remaining.py`, *"only S1 episodes exit early on success; S2+ run their
full 4x-optimal turn budget"*, so three of four levels are always full length.

**Unmeasured parameter.** The 15–35 s/turn spread swings the answer between 67 h
and 157 h. Measure it before scheduling, with the terminal path that already
works:

```
clem run -g matrixgame_covered_solo -m human -i instances -e Solo_S1_Silent_Compact
```

## 6. Prolific fallback

Recruitment stays out of the core. `humanweb_api.py`, the turn service and the
play UI never learn how a participant arrived. `ProlificLobby` adds consent,
demographics, a comprehension gate, a waiting room with over-recruitment, and
completion codes. Two mechanisms live there rather than in the core:

- **Hard deadline** (e.g. 180 s) marking a seat abandoned.
- **Seat substitution.** Because the perspective is Markovian and the board is
  fully re-described every turn, a fresh participant can take over an abandoned
  seat **with zero information loss**. Human hot-spares are therefore preferred;
  `HumanWebModel` delegating to a `backends.load_models([...])` fallback is the
  last resort, and any episode after a model substitution is flagged and
  excluded from the all-human corpus.

## 7. Testing

### The test that carries the comparability claim

Run one instance twice — through `CustomResponseModel` and through `humanweb`
driven by a scripted client posting the same moves — and assert the two
`interactions.json` files are identical with `timestamp` and `model_name`
masked:

```
test_human_episode_matches_programmatic_episode_structure()
```

It belongs in build step 1, before any UI exists.

### The stimulus-fidelity test

Assert the bytes rendered to the participant equal `messages[-1]["content"]`
exactly. `Board._render_raw` pads columns with runs of spaces, and HTML
collapses those unless the container sets `white-space: pre`. A silently
reflowed board is a different stimulus and would invalidate every comparison.

### Remaining tests

| file | covers |
|---|---|
| `test_seats.py` | N `get_model_for` calls → seats 0…N-1 despite a fresh Backend per call |
| `test_backend.py` | response returned verbatim, no stripping; abandonment raises so `sequential.py:42` catches it |
| `test_service.py` | one submission per seat per turn; late submission after abandonment → 410; episode boundary clears state |
| `test_console.py` | level interleaving, Latin-square rotation, stop-time guard — pure functions |
| `test_resume.py` | SIGTERM mid-episode → no `interactions.json` → episode reappears in the outstanding set |

## 8. Build order

0. **Measure per-turn time** with `clem run -m human`. No code. Sizes the study.
1. **Backend + headless service.** Driven by pytest, no UI. Ends with a solo
   episode producing a scorable results dir and the structural-identity test
   green — the science proven before any CSS.
2. **Play UI.** Ends with a browser-played S1 episode, and a re-measurement of
   per-turn time in the real interface.
3. **Multi-seat.** Six links, turn indicator. Ends with a 6p episode across six
   tabs on one machine.
4. **Console.** Outstanding episodes, interleaving, rotation, stop-time guard,
   spawn/kill. Ends with the ability to run an afternoon.
5. **`turns.jsonl` + manifest + join script.** Ends with the linked dataset.
6. **`ProlificLobby`** — optional, unblocked by everything above.

**Schedule, do not build:** a dress-rehearsal afternoon on S1 only with all six
friends, before committing to the full plan. It surfaces what no test catches —
real tutorial length, whether people can read `Label@RrowCcol` under time
pressure, how badly one slow player drags a 6p episode — while a bad answer
still costs one afternoon instead of twelve.

## Open questions

1. **Cell coverage.** 28 afternoons assumes the full n=3 crossing of all 56
   cells. If the schedule slips, which cells get cut is a research decision, not
   an engineering one. The findings most worth protecting are the solo-vs-6p
   coordination gap (`project_coordination_not_spatial`) and the S1 protocol
   ordering (`project_protocol_ordering_finding`).
2. **Solo thinking axis.** Solo has a Thinking/Silent split where the multiplayer
   suites have four protocols. Forcing a written note every turn roughly doubles
   solo per-turn cost. Keep both arms, or run Silent only?
3. **Tutorial content** is unspecified beyond a comprehension gate.
