# Enforced Hybrid Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `hybrid` communication protocol a genuine blend of `freeform` and `structured` — a free natural-language message that MUST end with the structured `PLAN: move … — REQUEST: …` coordinate clause, with the clause enforced by reprompt/abort exactly like the structured protocol.

**Architecture:** Add a `HYBRID_MSG_PATTERN` (anchored: prose then the structured clause) to each package's `master.py`, and replace the single structured-only enforcement `if` with a small `{protocol: (pattern, reminder)}` table so `structured` behaves byte-for-byte as today and `hybrid` reuses the same reprompt→abort path and `violated_request_counts` metric. Rewrite both `player_prompt_hybrid.template` files to instruct the combined format. Because each `game_instance` in `in/instances.json` bakes in a fully-rendered `player_prompt`, regenerate instances for both packages so the `hybrid` instances pick up the new prompt text (boards must stay byte-identical — same seed 42).

**Tech Stack:** Python 3, `re`, pytest. clembench legacy `DialogueGameMaster`.

## Global Constraints

- Base seed **42**, **8 instances per condition** (`--n 8`, the generator default), for both packages — verbatim from existing generators.
- Grid 8×8, objects `A–F`, `max_retries = 2` — unchanged.
- The `structured` protocol's runtime behavior, reprompt text, log strings, and metrics MUST remain byte-for-byte identical after the refactor.
- `hybrid` message enforcement MUST reuse the existing `violated_request_counts` / `reprompt_attempts` / `aborted` machinery — no new metric fields.
- Done declarations stay exempt from schema enforcement (handled earlier in `_validate_player_response`).
- The em-dash character in the PLAN/REQUEST clause is `—` (U+2014); the regexes accept `[—\-]+` (em-dash or hyphen). Templates use `—`.
- Run each package's tests from that package's `tests/` dir (its `conftest.py` inserts the package root onto `sys.path`).
- Only the `hybrid` `player_prompt` text may change in `instances.json`; `none`/`structured`/`freeform` instances and all board/wall/start/target data MUST be byte-identical to before.

---

### Task 1: Enforced hybrid in `matrixgame_covered_comm` (dyad)

**Files:**
- Modify: `matrixgame_covered_comm/master.py` (patterns block after line 101; enforcement block lines 374–400; module docstring line 8)
- Modify: `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_hybrid.template` (lines 24–35)
- Test: `matrixgame_covered_comm/tests/test_smoke.py` (add after the structured tests, ~line 246)

**Interfaces:**
- Consumes: existing `MatrixGameMaster._validate_player_response(player, response) -> bool`, `_setup_master(comm_protocol)` and `_reprompt_text(master, player)` test helpers, `STRUCTURED_MSG_PATTERN`, constants `COMM_STRUCTURED`/`COMM_HYBRID`.
- Produces: module-level `HYBRID_MSG_PATTERN` (compiled regex) and `ENFORCED_MSG_SCHEMAS` (dict `{str: (Pattern, str)}`), both importable from `master`.

- [ ] **Step 1: Write the failing tests**

Add to `matrixgame_covered_comm/tests/test_smoke.py` after `test_structured_done_message_exempt` (line 246). Player A owns A,B,C and A starts at R1,C1.

```python
def test_hybrid_pattern_requires_prose_and_clause():
    from master import HYBRID_MSG_PATTERN
    # prose + clause -> ok
    assert HYBRID_MSG_PATTERN.match(
        "I'll clear the left. PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none"
    )
    # structured-only (no prose) -> rejected: hybrid must combine both
    assert not HYBRID_MSG_PATTERN.match(
        "PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none"
    )
    # prose-only (no clause) -> rejected
    assert not HYBRID_MSG_PATTERN.match("let's go team")


def test_hybrid_wellformed_message_passes():
    master = _setup_master("hybrid")
    ok = master._validate_player_response(
        master.player_a,
        "message: I'll take the left so you have room. "
        "PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\n"
        "move: A to R2,C1 (down)",
    )
    assert ok is True


def test_hybrid_missing_clause_reprompts():
    master = _setup_master("hybrid")
    ok = master._validate_player_response(
        master.player_a, "message: let's go team\nmove: A to R2,C1 (down)"
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    assert "PLAN:" in _reprompt_text(master, master.player_a)


def test_hybrid_malformed_thrice_aborts():
    master = _setup_master("hybrid")
    for _ in range(3):
        master._validate_player_response(
            master.player_a, "message: chatty\nmove: A to R2,C1 (down)"
        )
    assert master.aborted is True


def test_hybrid_done_message_exempt():
    from master import MatrixGameMaster
    gi = _make_game_instance("hybrid")
    for o in ("A", "B", "C"):
        gi["target_positions"][o] = gi["start_positions"][o]
    master = MatrixGameMaster(
        game_spec=_make_game_spec(),
        experiment={"variant_config": {"comm_protocol": "hybrid"}},
        player_models=[_DummyModel("a"), _DummyModel("b")],
    )
    master._on_setup(**gi)
    ok = master._validate_player_response(
        master.player_a, "message: done\ndone: all my objects are placed"
    )
    assert ok is True


def test_hybrid_template_example_matches_pattern():
    """The hybrid template's example message must satisfy the enforced schema,
    so a compliant model is never spuriously reprompted."""
    import re as _re
    from master import HYBRID_MSG_PATTERN
    from pathlib import Path
    tmpl = Path(__file__).resolve().parents[1] / \
        "resources/initial_prompts/en/player_prompt_hybrid.template"
    text = tmpl.read_text(encoding="utf-8")
    lines = _re.findall(r"^message:\s*(.+)$", text, _re.MULTILINE)
    assert lines, "template must contain an example `message:` line"
    # The concrete example line must satisfy the enforced schema (the
    # `<...>` format-placeholder line is not expected to, and is ignored).
    assert any(HYBRID_MSG_PATTERN.match(m.strip()) for m in lines), \
        f"no example `message:` line matches HYBRID_MSG_PATTERN: {lines}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd matrixgame_covered_comm/tests && python -m pytest test_smoke.py -k hybrid -v`
Expected: FAIL — `ImportError: cannot import name 'HYBRID_MSG_PATTERN'` / `hybrid` messages currently accepted (no enforcement) so reprompt/abort assertions fail.

- [ ] **Step 3: Add the hybrid pattern and schema table**

In `matrixgame_covered_comm/master.py`, immediately after the `STRUCTURED_MSG_PATTERN` block (after line 101), insert:

```python
# Hybrid-protocol message schema, enforced (thesis: hybrid *combines* freeform
# and structured). The message must open with free natural-language prose and
# END with the same PLAN/REQUEST clause the structured protocol prescribes.
# Anchored full-match: >=1 char of prose, then the clause running to the end.
# A clause-only message (no prose) is rejected — that would just be structured.
HYBRID_MSG_PATTERN = re.compile(
    r"\A\s*\S.*?\s*PLAN:\s*move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\)\s*[—\-]+\s*REQUEST:\s*.+\Z",
    re.IGNORECASE | re.DOTALL,
)

# Protocols whose message body must match a prescribed schema, paired with the
# reprompt reminder shown on violation. `structured` keeps its exact prior
# behavior; `hybrid` reuses the same reprompt/abort path.
ENFORCED_MSG_SCHEMAS = {
    COMM_STRUCTURED: (
        STRUCTURED_MSG_PATTERN,
        'message: PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]',
    ),
    COMM_HYBRID: (
        HYBRID_MSG_PATTERN,
        'message: <natural language> PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]',
    ),
}
```

- [ ] **Step 4: Replace the structured-only enforcement block with the table-driven version**

In `matrixgame_covered_comm/master.py`, replace lines 374–400 (the `# Structured protocol:` comment through the `return False`) with:

```python
        # Structured & hybrid protocols prescribe a message schema; enforce it.
        # A malformed message is a violation like any parse failure: reprompt,
        # consume the retry budget, abort on exhaustion. Done declarations are
        # exempt (their message is the "done" marker, handled above).
        schema = ENFORCED_MSG_SCHEMAS.get(self.comm_protocol)
        if schema is not None:
            pattern, reminder = schema
            msg = parsed.get("message", "")
            if not pattern.match(msg):
                self.violated_request_counts += 1
                self.log_to_self(
                    "malformed_message",
                    f"{player.role_name}: {self.comm_protocol} format not followed — got: {msg[:80]}",
                )
                self.reprompt_attempts += 1
                if self.reprompt_attempts > self.max_retries:
                    self.aborted = True
                    self.log_to_self("abort", f"Too many malformed {self.comm_protocol} messages")
                else:
                    self.reprompt_pending = True
                    self.set_context_for(player, (
                        f"Your message did not follow the required {self.comm_protocol} format. "
                        "Use exactly:\n"
                        f"{reminder}\n\n"
                        f"{self._format_reminder()}\n\n"
                        f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                        f"GOAL BOARD:\n{self._render_targets_for(player)}"
                    ))
                return False
```

- [ ] **Step 5: Update the module docstring hybrid line**

In `matrixgame_covered_comm/master.py`, change line 8 from:

```
  hybrid     — free-form messaging plus an explicit shared coordinate frame in the prompt.
```
to:
```
  hybrid     — free-form messaging plus a required PLAN/REQUEST coordinate clause; message schema enforced (reprompt/abort) like structured.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd matrixgame_covered_comm/tests && python -m pytest test_smoke.py -k "hybrid or structured" -v`
Expected: PASS — all `hybrid` tests pass AND all `structured` tests still pass (byte-identical behavior).

- [ ] **Step 7: Rewrite the hybrid template**

Replace lines 24–35 of `matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_hybrid.template` (the `COMMUNICATION:` paragraph through the `move: A to R3,C3 (down)` example) with:

```
COMMUNICATION:
Before every move you must send your partner a short message (keep it under 100 words). Your message must combine BOTH of the following, in this order:
1. Natural language — share your plan and reflect on your partner's situation: where do you think their objects are heading? Are you about to create an obstacle for them, or free up space they need? Acknowledge what they might need from you.
2. A structured plan clause at the END of the message, in exactly this form:
   PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <what you need from your partner, or "none">
Use (Rr,Cc) coordinates for every cell in the PLAN clause.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `message:` on the very first line — do not output any preamble, thinking, or scratchpad text before it.
message: <natural language> PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <request or "none">
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
message: I'll clear the left side so you have room to bring your object down. PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: please clear (R3,C3) if you can.
move: A to R3,C3 (down)
```

(Leave lines 1–23 — intro, grid, task, rules, SHARED COORDINATE SYSTEM — and the closing "Think carefully…" paragraph on line 37 unchanged.)

- [ ] **Step 8: Run the template-consistency test**

Run: `cd matrixgame_covered_comm/tests && python -m pytest test_smoke.py::test_hybrid_template_example_matches_pattern -v`
Expected: PASS — the template's example `message:` line matches `HYBRID_MSG_PATTERN`.

- [ ] **Step 9: Run the full dyad suite**

Run: `cd matrixgame_covered_comm/tests && python -m pytest test_smoke.py -v`
Expected: PASS — no regressions.

- [ ] **Step 10: Commit**

```bash
cd /Users/tom/uni/clembench-spatial-reference
git add matrixgame_covered_comm/master.py \
        matrixgame_covered_comm/resources/initial_prompts/en/player_prompt_hybrid.template \
        matrixgame_covered_comm/tests/test_smoke.py
git commit -m "comm: enforce hybrid protocol as freeform prose + structured PLAN clause"
```

---

### Task 2: Enforced hybrid in `matrixgame_covered_comm_nplayer`

**Files:**
- Modify: `matrixgame_covered_comm_nplayer/master.py` (patterns block after line 99; enforcement block lines 374–400; module docstring line 11)
- Modify: `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_hybrid.template` (line 13 coordinate note; lines 26–37 communication/format/example)
- Test: `matrixgame_covered_comm_nplayer/tests/test_smoke.py` (add after `test_validate_response_rejects_invalid_format`, ~line 499)

**Interfaces:**
- Consumes: `NPlayerMatrixGameMaster._validate_player_response(player, response) -> bool`, `_make_game_instance(num_players, comm_protocol)`, `_make_game_spec(num_players)`, `_DummyModel`, `_reprompt_text(master, player)` (defined at line 501 — reference it, do not redefine), `STRUCTURED_MSG_PATTERN`, constants `COMM_STRUCTURED`/`COMM_HYBRID`.
- Produces: module-level `HYBRID_MSG_PATTERN` and `ENFORCED_MSG_SCHEMAS` in the n-player `master`.

- [ ] **Step 1: Write the failing tests**

Add to `matrixgame_covered_comm_nplayer/tests/test_smoke.py` immediately after `test_validate_response_rejects_invalid_format` (line 499) and BEFORE the `_reprompt_text` definition (so the helper is in module scope by call time — pytest collects all defs before running). In 6p mode player A (index 0) owns only "A" starting at R1,C1.

```python
def test_hybrid_pattern_requires_prose_and_clause():
    from master import HYBRID_MSG_PATTERN
    assert HYBRID_MSG_PATTERN.match(
        "heads up team. PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none"
    )
    assert not HYBRID_MSG_PATTERN.match(
        "PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none"
    )
    assert not HYBRID_MSG_PATTERN.match("go team")


def _setup_hybrid_master():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="hybrid")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "hybrid"}},
        player_models=models,
    )
    master._on_setup(**instance)
    return master


def test_hybrid_wellformed_message_passes():
    master = _setup_hybrid_master()
    players = master.get_players()
    ok = master._validate_player_response(
        players[0],
        "message: I'll head down the left so the rest of you have room. "
        "PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\n"
        "move: A to R2,C1 (down)",
    )
    assert ok is True


def test_hybrid_missing_clause_reprompts():
    master = _setup_hybrid_master()
    players = master.get_players()
    ok = master._validate_player_response(
        players[0], "message: go team\nmove: A to R2,C1 (down)"
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    assert "PLAN:" in _reprompt_text(master, players[0])


def test_hybrid_malformed_thrice_aborts():
    master = _setup_hybrid_master()
    players = master.get_players()
    for _ in range(3):
        master._validate_player_response(
            players[0], "message: chatty\nmove: A to R2,C1 (down)"
        )
    assert master.aborted is True


def test_hybrid_template_example_matches_pattern():
    import re as _re
    from master import HYBRID_MSG_PATTERN
    from pathlib import Path
    tmpl = Path(__file__).resolve().parents[1] / \
        "resources/initial_prompts/en/player_prompt_hybrid.template"
    text = tmpl.read_text(encoding="utf-8")
    lines = _re.findall(r"^message:\s*(.+)$", text, _re.MULTILINE)
    assert lines, "template must contain an example `message:` line"
    # The concrete example line must satisfy the enforced schema (the
    # `<...>` format-placeholder line is not expected to, and is ignored).
    assert any(HYBRID_MSG_PATTERN.match(m.strip()) for m in lines), \
        f"no example `message:` line matches HYBRID_MSG_PATTERN: {lines}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd matrixgame_covered_comm_nplayer/tests && python -m pytest test_smoke.py -k hybrid -v`
Expected: FAIL — `ImportError: cannot import name 'HYBRID_MSG_PATTERN'`.

- [ ] **Step 3: Add the hybrid pattern and schema table**

In `matrixgame_covered_comm_nplayer/master.py`, immediately after the `STRUCTURED_MSG_PATTERN` block (after line 99), insert the SAME block as Task 1 Step 3:

```python
# Hybrid-protocol message schema, enforced (thesis: hybrid *combines* freeform
# and structured). The message must open with free natural-language prose and
# END with the same PLAN/REQUEST clause the structured protocol prescribes.
# Anchored full-match: >=1 char of prose, then the clause running to the end.
# A clause-only message (no prose) is rejected — that would just be structured.
HYBRID_MSG_PATTERN = re.compile(
    r"\A\s*\S.*?\s*PLAN:\s*move\s+\S+\s+from\s*\(R\d+,C\d+\)\s+to\s*\(R\d+,C\d+\)\s*[—\-]+\s*REQUEST:\s*.+\Z",
    re.IGNORECASE | re.DOTALL,
)

# Protocols whose message body must match a prescribed schema, paired with the
# reprompt reminder shown on violation. `structured` keeps its exact prior
# behavior; `hybrid` reuses the same reprompt/abort path.
ENFORCED_MSG_SCHEMAS = {
    COMM_STRUCTURED: (
        STRUCTURED_MSG_PATTERN,
        'message: PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]',
    ),
    COMM_HYBRID: (
        HYBRID_MSG_PATTERN,
        'message: <natural language> PLAN: move [OBJECT] from (Rr,Cc) to (Rr,Cc) — REQUEST: [request or "none"]',
    ),
}
```

- [ ] **Step 4: Replace the structured-only enforcement block with the table-driven version**

In `matrixgame_covered_comm_nplayer/master.py`, replace lines 374–400 with the SAME table-driven block as Task 1 Step 4:

```python
        # Structured & hybrid protocols prescribe a message schema; enforce it.
        # A malformed message is a violation like any parse failure: reprompt,
        # consume the retry budget, abort on exhaustion. Done declarations are
        # exempt (their message is the "done" marker, handled above).
        schema = ENFORCED_MSG_SCHEMAS.get(self.comm_protocol)
        if schema is not None:
            pattern, reminder = schema
            msg = parsed.get("message", "")
            if not pattern.match(msg):
                self.violated_request_counts += 1
                self.log_to_self(
                    "malformed_message",
                    f"{player.role_name}: {self.comm_protocol} format not followed — got: {msg[:80]}",
                )
                self.reprompt_attempts += 1
                if self.reprompt_attempts > self.max_retries:
                    self.aborted = True
                    self.log_to_self("abort", f"Too many malformed {self.comm_protocol} messages")
                else:
                    self.reprompt_pending = True
                    self.set_context_for(player, (
                        f"Your message did not follow the required {self.comm_protocol} format. "
                        "Use exactly:\n"
                        f"{reminder}\n\n"
                        f"{self._format_reminder()}\n\n"
                        f"CURRENT BOARD:\n{self._render_for(player)}\n\n"
                        f"GOAL BOARD:\n{self._render_targets_for(player)}"
                    ))
                return False
```

- [ ] **Step 5: Update the module docstring hybrid line**

In `matrixgame_covered_comm_nplayer/master.py`, change line 11 from:

```
  hybrid     — free-form messaging plus an explicit shared coordinate frame in the prompt.
```
to:
```
  hybrid     — free-form messaging plus a required PLAN/REQUEST coordinate clause; message schema enforced (reprompt/abort) like structured.
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd matrixgame_covered_comm_nplayer/tests && python -m pytest test_smoke.py -k hybrid -v`
Expected: PASS.

- [ ] **Step 7: Fix the contradictory coordinate note (line 13)**

In `matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_hybrid.template`, replace line 13:

```
All players use the same coordinate notation: (Rr,Cc) where r is the row number (1–$GRID_SIZE$, top to bottom) and c is the column number (1–$GRID_SIZE$, left to right). For example, (R3,C5) is row 3, column 5. You are encouraged (but not required) to use this notation in your messages so your teammates can understand exactly which cells you mean.
```
with:
```
All players use the same coordinate notation: (Rr,Cc) where r is the row number (1–$GRID_SIZE$, top to bottom) and c is the column number (1–$GRID_SIZE$, left to right). For example, (R3,C5) is row 3, column 5. Use this notation for the cells in your PLAN clause so your teammates know exactly which cells you mean.
```

- [ ] **Step 8: Rewrite the communication / format / example (lines 26–37)**

Replace lines 26–37 (the `COMMUNICATION:` paragraph through the `move: A to R3,C3 (down)` example) with:

```
COMMUNICATION:
Before every move you must send your teammates a short broadcast message (keep it under 100 words). It will be shown to every teammate on their next turn. Your message must combine BOTH of the following, in this order:
1. Natural language — share your plan and reflect on your teammates' situations: where do you think their objects are heading? Are you about to create an obstacle for them, or open up space they need? Acknowledge what they might need from you.
2. A structured plan clause at the END of the message, in exactly this form:
   PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <what you need from a teammate, or "none">
Use (Rr,Cc) coordinates for every cell in the PLAN clause.

RESPONSE FORMAT:
You must respond using exactly this format — no other text:
Begin your response directly with `message:` on the very first line — do not output any preamble, thinking, or scratchpad text before it.
message: <natural language> PLAN: move <OBJECT> from (Rr,Cc) to (Rr,Cc) — REQUEST: <request or "none">
move: <OBJECT> to R<row>,C<col> (<DIRECTION>)

Example:
message: I'll work A down the left column so the middle stays open for you. PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: please keep (R3,C3) clear if you can.
move: A to R3,C3 (down)
```

(Leave line 1 role, grid/task/rules, and the closing "Think carefully…" paragraph on line 39 unchanged.)

- [ ] **Step 9: Run the template-consistency test and full suite**

Run: `cd matrixgame_covered_comm_nplayer/tests && python -m pytest test_smoke.py -v`
Expected: PASS — including `test_hybrid_template_example_matches_pattern` and the existing `test_message_protocols_mention_broadcast[hybrid]`.

- [ ] **Step 10: Commit**

```bash
cd /Users/tom/uni/clembench-spatial-reference
git add matrixgame_covered_comm_nplayer/master.py \
        matrixgame_covered_comm_nplayer/resources/initial_prompts/en/player_prompt_hybrid.template \
        matrixgame_covered_comm_nplayer/tests/test_smoke.py
git commit -m "nplayer: enforce hybrid protocol as freeform prose + structured PLAN clause"
```

---

### Task 3: Regenerate instances and verify boards unchanged

**Files:**
- Modify (regenerated): `matrixgame_covered_comm/in/instances.json`
- Modify (regenerated): `matrixgame_covered_comm_nplayer/in/instances.json`

**Interfaces:**
- Consumes: `instancegenerator.py` in each package (seed 42, `--n 8` default). No code changes — this task only regenerates data and verifies invariants.

- [ ] **Step 1: Snapshot the current instances for the board-invariance check**

```bash
cd /Users/tom/uni/clembench-spatial-reference
cp matrixgame_covered_comm/in/instances.json /tmp/comm_before.json
cp matrixgame_covered_comm_nplayer/in/instances.json /tmp/nplayer_before.json
```

- [ ] **Step 2: Regenerate the dyad instances**

```bash
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_comm
python instancegenerator.py
```
Expected: prints `Generated: exp2_s2_{none,structured,freeform,hybrid}_compact (n=8)` and the exp3 lines; writes `in/instances.json`.

- [ ] **Step 3: Regenerate the n-player instances**

```bash
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_comm_nplayer
python instancegenerator.py
```
Expected: writes `in/instances.json` for 3p and 6p.

- [ ] **Step 4: Verify only the hybrid player_prompt changed (dyad)**

Run:
```bash
cd /Users/tom/uni/clembench-spatial-reference
python3 - <<'PY'
import json
def index(path):
    d = json.load(open(path))
    out = {}
    for e in d["experiments"]:
        for gi in e["game_instances"]:
            out[(e["name"], gi["game_id"])] = gi
    return out
before = index("/tmp/comm_before.json")
after = index("matrixgame_covered_comm/in/instances.json")
assert before.keys() == after.keys(), "instance set changed!"
board_keys = ["walls","objects","start_positions","target_positions",
              "player_a_objects","player_b_objects","optimal_moves","spatial_level"]
changed_prompt, changed_board = [], []
for k, b in before.items():
    a = after[k]
    if any(a.get(x) != b.get(x) for x in board_keys):
        changed_board.append(k)
    if a.get("player_prompt") != b.get("player_prompt"):
        changed_prompt.append(k)
assert not changed_board, f"BOARD DATA CHANGED (must be identical): {changed_board[:5]}"
non_hybrid = [k for k in changed_prompt if after[k].get("comm_protocol") != "hybrid"]
assert not non_hybrid, f"non-hybrid prompt changed: {non_hybrid[:5]}"
hybrid_total = [k for k in after if after[k].get("comm_protocol") == "hybrid"]
assert set(changed_prompt) == set(hybrid_total), \
    f"expected every hybrid prompt to change: {len(changed_prompt)} vs {len(hybrid_total)}"
assert any("PLAN: move <OBJECT>" in after[k]["player_prompt"] for k in hybrid_total)
print(f"dyad OK: {len(hybrid_total)} hybrid prompts updated, all boards identical")
PY
```
Expected: `dyad OK: N hybrid prompts updated, all boards identical`.

- [ ] **Step 5: Verify only the hybrid player_prompt changed (n-player)**

Run the same script with paths swapped to `/tmp/nplayer_before.json` and `matrixgame_covered_comm_nplayer/in/instances.json`. Note: n-player instances have `player_objects` instead of `player_a_objects`/`player_b_objects` — replace those two entries in `board_keys` with `"player_objects"` and `"num_players"`.
Expected: `dyad OK: N hybrid prompts updated, all boards identical` (label text ignored — the assertions are what matter).

- [ ] **Step 6: Re-run both test suites against the regenerated instances**

```bash
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_comm/tests && python -m pytest test_smoke.py -q
cd /Users/tom/uni/clembench-spatial-reference/matrixgame_covered_comm_nplayer/tests && python -m pytest test_smoke.py -q
```
Expected: both suites pass (n-player has generator-fixture tests that read the regenerated data).

- [ ] **Step 7: Commit**

```bash
cd /Users/tom/uni/clembench-spatial-reference
git add matrixgame_covered_comm/in/instances.json \
        matrixgame_covered_comm_nplayer/in/instances.json
git commit -m "instances: regenerate hybrid prompts with enforced PLAN clause (boards unchanged)"
```

---

## Thesis follow-up (not a code task)

Record for the Method chapter edits — hybrid is now a **superset of structured** (structured schema embedded in free prose), enforced identically:
- The grammar-section sentence "The hybrid protocol combines the two" is now **correct** and should stay (previously flagged for deletion in `thesis_code_alignment_plan.md` Task 8).
- The method-comm definition ("natural language … encouraged but not forced to use coordinates") is now **wrong** and must be reworded to: free-form message that must include the structured PLAN/REQUEST clause, schema-enforced by reprompt/abort.
- Exp 2 note: `hybrid` now shares `structured`'s abort/violation failure mode; account for this when comparing protocols.
