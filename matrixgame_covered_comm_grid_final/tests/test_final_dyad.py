"""Final-arm invariants for matrixgame_covered_comm_grid_final.

Instances: only player_prompt differs from the updated-prompt arm, except that
S3 re-splits the objects (prompt_final/split_s3.py) with budgets on the new
A_k and S4 plays the S4x boards (prompt_final/s4x.py); every prompt states the
record window the master keeps, not the cycle rule.
Master: every turn and retry context carries the last ROUNDS rounds as a
MESSAGE BOARD (with senders) and a MOVE BOARD (no player named, only the
recipient's own moves in full), and nothing else: no warning ahead of the
cycle rule.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SOURCE = PKG.parent / "matrixgame_covered_comm_grid_updated_prompt"
CROSSED = PKG.parent / "matrixgame_covered_comm_crossed_grid"
PLACEHOLDER = re.compile(r"\$[A-Z_]+\$")
# Prose is colon-free; colons stay only in headers, `Example:` and the response
# grammar (message:/move:/done: lines, the PLAN clause, backticked keys).
GRAMMAR = re.compile(r"^\s*([A-Z][A-Z ]+:|Example:|(message|move|done|PLAN):.*)$")


def _prose_colons(prompt):
    return [l for l in prompt.splitlines() if ":" in re.sub(r"`\w+:`", "", l) and not GRAMMAR.match(l)]

_spec = importlib.util.spec_from_file_location("final_dyad_master", PKG / "master.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
ROUNDS = M.history.ROUNDS


def _instances(pkg):
    data = json.loads((pkg / "in" / "instances.json").read_text(encoding="utf-8"))
    return {(e["name"], i["game_id"]): i for e in data["experiments"] for i in e["game_instances"]}


def _source():
    """The updated-prompt arm, with S4 on the crossed package's S4x boards."""
    source = _instances(SOURCE)
    for (name, gid), inst in _instances(CROSSED).items():
        if name.startswith("S4x_"):
            source[(name.replace("S4x_", "S4_"), gid)] = {**inst, "spatial_level": "S4"}
    return source


S3_SPLIT = ("player_a_objects", "player_b_objects", "optimal_moves", "max_turns")


def test_only_the_prompt_and_the_s3_split_differ_from_the_updated_prompt_arm():
    arm, source = _instances(PKG), _source()
    assert arm.keys() == source.keys()
    for key, inst in arm.items():
        free = {"player_prompt": "", **({k: 0 for k in S3_SPLIT} if inst["spatial_level"] == "S3" else {})}
        assert {**inst, **free} == {**source[key], **free}, key
        assert PLACEHOLDER.findall(inst["player_prompt"]) == PLACEHOLDER.findall(source[key]["player_prompt"]), key


def test_s3_puts_entrance_holder_and_corridor_object_on_different_players():
    # prompt_final/split_s3.py: B (target on the entrance) and D (target on the
    # dead end) as at 3p/6p, so a `done` can seal the corridor in every team size.
    optima = {}
    for line in (PKG.parent / "analysis/efficiency_baseline/attainable_dyad_final.jsonl").open():
        row = json.loads(line)
        optima[(row["level"], row["game_id"])] = row["stuckfree"]
    s3 = [i for (name, _), i in _instances(PKG).items() if name.startswith("S3_")]
    assert len(s3) == 60
    for inst in s3:
        assert set(inst["player_a_objects"]) == {"A", "D", "E"}, inst["game_id"]
        assert set(inst["player_b_objects"]) == {"B", "C", "F"}, inst["game_id"]
        assert inst["optimal_moves"] == optima[("S3", inst["game_id"])]
        assert inst["max_turns"] == max(4 * inst["optimal_moves"], 20)


def test_every_prompt_states_the_window_but_not_the_cycle_rule():
    for key, inst in _instances(PKG).items():
        p = inst["player_prompt"]
        assert "third time" not in p, key  # the cycle rule is a progress check, not a rule to play to
        assert f"the last {ROUNDS} rounds of play" in p, key
        assert f"Turns older than the last {ROUNDS} rounds are not shown again" in p, key
        assert ("MESSAGE BOARD" in p) == (inst["comm_protocol"] != "none") and "MOVE BOARD" in p, key
        assert not _prose_colons(p), (key, _prose_colons(p))
        assert "Do not output" not in p, key


# ── master ───────────────────────────────────────────────────────────────

class _Dummy:
    name = "dummy"

    def set_gen_arg(self, *_):
        pass


def _master(protocol="freeform", a_placed=False):
    from clemcore.clemgame.registry import GameSpec
    objs = "ABCDEF"
    starts = {o: [0, i] for i, o in enumerate(objs)}
    inst = {
        "game_id": 1, "grid_size": 8, "walls": [], "objects": list(objs),
        "player_a_objects": ["A", "B", "C"], "player_b_objects": ["D", "E", "F"],
        "start_positions": starts,
        "target_positions": {o: (starts[o] if a_placed and o in "ABC" else [7, i]) for i, o in enumerate(objs)},
        "optimal_moves": 42, "max_turns": 100, "max_retries": 2, "strict": False,
        "use_masking": True, "spatial_level": "S1", "comm_protocol": protocol,
        "compact_board": True, "player_prompt": "PROMPT $YOUR_OBJECTS$ $GRID_SIZE$",
    }
    m = M.FinalMatrixGameMaster(GameSpec(game_name="g", game_path=str(PKG), players=2),
                                {"name": "x"}, [_Dummy(), _Dummy()])
    m._on_setup(**inst)
    return m


def _play(m, player, response):
    assert m._validate_player_response(player, response), response
    m._on_valid_player_response(player, m._parse_response(player, response))


def _ctx(m, player):
    return m.get_context_for(player)["content"]


def test_boards_separate_messages_from_anonymous_moves():
    m = _master()
    _play(m, m.player_a, "message: A goes first\nmove: A down")
    _play(m, m.player_b, "message: then D\nmove: D down")
    ctx = _ctx(m, m.player_a)
    messages, moves = ctx.split("\n\n")[:2]
    assert messages.splitlines() == [f"MESSAGE BOARD (the last {ROUNDS} rounds, oldest first):",
                                     'Player Green (you) said "A goes first"', 'Player Orange said "then D"']
    assert moves.splitlines() == [f"MOVE BOARD (the last {ROUNDS} rounds, oldest first):",
                                  "You moved A from (R1,C1) to (R2,C1)", "X moved from (R1,C4) to (R2,C4)"]
    assert "says:" not in ctx  # the boards replace the latest-message block
    assert ctx.index("MOVE BOARD") < ctx.index("CURRENT BOARD")


def test_window_keeps_the_last_rounds_only():
    m = _master("none")
    a, b = "ABC", "DEF"
    for t in range(2 * ROUNDS + 1):  # objects only move down: no board recurs
        player, objs = (m.player_a, a) if t % 2 == 0 else (m.player_b, b)
        _play(m, player, f"move: {objs[(t // 2) % 3]} down")
    ctx = _ctx(m, m.player_b)
    moves = ctx.split("\n\n")[0].splitlines()
    assert moves[0].startswith("MOVE BOARD") and len(moves) == 1 + 2 * ROUNDS
    assert "X moved from (R1,C1) to (R2,C1)" not in ctx  # turn 1 has left the window
    assert "MESSAGE BOARD" not in ctx


def test_retry_context_keeps_the_record():
    m = _master()
    _play(m, m.player_a, "message: hi\nmove: A down")
    assert not m._validate_player_response(m.player_b, "no idea")
    ctx = _ctx(m, m.player_b)
    assert "could not be parsed" in ctx and 'Player Green said "hi"' in ctx and "X moved from (R1,C1) to (R2,C1)" in ctx


def test_no_warning_before_the_cycle_rule_fires():
    # The cycle rule measures whether play progresses; it is not announced
    # turn by turn, so a second occurrence passes silently and a third loses.
    m = _master("none")
    for _ in range(2):
        _play(m, m.player_a, "move: A down")
        _play(m, m.player_b, "move: D down")
        _play(m, m.player_a, "move: A up")
        assert "occurred" not in _ctx(m, m.player_b).lower()
        _play(m, m.player_b, "move: D up")  # back to the start board
    assert m.lost


def test_done_is_no_move_and_its_message_is_dropped():
    m = _master(a_placed=True)
    _play(m, m.player_a, "message: done\ndone: all my objects are placed")
    ctx = _ctx(m, m.player_b)
    assert "MOVE BOARD" not in ctx and "MESSAGE BOARD" not in ctx and '"done"' not in ctx
    assert "Player Green has finished" in ctx


def test_seats_are_named_by_colour():
    m = _master()
    assert (m.player_a.role_name, m.player_b.role_name) == ("Player Green", "Player Orange")
