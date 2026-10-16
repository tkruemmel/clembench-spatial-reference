"""Final-arm invariants for matrixgame_covered_solo_grid_final.

Instances: the silent experiments of matrixgame_covered_solo_grid (the final
arm has no thinking variant), with only the prompt and the budget changed,
except that S4 plays the S4x boards (prompt_final/s4x.py); the budget comes
from the attainable solo optimum. Master: the MOVE BOARD keeps the last ROUNDS
turns, also on a retry.
"""

import importlib.util
import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SOURCE = PKG.parent / "matrixgame_covered_solo_grid"
CROSSED = PKG.parent / "matrixgame_covered_solo_crossed_grid"
ATTAINABLE = PKG.parent / "analysis" / "efficiency_baseline" / "attainable_solo.jsonl"
# Solo's attainable optimum is the joint optimum; the S4x boards have their own file.
JOINT_S4X = PKG.parent / "analysis" / "efficiency_baseline" / "joint_optimum_crossed.jsonl"
MUTABLE = {"player_prompt": "", "max_turns": 0, "optimal_moves": 0}
PLACEHOLDER = re.compile(r"\$[A-Z_]+\$")
# Prose is colon-free; colons stay only in headers, `Example:` and the response
# grammar (message:/move:/done: lines, the PLAN clause, backticked keys).
GRAMMAR = re.compile(r"^\s*([A-Z][A-Z ]+:|Example:|(message|move|done|PLAN):.*)$")


def _prose_colons(prompt):
    return [l for l in prompt.splitlines() if ":" in re.sub(r"`\w+:`", "", l) and not GRAMMAR.match(l)]

_spec = importlib.util.spec_from_file_location("final_solo_master", PKG / "master.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
ROUNDS = M.history.ROUNDS


def _instances(pkg):
    data = json.loads((pkg / "in" / "instances.json").read_text(encoding="utf-8"))
    return {(e["name"], i["game_id"]): i for e in data["experiments"] for i in e["game_instances"]}


def test_only_prompt_and_budget_differ_and_the_budget_is_attainable():
    arm = _instances(PKG)
    source = {k: i for k, i in _instances(SOURCE).items() if not i["thinking"]}
    for (name, gid), inst in _instances(CROSSED).items():
        if "_S4x_" in name and not inst["thinking"]:
            source[(name.replace("_S4x_", "_S4_"), gid)] = {**inst, "spatial_level": "S4"}
    rows = map(json.loads, ATTAINABLE.read_text(encoding="utf-8").splitlines())
    A = {(r["level"], r["game_id"]): r["stuckfree"] for r in rows if r["level"] != "S4"}
    rows = map(json.loads, JOINT_S4X.read_text(encoding="utf-8").splitlines())
    A.update({("S4", r["game_id"]): r["opt"] for r in rows if r["level"] == "S4x"})
    assert arm.keys() == source.keys()
    for key, inst in arm.items():
        assert {**inst, **MUTABLE} == {**source[key], **MUTABLE}, key
        assert PLACEHOLDER.findall(inst["player_prompt"]) == PLACEHOLDER.findall(source[key]["player_prompt"]), key
        assert inst["optimal_moves"] == A[(inst["spatial_level"], inst["game_id"])], key
        assert inst["max_turns"] == max(4 * inst["optimal_moves"], 20), key


def test_every_prompt_states_the_window_but_not_the_cycle_rule():
    for key, inst in _instances(PKG).items():
        p = inst["player_prompt"]
        assert "third time" not in p, key  # the cycle rule is a progress check, not a rule to play to
        assert "stay in the conversation" not in p, key
        assert f"the last {ROUNDS} rounds of play" in p, key
        assert f"Turns older than the last {ROUNDS} rounds are not shown again" in p, key
        assert not inst["thinking"] and "MOVE BOARD" in p and "NOTE BOARD" not in p, key
        assert not _prose_colons(p), (key, _prose_colons(p))
        assert "Do not output" not in p, key


# ── master ───────────────────────────────────────────────────────────────

class _Dummy:
    name = "dummy"

    def set_gen_arg(self, *_):
        pass


# The final masters' grid header: each label over its cell (history.aligned).
HEADER = "       C1   C2   C3   C4   C5   C6   C7   C8"


def _master(grid=False):
    from clemcore.clemgame.registry import GameSpec
    objs = "ABCDEF"
    inst = {
        "grid_size": 8, "walls": [], "objects": list(objs),
        "start_positions": {o: [0, i] for i, o in enumerate(objs)},
        "target_positions": {o: [7, i] for i, o in enumerate(objs)},
        "optimal_moves": 42, "max_turns": 100, "max_retries": 2, "strict": False,
        "thinking": False, "spatial_level": "S1", "compact_board": not grid,
        "player_prompt": "PROMPT $YOUR_OBJECTS$ $GRID_SIZE$",
    }
    m = M.FinalSoloMatrixGameMaster(GameSpec(game_name="g", game_path=str(PKG), players=1),
                                    {"name": "x"}, [_Dummy()])
    m._on_setup(**inst)
    return m


def _play(m, response):
    assert m._validate_player_response(m.player, response), response
    m._on_valid_player_response(m.player, m._parse_response(m.player, response))


def test_move_board_keeps_the_last_turns():
    m = _master()
    for t in range(ROUNDS + 1):
        _play(m, f"move: {'ABCDEF'[t]} down")
    ctx = m.get_context_for(m.player)["content"]
    assert ctx.split("\n\n")[0].splitlines() == [f"MOVE BOARD (the last {ROUNDS} rounds, oldest first):"] + [
        f"You moved {'ABCDEF'[t]} from (R1,C{t + 1}) to (R2,C{t + 1})" for t in range(1, ROUNDS + 1)]  # turn 1 left
    assert "MESSAGE BOARD" not in ctx


def test_retry_context_keeps_the_record():
    m = _master()
    _play(m, "move: A down")
    assert not m._validate_player_response(m.player, "blah")
    ctx = m.get_context_for(m.player)["content"]
    assert "could not be parsed" in ctx and "You moved A from (R1,C1) to (R2,C1)" in ctx


def test_grid_header_sits_over_its_cells():
    m = _master(grid=True)
    assert m.get_context_for(m.player)["content"].count(HEADER + "\n") == 2  # first turn
    _play(m, "move: A down")
    assert m.get_context_for(m.player)["content"].count(HEADER + "\n") == 2
