"""Final-arm invariants for matrixgame_covered_comm_nplayer_grid_final (3p and 6p).

Instances: only player_prompt differs from the updated-prompt arm, except that
S4 plays the S4x boards (prompt_final/s4x.py), and every prompt states the record window the master keeps, not the cycle rule.
Master: the MESSAGE BOARD and MOVE BOARD span ROUNDS rounds of all players,
moves name no player and keep their order, teammates' messages are shuffled
within a round, seats are named by colour, and finished teammates stay listed
after the one-off notice.
"""

import importlib.util
import json
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
SOURCE = PKG.parent / "matrixgame_covered_comm_nplayer_grid_updated_prompt"
CROSSED = PKG.parent / "matrixgame_covered_comm_nplayer_crossed_grid"
FILES = ("instances_3p.json", "instances_6p.json")
PLACEHOLDER = re.compile(r"\$[A-Z_]+\$")
# Prose is colon-free; colons stay only in headers, `Example:` and the response
# grammar (message:/move:/done: lines, the PLAN clause, backticked keys).
GRAMMAR = re.compile(r"^\s*([A-Z][A-Z ]+:|Example:|(message|move|done|PLAN):.*)$")


def _prose_colons(prompt):
    return [l for l in prompt.splitlines() if ":" in re.sub(r"`\w+:`", "", l) and not GRAMMAR.match(l)]

_spec = importlib.util.spec_from_file_location("final_nplayer_master", PKG / "master.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)
ROUNDS = M.history.ROUNDS


def _instances(pkg, filename):
    data = json.loads((pkg / "in" / filename).read_text(encoding="utf-8"))
    return {(e["name"], i["game_id"]): i for e in data["experiments"] for i in e["game_instances"]}


def _source(filename):
    """The updated-prompt arm, with S4 on the crossed package's S4x boards."""
    source = _instances(SOURCE, filename)
    for (name, gid), inst in _instances(CROSSED, filename).items():
        if name.startswith("S4x_"):
            source[(name.replace("S4x_", "S4_"), gid)] = {**inst, "spatial_level": "S4"}
    return source


def test_only_the_prompt_differs_from_the_updated_prompt_arm():
    for f in FILES:
        arm, source = _instances(PKG, f), _source(f)
        assert arm.keys() == source.keys(), f
        for key, inst in arm.items():
            assert {**inst, "player_prompt": ""} == {**source[key], "player_prompt": ""}, key
            assert PLACEHOLDER.findall(inst["player_prompt"]) == PLACEHOLDER.findall(source[key]["player_prompt"]), key


def test_every_prompt_states_the_window_but_not_the_cycle_rule():
    for f in FILES:
        for key, inst in _instances(PKG, f).items():
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


def _master(a_placed=False):
    from clemcore.clemgame.registry import GameSpec
    objs = "ABCDEF"
    starts = {o: [0, i] for i, o in enumerate(objs)}
    inst = {
        "game_id": 1, "grid_size": 8, "walls": [], "objects": list(objs), "num_players": 3,
        "player_objects": [["A", "B"], ["C", "D"], ["E", "F"]],
        "start_positions": starts,
        "target_positions": {o: (starts[o] if a_placed and o in "AB" else [7, i]) for i, o in enumerate(objs)},
        "optimal_moves": 42, "max_turns": 100, "max_retries": 2, "strict": False,
        "use_masking": True, "spatial_level": "S1", "comm_protocol": "freeform",
        "compact_board": True, "player_prompt": "PROMPT $YOUR_ROLE$ $YOUR_OBJECTS$ $GRID_SIZE$ $N_OTHER_PLAYERS$",
    }
    m = M.FinalNPlayerMatrixGameMaster(GameSpec(game_name="g", game_path=str(PKG), players=3),
                                       {"name": "x"}, [_Dummy() for _ in range(3)])
    m._on_setup(**inst)
    return m


def _play(m, player, response):
    assert m._validate_player_response(player, response), response
    m._on_valid_player_response(player, m._parse_response(player, response))


def _rounds(m, n_turns):
    players = m.get_players()
    owned = ["AB", "CD", "EF"]
    for t in range(n_turns):  # objects only move down: no board recurs
        i = t % 3
        _play(m, players[i], f"message: turn {t + 1}\nmove: {owned[i][(t // 3) % 2]} down")
    messages, moves = m.get_context_for(players[1])["content"].split("\n\n")[:2]
    return messages.splitlines()[1:], moves.splitlines()[1:]


def test_window_spans_rounds_of_all_players():
    messages, moves = _rounds(_master(), 3 * ROUNDS + 1)
    assert len(messages) == len(moves) == 3 * ROUNDS
    # Orange's own messages open each round; the teammates' two that follow are
    # shuffled within the round, so only the round's set is fixed.
    for r in range(ROUNDS):
        t = 3 * r + 2
        assert messages[3 * r] == f'Player Orange (you) said "turn {t}"'
        assert sorted(messages[3 * r + 1:3 * r + 3]) == [f'Player Blue said "turn {t + 1}"',
                                                         f'Player Green said "turn {t + 2}"']
    # Moves keep the order they were made and name no player.
    assert moves[0] == "You moved C from (R1,C3) to (R2,C3)"
    assert moves[1:3] == ["X moved from (R1,C5) to (R2,C5)", "X moved from (R1,C2) to (R2,C2)"]
    assert all(l.startswith("X moved from (R") for l in moves if not l.startswith("You moved "))


def test_message_shuffle_is_seeded_and_not_the_turn_order():
    def order(game_id):
        m = _master()
        m.seed = str(game_id)
        return _rounds(m, 3 * ROUNDS + 1)[0]
    assert order(1) == order(1)  # a replay renders the same board
    chrono = order(1)  # rebuilt below as turn order to compare against
    chrono = [l for r in range(ROUNDS) for l in sorted(chrono[3 * r:3 * r + 3], key=lambda l: int(l.split()[-1][:-1]))]
    assert any(order(g) != chrono for g in range(1, 6))


def test_window_counts_rounds_not_turns():
    # A finished teammate takes no turns, so a round is shorter; the window
    # still holds ROUNDS of the viewer's rounds, as the prompt says.
    m = _master(a_placed=True)
    g, o, b = m.get_players()
    _play(m, g, "message: done\ndone: all my objects are placed")
    for t in range(12):  # Orange and Blue alternate; objects only move down
        p, objs = (o, "CD") if t % 2 == 0 else (b, "EF")
        _play(m, p, f"message: t{t}\nmove: {objs[(t // 2) % 2]} down")
    messages, moves = m.get_context_for(o)["content"].split("\n\n")[:2]
    assert sum("(you)" in l for l in messages.splitlines()) == ROUNDS
    assert len(moves.splitlines()) == 1 + 2 * ROUNDS


def test_seats_are_named_by_colour():
    names = [p.role_name for p in _master().get_players()]
    assert names == ["Player Green", "Player Orange", "Player Blue"]
    assert len(set(M.history.SEATS)) == 6 and all(len(n.split()[1]) > 1 for n in M.history.SEATS)


def test_finished_teammates_stay_listed():
    m = _master(a_placed=True)
    a, b, c = m.get_players()
    _play(m, a, "message: done\ndone: all my objects are placed")
    _play(m, b, "message: ok\nmove: C down")
    _play(m, c, "message: ok\nmove: E down")
    ctx = m.get_context_for(b)["content"]
    assert "All their objects are at their target positions" not in ctx  # the base notice is one-off
    assert "Player Green has finished. Their objects will not move again." in ctx
