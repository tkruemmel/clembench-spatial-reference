"""Smoke tests for matrixgame_covered_comm_nplayer."""

import pytest
from pathlib import Path

PKG = Path(__file__).parent.parent


def test_board_module_imports():
    from utils.board import Board, DIRECTIONS
    assert DIRECTIONS  # non-empty dict/list
    board = Board(grid_size=8, walls=set())
    assert board.grid_size == 8


def test_astar_module_imports():
    from utils import astar
    assert any(callable(getattr(astar, n)) for n in dir(astar) if not n.startswith("_"))


def test_common_config_shape():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "common_config.json") as f:
        cfg = json.load(f)
    assert cfg["grid_size"] == 8
    assert cfg["objects"] == ["A", "B", "C", "D", "E", "F"]
    assert cfg["directions"] == ["up", "down", "left", "right"]
    assert cfg["max_retries"] == 2
    assert cfg["strict"] is False


def test_config_has_32_variants():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    assert len(cfg) == 32, f"expected 32 variants, got {len(cfg)}"


def test_every_variant_has_required_fields():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    for key, variant in cfg.items():
        for field in ("name", "num_players", "spatial_level", "comm_protocol", "use_masking", "compact_board"):
            assert field in variant, f"variant {key} missing {field}"
        assert variant["num_players"] in (3, 6), f"{key}: num_players must be 3 or 6"
        assert variant["spatial_level"] in ("S1", "S2", "S3", "S4"), f"{key}: invalid spatial_level"
        assert variant["comm_protocol"] in ("none", "structured", "freeform", "hybrid"), f"{key}: invalid comm_protocol"
        assert variant["use_masking"] is True, f"{key}: use_masking must be true"
        assert variant["compact_board"] is True, f"{key}: compact_board must be true"


def test_variants_cover_full_matrix():
    import json
    from pathlib import Path

    PKG = Path(__file__).parent.parent

    with open(PKG / "resources" / "config.json") as f:
        cfg = json.load(f)
    seen = set()
    for variant in cfg.values():
        seen.add((variant["spatial_level"], variant["comm_protocol"], variant["num_players"]))
    expected = {
        (s, p, n)
        for s in ("S1", "S2", "S3", "S4")
        for p in ("none", "structured", "freeform", "hybrid")
        for n in (3, 6)
    }
    assert seen == expected, f"missing combinations: {expected - seen}"


@pytest.mark.parametrize("protocol", ["none", "structured", "freeform", "hybrid"])
def test_prompt_template_exists(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    assert path.exists(), f"missing template for {protocol}"
    text = path.read_text()
    assert "$YOUR_OBJECTS$" in text
    assert "$GRID_SIZE$" in text
    assert "$YOUR_ROLE$" in text
    assert "$N_OTHER_PLAYERS$" in text


@pytest.mark.parametrize("protocol", ["structured", "freeform", "hybrid"])
def test_message_protocols_mention_broadcast(protocol):
    path = PKG / "resources" / "initial_prompts" / "en" / f"player_prompt_{protocol}.template"
    text = path.read_text().lower()
    assert "broadcast" in text or "teammates" in text, \
        f"{protocol} template should mention teammates/broadcast"


def test_none_template_has_no_message_line():
    path = PKG / "resources" / "initial_prompts" / "en" / "player_prompt_none.template"
    text = path.read_text()
    assert "message:" not in text.lower(), "none template must not include 'message:'"


def test_instance_generator_module_imports():
    from instancegenerator import MatrixGameNPlayerInstanceGenerator
    gen = MatrixGameNPlayerInstanceGenerator()
    assert gen is not None


import json


@pytest.fixture(scope="module")
def generated_instances(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("nplayer_in")
    import sys
    sys.path.insert(0, str(PKG))
    from instancegenerator import MatrixGameNPlayerInstanceGenerator

    gen = MatrixGameNPlayerInstanceGenerator()
    gen.generate(filename=str(out_dir / "instances.json"), seed=42, num_instances=2)
    with open(out_dir / "instances.json") as f:
        return json.load(f)


def test_generator_emits_all_32_experiments(generated_instances):
    experiments = generated_instances["experiments"]
    assert len(experiments) == 32, f"expected 32 experiments, got {len(experiments)}"


def test_instance_schema(generated_instances):
    required = (
        "grid_size", "walls", "objects",
        "num_players", "player_objects",
        "start_positions", "target_positions",
        "optimal_moves", "max_turns", "max_retries", "strict",
        "use_masking", "spatial_level", "comm_protocol",
        "compact_board", "player_prompt",
    )
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            for field in required:
                assert field in inst, f"missing {field} in {exp['name']} instance {inst.get('game_id')}"


def test_player_partition_3p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_3p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 3
            po = inst["player_objects"]
            assert len(po) == 3, f"3p should have 3 groups, got {len(po)}"
            assert all(len(g) == 2 for g in po), f"3p: every group must own 2 objects, got {po}"
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            # Round-robin partition: objects[j] -> player j % num_players
            assert po == [["A", "D"], ["B", "E"], ["C", "F"]]


def test_player_partition_6p(generated_instances):
    for exp in generated_instances["experiments"]:
        if "_6p" not in exp["name"].lower():
            continue
        for inst in exp["game_instances"]:
            assert inst["num_players"] == 6
            po = inst["player_objects"]
            assert len(po) == 6
            assert all(len(g) == 1 for g in po)
            flat = [obj for group in po for obj in group]
            assert sorted(flat) == ["A", "B", "C", "D", "E", "F"]
            assert po == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]


def test_compact_board_always_true(generated_instances):
    for exp in generated_instances["experiments"]:
        for inst in exp["game_instances"]:
            assert inst["compact_board"] is True


def test_experiment_naming(generated_instances):
    names = {exp["name"] for exp in generated_instances["experiments"]}
    # 32 unique names of form "S<level>_<protocol>_<n>p"
    assert len(names) == 32
    for name in names:
        parts = name.split("_")
        assert len(parts) == 3, f"unexpected experiment name shape: {name}"
        level, protocol, mode = parts
        assert level in ("S1", "S2", "S3", "S4")
        assert protocol.lower() in ("none", "structured", "freeform", "hybrid")
        assert mode in ("3p", "6p")


def test_master_module_imports():
    from master import (
        NPlayerMatrixGameMaster,
        NPlayerMatrixGameScorer,
        NPlayerMatrixGameBenchmark,
        MatrixPlayer,
        _parse_response,
    )
    assert NPlayerMatrixGameMaster is not None
    assert MatrixPlayer is not None


def test_parse_response_move_only():
    from master import _parse_response
    text = "move: A down"
    parsed = _parse_response(text, with_message=False)
    assert parsed["object"] == "A"
    assert parsed["direction"] == "down"


def test_parse_response_with_message():
    from master import _parse_response
    text = "message: moving A\nmove: A down"
    parsed = _parse_response(text, with_message=True)
    assert parsed["message"] == "moving A"
    assert parsed["object"] == "A"


def test_grammar_anchored_no_reason():
    from master import _parse_response
    # channel off
    assert _parse_response("move: A down", with_message=False) is not None
    assert _parse_response("thinking first\nmove: A down", with_message=False) is None
    assert _parse_response("reason: r\nmove: A down", with_message=False) is None
    assert _parse_response("done: all my objects are placed", with_message=False)["is_done"] is True
    # channel on: messages MAY span multiple lines (single-line boundary
    # removed); the parser still never produces a `reason` field, and the move
    # line remains the required terminator. Extra lines fold into the message.
    p = _parse_response("message: hi\nmove: A down", with_message=True)
    assert p["message"] == "hi" and "reason" not in p
    ml = _parse_response("message: hi\nmore context\nmove: A down", with_message=True)
    assert ml is not None and ml["message"] == "hi\nmore context" and "reason" not in ml
    d = _parse_response("message: done\ndone: all placed", with_message=True)
    assert d["is_done"] is True and d["message"] == "done"
    # a preamble BEFORE `message:` still fails the \A-anchored parse
    assert _parse_response("preamble\nmessage: hi\nmove: A down", with_message=True) is None


def _make_game_instance(num_players=3, comm_protocol="freeform"):
    """Build a minimal game_instance dict suitable for setup tests."""
    objs = ["A", "B", "C", "D", "E", "F"]
    objs_per = len(objs) // num_players
    player_objects = [objs[i*objs_per:(i+1)*objs_per] for i in range(num_players)]
    return {
        "grid_size": 8,
        "walls": [],
        "objects": objs,
        "num_players": num_players,
        "player_objects": player_objects,
        "start_positions": {o: [0, i] for i, o in enumerate(objs)},
        "target_positions": {o: [7, i] for i, o in enumerate(objs)},
        "optimal_moves": 42,
        "max_turns": 100,
        "max_retries": 2,
        "strict": False,
        "use_masking": True,
        "spatial_level": "S1",
        "comm_protocol": comm_protocol,
        "compact_board": True,
        "player_prompt": (
            "You are $YOUR_ROLE$. You control $YOUR_OBJECTS$ on a $GRID_SIZE$x$GRID_SIZE$ "
            "grid with $N_OTHER_PLAYERS$ teammates."
        ),
    }


class _DummyModel:
    """Stand-in for clemcore.backends.Model with the methods setup touches."""
    def __init__(self, name="dummy"):
        self.name = name
        self._gen_args = {}

    def set_gen_arg(self, key, value):
        self._gen_args[key] = value

    def generate_response(self, *_args, **_kwargs):
        return "move: A down"


def _make_game_spec(num_players=6):
    """Build a minimal GameSpec suitable for NPlayerMatrixGameMaster constructor."""
    from clemcore.clemgame.registry import GameSpec
    return GameSpec(
        game_name="matrixgame_covered_comm_nplayer",
        game_path="matrixgame_covered_comm_nplayer",
        players=num_players,
    )


def test_timeout_is_lose_not_abort():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3)
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    master.move_log = [{}] * master.max_turns
    assert master._does_game_proceed() is False
    assert master.lost is True
    assert master.aborted is False


@pytest.mark.parametrize("num_players", [3, 6])
def test_master_setup_builds_n_players(num_players):
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=num_players)
    models = [_DummyModel(f"m{i}") for i in range(num_players)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=num_players),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=models,
    )
    master._on_setup(**instance)
    assert master.num_players == num_players
    assert len(master.player_objects) == num_players
    assert len(master.player_done) == num_players
    assert all(d is False for d in master.player_done)
    assert len(master.last_message) == num_players
    assert all(v is None for v in master.last_message.values())


def test_master_setup_single_model_self_play():
    """If only one model is supplied for N>1, every player shares it."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    one_model = _DummyModel("solo")
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=6),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=[one_model],
    )
    master._on_setup(**instance)
    players = master.get_players()
    assert len(players) == 6
    assert all(p.model is one_model for p in players)


def test_master_setup_initial_context_substitutions():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3)
    models = [_DummyModel(f"m{i}") for i in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # In 3p mode, $N_OTHER_PLAYERS$ should equal 2
    for p in players:
        # get_context_for returns a dict with "content" key
        ctx_dict = master.get_context_for(p)
        ctx = ctx_dict["content"] if ctx_dict is not None else ""
        assert "$YOUR_ROLE$" not in ctx, "role placeholder not substituted"
        assert "$N_OTHER_PLAYERS$" not in ctx, "n_other placeholder not substituted"
        assert "$YOUR_OBJECTS$" not in ctx
        assert "$GRID_SIZE$" not in ctx
        # And the substituted role/objects should appear
        assert p.role_name in ctx


def test_render_for_uses_compact():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    rendered = master._render_for(players[0])
    # Compact rendering produces a single-line-per-row format; just check it's a string
    assert isinstance(rendered, str)
    assert len(rendered) > 0


def test_render_for_masks_other_players_objects():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A (index 0) owns only "A". B-F's objects should be rendered as X.
    rendered_a = master._render_for(players[0])
    assert "B" not in rendered_a, "Player A should not see B"
    assert "X" in rendered_a, "other players' objects should be masked as X"


def test_render_targets_for_shows_only_own_targets():
    """The GOAL BOARD must list only the player's own object targets — never
    teammates' targets (which previously appeared masked as 'X->RxCy' and lured
    players into submitting moves for objects they cannot control)."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6)
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A (index 0) owns only "A".
    rendered_a = master._render_targets_for(players[0])
    assert "A->" in rendered_a, "Player A should see its own target"
    assert "X" not in rendered_a, "teammates' targets must be omitted, not masked as X"
    for other in ("B", "C", "D", "E", "F"):
        assert f"{other}->" not in rendered_a, f"Player A should not see {other}'s target"


def test_validate_response_accepts_valid_move():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A starts at (0, 0). Moving down means row index +1 → final position (1, 0).
    resp = "move: A down"
    assert master._validate_player_response(players[0], resp)


def test_validate_move_has_no_declared_target_param():
    import inspect
    from utils.board import Board
    sig = inspect.signature(Board.validate_move)
    assert "declared_target" not in sig.parameters


def test_old_coordinate_move_grammar_does_not_parse():
    """Clean break: the old `move: A to Rr,Cc (dir)` form must not parse; it
    falls through to the standard reprompt path (invalid_format), not a move."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    ok = master._validate_player_response(players[0], "move: A to R2,C1 (down)")
    assert ok is False
    assert master.reprompt_pending is True
    assert "could not be parsed" in _reprompt_text(master, players[0])


def test_validate_response_rejects_unowned_object():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Player A tries to move B (owned by Player B in 6p mode)
    resp = "move: B down"
    assert not master._validate_player_response(players[0], resp)


def test_validate_response_rejects_invalid_format():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="none")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    assert not master._validate_player_response(players[0], "no move here")


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
        "move: A down",
    )
    assert ok is True


def test_hybrid_missing_clause_reprompts():
    master = _setup_hybrid_master()
    players = master.get_players()
    ok = master._validate_player_response(
        players[0], "message: go team\nmove: A down"
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
            players[0], "message: chatty\nmove: A down"
        )
    assert master.aborted is True


def test_hybrid_multiline_message_passes():
    master = _setup_hybrid_master()
    players = master.get_players()
    ok = master._validate_player_response(
        players[0],
        "message: I'll head down the left so the rest of you have room.\n"
        "PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\n"
        "move: A down",
    )
    assert ok is True


def test_hybrid_prose_over_100_words_reprompts():
    master = _setup_hybrid_master()
    players = master.get_players()
    prose = " ".join(["word"] * 101)
    ok = master._validate_player_response(
        players[0],
        f"message: {prose} PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\n"
        "move: A down",
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert "100 words" in _reprompt_text(master, players[0])


def test_hybrid_request_over_10_words_reprompts():
    master = _setup_hybrid_master()
    players = master.get_players()
    req = " ".join(["x"] * 11)
    ok = master._validate_player_response(
        players[0],
        f"message: heads up. PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: {req}\n"
        "move: A down",
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert "REQUEST" in _reprompt_text(master, players[0])


def _setup_structured_master():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="structured")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=models,
    )
    master._on_setup(**instance)
    return master


def test_structured_request_over_10_words_reprompts():
    master = _setup_structured_master()
    players = master.get_players()
    req = " ".join(["x"] * 11)
    ok = master._validate_player_response(
        players[0],
        f"message: PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: {req}\n"
        "move: A down",
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert "REQUEST" in _reprompt_text(master, players[0])


def test_hybrid_template_example_matches_pattern():
    import re as _re
    from master import HYBRID_MSG_PATTERN, _schema_violation
    from pathlib import Path
    tmpl = Path(__file__).resolve().parents[1] / \
        "resources/initial_prompts/en/player_prompt_hybrid.template"
    text = tmpl.read_text(encoding="utf-8")
    lines = _re.findall(r"^message:\s*(.+)$", text, _re.MULTILINE)
    assert lines, "template must contain an example `message:` line"
    concrete = [m.strip() for m in lines if HYBRID_MSG_PATTERN.match(m.strip())]
    assert concrete, f"no example `message:` line matches HYBRID_MSG_PATTERN: {lines}"
    assert all(_schema_violation("hybrid", m) is None for m in concrete), \
        f"example message violates word caps: {concrete}"


def _reprompt_text(master, player):
    ctx = master.get_context_for(player)
    return (ctx.get("content") if isinstance(ctx, dict) else ctx) or ""


def test_masked_move_attempt_position_qualified_gives_semantic_hint():
    """`move: X@RxCy ...` (what models actually emit) must trigger a semantic
    redirect naming the player's own objects — not a generic 'could not parse'."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()  # Player A owns only "A"
    accepted = master._validate_player_response(players[0], "move: X@R1C2 right")
    assert accepted is False
    msg = _reprompt_text(master, players[0])
    assert "teammates" in msg, "should explain X marks teammates' objects"
    assert "A" in msg, "should name the player's own object(s)"
    assert "could not be parsed" not in msg, "must not fall back to the generic format error"


def test_masked_move_attempt_clean_gives_semantic_hint():
    """`move: X to ...` (parses, but X is a masked object) must also trigger the
    semantic redirect rather than the confusing 'Unknown object X'."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    accepted = master._validate_player_response(players[0], "move: X right")
    assert accepted is False
    msg = _reprompt_text(master, players[0])
    assert "teammates" in msg
    assert "A" in msg
    assert "Unknown object" not in msg


def _setup_master_hints(comm_protocol, masking_hints, num_players=3):
    from master import NPlayerMatrixGameMaster
    inst = _make_game_instance(num_players=num_players, comm_protocol=comm_protocol)
    inst["masking_hints"] = masking_hints
    models = [_DummyModel(f"m{i}") for i in range(num_players)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=num_players),
        experiment={"variant_config": {"comm_protocol": comm_protocol}},
        player_models=models,
    )
    master._on_setup(**inst)
    return master


def test_masking_hint_shown_when_flag_on():
    master = _setup_master_hints("none", masking_hints=True)
    player0 = master.get_players()[0]
    ok = master._validate_player_response(player0, "move: X down")
    assert ok is False
    assert "marks your teammates'" in _reprompt_text(master, player0)
    assert master.illegal_attempts == 1  # counting unaffected


def test_masking_hint_suppressed_when_flag_off():
    master = _setup_master_hints("none", masking_hints=False)
    player0 = master.get_players()[0]
    ok = master._validate_player_response(player0, "move: X down")
    assert ok is False
    assert "marks your teammates'" not in _reprompt_text(master, player0)
    assert master.illegal_attempts == 1  # counting still identical


def test_validate_move_unowned_object_message_lists_own_objects():
    """Board-level ownership error must name the player's own objects and must
    not use the 2-player legacy phrasing 'the other player'."""
    from utils.board import Board
    b = Board(grid_size=8, walls=set())
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    result = b.validate_move("B", "down", allowed_objects={"A"})
    assert result is not None
    code, err = result
    assert code == "not_owner"
    assert "A" in err, "error should name the player's own objects"
    assert "other player" not in err, "N-player phrasing must not say 'the other player'"


def test_structured_wellformed_message_passes():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="structured")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    ok = master._validate_player_response(
        players[0],
        "message: PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\nmove: A down",
    )
    assert ok is True


def test_structured_malformed_message_reprompts():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="structured")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    ok = master._validate_player_response(
        players[0], "message: let's go team\nmove: A down"
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    assert "PLAN:" in _reprompt_text(master, players[0])


def test_structured_malformed_thrice_aborts():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="structured")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    for _ in range(3):
        master._validate_player_response(
            players[0], "message: chatty\nmove: A down"
        )
    assert master.aborted is True


def test_structured_done_message_exempt():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="structured")
    for o in instance["player_objects"][0]:
        instance["target_positions"][o] = instance["start_positions"][o]
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    ok = master._validate_player_response(
        players[0], "message: done\ndone: all my objects are placed"
    )
    assert ok is True


def test_next_player_skips_done():
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    # Mark players B (index 1) and C (index 2) done
    master.player_done[1] = True
    master.player_done[2] = True
    # Current player is at index 0 → next should skip 1, 2 and land on 3
    master._current_player_idx = 0
    nxt = master._next_player()
    assert nxt is players[3]


def test_broadcast_after_valid_move_updates_other_contexts():
    """After Player A makes a valid move with a message, all other active
    players should see Player A's message in their next context."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="freeform")
    models = [_DummyModel() for _ in range(3)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(3) if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()

    # Player A produces a valid response with a message
    resp = (
        "message: hello teammates\n"
        "move: A down"
    )
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A down")

    # last_message[0] should now hold the message
    assert master.last_message[0] == "hello teammates"

    # Players B and C should have new contexts mentioning the message and the move.
    # Note: master.get_context_for returns a dict in this clemcore version.
    for idx in (1, 2):
        ctx_obj = master.get_context_for(players[idx])
        ctx = ctx_obj["content"] if isinstance(ctx_obj, dict) else ctx_obj
        assert "Player A says: hello teammates" in ctx
        assert "moved" not in ctx


def test_broadcast_with_masking_hides_unseen_object():
    """In 6p mode (every object owned by a different player), Player B learns
    of Player A's move only from the (masked) board listing — the master
    narrates neither the object's identity nor the move itself."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=6, comm_protocol="none")
    models = [_DummyModel() for _ in range(6)]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec() if "_make_game_spec" in globals() else None,
        experiment={"variant_config": {"comm_protocol": "none"}},
        player_models=models,
    )
    master._on_setup(**instance)
    players = master.get_players()
    resp = "move: A down"
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A down")

    # Player B (index 1) should not see any move narration at all.
    ctx_obj = master.get_context_for(players[1])
    ctx_b = ctx_obj["content"] if isinstance(ctx_obj, dict) else ctx_obj
    assert "moved" not in ctx_b


def test_move_is_not_narrated_to_recipient():
    """After a normal move, the recipient's context carries no master
    narration of the move — only messages + boards."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="freeform")
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    ctx = master._build_broadcast_context(master.get_players()[1])
    assert "moved" not in ctx
    assert "CURRENT BOARD:" in ctx


def test_scorer_success_score_capped():
    import numpy as np
    from master import NPlayerMatrixGameScorer, MOVE_COUNT
    from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE
    scorer = NPlayerMatrixGameScorer(
        game_name="matrixgame_covered_comm_6p",
        experiment={},
        game_instance={"optimal_moves": 20, "start_positions": {},
                       "target_positions": {}, "num_players": 6},
    )
    captured = {}
    scorer.log_episode_score = lambda k, v: captured.__setitem__(k, v)
    scorer.compute_scores({
        METRIC_ABORTED: 0,
        METRIC_SUCCESS: 1,
        MOVE_COUNT: 30,
    })
    # 20 / 30 * 100 = 66.67
    assert captured[BENCH_SCORE] == pytest.approx(66.67, rel=0.01)


def test_scorer_abort_returns_nan():
    import numpy as np
    from master import NPlayerMatrixGameScorer, MOVE_COUNT
    from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE
    scorer = NPlayerMatrixGameScorer(
        game_name="matrixgame_covered_comm_6p",
        experiment={},
        game_instance={"optimal_moves": 20, "start_positions": {},
                       "target_positions": {}, "num_players": 6},
    )
    captured = {}
    scorer.log_episode_score = lambda k, v: captured.__setitem__(k, v)
    scorer.compute_scores({
        METRIC_ABORTED: 1,
        METRIC_SUCCESS: 0,
        MOVE_COUNT: 5,
    })
    assert np.isnan(captured[BENCH_SCORE])


def test_clemgame_json_has_both_entries():
    with open(PKG / "clemgame.json") as f:
        entries = json.load(f)
    assert isinstance(entries, list)
    assert len(entries) == 2
    names = {e["game_name"] for e in entries}
    assert names == {"matrixgame_covered_comm_3p", "matrixgame_covered_comm_6p"}
    by_name = {e["game_name"]: e for e in entries}
    assert by_name["matrixgame_covered_comm_3p"]["players"] == 3
    assert by_name["matrixgame_covered_comm_6p"]["players"] == 6
    assert by_name["matrixgame_covered_comm_3p"]["roles"] == ["Player A", "Player B", "Player C"]
    assert by_name["matrixgame_covered_comm_6p"]["roles"] == [
        "Player A", "Player B", "Player C", "Player D", "Player E", "Player F"
    ]
    for e in entries:
        assert e["main_game"] == "matrixgame_covered_comm_nplayer"


def test_render_uses_ascii_grid_when_compact_board_false():
    """compact_board=false instances (the *_grid packages) must be sent the ASCII
    grid: masked teammates as X, walls as ##, and no inventory listing."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3)
    instance["compact_board"] = False
    instance["walls"] = [[4, 4]]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": instance["comm_protocol"]}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    player = master.get_players()[0]

    board = master._render_for(player)
    assert "Objects:" not in board
    assert "C1" in board and "+----+" in board, "expected the boxed grid header"
    assert "| ## |" in board, "walls must render as ##"
    assert "| X  |" in board, "teammates' objects must be masked as X on the grid"
    assert "| A  |" in board and "| B  |" in board, "own objects keep their letters"

    goal = master._render_targets_for(player)
    assert "Targets:" not in goal
    assert "+----+" in goal
    assert "X" not in goal, "teammates' targets are omitted, never masked"
    assert "| A  |" in goal and "| B  |" in goal
    for other in "CDEF":
        assert f"| {other}  |" not in goal, f"teammate target {other} must be omitted"


def _setup_lone_player_master(comm_protocol="none"):
    """3p master where Players B and C have already declared done, leaving
    Player A to finish alone. Returns (master, player_a)."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol=comm_protocol)
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": comm_protocol}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    master.player_done[1] = True
    master.player_done[2] = True
    return master, master.get_players()[0]


def test_lone_player_sees_fresh_board_after_own_move():
    """Once every teammate is done, the remaining player's next context must
    show the board AFTER its own move. (Regression: the master only refreshed
    the *other* players' contexts, so the lone player was re-served its
    previous context with a stale board.)"""
    master, player_a = _setup_lone_player_master()
    assert master._validate_player_response(player_a, "move: A down")
    master._on_valid_player_response(player_a, "A down")

    ctx = master.get_context_for(player_a)["content"]
    assert "A@R2C1" in ctx, ctx      # A started at R1C1 and moved down
    assert "A@R1C1" not in ctx, ctx
    assert "Your turn." in ctx, ctx  # a proper turn context, not the initial one


def test_lone_player_context_drops_stale_reprompt_error():
    """A lone player that was reprompted and then moved validly must not be
    shown the old 'Invalid move' error again on its next turn."""
    master, player_a = _setup_lone_player_master()
    # First an invalid move (up from the top row) -> reprompt context stored.
    assert not master._validate_player_response(player_a, "move: A up")
    assert "Invalid move" in master.get_context_for(player_a)["content"]
    # Then a valid move.
    assert master._validate_player_response(player_a, "move: A down")
    master._on_valid_player_response(player_a, "A down")

    ctx = master.get_context_for(player_a)["content"]
    assert "Invalid move" not in ctx, ctx
    assert "A@R2C1" in ctx, ctx


def _setup_master_with_player_a_placed(comm_protocol="none"):
    """3p master where Player A's objects already sit on their targets."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol=comm_protocol)
    for o in instance["player_objects"][0]:
        instance["target_positions"][o] = instance["start_positions"][o]
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": comm_protocol}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    return master, master.get_players()[0]


def test_done_after_reprompt_passes_the_turn():
    """A valid done declaration must pass the turn even if the same player was
    reprompted just before. (Regression: reprompt_pending stayed True, so the
    done player kept being asked, could then move an object off its target,
    and the game never terminated cleanly.)"""
    master, player_a = _setup_master_with_player_a_placed()
    assert not master._validate_player_response(player_a, "move: A up")   # off-grid -> reprompt
    assert master._should_pass_turn() is False
    assert master._validate_player_response(player_a, "done: all my objects are placed")
    master._on_valid_player_response(player_a, "done")
    assert master.player_done[0] is True
    assert master._should_pass_turn() is True
    assert master._next_player() is master.get_players()[1]


def test_game_ends_when_all_players_done_but_board_unsolved():
    """Defensive: if every player is marked done while the board is not solved,
    the game must stop instead of cycling through done players forever."""
    master, _ = _setup_master_with_player_a_placed()
    master.player_done = [True] * master.num_players
    assert master.board.is_solved() is False
    assert master._does_game_proceed() is False
    assert master.lost is True
    assert master.aborted is False


def test_bare_done_accepted_under_message_protocol():
    """Under a messaging protocol a bare `done:` line (no message: line) is a
    valid done declaration: the DONE OPTION is about ending participation, and
    the message part carries no information."""
    from master import _parse_response
    parsed = _parse_response("done: all my objects are placed", with_message=True)
    assert parsed is not None and parsed["is_done"] is True
    assert not parsed.get("message")
    master, player_a = _setup_master_with_player_a_placed(comm_protocol="freeform")
    assert master._validate_player_response(player_a, "done: all my objects are placed") is True
    master._on_valid_player_response(player_a, "done")
    assert master.player_done[0] is True
    assert master.last_message[0] is None


def test_bare_done_still_loses_when_objects_not_placed():
    """Accepting the bare form must not weaken the premature-done rule."""
    from master import NPlayerMatrixGameMaster
    instance = _make_game_instance(num_players=3, comm_protocol="freeform")
    master = NPlayerMatrixGameMaster(
        game_spec=_make_game_spec(num_players=3),
        experiment={"variant_config": {"comm_protocol": "freeform"}},
        player_models=[_DummyModel(f"m{i}") for i in range(3)],
    )
    master._on_setup(**instance)
    assert master._validate_player_response(master.get_players()[0], "done: all my objects are placed") is False
    assert master.lost is True
