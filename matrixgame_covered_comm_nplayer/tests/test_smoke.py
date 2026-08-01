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
    text = "move: A to R3,C5 (down)"
    parsed = _parse_response(text, with_message=False)
    assert parsed["object"] == "A"
    assert parsed["direction"] == "down"
    assert parsed["target"] == "R3,C5"


def test_parse_response_with_message():
    from master import _parse_response
    text = "message: moving A\nmove: A to R3,C5 (down)"
    parsed = _parse_response(text, with_message=True)
    assert parsed["message"] == "moving A"
    assert parsed["object"] == "A"


def test_grammar_anchored_no_reason():
    from master import _parse_response
    # channel off
    assert _parse_response("move: A to R2,C1 (down)", with_message=False) is not None
    assert _parse_response("thinking first\nmove: A to R2,C1 (down)", with_message=False) is None
    assert _parse_response("reason: r\nmove: A to R2,C1 (down)", with_message=False) is None
    assert _parse_response("done: all my objects are placed", with_message=False)["is_done"] is True
    # channel on
    p = _parse_response("message: hi\nmove: A to R2,C1 (down)", with_message=True)
    assert p["message"] == "hi" and "reason" not in p
    assert _parse_response("message: hi\nreason: r\nmove: A to R2,C1 (down)", with_message=True) is None
    d = _parse_response("message: done\ndone: all placed", with_message=True)
    assert d["is_done"] is True and d["message"] == "done"
    assert _parse_response("preamble\nmessage: hi\nmove: A to R2,C1 (down)", with_message=True) is None


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
        return "move: A to R1,C1 (down)"


def _make_game_spec(num_players=6):
    """Build a minimal GameSpec suitable for NPlayerMatrixGameMaster constructor."""
    from clemcore.clemgame.registry import GameSpec
    return GameSpec(
        game_name="matrixgame_covered_comm_nplayer",
        game_path="matrixgame_covered_comm_nplayer",
        players=num_players,
    )


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
    # Player A starts at (0, 0). Moving down means row index +1 → final position (1, 0)
    # which in 1-indexed is R2,C1.
    resp = "move: A to R2,C1 (down)"
    assert master._validate_player_response(players[0], resp)


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
    resp = "move: B to R2,C2 (down)"
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
    accepted = master._validate_player_response(players[0], "move: X@R1C2 to R1,C3 (right)")
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
    accepted = master._validate_player_response(players[0], "move: X to R1,C3 (right)")
    assert accepted is False
    msg = _reprompt_text(master, players[0])
    assert "teammates" in msg
    assert "A" in msg
    assert "Unknown object" not in msg


def test_validate_move_unowned_object_message_lists_own_objects():
    """Board-level ownership error must name the player's own objects and must
    not use the 2-player legacy phrasing 'the other player'."""
    from utils.board import Board
    b = Board(grid_size=8, walls=set())
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    err = b.validate_move("B", "down", allowed_objects={"A"})
    assert err is not None
    assert "A" in err, "error should name the player's own objects"
    assert "other player" not in err, "N-player phrasing must not say 'the other player'"


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
        "move: A to R2,C1 (down)"
    )
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A to R2,C1 (down)")

    # last_message[0] should now hold the message
    assert master.last_message[0] == "hello teammates"

    # Players B and C should have new contexts mentioning the message and the move.
    # Note: master.get_context_for returns a dict in this clemcore version.
    for idx in (1, 2):
        ctx_obj = master.get_context_for(players[idx])
        ctx = ctx_obj["content"] if isinstance(ctx_obj, dict) else ctx_obj
        assert "Player A says: hello teammates" in ctx
        assert "Player A moved A down" in ctx or "Player A moved a blocked object down" in ctx


def test_broadcast_with_masking_hides_unseen_object():
    """In 6p mode (every object owned by a different player), Player B should
    see Player A's move as 'a blocked object' because B can't see A."""
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
    resp = "move: A to R2,C1 (down)"
    assert master._validate_player_response(players[0], resp)
    master._on_valid_player_response(players[0], "A to R2,C1 (down)")

    # Player B (index 1) should see masked move
    ctx_obj = master.get_context_for(players[1])
    ctx_b = ctx_obj["content"] if isinstance(ctx_obj, dict) else ctx_obj
    assert "Player A moved a blocked object down" in ctx_b
    assert "Player A moved A down" not in ctx_b


def test_scorer_success_score_capped():
    import numpy as np
    from master import NPlayerMatrixGameScorer, MOVE_COUNT
    from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, BENCH_SCORE
    scorer = NPlayerMatrixGameScorer(
        game_name="matrixgame_covered_comm_6p",
        experiment={},
        game_instance={"optimal_moves": 20},
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
        game_instance={"optimal_moves": 20},
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
