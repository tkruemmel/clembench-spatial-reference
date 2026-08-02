"""Tests for the two masked-object fixes in matrixgame_covered_comm:

1. The goal board shows only the player's own targets (partner's targets are
   omitted, not masked as 'X').
2. A move on a masked partner object ('X' or 'X@RxCy') triggers a targeted
   semantic redirect naming the player's own objects, and the board-level
   ownership error lists the player's own objects.
"""

import pytest
from pathlib import Path

PKG = Path(__file__).parent.parent


# ── Board-level (pure) ───────────────────────────────────────────────────

def test_render_targets_compact_omits_partner_targets():
    from utils.board import Board
    b = Board(grid_size=8, walls=set())
    for i, o in enumerate(["A", "B", "C", "D", "E", "F"]):
        b.set_target(o, 0, i)
    rendered = b.render_targets_compact(visible_objects={"A", "B", "C"})
    for own in ("A->", "B->", "C->"):
        assert own in rendered, f"own target {own} should be shown"
    assert "X" not in rendered, "partner targets must be omitted, not masked as X"
    for other in ("D->", "E->", "F->"):
        assert other not in rendered, f"partner target {other} should be omitted"


def test_render_targets_for_player_omits_partner_targets():
    """Non-compact (grid) target view must also omit partner targets."""
    from utils.board import Board, BLOCKED_MARKER
    b = Board(grid_size=8, walls=set())
    for i, o in enumerate(["A", "B", "C", "D", "E", "F"]):
        b.set_target(o, i, i)
    rendered = b.render_targets_for_player({"A", "B", "C"})
    assert BLOCKED_MARKER not in rendered, "partner targets must not appear as X in the grid"
    for own in ("A", "B", "C"):
        assert own in rendered


def test_validate_move_unowned_object_message_lists_own_objects():
    from utils.board import Board
    b = Board(grid_size=8, walls=set())
    b.place_object("A", 0, 0)
    b.place_object("D", 0, 1)
    result = b.validate_move("D", "down", allowed_objects={"A", "B", "C"})
    assert result is not None
    code, err = result
    assert code == "not_owner"
    assert "A" in err, "error should name the player's own objects"
    assert "other player" not in err, "legacy 2-player phrasing should be gone"


def test_validate_move_returns_typed_codes():
    from utils.board import Board
    b = Board(grid_size=8, walls={(0, 3)})
    b.place_object("A", 0, 0)
    b.place_object("B", 0, 1)
    assert b.validate_move("A", "right")[0] == "occupied"   # B is there
    assert b.validate_move("A", "up")[0] == "off_grid"
    assert b.validate_move("B", "right", allowed_objects={"A"})[0] == "not_owner"
    assert b.validate_move("Z", "down")[0] == "unknown_object"
    b2 = Board(grid_size=8, walls={(1, 0)})
    b2.place_object("A", 0, 0)
    assert b2.validate_move("A", "down")[0] == "wall"
    assert b2.validate_move("A", "right") is None


def test_masked_object_move_pattern():
    from master import MASKED_OBJECT_MOVE_PATTERN as P
    assert P.search("move: X@R2C3 to R2,C4 (right)")
    assert P.search("move: X to R2,C4 (right)")
    assert P.search("move: x@r2c3 to r2,c4 (right)")  # case-insensitive
    assert not P.search("move: A to R2,C4 (right)")
    assert not P.search("the X at R2C3 is blocking\nmove: A to R2,C4 (right)")


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


# ── Master-driven ────────────────────────────────────────────────────────

def _make_game_instance(comm_protocol="none"):
    objs = ["A", "B", "C", "D", "E", "F"]
    return {
        "grid_size": 8,
        "walls": [],
        "objects": objs,
        "player_a_objects": ["A", "B", "C"],
        "player_b_objects": ["D", "E", "F"],
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
            "You are playing with a partner. You control $YOUR_OBJECTS$ on a "
            "$GRID_SIZE$x$GRID_SIZE$ grid."
        ),
    }


class _DummyModel:
    def __init__(self, name="dummy"):
        self.name = name
        self._gen_args = {}

    def set_gen_arg(self, key, value):
        self._gen_args[key] = value

    def generate_response(self, *_a, **_k):
        return "move: A to R1,C1 (down)"


def _make_game_spec():
    from clemcore.clemgame.registry import GameSpec
    return GameSpec(
        game_name="matrixgame_covered_comm",
        game_path="matrixgame_covered_comm",
        players=2,
    )


def _setup_master(comm_protocol="none"):
    from master import MatrixGameMaster
    master = MatrixGameMaster(
        game_spec=_make_game_spec(),
        experiment={"variant_config": {"comm_protocol": comm_protocol}},
        player_models=[_DummyModel("a"), _DummyModel("b")],
    )
    master._on_setup(**_make_game_instance(comm_protocol))
    return master


def _reprompt_text(master, player):
    ctx = master.get_context_for(player)
    return (ctx.get("content") if isinstance(ctx, dict) else ctx) or ""


def test_render_targets_for_shows_only_own_targets():
    master = _setup_master()
    rendered = master._render_targets_for(master.player_a)  # owns A, B, C
    for own in ("A->", "B->", "C->"):
        assert own in rendered
    assert "X" not in rendered
    for other in ("D->", "E->", "F->"):
        assert other not in rendered


def test_masked_move_attempt_position_qualified_gives_semantic_hint():
    master = _setup_master()
    accepted = master._validate_player_response(
        master.player_a, "move: X@R1C5 to R1,C6 (right)"
    )
    assert accepted is False
    msg = _reprompt_text(master, master.player_a)
    assert "partner" in msg, "should explain X marks the partner's objects"
    assert "A, B, C" in msg, "should name the player's own objects"
    assert "could not be parsed" not in msg


def test_masked_move_attempt_clean_gives_semantic_hint():
    master = _setup_master()
    accepted = master._validate_player_response(
        master.player_a, "move: X to R1,C6 (right)"
    )
    assert accepted is False
    msg = _reprompt_text(master, master.player_a)
    assert "partner" in msg
    assert "A, B, C" in msg
    assert "Unknown object" not in msg


def test_structured_wellformed_message_passes():
    master = _setup_master("structured")
    ok = master._validate_player_response(
        master.player_a,
        "message: PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none\nmove: A to R2,C1 (down)",
    )
    assert ok is True


def test_structured_malformed_message_reprompts():
    master = _setup_master("structured")
    ok = master._validate_player_response(
        master.player_a, "message: let's go team\nmove: A to R2,C1 (down)"
    )
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    assert "PLAN:" in _reprompt_text(master, master.player_a)


def test_structured_malformed_thrice_aborts():
    master = _setup_master("structured")
    for _ in range(3):
        master._validate_player_response(
            master.player_a, "message: chatty\nmove: A to R2,C1 (down)"
        )
    assert master.aborted is True


def test_structured_done_message_exempt():
    from master import MatrixGameMaster
    gi = _make_game_instance("structured")
    for o in ("A", "B", "C"):
        gi["target_positions"][o] = gi["start_positions"][o]
    master = MatrixGameMaster(
        game_spec=_make_game_spec(),
        experiment={"variant_config": {"comm_protocol": "structured"}},
        player_models=[_DummyModel("a"), _DummyModel("b")],
    )
    master._on_setup(**gi)
    ok = master._validate_player_response(
        master.player_a, "message: done\ndone: all my objects are placed"
    )
    assert ok is True


def test_blocked_vs_illegal_counters():
    master = _setup_master()
    # A at R1,C1; right into B at R1,C2 -> blocked
    master._validate_player_response(master.player_a, "move: A to R1,C2 (right)")
    assert master.blocked_attempts == 1 and master.illegal_attempts == 0
    # A up leaves the grid -> illegal
    master._validate_player_response(master.player_a, "move: A to R0,C1 (up)")
    assert master.illegal_attempts == 1
    # legal move counts as an attempt but neither illegal nor blocked
    master._validate_player_response(master.player_a, "move: A to R2,C1 (down)")
    assert master.move_attempts == 3


def test_move_log_has_from_to():
    master = _setup_master()
    master._validate_player_response(master.player_a, "move: A to R2,C1 (down)")
    master._on_valid_player_response(master.player_a, "A to R2,C1 (down)")
    entry = master.move_log[-1]
    assert entry["from"] == [0, 0] and entry["to"] == [1, 0]
