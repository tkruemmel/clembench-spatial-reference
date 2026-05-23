"""Tests for SoloMatrixGameMaster setup, turn handling, and scoring."""

from pathlib import Path

import pytest

from clemcore.clemgame import GameSpec
from matrixgame_covered_solo.master import (
    SoloMatrixGameMaster,
    SoloMatrixPlayer,
)


class _StubModel:
    """Minimal Model substitute for masters that don't actually call the model."""
    def __init__(self):
        self.model_spec = type("Spec", (), {"model_name": "stub"})()

    def set_gen_arg(self, key, value):
        pass


def _instance(thinking: bool, prompt: str = "PROMPT $YOUR_OBJECTS$ $GRID_SIZE$") -> dict:
    return {
        "grid_size": 4,
        "walls": [],
        "objects": ["A", "B", "C", "D", "E", "F"],
        "start_positions": {
            "A": [0, 0], "B": [0, 1], "C": [0, 2],
            "D": [3, 0], "E": [3, 1], "F": [3, 2],
        },
        "target_positions": {
            "A": [2, 0], "B": [2, 1], "C": [2, 2],
            "D": [1, 0], "E": [1, 1], "F": [1, 2],
        },
        "optimal_moves": 8,
        "max_turns": 32,
        "max_retries": 2,
        "strict": False,
        "thinking": thinking,
        "spatial_level": "S1",
        "compact_board": False,
        "player_prompt": prompt,
    }


def _make_master(thinking: bool):
    game_spec = GameSpec(
        allow_underspecified=True,
        game_name="matrixgame_covered_solo",
        main_game="matrixgame_covered_solo",
        game_path=str(Path(__file__).parent.parent),
        players=1,
    )
    experiment = {"name": f"Solo_{'T' if thinking else 'S'}"}
    master = SoloMatrixGameMaster(game_spec, experiment, [_StubModel()])
    master._on_setup(**_instance(thinking))
    return master


def test_setup_registers_single_player_with_all_six_owned():
    master = _make_master(thinking=False)
    players = master.get_players()
    assert len(players) == 1
    assert isinstance(players[0], SoloMatrixPlayer)
    assert players[0].own_objects == set("ABCDEF")


def test_setup_board_has_all_six_objects_placed():
    master = _make_master(thinking=False)
    assert set(master.board.object_positions.keys()) == set("ABCDEF")


def test_setup_records_thinking_flag():
    master_t = _make_master(thinking=True)
    assert master_t.thinking is True
    assert master_t.with_message is True

    master_s = _make_master(thinking=False)
    assert master_s.thinking is False
    assert master_s.with_message is False


def test_initial_context_substitutes_placeholders():
    master = _make_master(thinking=False)
    # The initial context is on player[0]'s message queue. Look it up via the master.
    # DialogueGameMaster stores initial context per player; we can inspect it
    # via player.messages or master.messages_by_names depending on clemcore version.
    # Easier: just check that the build helper produces a string containing both
    # the placeholder expansion and the rendered board.
    ctx = master._build_initial_context(
        "PROMPT objects=$YOUR_OBJECTS$ grid=$GRID_SIZE$",
        "A, B, C, D, E, F",
    )
    assert "objects=A, B, C, D, E, F" in ctx
    assert "grid=4" in ctx
    assert "CURRENT BOARD:" in ctx
    assert "GOAL BOARD" in ctx


# ── Turn loop ──────────────────────────────────────────────────────────

def test_invalid_format_triggers_reprompt_then_abort():
    master = _make_master(thinking=False)
    # 1st bad response → reprompt
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.reprompt_pending is True
    assert master.aborted is False
    # 2nd bad response → reprompt
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.aborted is False
    # 3rd bad response (over max_retries=2) → abort
    assert master._validate_player_response(master.player, "garbage") is False
    assert master.aborted is True


def test_valid_move_applies_and_does_not_end_episode():
    master = _make_master(thinking=False)
    # A is at (0,0), all targets at (2,*); moving A down once is valid and not solving.
    ok = master._validate_player_response(master.player, "reason: down\nmove: A to R1,C0 (down)")
    assert ok is True
    master._on_valid_player_response(master.player, "A to R1,C0 (down)")
    assert master.board.object_positions["A"] == (1, 0)
    assert master.success is False
    assert master.aborted is False


def test_success_triggers_when_all_objects_at_targets():
    """Auto-success when every object reaches its target."""
    master = _make_master(thinking=False)
    # Hand-craft the board so only one move is needed to solve.
    # Starts: A,B,C @ row 0; D,E,F @ row 3. Targets: A,B,C @ row 2; D,E,F @ row 1.
    # Move every object except F to its target manually via board.apply_move,
    # then issue the final move (F up) through the master so it triggers the check.
    b = master.board
    # A: (0,0)->(2,0): down twice
    b.apply_move("A", "down"); b.apply_move("A", "down")
    # B: (0,1)->(2,1)
    b.apply_move("B", "down"); b.apply_move("B", "down")
    # C: (0,2)->(2,2)
    b.apply_move("C", "down"); b.apply_move("C", "down")
    # D: (3,0)->(1,0)
    b.apply_move("D", "up"); b.apply_move("D", "up")
    # E: (3,1)->(1,1)
    b.apply_move("E", "up"); b.apply_move("E", "up")
    # F: (3,2)->(2,2) is blocked — but F target is (1,2), so:
    # F at (3,2): up to (2,2) — occupied by C. So move sequence is more complex.
    # Simpler approach: jiggle starts so the final state is one move away from solved.
    # Instead, reset: hand-place everything except F one step from its target.
    # Just override positions directly via board internals for test purposes:
    b.object_positions = {
        "A": (2, 0), "B": (2, 1), "C": (2, 2),
        "D": (1, 0), "E": (1, 1), "F": (2, 2),  # F intentionally off-target
    }
    # Rebuild the grid array to match
    for r in range(b.grid_size):
        for c in range(b.grid_size):
            b.grid[r][c] = None
    for obj, (r, c) in b.object_positions.items():
        if b.grid[r][c] is None:
            b.grid[r][c] = obj
    # F is "stuck" on top of C at (2,2) — that's an invalid board state for the test.
    # Cleaner: just place F at (2,2) is wrong; use a free cell instead.
    # Reset properly: place F at (2,3) — one move LEFT from target (1,2)? No, (2,3)->(1,3) is up, not (1,2).
    # F's target is (1,2). Place F at (2,2)? C is there. Place F at (1,3)? then move left to (1,2).
    b.object_positions["F"] = (1, 3)
    b.grid = [[None] * 4 for _ in range(4)]
    for obj, (r, c) in b.object_positions.items():
        b.grid[r][c] = obj

    # Now the only object not on target is F at (1,3); F's target is (1,2). One move left.
    ok = master._validate_player_response(master.player, "reason: left\nmove: F to R1,C2 (left)")
    assert ok is True
    master._on_valid_player_response(master.player, "F to R1,C2 (left)")
    assert master.success is True
    assert master.aborted is False


def test_does_game_proceed_false_after_max_turns():
    master = _make_master(thinking=False)
    master.max_turns = 1
    # Apply one valid move
    master._validate_player_response(master.player, "reason: down\nmove: A to R1,C0 (down)")
    master._on_valid_player_response(master.player, "A to R1,C0 (down)")
    assert master._does_game_proceed() is False
    assert master.aborted is True


def test_invalid_move_reprompts_with_format_reminder():
    """Moving into a wall is rejected; reprompt context is set."""
    master = _make_master(thinking=False)
    # Place a wall directly below A's start so moving A down hits it.
    master.board.walls.add((1, 0))
    ok = master._validate_player_response(master.player, "reason: down\nmove: A to R1,C0 (down)")
    assert ok is False
    assert master.reprompt_pending is True
    assert master.aborted is False


# ── Scorer ─────────────────────────────────────────────────────────────

import math
from clemcore.clemgame.metrics import BENCH_SCORE
from matrixgame_covered_solo.master import SoloMatrixGameScorer


def _episode(aborted=0, success=0, move_count=0) -> dict:
    return {
        "Aborted": aborted,
        "Success": success,
        "Move Count": move_count,
    }


def _scorer(optimal=10) -> SoloMatrixGameScorer:
    return SoloMatrixGameScorer(
        game_name="matrixgame_covered_solo",
        experiment={"name": "test"},
        game_instance={"optimal_moves": optimal},
    )


def _episode_score(scorer: SoloMatrixGameScorer, key: str):
    """Get the most recent value logged via log_episode_score for `key`."""
    # clemcore's GameScorer keeps scores in self.scores under 'episode scores'.
    return scorer.scores["episode scores"].get(key)


def test_bench_score_nan_on_abort():
    s = _scorer()
    s.compute_scores(_episode(aborted=1))
    value = _episode_score(s, BENCH_SCORE)
    assert isinstance(value, float) and math.isnan(value)


def test_bench_score_zero_on_lose():
    s = _scorer()
    s.compute_scores(_episode(aborted=0, success=0, move_count=15))
    assert _episode_score(s, BENCH_SCORE) == 0


def test_bench_score_full_on_optimal_success():
    s = _scorer(optimal=10)
    s.compute_scores(_episode(aborted=0, success=1, move_count=10))
    assert _episode_score(s, BENCH_SCORE) == 100


def test_bench_score_partial_on_overrun_success():
    s = _scorer(optimal=10)
    s.compute_scores(_episode(aborted=0, success=1, move_count=20))
    assert _episode_score(s, BENCH_SCORE) == 50.0


# ── End-to-end smoke ──────────────────────────────────────────────────

from matrixgame_covered_solo.master import SoloMatrixGameBenchmark


def test_benchmark_creates_master_and_scorer():
    game_spec = GameSpec(
        allow_underspecified=True,
        game_name="matrixgame_covered_solo",
        main_game="matrixgame_covered_solo",
        game_path=str(Path(__file__).parent.parent),
        players=1,
    )
    bench = SoloMatrixGameBenchmark(game_spec)
    master = bench.create_game_master({"name": "x"}, [_StubModel()])
    assert isinstance(master, SoloMatrixGameMaster)
    scorer = bench.create_game_scorer({"name": "x"}, {"optimal_moves": 5})
    assert isinstance(scorer, SoloMatrixGameScorer)


def test_one_turn_smoke():
    """Run one validate+apply cycle using the stub player's _custom_response."""
    master = _make_master(thinking=False)
    response = master.player._custom_response(context=None)
    ok = master._validate_player_response(master.player, response)
    # The stub _custom_response targets R1,C1 — may be invalid depending on start position,
    # but it must at least parse without raising.
    assert isinstance(ok, bool)
