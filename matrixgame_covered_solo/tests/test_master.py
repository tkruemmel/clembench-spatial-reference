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
