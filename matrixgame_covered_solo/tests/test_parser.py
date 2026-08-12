"""Tests for the response parser in matrixgame_covered_solo.master."""

import pytest

from matrixgame_covered_solo.master import _parse_response


# ── silent mode (with_message=False) ──────────────────────────────────

def test_parse_silent_valid_lowercase_dir():
    response = "move: A right"
    parsed = _parse_response(response, with_message=False)
    assert parsed == {
        "object": "A",
        "direction": "right",
    }


def test_parse_silent_valid_uppercase_dir():
    response = "MOVE: B UP"
    parsed = _parse_response(response, with_message=False)
    assert parsed is not None
    assert parsed["object"] == "B"
    assert parsed["direction"] == "up"


def test_parse_silent_rejects_message_line():
    """In silent mode, a response containing 'message:' must not parse."""
    response = "message: hi\nmove: A right"
    parsed = _parse_response(response, with_message=False)
    assert parsed is None


def test_parse_silent_rejects_malformed():
    assert _parse_response("just some text", with_message=False) is None
    assert _parse_response("move: A", with_message=False) is None  # bad move format


# ── thinking mode (with_message=True) ──────────────────────────────────

def test_parse_thinking_valid():
    response = (
        "message: plan B last.\n"
        "move: A right"
    )
    parsed = _parse_response(response, with_message=True)
    assert parsed == {
        "message": "plan B last.",
        "object": "A",
        "direction": "right",
    }


def test_parse_thinking_rejects_missing_message():
    """In thinking mode, the message: line is required."""
    response = "move: A right"
    parsed = _parse_response(response, with_message=True)
    assert parsed is None


def test_parse_thinking_rejects_malformed():
    assert _parse_response("just text", with_message=True) is None


def test_silent_is_bare_anchored_move_line():
    from master import _parse_response
    parsed = _parse_response("move: A right", with_message=False)
    assert parsed == {"object": "A", "direction": "right"}
    # any preamble (including a reason: line) fails the anchored parse
    assert _parse_response("I will now move.\nmove: A right", with_message=False) is None
    assert _parse_response("reason: x\nmove: A right", with_message=False) is None


def test_thinking_is_message_then_move():
    from master import _parse_response
    parsed = _parse_response("message: note to self\nmove: B up", with_message=True)
    assert parsed["message"] == "note to self"
    assert parsed["object"] == "B"
    assert "reason" not in parsed
    # a reason: field between message and move is no longer part of the grammar
    assert _parse_response("message: m\nreason: r\nmove: B up", with_message=True) is None
    # missing message: fails
    assert _parse_response("move: B up", with_message=True) is None


def test_parses_direction_only_move():
    from master import _parse_response
    r = _parse_response("move: A right", with_message=False)
    assert r == {"object": "A", "direction": "right"}


def test_old_coordinate_form_no_longer_parses():
    from master import _parse_response
    assert _parse_response("move: A to R3,C5 (right)", with_message=False) is None


def test_thinking_message_then_direction_move():
    from master import _parse_response
    r = _parse_response("message: go\nmove: B down", with_message=True)
    assert r["object"] == "B" and r["direction"] == "down" and r["message"] == "go"
