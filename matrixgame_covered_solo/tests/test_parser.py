"""Tests for the response parser in matrixgame_covered_solo.master."""

import pytest

from matrixgame_covered_solo.master import _parse_response


# ── silent mode (with_message=False) ──────────────────────────────────

def test_parse_silent_valid_lowercase_dir():
    response = "reason: A needs to move right.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=False)
    assert parsed == {
        "reason": "A needs to move right.",
        "object": "A",
        "direction": "right",
        "target": "R3,C5",
    }


def test_parse_silent_valid_uppercase_dir():
    response = "REASON: B goes up.\nMOVE: B to R1,C2 (UP)"
    parsed = _parse_response(response, with_message=False)
    assert parsed is not None
    assert parsed["object"] == "B"
    assert parsed["direction"] == "up"


def test_parse_silent_rejects_message_line():
    """In silent mode, a response containing 'message:' must not parse."""
    response = "message: hi\nreason: A needs to move.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=False)
    assert parsed is None


def test_parse_silent_rejects_malformed():
    assert _parse_response("just some text", with_message=False) is None
    assert _parse_response("move: A to R3,C5 (right)", with_message=False) is None  # missing reason
    assert _parse_response("reason: hi\nmove: A right", with_message=False) is None  # bad move format


# ── thinking mode (with_message=True) ──────────────────────────────────

def test_parse_thinking_valid():
    response = (
        "message: plan B last.\n"
        "reason: move A right to target.\n"
        "move: A to R3,C5 (right)"
    )
    parsed = _parse_response(response, with_message=True)
    assert parsed == {
        "message": "plan B last.",
        "reason": "move A right to target.",
        "object": "A",
        "direction": "right",
        "target": "R3,C5",
    }


def test_parse_thinking_rejects_missing_message():
    """In thinking mode, the message: line is required."""
    response = "reason: hi.\nmove: A to R3,C5 (right)"
    parsed = _parse_response(response, with_message=True)
    assert parsed is None


def test_parse_thinking_rejects_malformed():
    assert _parse_response("just text", with_message=True) is None
