"""Self-check and visibility tests for corpus.py (spec §3, §9.1)."""
import copy
import os

import pytest

from analysis.messages import corpus


@pytest.fixture(scope="module")
def ep():
    p = next(p for p in corpus.episode_paths("final", "dsv41", "3p") if "/S3_hybrid" in p)
    return corpus.parse(p, "final", "dsv41")


def test_real_episode_passes(ep):
    corpus.selfcheck(ep)
    assert ep["meta"]["suite"] == "3p" and ep["meta"]["proto"] == "hybrid"
    assert all(t["slots"] for t in ep["turns"] if t["is_msg"])


@pytest.mark.parametrize("corrupt, check", [
    (lambda e: e["owner"].__setitem__(e["turns"][0]["obj"], "Player Z"), "A2"),
    (lambda e: e["turns"][0].__setitem__("dst", (-1, 0)), "A4"),
    (lambda e: e["it"].__setitem__("Request Count", -1), "A8"),
    (lambda e: e["turns"][3].__setitem__("vis", []), "V1"),
])
def test_selfcheck_catches(ep, corrupt, check):
    bad = copy.deepcopy(ep)
    corrupt(bad)
    with pytest.raises(AssertionError, match=check):
        corpus.selfcheck(bad)


def _fake(arm, seats, plies):
    """plies: [(seat, msg, done)]"""
    turns = [dict(ply=i, seat=s, msg=m, done=d, retry=False) for i, (s, m, d) in enumerate(plies)]
    return dict(meta=dict(arm=arm, proto="freeform"), turns=turns, seats=seats)


def test_visible_dev_latest_per_seat():
    e = _fake("dev", ["Player A", "Player B"], [("Player A", "a1", False), ("Player B", "b1", False),
                                               ("Player A", "a2", False), ("Player B", "b2", False)])
    assert corpus.visible(e, 3, False) == [1, 2]  # own latest (b1) and A's latest (a2)
    assert corpus.visible(e, 3, True) == []       # retry contexts carry no messages


def test_visible_final_window():
    plies = [(s, f"{s[-1]}{k}", False) for k in range(6) for s in ("Player Green", "Player Orange")]
    e = _fake("final", ["Player Green", "Player Orange"], plies)
    # Green's 6th turn (ply 10): window starts at its 4th-last own turn (ply 2), as history.py
    assert corpus.visible(e, 10, False) == list(range(2, 10))
    assert corpus.visible(e, 10, True) == list(range(2, 10))
    e["turns"][5]["done"], e["turns"][5]["msg"] = True, "done"
    assert 5 not in corpus.visible(e, 10, False)  # a bare done shows as `declared done`


def test_scope_from_path():
    assert corpus.scope_of("final", "dsv41", "3p", "structured", "S2", 1) == "test"
    assert corpus.scope_of("final", "dsv41", "6p", "hybrid", "S1", 9) == "calib"
    assert corpus.scope_of("final", "dsv41", "6p", "hybrid", "S4", 9) == "test"


def test_master_regexes_loaded():
    assert corpus.SCHEMA["structured"].match("PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none")
    assert corpus.SCHEMA["hybrid"].match("Going down. PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none")
    assert not corpus.SCHEMA["hybrid"].match("PLAN: move A from (R2,C3) to (R4,C3) — REQUEST: none")
