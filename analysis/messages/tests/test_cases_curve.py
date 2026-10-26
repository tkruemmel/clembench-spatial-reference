"""Tests for cases.py (spec §10) and curve.py (spec §9.3) on constructed episodes."""
from analysis.messages import cases, curve, grammar


def _ep(eid, model="qwen38", proto="freeform", suite="dyad", level="S1", board=1, success=1, aborted=0,
        moves=10, optimal=8):
    turns = [dict(done=False, is_msg=False) for _ in range(moves)]
    return dict(meta=dict(id=eid, model=model, proto=proto, suite=suite, level=level, board=board,
                          success=success, aborted=aborted), turns=turns, ins=dict(optimal_moves=optimal))


def test_quality_scores_aborts_zero():
    assert cases.quality(_ep("a")) == 80.0
    assert cases.quality(_ep("b", success=0)) == 0.0
    assert cases.quality(_ep("c", success=0, aborted=1)) == 0.0
    assert cases.quality(_ep("d", moves=5)) == 100.0  # capped


def test_deviant_thresholds():
    # a cell succeeding 4 of 5 (80 %): a featured failure is deviant
    eps = [_ep(f"s{i}", board=i) for i in range(4)] + [_ep("f", board=9, success=0)]
    out = cases.select(eps, {"f": True, "s0": True})
    kinds = {(r["kind"], r["episode"]) for r in out}
    assert ("deviant", "f") in kinds and ("typical", "s0") in kinds
    # at 3 of 4 (75 %) nothing is deviant
    out = cases.select(eps[1:], {"f": True})
    assert not any(r["kind"] == "deviant" for r in out)


def test_most_similar_varies_one_factor():
    eps = [_ep("a", proto="freeform", success=1), _ep("b", proto="hybrid", success=0),
           _ep("c", proto="hybrid", model="dsv41", success=0)]
    out = [r for r in cases.select(eps, {"a": True}) if r["kind"].startswith("most similar")]
    assert [(r["kind"], r["episode"], r["pair"]) for r in out] == [("most similar (proto)", "a", "b")]
    assert out[0]["p_discordant_null"] == 1.0


def test_d2_values_truth_and_frames():
    an = grammar.analyse("E is at (R2,C6). B wait at (R1,C1)", "dev", "freeform")
    ep = dict(meta=dict(id="x"), owner={"E": "Player B", "B": "Player B", "A": "Player A"},
              turns=[dict(is_msg=True, dup=False, seat="Player A", ply=0, board={"E": (1, 5), "B": (0, 0)})])
    A = {("x", 0): an}
    assert curve.d2_values([ep], A, "strict") == {"x": (1, 1)}  # the directive is excluded
    assert curve.d2_values([ep], A, "lenient") == {"x": (1, 1)}  # lenient also drops directives
    ep["turns"][0]["board"]["E"] = (0, 0)
    assert curve.d2_values([ep], A, "strict") == {"x": (0, 1)}
