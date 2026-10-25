"""§5.4 accuracy, §9.2 bounds and test-arm rule, on constructed episodes."""
import collections

import numpy as np

from analysis.messages import corpus, grammar
from analysis.messages.measures import accuracy as acc
from analysis.messages.measures import coverage as cov

OWNER = {"A": "Player A", "B": "Player A", "C": "Player B", "D": "Player B"}
START = {"A": (0, 0), "B": (4, 4), "C": (2, 2), "D": (4, 0)}


def episode(moves, arm="dev", model="qwen38", proto="freeform", eid="e"):
    """Minimal episode: moves = [(seat, obj, dir, message)] replayed from START."""
    pos, turns = dict(START), []
    for ply, (seat, obj, d, msg) in enumerate(moves):
        src = pos[obj]
        dst = (src[0] + corpus.DIRS[d][0], src[1] + corpus.DIRS[d][1])
        turns.append(dict(ply=ply, seat=seat, si=0, obj=obj, dir=d, src=src, dst=dst, done=False, msg=msg,
                          board=dict(pos), is_msg=True, dup=False, retry=False,
                          slots=corpus.slots(msg, proto)))
        pos[obj] = dst
    for i, t in enumerate(turns):
        t["after"] = turns[i + 1]["board"] if i + 1 < len(turns) else dict(pos)
    meta = dict(id=eid, arm=arm, model=model, tier=corpus.TIER[model], suite="dyad", level="S1", board=1,
                proto=proto)
    ep = dict(meta=meta, owner=OWNER, seats=["Player A", "Player B"], turns=turns, done_ply={},
              objs={"Player A": ["A", "B"], "Player B": ["C", "D"]})
    for i, t in enumerate(turns):
        t["vis"] = corpus.visible(ep, i, False)
    return ep


def analyse(eps):
    return {(ep["meta"]["id"], t["ply"]): grammar.analyse(t["msg"], ep["meta"]["arm"], ep["meta"]["proto"],
                                                         t["slots"])
            for ep in eps for t in ep["turns"]}


def pairs(ep):
    cands = acc.candidates([ep], analyse([ep]))
    return [(c["i"], c["kind"], c["letter"], c["cell"], c["own"], c["true"]) for c in cands if c["kept"]]


def test_truth_rule_own_before_or_after_teammate_before():
    ep = episode([("Player A", "A", "right", "A is at (R1,C2). C is at (R3,C3)."),     # A moves (0,0)->(0,1)
                  ("Player B", "C", "up", "A is at (R1,C1). C is at (R2,C3)."),       # C moves (2,2)->(1,2)
                  ("Player A", "A", "right", "A is at (R1,C2). C is at (R3,C3).")])   # A (0,1)->(0,2)
    got = pairs(ep)
    assert (0, "loc", "A", (0, 1), True, True) in got     # own, true at B_t+ only
    assert (0, "loc", "C", (2, 2), False, True) in got    # teammate, true at B_t-
    assert (1, "loc", "A", (0, 0), False, False) in got   # teammate, stale: A left (R1,C1) at ply 0
    assert (1, "loc", "C", (1, 2), True, True) in got     # own, true at B_t+
    assert (2, "loc", "A", (0, 1), True, True) in got     # own, true at B_t- only
    assert (2, "loc", "C", (2, 2), False, False) in got   # teammate, C moved away
    # a teammate pair is never scored at B_t+ (teammates do not move in S's turn; forced here)
    t = ep["turns"][0]
    t["after"] = dict(t["after"], C=(3, 3))
    c = acc._pair(ep, 0, "loc", "prose", "rev", "C", (3, 3), set(), True, None, None, grammar.DEFAULT)
    assert not c["true"]
    o = acc._pair(ep, 0, "loc", "prose", "rev", "A", (0, 1), set(), True, None, None, grammar.DEFAULT)
    assert o["own"] and o["true"]


def test_plan_from_primary_channel():
    msg = "PLAN: move A from (R1,C1) to (R1,C2) — REQUEST: none"
    ep = episode([("Player A", "A", "right", msg)], proto="structured")
    assert ep["turns"][0]["slots"]["plan_from"] == (0, 0)
    rows = acc.own_rows([ep], acc.candidates([ep], analyse([ep])))
    plan = next(r for r in rows if r["channel"] == "plan_from")
    assert (plan["n"], plan["k"], plan["prop"]) == (1, 1, 1.0)


def canon(s, letter, cell, kind="loc"):
    an = grammar.analyse(s, "dev", "freeform")
    p = next(p for p in an[kind] if p["letter"] == letter)
    return acc.canonical(acc.clause_tokens(an, p["clause"]), kind, letter, cell)


def test_canonical_form_filter():
    assert canon("C at (R2,C3)", "C", (1, 2))
    assert canon("Your C is at (R2,C3) now", "C", (1, 2))
    assert canon("I see C from (R2,C3) moving", "C", (1, 2), "frm")
    assert not canon("C is currently at (R2,C3)", "C", (1, 2))
    assert not canon("C, at (R2,C3)", "C", (1, 2))
    assert not canon("C sits at (R2,C3)", "C", (1, 2))
    # the reverse rule is never canonical, and canon is set only on kept teammate forward pairs
    ep = episode([("Player A", "A", "right", "(R3,C3) is occupied by C. D is at (R5,C1).")])
    cands = acc.candidates([ep], analyse([ep]))
    assert sorted((c["letter"], c["rule"], c["canon"]) for c in cands if c["kind"] == "loc") == \
        [("C", "rev", False), ("D", "fwd", True)]


def test_old_scorer_claim():
    ep = episode([("Player A", "A", "right", "C at (R3,C3); D is near at R5C1, A at (R9,C9)")])
    assert acc.old_claims(ep, ep["turns"][0]) == [True, True]  # own A is skipped


def source(moves, arm="dev"):
    ep = episode(moves, arm=arm)
    cands = acc.candidates([ep], analyse([ep]))
    by = collections.defaultdict(list)
    for c in cands:
        by[(ep["meta"]["id"], c["i"])].append(c)
    last = [c for c in acc.teammate(cands) if c["i"] == len(moves) - 1]
    return [acc.source_class(ep, c["i"], c["letter"], c["cell"], c["true"], by) for c in last]


def test_source_split():
    owner_says = ("Player B", "C", "up", "C is at (R2,C3).")  # true at B_t+ of ply 1
    a = ("Player A", "A", "right", "ok")
    assert source([a, owner_says, ("Player A", "A", "right", "C is at (R2,C3).")]) == [acc.SOURCE[0]]
    silent = ("Player B", "C", "up", "moving on")  # the owner moves C on without naming a cell
    # on the development arm only the latest message per seat is visible: the owner's "moving on"
    # replaced its statement, so there is no visible owner cell and a true cell is "updated"
    got = source([a, owner_says, a, silent, ("Player A", "A", "down", "C is at (R1,C3).")])
    assert got == [acc.SOURCE[1]]
    assert source([a, owner_says, ("Player A", "A", "right", "C is at (R5,C5).")]) == [acc.SOURCE[2]]
    # the latest owner message that pairs C with a cell counts, PLAN-from included (final arm: 4 rounds visible)
    fin = [a, owner_says, a, silent, ("Player A", "A", "down", "C is at (R2,C3).")]
    assert source(fin, arm="final") == [acc.SOURCE[0]]
    # final arm: the MOVE BOARD line "X moved (R2,C3) -> (R1,C3)" advances the owner's cell
    fin[-1] = ("Player A", "A", "down", "C is at (R1,C3).")
    assert source(fin, arm="final") == [acc.SOURCE[3]]


def test_freshness():
    ep = episode([("Player A", "A", "right", "x"), ("Player B", "C", "up", "x"), ("Player A", "A", "right", "x"),
                  ("Player B", "D", "up", "x"), ("Player A", "A", "down", "x")])
    assert acc.moved_since(ep, 2, "C") and not acc.moved_since(ep, 4, "C") and acc.moved_since(ep, 4, "D")
    assert acc.moved_since(ep, 1, "A") and not acc.moved_since(ep, 0, "C")  # no previous turn: since the start


def test_d2_grammar_and_union():
    strong = episode([("Player A", "A", "right", "C is at (R3,C3). D is at (R1,C1).")], eid="s")
    weak = episode([("Player A", "A", "right", "C is at (R9,C9).")], model="nemotron", eid="w")
    eps = [strong, weak]
    A = analyse(eps)
    rows = acc.d2_episodes(eps, A, acc.candidates(eps, A), extra={("w", 0, "D", (4, 0)), ("s", 0, "C", (2, 2))})
    r = {x["episode"]: x for x in rows}
    assert (r["s"]["all_n"], r["s"]["all_k"], r["s"]["union_n"]) == (2, 1, 2)  # the extra C is the grammar's
    assert (r["w"]["all_prop"], r["w"]["union_n"], r["w"]["union_k"]) == (0.0, 2, 1)
    res = acc.d2(eps, A, extra={("w", 0, "D", (4, 0))}, validation="both")
    assert res["union"]["primary"] and res["union"]["status"] == "confirmatory"
    assert res["grammar"]["status"] == "exploratory" and res["grammar"]["name"] == acc.D2_NAMES["validated"]
    assert (res["grammar"]["strong"], res["grammar"]["weak"]) == (0.5, 0.0)
    assert np.isnan(res["grammar"]["p"])  # one episode per tier: no permutation test
    assert acc.d2(eps, A)["grammar"]["name"] == acc.D2_NAMES["manifest"]
    g = acc.d2(eps, A, extra={("w", 0, "D", (4, 0))}, validation="grammar")
    assert g["grammar"]["name"] == acc.D2_NAMES["validated"] and g["grammar"]["status"] == "confirmatory"
    assert g["union"]["name"] == acc.D2_NAMES["manifest"] and g["union"]["status"] == "exploratory"
    import pytest
    with pytest.raises(AssertionError, match="no E5 records"):  # the confirmatory union row must not vanish
        acc.d2(eps, A, validation="both")


def test_bounds():
    rows = [dict(model="m", suite="dyad", level="S1", board=b, proto=p, v=v, r=r, n=n)
            for b, p, v, r, n in [(1, "hybrid", 3, 1, 4), (1, "structured", 1, 0, 4),
                                  (2, "hybrid", 2, 2, 2), (2, "structured", 2, 1, 3)]]
    b = cov.bounds(rows, "v", "r", "proto", "hybrid", "structured")
    assert b["pairs"] == 2
    assert np.isclose(b["est_lower"], ((3 - 1) + (2 - 2)) / 2)            # residuals as non-events
    assert np.isclose(b["est_upper"], ((4 - 1) + (4 - 3)) / 2)            # residuals as events
    assert np.isclose(b["worst_lo"], ((3 - 1) + (2 - 3)) / 2) and np.isclose(b["worst_hi"], ((4 - 1) + (4 - 2)) / 2)
    br = cov.bounds(rows, "v", "r", "proto", "hybrid", "structured", denom="n")
    assert np.isclose(br["est_lower"], ((3 / 5 - 1 / 4) + (2 / 4 - 2 / 4)) / 2)
    assert np.isclose(br["est_upper"], ((4 / 5 - 1 / 4) + (4 / 4 - 3 / 4)) / 2)


def test_two_x_rule():
    assert cov.coverage_limited(2.01, 1.0) and not cov.coverage_limited(2.0, 1.0)
    assert cov.coverage_limited(0.5, 0.0) and cov.coverage_limited(None, 1.0) is None
    ep = episode([("Player A", "A", "right", "C goes north, then west"), ("Player B", "C", "up", "fine")])
    A = analyse([ep])
    key = lambda r: (r["block"], r["quantity"], r["model"], r["proto"])
    dev = cov.run("dev", [ep], A)["audit_rates"]
    g = {key(r): r for r in dev}
    assert g[(cov.GRAM, "G-dir", "all", "all")]["value"] == 1000.0  # two compass words in two messages
    assert g[(cov.GRAM, "G-dir", "all", "all")]["coverage_limited"] is None  # no rule on the development arm
    # run.py passes the development table back as CSV rows (strings)
    dev_rows = [{k: "" if v is None else str(v / 4 if k == "value" and r["block"] == cov.GRAM else v)
                 for k, v in r.items()} for r in dev]
    out = cov.run("calib", [ep], A, dev_rates=dev_rows, opportunity_counts=[
        dict(model="qwen38", tier="strong", suite="dyad", proto="freeform", opp_primary=3)])
    g = {key(r): r for r in out["audit_rates"]}
    assert g[(cov.GRAM, "G-dir", "all", "all")]["threshold"] == 500.0
    assert g[(cov.GRAM, "G-dir", "all", "all")]["coverage_limited"] is True
    assert g[(cov.GRAM, "G-cell", "all", "all")]["coverage_limited"] is False  # 0 against 0
    assert g[(cov.OPP, "opp_primary", "qwen38", "freeform")]["value"] == 3.0
    side = {(r["block"], r["quantity"], r["model"], r["proto"]): r for r in out["cross_arm_equivalence"]}
    assert side[(cov.GRAM, "G-dir", "all", "all")]["dev"] == 250.0
    assert side[(cov.GRAM, "G-dir", "all", "all")]["calib"] == 1000.0
    assert side[(cov.OPP, "opp_primary", "qwen38", "freeform")]["dev"] is None  # dev had no counts here
