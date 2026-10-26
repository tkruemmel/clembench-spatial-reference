"""failure.py (spec §6) on constructed minimal episodes: spell segmentation,
feature timing (a, d), reachability for b, the free-pinch risk set."""
import numpy as np

from analysis.messages import corpus, failure, grammar


def build(n, free_cells, start, targets, groups, script, proto="freeform"):
    """A dev-arm episode on an n x n grid whose non-free cells are walls.
    groups: object lists in seat order; script: [(seat index, (obj, dir) | "done", message)]."""
    seats = list(corpus.DEV_SEATS[:len(groups)])
    owner = {o: seats[i] for i, g in enumerate(groups) for o in g}
    pos, turns = dict(start), []
    for i, (si, act, msg) in enumerate(script):
        t = dict(ply=i, seat=seats[si], board=dict(pos), msg=msg, retry=False, slots=None,
                 done=act == "done", obj=None, dir=None)
        if act != "done":
            o, d = act
            t.update(obj=o, dir=d)
            pos[o] = (pos[o][0] + corpus.DIRS[d][0], pos[o][1] + corpus.DIRS[d][1])
        t["is_msg"] = bool(msg) and not t["done"]
        turns.append(t)
    meta = dict(id="t", arm="dev", proto=proto, model="qwen38", tier="strong", suite=corpus.SUITES[len(groups)],
                level="S1", board=1)
    ep = dict(meta=meta, seats=seats, owner=owner, objs={s: sorted(o for o in owner if owner[o] == s) for s in seats},
              grid=n, walls=frozenset((r, c) for r in range(n) for c in range(n)) - set(free_cells),
              targets=targets, turns=turns, done_ply={t["seat"]: t["ply"] for t in turns if t["done"]})
    for i, t in enumerate(turns):
        t["vis"] = corpus.visible(ep, i, False)
    A = {("t", t["ply"]): grammar.analyse(t["msg"], "dev", proto) for t in turns if t["is_msg"]}
    return ep, A


# A 1x5 corridor on row 0, a side pocket (1,3), and row 4 for C. A steps home
# to (0,2) and stands between B and B's target (0,4): at A's next decision
# (ply 3) declaring done would cut B off.
ROW0 = [(0, c) for c in range(5)]
FREE = ROW0 + [(1, 3)] + [(4, c) for c in range(5)]
START = {"A": (0, 3), "B": (0, 0), "C": (4, 0)}
TARGETS = {"A": (0, 2), "B": (0, 4), "C": (4, 4)}


def corridor(m0="ok", m1="ok", m2="ok", m3="ok", m4="ok", m5="ok"):
    script = [(0, ("A", "left"), m0), (1, ("B", "right"), m1), (2, ("C", "right"), m2),
              (0, ("A", "right"), m3), (1, ("B", "left"), m4), (2, ("C", "right"), m5),
              (0, ("A", "left"), "ok")]
    return build(5, FREE, START, TARGETS, [["A"], ["B"], ["C"]], script)


def fork(ep):
    (h,) = corpus.hazards(ep)
    return h


def test_fixture_has_one_fork():
    ep, _ = corridor()
    h = fork(ep)
    assert (h["kind"], h["ply"], h["seat"], h["case"]) == ("fork", 3, "Player A", False)
    assert h["victims"] == ["B"] and h["critical"] == [(0, 2)]
    assert ep["turns"][3]["vis"] == [0, 1, 2]  # dev: every seat's latest, own included


def test_spell_segmentation():
    seats = ["Player A", "Player B", "Player C"]
    ep = dict(seats=seats, turns=[dict(ply=i, seat=seats[i % 3]) for i in range(15)])
    recs = [dict(seat="Player A", kind="pinch", ply=p) for p in (0, 3, 9, 12)] + \
           [dict(seat="Player A", kind="fork", ply=6), dict(seat="Player B", kind="pinch", ply=1)]
    sp = failure.spells(ep, recs)
    # A's own decisions are 0, 3, 6, 9, 12: pinch live at 0, 3 | not at 6 | 9, 12
    assert sp == {("Player A", "pinch", 0): 0, ("Player A", "pinch", 3): 0, ("Player A", "pinch", 9): 9,
                  ("Player A", "pinch", 12): 9, ("Player A", "fork", 6): 6, ("Player B", "pinch", 1): 1}
    D = [dict(episode="t", seat="Player A", kind="pinch", ply=p, spell=sp[("Player A", "pinch", p)], case=c)
         for p, c in ((0, 0), (3, 1), (9, 0), (12, 0))]
    S = failure.spell_rows(D)
    assert [(r["ply"], r["case"], r["spell_len"], r["spell_taken"]) for r in S] == [(0, 0, 2, 1), (9, 0, 2, 0)]


def test_feature_a_teammates_only_and_never_the_same_response():
    crit = "(R1,C3)"  # (0, 2), A's cell at the fork
    for kw, a in ((dict(m1=f"Do not stop at {crit}."), 1),   # teammate B
                  (dict(m2=f"I see {crit}."), 1),              # teammate C
                  (dict(m0=f"Home at {crit}."), 0),            # A's own earlier message, visible on dev
                  (dict(m3=f"Stepping off {crit}."), 0),       # A's message in the deciding response
                  (dict(m4=f"Thanks for {crit}."), 0)):        # B, but after the decision
        ep, A = corridor(**kw)
        assert failure.features(ep, fork(ep), A, start=3)["a"] == a, kw


def test_feature_d_uses_only_messages_before_the_first_live_ply():
    crit = "(R1,C3)"
    ep, A = corridor(m0=f"Home at {crit}.")
    assert failure.features(ep, fork(ep), A, start=3)["d"] == 1
    ep, A = corridor(m3=f"Leaving {crit}.")           # the response at the first live ply
    assert failure.features(ep, fork(ep), A, start=3)["d"] == 0
    # a later decision of the same spell keeps the spell's message: A's message at
    # the first live ply (3) does not count, the one before it (0) does
    later = dict(fork(ep), ply=6)
    assert failure.features(ep, later, A, start=3)["d"] == 0
    ep, A = corridor(m0=f"Home at {crit}.", m3="ok")
    assert failure.features(ep, dict(fork(ep), ply=6), A, start=3)["d"] == 1
    assert failure.features(ep, dict(fork(ep), ply=6), A, start=6)["d"] == 0  # last own message is ply 3


def test_feature_b_reachability():
    ep, _ = corridor()
    # B at (0,1) to (0,4): the route beyond A's cell is cut; (0,0) is behind B and
    # the pocket (1,3) is cut off but on no shortest route
    assert failure.cut_cells(ep, fork(ep)) == {(0, 2), (0, 3), (0, 4)}
    for msg, b in (("Heading for (R1,C4).", 1), ("Room at (R1,C1).", 0), ("The pocket (R2,C4) is empty.", 0)):
        ep, A = corridor(m1=msg)
        assert failure.features(ep, fork(ep), A, start=3)["b"] == b, msg
    # finished objects are walls: one in the pocket changes nothing (off-route);
    # one on B's route leaves B no route to cut
    ep, _ = corridor()
    ep["turns"][3]["board"]["C"] = (1, 3)
    ep["done_ply"] = {"Player C": 2}
    assert failure.cut_cells(ep, fork(ep)) == {(0, 2), (0, 3), (0, 4)}  # the pocket is off-route anyway
    ep["turns"][3]["board"]["C"] = (0, 3)
    assert failure.cut_cells(ep, fork(ep)) == set()


def test_feature_b_target_stated_by_the_victims_owner(monkeypatch):
    monkeypatch.setattr(failure, "cut_cells", lambda ep, h: set())  # isolate the target path
    ep, A = corridor(m1="My target is (R1,C5).")  # B states its own target
    assert failure.features(ep, fork(ep), A, start=3)["b"] == 1
    ep, A = corridor(m2="My target is (R1,C5).")  # C is not the victim's owner
    assert failure.features(ep, fork(ep), A, start=3)["b"] == 0


def test_feature_e_victim_letter_with_its_current_cell():
    for msg, e in (("B is at (R1,C2).", 1), ("B is at (R1,C1).", 0), ("B will be at (R1,C2).", 0)):
        ep, A = corridor(m2=msg)
        assert failure.features(ep, fork(ep), A, start=3)["e"] == e, msg


def test_visible_done_turn_is_no_message():
    ep, A = corridor()
    t = ep["turns"][2]  # C declares done instead, saying the critical cell
    t.update(done=True, obj=None, dir=None, is_msg=False, msg="done (R1,C3)")
    ep["done_ply"] = {"Player C": 2}
    A.pop(("t", 2))
    assert 2 in ep["turns"][3]["vis"]
    assert failure.features(ep, fork(ep), A, start=3)["a"] == 0


def test_info_visible_is_a_or_b_or_e():
    for kw, v in ((dict(m1="Do not stop at (R1,C3)."), True),      # a
                  (dict(m2="B is at (R1,C2)."), True),            # e
                  (dict(m0="Home at (R1,C3)."), False),           # d only
                  (dict(), False)):
        ep, A = corridor(**kw)
        assert failure.info_visible(ep, fork(ep)) is v, kw      # analyses built on the fly
        assert failure.info_visible(ep, fork(ep), A) is v, kw


def test_none_has_no_features():
    ep, A = corridor()
    ep["meta"]["proto"] = "none"
    assert failure.features(ep, fork(ep), A, start=3) == dict.fromkeys(failure.FEATURES)


def test_risk_set_keeps_free_pinches_only():
    # B sits in the pocket (1,2) under (0,2); A at (0,1) can seal it by moving right.
    # With (0,0) free A has a harmless alternative; with C on (0,0) it has none.
    free_cells = [(0, c) for c in range(5)] + [(1, 2)] + [(4, c) for c in range(5)]
    tg = {"A": (0, 4), "B": (0, 3), "C": (4, 4)}
    for c_at, act, free in (((4, 0), ("A", "left"), True), ((0, 0), ("A", "right"), False)):
        ep, _ = build(5, free_cells, {"A": (0, 1), "B": (1, 2), "C": c_at}, tg, [["A"], ["B"], ["C"]],
                      [(0, act, "ok")])
        (h,) = corpus.hazards(ep)
        assert h["kind"] == "pinch" and h["victims"] == ["Player B"] and h["critical"] == [(0, 2)]
        assert h["case"] == (not free)
        assert failure.free(ep, h) == free
        assert len(failure.risk_set(ep)) == int(free)
        if free:  # the pocket is B's only way out, and (0,2) is on its route
            assert failure.cut_cells(ep, h) == {(0, 2), (0, 3)}


def test_block_permuter_moves_episode_blocks_within_strata():
    strata = np.array([0, 0, 0, 0, 1, 1, 1])
    eps = ["x", "x", "y", "y", "z", "w", "v"]
    perm = failure.block_permuter(strata, eps)
    y = np.array([1, 0, 0, 0, 1, 0, 0])
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(50):
        p = perm(y, rng)
        assert p[:4].sum() == 1 and p[4:].sum() == 1
        assert tuple(p[:4]) in {(1, 0, 0, 0), (0, 0, 1, 0)}  # x's block moves whole onto y's
        seen.add(tuple(p))
    assert len(seen) == 6
