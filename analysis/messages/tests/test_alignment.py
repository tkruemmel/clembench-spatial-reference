"""Constructed-episode tests for measures/alignment.py (spec §5.5)."""
from analysis.messages import corpus, grammar
from analysis.messages.measures import alignment as al

DEV = ("Player A", "Player B")
FIN = ("Player Green", "Player Orange")


def make(plies, arm="dev", proto="freeform", seats=DEV, start=None, targets=None, grid=6, walls=(), objs=("B",)):
    """plies: [(seat index, (obj, dir) | 'done', msg, retry)]. Seat 0 owns A, seat 1 owns `objs`."""
    start = start or {"A": (0, 0), "B": (3, 3)}
    targets = targets or {"A": (0, 5), "B": (0, 3)}
    objs = {seats[0]: ["A"], seats[1]: list(objs)}
    ep = dict(meta=dict(id=f"{arm}|qwen38|dyad|S1|{proto}|01", arm=arm, model="qwen38", tier="strong", suite="dyad",
                        level="S1", board=1, proto=proto, success=0),
              seats=list(seats), owner={o: s for s, os in objs.items() for o in os}, objs=objs, grid=grid,
              walls=frozenset(walls), start=dict(start), targets=dict(targets), turns=[], device={})
    pos, seen, own = dict(start), {s: set() for s in seats}, {s: 0 for s in seats}
    for i, (si, mv, msg, retry) in enumerate(plies):
        s = seats[si]
        t = dict(ply=i, seat=s, si=si, own_turn=own[s], done=mv == "done", msg=msg, retry=retry, board=dict(pos),
                 obj=None, dir=None, src=None, dst=None, call=i)
        own[s] += 1
        if mv != "done":
            o, d = mv
            t.update(obj=o, dir=d, src=pos[o], dst=corpus.step(pos, o, d)[o])
            pos[o] = t["dst"]
        t["is_msg"] = bool(msg) and not t["done"] and proto != "none"
        t["slots"] = corpus.slots(msg, proto) if t["is_msg"] else None
        t["dup"] = t["is_msg"] and msg in seen[s]
        if t["is_msg"]:
            seen[s].add(msg)
        ep["turns"].append(t)
    for i, t in enumerate(ep["turns"]):
        t["after"] = ep["turns"][i + 1]["board"] if i + 1 < len(plies) else pos
        t["vis"] = corpus.visible(ep, i, t["retry"])
    ep["done_ply"] = {t["seat"]: t["ply"] for t in ep["turns"] if t["done"]}
    ep["final_pos"] = pos
    return ep


def opps(ep, bidx=None):
    X = al._Ep(ep, grammar.analyse_all([ep]), bidx)
    return al.opportunities(X)


def test_first_visible_exposure():
    ep = make([(0, ("A", "right"), "B, move up please.", False),
               (1, ("B", "up"), "ok", False),
               (0, ("A", "right"), "nice", False),
               (1, ("B", "up"), "ok again", False)])
    rows, _ = opps(ep)
    assert [(r["exposure"], r["decision_ply"], r["letter"], r["dir"], r["o1"]) for r in rows] == \
        [("visible", 1, "B", "up", True)]
    assert rows[0]["listener"] == "Player B" and rows[0]["goalward"] is True and rows[0]["o2"]
    fin = make([(0, ("A", "right"), "B, move up please.", False), (1, ("B", "left"), "ok", False)],
               arm="final", seats=FIN)
    rows, _ = opps(fin)
    assert [(r["exposure"], r["decision_ply"], r["o1"]) for r in rows] == [("visible", 1, False)]


def test_negated_and_question_frames_excluded():
    ep = make([(0, ("A", "right"), "Don't move B up.", False), (1, ("B", "up"), "ok", False)])
    rows, cnt = opps(ep)
    assert rows == [] and cnt["frame_excluded"] == 1


def test_echo_detection():
    def run(own_msg):
        ep = make([(0, ("A", "right"), "hi", False),
                   (1, ("B", "left"), own_msg, False),
                   (0, ("A", "right"), "B, go up now.", False),
                   (1, ("B", "up"), "ok", False)])
        return [r for r in opps(ep)[0] if r["decision_ply"] == 3]
    assert run("I will move B up next.")[0]["echo_src"] == "text"
    assert run("I will move B down next.")[0]["echo"] is False
    # a PLAN whose displacement has an upward component states (B, up)
    ep = make([(0, ("A", "right"), "PLAN: move A from (R1,C1) to (R1,C2) — REQUEST: none", False),
               (1, ("B", "left"), "PLAN: move B from (R4,C4) to (R2,C3) — REQUEST: none", False),
               (0, ("A", "right"), "PLAN: move A from (R1,C2) to (R1,C3) — REQUEST: move B up", False),
               (1, ("B", "up"), "PLAN: move B from (R4,C3) to (R3,C3) — REQUEST: none", False)], proto="structured")
    r = [r for r in opps(ep)[0] if r["kind"] == "move"]
    assert len(r) == 1 and r[0]["echo"] is True and r[0]["echo_src"] == "plan" and r[0]["src"] == "slots"


def test_legality_filter():
    # B at (3,3): up is blocked by a wall at (2,3); right is free
    ep = make([(0, ("A", "right"), "B up, or B right.", False), (1, ("B", "right"), "ok", False)], walls=[(2, 3)])
    rows, cnt = opps(ep)
    assert [(r["letter"], r["dir"], r["o1"]) for r in rows] == [("B", "right", True)]
    assert cnt["illegal_visible"] == 1
    # occupied by a teammate's object and off the grid
    ep = make([(0, ("A", "right"), "B left. B down.", False), (1, ("B", "up"), "ok", False)],
              start={"A": (3, 1), "B": (5, 3)})
    rows, cnt = opps(ep)  # A moves to (3,2), B at (5,3): left is free, down is off-grid
    assert [(r["letter"], r["dir"]) for r in rows] == [("B", "left")] and cnt["illegal_visible"] == 1


def test_matching_key():
    none = make([(1, ("B", "up"), None, False), (0, ("A", "right"), None, False),
                 (1, ("B", "down"), None, False), (0, ("A", "left"), None, False),
                 (1, ("B", "up"), None, False)], proto="none")
    idx = al.baseline_index(none)
    b0 = none["turns"][0]["board"]
    # ply 0: B on (3,3), never moved; ply 2: B on (2,3) with momentum up; ply 4: B on (3,3), momentum down
    m = al.match(idx, b0, "B", None, "up")
    assert [r[0] for r in m] == [0] and m[0][1] == "up"
    assert [r[0] for r in al.match(idx, b0, "B", "down", "up")] == [4]  # momentum is part of the key
    assert al.match(idx, b0, "B", "up", "up") == []
    # the cell two steps along d: occupied in the opportunity, empty at the moment -> no match
    b = dict(b0, A=(1, 3))
    assert al.match(idx, b, "B", None, "up") == [] and len(al.match(idx, b, "B", None, "left")) == 1
    # a neighbour's occupancy differs -> no match
    assert al.match(idx, dict(b0, A=(2, 2)), "B", None, "down") == []
    # the direction-free key (vacate) ignores the two-step cell
    assert [r[0] for r in al.match(idx, b, "B", None, None)] == [0]
    f = al._base_fields(none, al.match(idx, b0, "B", None, "up"), b0, "B", "up", "Player B", 0)
    assert (f["base_n"], f["base_k"], f["covered"]) == (1, 1, True)


def test_negative_control_retry_and_superseded():
    ep = make([(0, ("A", "right"), "B, go up.", False), (1, ("B", "up"), "ok", True)])
    rows, _ = opps(ep)
    assert [(r["exposure"], r["decision_ply"], r["o1"]) for r in rows] == [("nc_retry", 1, True)]
    ep = make([(0, ("A", "right"), "B, go up.", False), (0, ("A", "right"), "hello", False),
               (1, ("B", "left"), "ok", False)])
    rows, _ = opps(ep)
    assert [(r["exposure"], r["decision_ply"]) for r in rows] == [("nc_superseded", 2)]
    # not a control when another visible message still names (B, up)
    ep = make([(0, ("A", "right"), "B, go up.", False), (0, ("A", "right"), "B up!", False),
               (1, ("B", "left"), "ok", False)])
    rows, cnt = opps(ep)
    assert [r["exposure"] for r in rows] == ["visible"] and cnt["contaminated_nc_superseded"] == 1


def test_negative_control_rolled_off():
    plies = [(0, ("A", "right"), "Orange, B up please.", False)]
    for k in range(5):
        plies += [(1, ("B", "left" if k % 2 else "right"), f"b{k}", False),
                  (0, ("A", "down" if k % 2 else "up"), f"a{k}", False)]
    ep = make(plies, arm="final", seats=FIN, start={"A": (1, 0), "B": (3, 3)})
    rows, _ = opps(ep)
    # Orange decides at 1, 3, 5, 7, 9; at 9 the window starts at its 4th-last own turn (ply 1)
    assert [(r["exposure"], r["decision_ply"]) for r in rows] == [("visible", 1), ("nc_rolled_off", 9)]


def test_negative_control_outcome_and_specificity():
    ep = make([(1, ("B", "up"), "hi", False),
               (0, ("A", "right"), "B, up again.", False),
               (1, ("B", "left"), "ok", False)])
    r = opps(ep)[0][0]
    assert (r["nc_prev_ply"], r["nc_prev_legal"], r["nc_prev_o1"]) == (0, True, True)
    # B at (2,3), target (0,3): up is goal-ward, the others are not -> no alternative with the same status
    assert r["goalward"] is True and r["alt_n"] == 0 and r["o1"] is False


def test_request_chain():
    req = "PLAN: move A from (R1,C1) to (R1,C2) — REQUEST: clear (R4,C4)"
    again = "PLAN: move A from (R1,C2) to (R1,C3) — REQUEST: clear (R4,C4)"
    none = "PLAN: move A from (R1,C2) to (R1,C3) — REQUEST: none"
    vac = (1, ("B", "up"), "PLAN: move B from (R4,C4) to (R3,C4) — REQUEST: none", False)
    plan_b = (1, ("C", "down"), "PLAN: move B from (R4,C4) to (R1,C4) — REQUEST: none", False)
    plan_c = (1, ("C", "down"), "PLAN: move C from (R5,C1) to (R6,C1) — REQUEST: none", False)

    def chain(holder, presenter):
        ep = make([(0, ("A", "right"), req, False), holder, (0, ("A", "right"), presenter, False)],
                  proto="structured", objs=["B", "C"], start={"A": (0, 0), "B": (3, 3), "C": (4, 0)},
                  targets={"A": (0, 5), "B": (0, 3), "C": (5, 1)})
        X = al._Ep(ep, grammar.analyse_all([ep]), None)
        return [(r["holder"], r["holder_next"], r["presenter_next"]) for r in al.request_chain(X)
                if r["msg_ply"] == 0], \
            al.opportunities(X)[0]

    rows, opp = chain(vac, none)
    assert rows == [("B", "E", "closes")]
    assert [(r["kind"], r["exposure"], r["o1"], r["dir"]) for r in opp] == [("vacate", "visible", True, None)]
    assert chain(vac, again)[0] == [("B", "E", "errs")]
    assert chain(plan_b, again)[0] == [("B", "P", "re-presents")]
    assert chain(plan_c, none)[0] == [("B", "N", "closes")]


def test_contrast_gating():
    def ep(board, obs, base, n):
        return dict(model="qwen38", suite="dyad", proto="freeform", level="S1", board=board, episode=str(board),
                    obs_covered=obs, base_covered=base, n_covered=n, o1=obs, o1_nc=None, n_nc=0, o1_prevset=None,
                    prev=None, n_prev=0, o1_altset=None, alt=None, n_alt=0, tgt_obs_covered=None,
                    tgt_base_covered=None, tgt_n_visible=0)
    rows = [ep(g, 0.6, 0.2, 3) for g in range(1, 7)]
    cov = [dict(model="qwen38", suite="dyad", proto="freeform", kind="move", coverage=0.5)]
    out = {r["comparison"]: r for r in al.contrasts(rows, cov, {})}
    assert out["silent_baseline"]["reported"] and out["silent_baseline"]["pairs"] == 6
    assert abs(out["silent_baseline"]["diff"] - 0.4) < 1e-9 and out["silent_baseline"]["p"] == 2 / 64
    assert not out["nc_exposure"]["reported"]  # no pairs
    cov[0]["coverage"] = 0.29
    out = {r["comparison"]: r for r in al.contrasts(rows, cov, {})}
    assert out["silent_baseline"]["reported"] is False and "coverage" in out["silent_baseline"]["note"]
