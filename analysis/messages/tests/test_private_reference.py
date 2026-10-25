"""Constructed-episode tests for measures/private_reference.py (spec §5.1-§5.3, §9.1),
including the synthetic MOVE-BOARD linking tests the spec requires before the freeze (§5.3)."""
import math
import re

from analysis.messages import corpus, grammar
from analysis.messages.measures import private_reference as pr
from matrixgame_covered_shared import history

INF = math.inf


def make_ep(arm, proto, objs, start, targets, plies, grid=6, device=None, walls=()):
    """objs: seat -> objects; plies: [(seat, 'A down' | 'done', message)]."""
    owner = {o: s for s, os_ in objs.items() for o in os_}
    pos, turns = dict(start), []
    for i, (seat, mv, msg) in enumerate(plies):
        o, d = (None, None) if mv == "done" else mv.split()
        src = pos[o] if o else None
        dst = (src[0] + corpus.DIRS[d][0], src[1] + corpus.DIRS[d][1]) if o else None
        turns.append(dict(ply=i, seat=seat, obj=o, dir=d, src=src, dst=dst, done=mv == "done", msg=msg,
                          board=dict(pos), call=i))
        if o:
            pos[o] = dst
    suite = corpus.SUITES[len(objs)]
    meta = dict(id=f"{arm}|qwen38|{suite}|S3|{proto}|01", arm=arm, model="qwen38", tier="strong", suite=suite,
                n=len(objs), level="S3", board=1, proto=proto)
    ep = dict(meta=meta, seats=list(objs), owner=owner, objs={s: sorted(v) for s, v in objs.items()}, grid=grid,
              walls=frozenset(walls), start=dict(start), targets=targets, device=device or {}, turns=turns,
              calls=[], final_pos=pos)
    corpus._annotate(ep)
    A = grammar.analyse_all([ep])
    M = {t["ply"]: pr.msg_info(ep, t, A[(meta["id"], t["ply"])]) for t in turns if t["is_msg"]}
    return ep, A, M


def row(rows, part):
    return next(r for r in rows if r["part"] == part)


# a development-arm dyad: Player A owns A B C (top row), Player B owns D E F (bottom row)
DYAD = {"Player A": ["A", "B", "C"], "Player B": ["D", "E", "F"]}
START = {"A": (0, 0), "B": (0, 2), "C": (0, 4), "D": (5, 0), "E": (5, 2), "F": (5, 4)}
TARGETS = {"A": (2, 0), "B": (2, 2), "C": (2, 4), "D": (3, 0), "E": (3, 2), "F": (3, 4)}


# ---------------------------------------------------------------- §5.2

def test_own_cell_target_and_facts():
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "A is at (R1,C1). My goal for A is (R3,C1)."),
        ("Player B", "D up", "D is at (R5,C1)."),       # true at B_t+ only
        ("Player A", "A down", "B is at (R4,C4)."),     # false
    ])
    p0, p1, p2 = (M[i]["prose"] for i in range(3))
    assert p0["own_true"] == {"A"} and p0["stated"] == {"A"} and p0["named"] == {"A"}
    assert pr.facts_of(ep, ep["turns"][0], M[0], "prose") == {("own", "A"), ("tgt", "A")}
    assert p1["own_true"] == {"D"}
    assert p2["own_true"] == set() and p2["true"] == set()
    assert M[0]["slots"]["own_true"] == set()  # freeform has no slots
    # 3p development arm: seat letters reveal owners, so ownership is no hidden fact
    ep3, _, M3 = make_ep("dev", "freeform", {"Player A": ["A", "D"], "Player B": ["B", "E"], "Player C": ["C", "F"]},
                         START, TARGETS, [("Player A", "A down", "A is at (R1,C1). My goal for A is (R3,C1).")])
    assert pr.facts_of(ep3, ep3["turns"][0], M3[0], "prose") == {("tgt", "A")}
    assert len(pr.all_facts(ep3)) == 6 and len(pr.all_facts(ep)) == 12


def test_disclosure_redundancy_copresence_described():
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "A is at (R1,C1). My goal for A is (R3,C1)."),
        ("Player B", "D up", "Moving D up."),
        ("Player A", "A down", "My goal for A is (R3,C1)."),   # re-disclosure, seen by B before
        ("Player B", "E up", "E is at (R6,C3)."),
        ("Player A", "A right", "Moving A right."),
    ])
    rows, drows, vrows = pr.private_rows(ep, A, M)
    r = row(rows, "prose")
    assert r["own_cell"] == 2 / 6 and r["own_cell_first"] == 0.5     # A at A's turn 0, E at B's turn 1
    assert r["target_stated"] == 1 / 6 and r["redisc"] == 1
    D = {x["r"]: x for x in drows if x["part"] == "prose"}
    assert D[1]["D"] == 2 / 12 and D[2]["D"] == 3 / 12 and D[1]["redisc"] == 0 and D[2]["redisc"] == 1
    # statements x listeners: ply0 own A, tgt A (B has no earlier view), ply2 tgt A (in B's view at ply 1),
    # ply3 own E (A's view at ply 2 shows nothing about E)
    assert r["n_red"] == 4 and r["redundant"] == 0.25
    # V at plies 0..4: 0, 2/6, 0, 1/6, 1/6
    assert abs(r["v_mean"] - (4 / 6) / 5) < 1e-12
    assert {x["r"]: x["V"] for x in vrows if x["part"] == "prose"} == {1: (0 + 2 / 6) / 2, 2: (0 + 1 / 6) / 2, 3: 1 / 6}
    assert r["move_described"] == 2 / 5 and r["move_consistent"] == 1.0


def test_move_described_inconsistent_and_teammate_letter():
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "Moving A right."),     # own, present, not the move made
        ("Player B", "D up", "Moving A right."),       # a teammate's letter: not a self-description
    ])
    r = row(pr.private_rows(ep, A, M)[0], "prose")
    assert r["move_described"] == 0.5 and r["move_consistent"] == 0.0


def test_later_move_plan_and_prose():
    plan = "PLAN: move {} from ({}) to ({}) — REQUEST: none"
    ep, A, M = make_ep("dev", "structured", DYAD, START, TARGETS, [
        ("Player A", "A down", plan.format("A", "R1,C1", "R4,C1")),   # later: A to (R4,C1)
        ("Player B", "D up", plan.format("D", "R6,C1", "R5,C1")),     # one step: no later move
        ("Player A", "A down", plan.format("A", "R2,C1", "R4,C1")),   # later; next own turn moves B
        ("Player B", "E up", plan.format("E", "R6,C3", "R4,C3")),     # later; B never moves again
        ("Player A", "B right", plan.format("B", "R1,C3", "R1,C4")),
        ("Player A", "A down", plan.format("A", "R3,C1", "R3,C1")),   # later (A ends at R4), unscored
    ])
    T = ep["turns"]
    sts = pr.later_statements(ep, T[0], M[0], "slots")
    assert sts == [("plan", "A", (3, 0), None)]
    assert pr.later_statements(ep, T[1], M[1], "slots") == []
    nxt = pr._next_turns(ep)
    assert pr.kept(ep, sts[0], nxt[0], 1) is True
    s2 = pr.later_statements(ep, T[2], M[2], "slots")[0]
    assert pr.kept(ep, s2, nxt[2], 1) is False and pr.kept(ep, s2, nxt[2], 2) is True
    assert pr.kept(ep, pr.later_statements(ep, T[3], M[3], "slots")[0], nxt[3], 1) is None
    r = row(pr.private_rows(ep, A, M)[0], "slots")
    assert r["n_later"] == 4 and r["later_kept1"] == 0.5 and r["later_kept2"] == 1.0
    # prose: future and sequence frames of own letters only
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "Next I will move B right."),
        ("Player B", "D up", "Once D clears, I will move A down."),
        ("Player A", "B right", "Fine."),
    ])
    assert pr.later_statements(ep, ep["turns"][0], M[0], "prose") == [("move", "B", "right", "future")]
    assert pr.later_statements(ep, ep["turns"][1], M[1], "prose") == []   # A is not B's
    assert pr.kept(ep, ("move", "B", "right", "future"), pr._next_turns(ep)[0], 1) is True


def test_cell_named_before_entry_and_permutation_baseline():
    assert pr.hit([2, 5], 0, 1) is None and pr.hit([2, 5], 0, 4) == 2
    assert pr.hit([2, 5], 2, INF) == 3 and pr.hit([], 0, INF) is None
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "Heading for (R3,C1)."),   # A's own turn 0; A enters (R3,C1) at own turn 2
        ("Player B", "D up", "Fine."),
        ("Player A", "B right", "Good."),
        ("Player B", "D up", "Next."),
        ("Player A", "A down", "Onward."),
        ("Player B", "E up", "Right."),
        ("Player A", "B right", "Still."),
    ])
    r = row(pr.private_rows(ep, A, M)[0], "prose")
    # one (message, cell) pair; A's message times are own turns 0..3
    assert r["entry_W1"] == 0.0 and r["chance_W1"] == 1 / 4    # only tau = 1 is one turn before entry 2
    assert r["entry_W4"] == 1.0 and r["chance_W4"] == 2 / 4
    assert r["entry_Wall"] == 1.0 and r["chance_Wall"] == 2 / 4 and r["lead"] == 2.0


# ---------------------------------------------------------------- §5.1 and §5.3 board

DEVICE = {"entrance": [2, 3], "corridor": [[3, 3], [4, 3]], "dead_end": [4, 3]}


def test_device_line():
    ep, _, _ = make_ep("dev", "freeform", DYAD, START, TARGETS, [], device=DEVICE)
    assert pr.device(ep) == ({(2, 3), (3, 3), (4, 3)}, ("C", 3))
    ep["device"] = {"entrance": [1, 1], "corridor": [[1, 2], [2, 2]]}
    assert pr.device(ep)[1] is None
    ep["device"] = {}
    assert pr.device(ep) == (None, None)


def test_board_classes_and_shares():
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "a"), ("Player B", "done", "done"), ("Player A", "A down", "b")],
        device=DEVICE, walls=[(1, 5)])
    t, dev = ep["turns"][2], pr.device(ep)[0]
    on = lambda c: {k for k, v in pr.classes(ep, t, c, dev).items() if v}
    assert on((1, 0)) == {"own_obj"} and on((5, 2)) == {"mate_finished"}
    assert on((2, 0)) == {"empty", "own_target"} and on((3, 0)) == {"empty", "mate_target"}
    assert on((3, 3)) == {"empty", "device"} and on((1, 5)) == set() and on((9, 9)) == set()
    assert pr.classes(ep, ep["turns"][0], (5, 2), dev)["mate_active"]  # before B's done
    sh, N = pr.board_shares(ep, t, dev), 35
    assert sh["own_obj"] == 3 / N and sh["mate_finished"] == 3 / N and sh["mate_active"] == 0
    assert sh["own_obj"] + sh["mate_active"] + sh["mate_finished"] + sh["empty"] == 1
    assert sh["device"] == 3 / N and sh["own_target"] == sh["mate_target"] == 3 / N
    assert pr.classes(ep, t, (3, 3), None)["device"] is None


def test_form_rows_slots():
    ep, A, M = make_ep("dev", "structured", DYAD, START, TARGETS, [
        ("Player A", "A down", "PLAN: move A from (R1,C1) to (R3,C1) — REQUEST: none"),
        ("Player B", "D up", "PLAN: move D from (R6,C1) to (R5,C1) — REQUEST: please clear (R1,C3) then (R3,C1)"),
        ("Player A", "A down", "PLAN: move A from (R2,C1) to (R3,C1) — REQUEST: none"),
    ])
    msg, rows = pr.form_rows(ep, A, M)
    r = row(rows, "slots")
    assert r["plan_to_goal"] == 2 / 3 and r["plan_own"] == 1.0 and r["plan_moved"] == 1.0
    assert r["request_none"] == 2 / 3 and r["req_cell"] == 1.0
    assert r["req_mate_active"] == 1.0 and r["req_mate_target"] == 0.0   # first cell (R1,C3) holds B
    assert r["words"] == (7 + 11 + 7) / 3   # the PLAN and REQUEST groups, as the master counts
    assert msg[0]["plan_present"] == 1.0 and msg[0]["dup"] == 0.0 and msg[0]["retry"] == 0.0


def test_letter_bound_teammate_target_and_device_named():
    ep, A, M = make_ep("dev", "freeform", DYAD, START, TARGETS, [
        ("Player A", "A down", "D goes to (R4,C1). E near (R4,C1). Column C4 is busy."),
    ], device=DEVICE)
    br, _ = pr.reference_rows(ep, A, M)
    r = row(br, "prose")
    assert r["n_letter_cell"] == 2 and r["mate_tgt_letter"] == 0.5 and r["mate_tgt_letter_perm"] == 1 / 3
    assert r["mate_target"] == 1.0 and r["device_named"] == 1.0   # the column of the device line


# ---------------------------------------------------------------- §5.3 resolvable reference

THREE_DEV = {"Player A": ["A", "D"], "Player B": ["B", "E"], "Player C": ["C", "F"]}


def test_listener_resolvable_reference():
    ep, A, M = make_ep("dev", "freeform", THREE_DEV, START, TARGETS, [
        ("Player A", "A down", "E is at (R6,C3)."),           # E true: resolvable for both listeners
        ("Player B", "B down", "Player C, watch F and (R2,C2)."),
        ("Player C", "C down", "Fine."),
        ("Player A", "A down", "Good."),
    ])
    _, rr = pr.reference_rows(ep, A, M)
    r = row(rr, "prose")
    # ply 0: refs E + (R6,C3), resolvable for B (own) and C (paired) -> 0, 0
    # ply 1: refs F + (R2,C2); listener C owns F -> 0; listener A: F unbound -> 1/2
    assert r["n_pairs"] == 4 and r["unres"] == (0 + 0 + 0 + 0.5) / 4
    assert r["n_pairs_addr"] == 1 and r["unres_addr"] == 0.0           # addressee: Player C only


def test_resolvable_through_listener_view():
    ep, A, M = make_ep("dev", "freeform", THREE_DEV, START, TARGETS, [
        ("Player A", "A down", "Hi."),
        ("Player B", "B down", "F is at (R6,C5)."),     # true: binds F in the next views
        ("Player C", "C down", "Mind F and B."),        # F, B unpaired here
        ("Player A", "A down", "Ok."),
        ("Player B", "B down", "Sure."),
    ])
    _, rr = pr.reference_rows(ep, A, M)
    # ply 1 -> C (owns F), A (paired): 0, 0
    # ply 2 -> A: F bound by ply 1 in its view, B unbound -> 1/2; B: F bound by its own shown message, B its own -> 0
    assert row(rr, "prose")["unres"] == (0 + 0 + 0.5 + 0) / 4


# ---------------------------------------------------------------- §5.3 owners resolvable: MOVE BOARD rule

def msg(seat, plan=None, prose=(), src=()):
    return dict(seat=seat, plan=plan, prose=set(prose), src=set(src))


def test_owner_claims_plan_and_prose():
    rounds = [dict(msgs=[], moves=[]),
              dict(msgs=[msg("Orange", plan="B"), msg("Blue", prose={"C"})], moves=[])]
    assert dict(pr.owner_claims(rounds, 3, "plan")) == {"B": {"Orange"}}
    assert dict(pr.owner_claims(rounds, 3, "plan_prose")) == {"B": {"Orange"}, "C": {"Blue"}}


def test_moveboard_direct_and_position_links():
    rounds = [dict(msgs=[], moves=[]),
              dict(msgs=[msg("Orange", src={(1, 1)}), msg("Blue")], moves=[((1, 1), "B"), ((4, 4), "C")]),
              dict(msgs=[], moves=[((2, 2), "E"), ((4, 5), "F")])]
    assert dict(pr.owner_claims(rounds, 3, "plan_prose")) == {}
    # direct link fixes line 0 of round 1 to Orange; full rounds keep the seat order, so E is Orange's too
    assert dict(pr.owner_claims(rounds, 3, "plan_prose_mb")) == {"B": {"Orange"}, "E": {"Orange"}}


def test_moveboard_ambiguous_and_partial_rounds():
    # two lines of the round share the stated source cell: no link
    amb = [dict(msgs=[], moves=[]),
           dict(msgs=[msg("Orange", src={(1, 1)})], moves=[((1, 1), "B"), ((1, 1), "C")]),
           dict(msgs=[], moves=[((2, 2), "E"), ((4, 5), "F")])]
    assert dict(pr.owner_claims(amb, 3, "plan_prose_mb")) == {}
    # a round with fewer than n-1 lines (a teammate finished) links directly but fixes no position
    part = [dict(msgs=[], moves=[]),
            dict(msgs=[msg("Orange", src={(1, 1)})], moves=[((1, 1), "B")]),
            dict(msgs=[], moves=[((2, 2), "E"), ((4, 5), "F")])]
    assert dict(pr.owner_claims(part, 3, "plan_prose_mb")) == {"B": {"Orange"}}
    # the prefix round (before the viewer's first own turn) fixes no position either
    pre = [dict(msgs=[msg("Orange", src={(1, 1)})], moves=[((1, 1), "B"), ((4, 4), "C")]),
           dict(msgs=[], moves=[((2, 2), "E"), ((4, 5), "F")])]
    assert dict(pr.owner_claims(pre, 3, "plan_prose_mb")) == {"B": {"Orange"}}


def test_moveboard_conflicts_fix_nothing():
    # two senders on one position
    two = [dict(msgs=[], moves=[]),
           dict(msgs=[msg("Orange", src={(1, 1)})], moves=[((1, 1), "B"), ((4, 4), "C")]),
           dict(msgs=[msg("Blue", src={(2, 2)})], moves=[((2, 2), "E"), ((4, 5), "F")]),
           dict(msgs=[], moves=[((3, 3), "B"), ((5, 5), "F")])]
    cl = pr.owner_claims(two, 3, "plan_prose_mb")
    assert dict(cl) == {"B": {"Orange"}, "E": {"Blue"}}
    # one sender on two positions
    two_pos = [dict(msgs=[], moves=[]),
               dict(msgs=[msg("Orange", src={(1, 1)})], moves=[((1, 1), "B"), ((4, 4), "C")]),
               dict(msgs=[msg("Orange", src={(4, 5)})], moves=[((2, 2), "E"), ((4, 5), "F")])]
    assert dict(pr.owner_claims(two_pos, 3, "plan_prose_mb")) == {"B": {"Orange"}, "F": {"Orange"}}


# a final-arm 3p episode: Green owns A D (row 1), Orange B E (row 3), Blue C F (row 5); Blue finishes at ply 8
GREEN, ORANGE, BLUE = corpus.FINAL_SEATS[:3]
THREE_FINAL = {GREEN: ["A", "D"], ORANGE: ["B", "E"], BLUE: ["C", "F"]}
START3 = {"A": (0, 0), "D": (0, 5), "B": (2, 0), "E": (2, 5), "C": (4, 0), "F": (4, 5)}
PLIES3 = [
    (GREEN, "A right", "Start."),
    (ORANGE, "B right", "Moving B from (R3,C1) right."),
    (BLUE, "C right", "Going on."),
    (GREEN, "D left", "Next."),
    (ORANGE, "E left", "Now this."),
    (BLUE, "F left", "And this."),
    (GREEN, "A right", "Again."),
    (ORANGE, "B right", "More."),
    (BLUE, "done", "done"),
    (GREEN, "A down", "Fine."),
    (ORANGE, "B right", "Last."),
]


def rendered(ep, idx):
    """history.boards for the viewer of turn idx, split into rounds at the viewer's own lines:
    ([X-line sources per round], [sorted teammate senders per round])."""
    rec = []
    for t in ep["turns"][:idx]:
        history.capture(rec, t["seat"], dict(message=t["msg"], is_done=t["done"], object=t["obj"], direction=t["dir"]),
                        type("B", (), {"object_positions": t["board"]})(), corpus.DIRS)
    text = history.boards(rec, ep["turns"][idx]["seat"], seed="7", messages=True)
    block = lambda name: text.split(name, 1)[1].split("\n\n", 1)[0].splitlines()[1:] if name in text else []
    moves, msgs = [[]], [[]]
    for line in block("MOVE BOARD"):
        if line.startswith("You moved"):
            moves.append([])
        else:
            moves[-1].append(corpus.uncell(re.search(r"from \((R\d+,C\d+)\)", line).group(1)))
    for line in block("MESSAGE BOARD"):
        m = re.match(r'^(Player \w+)( \(you\))? said', line)
        if m and m.group(2):
            msgs.append([])
        elif m:
            msgs[-1].append(m.group(1))
    return moves, [sorted(x) for x in msgs]


def test_mb_view_matches_history_rendering():
    ep, A, M = make_ep("final", "freeform", THREE_FINAL, START3, TARGETS, PLIES3)
    for idx in range(1, len(PLIES3)):
        view = pr.mb_view(ep, idx, M)
        moves, msgs = rendered(ep, idx)
        assert [[s for s, _ in r["moves"]] for r in view] == moves, idx
        assert [sorted(m["seat"] for m in r["msgs"]) for r in view] == msgs, idx


def test_owners_resolved_on_episode():
    ep, A, M = make_ep("final", "freeform", THREE_FINAL, START3, TARGETS, PLIES3)
    cache = {}
    # Green at ply 9: rounds after its own turns 0, 3, 6; the first two are full (both teammates moved)
    assert pr.resolved(ep, 9, M, "plan", cache) == set()
    assert pr.resolved(ep, 9, M, "plan_prose", cache) == {"B"}           # present-frame G-move by Orange
    assert pr.resolved(ep, 9, M, "plan_prose_mb", cache) == {"B", "E"}   # E by Orange's position
    rows = pr.owners_rows(ep, M)
    assert [r["part"] for r in rows] == list(pr.VARIANTS) and rows[0]["n_views"] == len(PLIES3)
    assert rows[0]["owners"] <= rows[1]["owners"] <= rows[2]["owners"]
    # development arm and dyads are left out
    ep2, _, M2 = make_ep("dev", "freeform", DYAD, START, TARGETS, [("Player A", "A down", "Hi.")])
    assert pr.owners_rows(ep2, M2) == []


def test_run_tables_on_constructed_episodes():
    ep, A, _ = make_ep("final", "freeform", THREE_FINAL, START3, TARGETS, PLIES3)
    out = pr.run("calib", [ep], A)
    assert {"pr_form", "pr_form_msg", "pr_private", "pr_disclosure", "pr_copresence", "pr_boardref",
            "pr_resolvable", "pr_owners", "pr_cells", "pr_contrasts"} == set(out)
    assert all(r["episode"] == ep["meta"]["id"] and r["level"] == "S3" for r in out["pr_private"])
    assert {r["part"] for r in out["pr_private"]} == {"prose", "all"}
    assert out["pr_contrasts"] == []   # no hybrid partner
