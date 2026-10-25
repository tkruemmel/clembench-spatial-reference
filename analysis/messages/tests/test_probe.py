"""E3/E4 probe: prompt edits on synthetic and logged prompts, parsers, scoring,
sampling eligibility, the noise floor, the cache and the cap. No API call is made."""
import datetime
import json

import pytest

from analysis.messages import corpus, grammar
from analysis.messages.probe import client, edit, run
from matrixgame.utils import history

TAIL = ("CURRENT BOARD:\n<grid>\n\nGOAL BOARD:\n<grid>\n\nYour turn. Respond with:\nmessage: <your message>\n"
        "move: <OBJECT> <DIRECTION>\n\nOR, if ALL your objects are at their target positions:\nmessage: done\n"
        "done: all my objects are placed")
PRE = "RULES ... The best move serves several at once.\n"


def dev_prompt(lines, merged=True):
    """As the dev masters build it: lines, a blank line, boards, response block."""
    ctx = ("\n".join(lines) + "\n\n" if lines else "") + TAIL
    return [{"role": "user", "content": PRE + "\n\n" + ctx}] if merged else \
        [{"role": "user", "content": PRE}, {"role": "user", "content": ctx}]


def final_prompt(record, viewer):
    """As the final masters build it, from history.boards (the game's own code)."""
    return [{"role": "user", "content": PRE + "\n\n" + history.boards(record, viewer, "7", True) + TAIL}]


REC = [dict(turn=1, player="Player Green", message="I go first, A to (R2,C3).", object="A", direction="down",
            src=(0, 2), dst=(1, 2)),
       dict(turn=2, player="Player Orange", message="Green, B  sits at\n(R4,C4).", object="B", direction="up",
            src=(3, 3), dst=(2, 3)),
       dict(turn=3, player="Player Blue", message="done", done=True)]


# ---------------------------------------------------------------- edits

def test_ask_replaces_only_the_response_block():
    for msgs in (dev_prompt(["Player A says: hi"]), dev_prompt(["Player A says: hi"], merged=False),
                 final_prompt(REC, "Player Blue")):
        out = edit.ask(msgs, edit.q_loc("E"))
        c, o = msgs[-1]["content"], out[-1]["content"]
        assert o == c[:c.rfind(edit.RESPONSE)] + edit.q_loc("E") and out[:-1] == msgs[:-1]
        assert "Respond with" not in o and o.endswith("`cell: unknown`.")
    with pytest.raises(ValueError):
        edit.ask([{"role": "user", "content": "no block"}], "Q")


def test_question_is_colon_free_outside_the_answer_format():
    for q in (edit.q_loc, edit.q_goal):
        assert ":" not in q("E").split("Answer `")[0]


def test_strip_messages_dev_and_final():
    lines = ["Player A says: one\nwith a newline", "Player B (you) says: two", "Player C says: done"]
    entries = [("Player A", "one\nwith a newline"), ("Player B (you)", "two"), ("Player C", "done")]
    for merged in (True, False):
        assert edit.strip_messages(dev_prompt(lines, merged), "dev", entries) == dev_prompt([], merged)
    out = edit.strip_messages(final_prompt(REC, "Player Blue"), "final", [])
    assert out == final_prompt([dict(r, message=None) for r in REC[:2]], "Player Blue")
    assert "MESSAGE BOARD" not in out[0]["content"] and "MOVE BOARD" in out[0]["content"]


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_delete_dev_line_positions(pos):
    lines = ["Player A says: one\nwith a newline", "Player B (you) says: two", "Player C says: three"]
    seat, msg = ["Player A", "Player B (you)", "Player C"][pos], ["one\nwith a newline", "two", "three"][pos]
    out = edit.delete(dev_prompt(lines), "dev", seat, msg)
    assert out == dev_prompt(lines[:pos] + lines[pos + 1:])


def test_delete_dev_only_line_takes_the_blank_line():
    for merged in (True, False):
        assert edit.delete(dev_prompt(["Player A says: x"], merged), "dev", "Player A", "x") == dev_prompt([], merged)


def test_delete_final_lines_and_only_line_takes_the_header():
    msgs = final_prompt(REC, "Player Blue")
    out = edit.delete(msgs, "final", "Player Orange", REC[1]["message"])
    assert out == final_prompt([REC[0], dict(REC[1], message=None), REC[2]], "Player Blue")
    assert 'said "Green, B sits at (R4,C4)."' in msgs[0]["content"]  # whitespace collapsed, as history.py
    one = [REC[0]]
    out = edit.delete(final_prompt(one, "Player Orange"), "final", "Player Green", REC[0]["message"])
    assert "MESSAGE BOARD" not in out[0]["content"] and out == final_prompt([dict(REC[0], message=None)], "Player Orange")
    assert "MOVE BOARD" in out[0]["content"]


def test_delete_refuses_a_missing_or_repeated_entry():
    with pytest.raises(ValueError):
        edit.delete(dev_prompt(["Player A says: x"]), "dev", "Player B", "x")
    with pytest.raises(ValueError):
        edit.delete(dev_prompt(["Player A says: x", "Player A says: x"]), "dev", "Player A", "x")


def test_replace_dev_and_final():
    out = edit.replace(dev_prompt(["Player A says: x", "Player B says: y"]), "dev", "Player A", "x", "z")
    assert out == dev_prompt(["Player A says: z", "Player B says: y"])
    out = edit.replace(final_prompt(REC, "Player Blue"), "final", "Player Green", REC[0]["message"], "new  text")
    assert out == final_prompt([dict(REC[0], message="new text")] + REC[1:], "Player Blue")


def test_swap_every_g_cell_form_same_length_in_grammar():
    msg = "D to (R3,C4) then R3C4, r3, c4 again; (R3, C4) and (R5,C6) stay. PLAN: move A from (R1,C1) to (R3,C4) — REQUEST: clear (R3,C4)"
    new, n = edit.swap_cell(msg, "dev", (2, 3), (6, 1))
    assert n == 6 and len(new) == len(msg)
    cells = lambda s: [t.val for t in grammar.tokenize(s, "dev", grammar.DEFAULT) if t.kind == "cell"]
    assert cells(new) == [(6, 1) if c == (2, 3) else c for c in cells(msg)]
    assert "(R7,C2)" in new and "R7C2" in new and "r7, c2" in new and "(R7, C2)" in new and "(R5,C6)" in new
    assert corpus.slots(new, "hybrid")["request"] == "clear (R7,C2)"


# ---------------------------------------------------------------- answers

def test_parsers():
    assert edit.parse_cell("cell: (R3,C4)") == (2, 3)
    assert edit.parse_cell("I think...\n`cell: (r3, c4)`") == (2, 3)
    assert edit.parse_cell("thinking cell: (R1,C1) </think>\ncell: R5C6") == (4, 5)  # after the last </think>
    assert edit.parse_cell("cell: (R1,C1)\nno wait, cell: unknown") == "unknown"   # the last answer counts
    assert edit.parse_cell("**cell:** unknown") == "unknown" and edit.parse_cell("(R3,C4)") is None
    assert edit.parse_move("message: hi\nmove: a Up") == ("A", "up")
    assert edit.parse_move("message: done\ndone: all my objects are placed") == "done"
    assert edit.parse_move("done: placed") == "done" and edit.parse_move("move: A up") is None


# ---------------------------------------------------------------- constructed episodes

OWN = {"A": "Player A", "D": "Player A", "B": "Player B", "E": "Player B", "C": "Player C", "F": "Player C"}
BOARD = {"A": (0, 0), "D": (0, 3), "B": (2, 2), "E": (5, 5), "C": (7, 1), "F": (4, 6)}


def make_ep(plies, suite="3p", proto="freeform", owner=OWN, board=BOARD, eid="dev|dsv41|3p|S1|freeform|01", targets=BOARD):
    """plies: [(seat, message, object, direction)]; object None is a done turn."""
    seats = list(dict.fromkeys(owner.values()))
    b, turns = dict(board), []
    for i, (seat, msg, o, d) in enumerate(plies):
        src = b.get(o)
        dst = (src[0] + corpus.DIRS[d][0], src[1] + corpus.DIRS[d][1]) if o else None
        turns.append(dict(ply=i, seat=seat, done=o is None, retry=False, msg=msg, obj=o, dir=d, src=src, dst=dst,
                          board=dict(b), call=i, attempts=[i]))
        if o:
            b[o] = dst
    meta = dict(id=eid, arm="dev", model=eid.split("|")[1], tier="strong", suite=suite, level=eid.split("|")[3],
                board=int(eid.split("|")[5]), proto=proto, path="")
    ep = dict(meta=meta, seats=seats, owner=owner, turns=turns, targets=dict(targets),
              objs={s: sorted(o for o in owner if owner[o] == s) for s in seats},
              calls=[dict(player=f"Player {seats.index(t['seat']) + 1}") for t in turns])
    for i, t in enumerate(turns):
        t["is_msg"] = bool(t["msg"]) and not t["done"]
        t["slots"] = corpus.slots(t["msg"], proto) if t["is_msg"] else None
        t["vis"] = corpus.visible(ep, i, False)
    return ep


def test_e3_eligibility_every_teammate_object_and_named():
    ep = make_ep([("Player A", "I move A down.", "A", "down"),
                  ("Player B", "B goes right, C is at (R8,C2).", "B", "right"),
                  ("Player C", "ok", "C", "up")])
    A = grammar.analyse_all([ep])
    assert run.e3_decision(ep, A, ep["turns"][0]) is None             # no teammate message in view
    d = run.e3_decision(ep, A, ep["turns"][2])
    assert d["letters"] == ["A", "B", "D", "E"] and d["named"] == ["A", "B"]  # C is the listener's own
    only_own = make_ep([("Player A", "A goes.", "A", "down"), ("Player A", "again", "A", "down")])
    assert run.e3_decision(only_own, grammar.analyse_all([only_own]), only_own["turns"][1]) is None


def _e4_ep(req, suite="3p", proto="structured"):
    msg = f"PLAN: move C from (R8,C2) to (R7,C2) — REQUEST: {req}"
    return make_ep([("Player C", msg, "C", "up"), ("Player A", "PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: none",
                                                    "A", "down")], suite=suite, proto=proto,
                   eid=f"dev|dsv41|{suite}|S1|{proto}|01")


def test_e4_eligibility():
    ep = _e4_ep("A please leave (R1,C1)")
    d = run.e4_decision(ep, grammar.analyse_all([ep]), ep["turns"][1])
    assert (d["O1"], d["c1"], d["O2"], d["c2"]) == ("A", (0, 0), "D", (0, 3))
    assert d["c1_elsewhere"] is False
    for bad in (_e4_ep("move off (R1,C1) and (R1,C4)"),       # O2's cell is named
                _e4_ep("keep (R6,C6) clear"),                  # no listener object on a request cell
                _e4_ep("none"),                                # no REQUEST
                _e4_ep("A please leave (R1,C1)", suite="6p")):  # 6p is out
        assert run.e4_decision(bad, grammar.analyse_all([bad]), bad["turns"][1]) is None
    ff = make_ep([("Player C", "A, leave (R1,C1)", "C", "up"), ("Player A", "x", "A", "down")])
    assert run.e4_decision(ff, grammar.analyse_all([ff]), ff["turns"][1]) is None  # freeform has no REQUEST


def test_sample_caps_two_per_episode_and_interleaves():
    eps = []
    for g in range(1, 4):
        for proto in ("freeform", "hybrid"):
            plies = [("Player A", "Go. PLAN: move A from (R1,C1) to (R2,C1) — REQUEST: B clear (R3,C3)" if proto == "hybrid"
                      else "B is at (R3,C3).", "A", "down")]
            plies += [("Player B", "B moves. PLAN: move B from (R3,C3) to (R3,C4) — REQUEST: none" if proto == "hybrid"
                       else "fine", "E", "up"), ("Player C", "C.", "F", "up")] * 3
            eps.append(make_ep(plies, proto=proto, eid=f"dev|dsv41|3p|S1|{proto}|{g:02d}"))
    A = grammar.analyse_all(eps)
    order, pool = run.sample("e3", eps, A, n=100)
    per_ep = {}
    for d in order:
        per_ep[d["eid"]] = per_ep.get(d["eid"], 0) + 1
    assert max(per_ep.values()) == 2 and len(order) == 12
    assert {d["eid"].split("|")[4] for d in order[:2]} == {"freeform", "hybrid"}  # prefixes span the strata
    assert order == run.sample("e3", eps, A, n=100)[0]  # seeded
    goal = run.sample("e3g", eps, A, n=100)[0]
    assert [(d["eid"], d["ply"]) for d in goal] == [(d["eid"], d["ply"]) for d in order]  # E3g asks at E3's decisions


# ---------------------------------------------------------------- scoring

def test_score_e3_contribution_and_split():
    ep = make_ep([("Player A", "B is at (R3,C3).", "A", "down"), ("Player C", "x", "C", "up")])
    base = dict(eid=ep["meta"]["id"], ply=1, prompt_ply=1, kind="loc", k=1)
    items = [dict(base, arm=a, letter=L, named=L == "B", sha=f"{L}{a}") for L in ("B", "D") for a in ("orig", "nomsg")]
    replies = {"Borig": "cell: (R3,C3)", "Bnomsg": "cell: unknown", "Dorig": "cell: (R1,C4)", "Dnomsg": "cell: (R1,C4)"}
    cache = {(k, 0): dict(text=v, finish="stop") for k, v in replies.items()}
    ans, rows, summ = run.score_e3(items, cache, {ep["meta"]["id"]: ep})
    r = rows[0]
    assert len(ans) == 4 and r["objects"] == 2 and r["named_share"] == 0.5
    assert (r["correct_orig"], r["correct_nomsg"], r["contribution"]) == (1.0, 0.5, 0.5)
    assert (r["named_contribution"], r["unnamed_contribution"], r["unknown_nomsg"]) == (1.0, 0.0, 0.5)
    assert {x["estimand"] for x in summ} >= {"contribution", "named_contribution", "unnamed_contribution"}


GOALS = dict(BOARD, A=(1, 1), B=(2, 3), E=(6, 6))  # D, C, F start home


def test_e3g_stated_home_and_score():
    ep = make_ep([("Player B", "My target for B is (R3,C4).", "E", "up"), ("Player A", "A target (R3,C4)?", "A", "down"),
                  ("Player C", "x", "C", "up")], targets=GOALS)
    A = grammar.analyse_all([ep])
    d = run.e3g_decision(ep, A, ep["turns"][2])
    assert d["letters"] == ["A", "B", "D", "E"] and d["stated"] == ["B"] and d["home"] == ["D"]  # only the owner states
    base = dict(eid=ep["meta"]["id"], ply=2, prompt_ply=2, kind="goal", k=1)
    items = [dict(base, arm=a, letter=L, stated=L == "B", home=L == "D", sha=f"{L}{a}") for L in ("B", "D", "E")
             for a in ("orig", "nomsg")]
    replies = {"Borig": "cell: (R3,C4)", "Bnomsg": "cell: unknown", "Dorig": "cell: (R1,C4)", "Dnomsg": "cell: (R1,C4)",
               "Eorig": "cell: unknown", "Enomsg": "cell: (R5,C6)"}  # E's current cell is not its target
    cache = {(k, 0): dict(text=v, finish="stop") for k, v in replies.items()}
    ans, rows, summ = run.score_e3(items, cache, {ep["meta"]["id"]: ep}, "goal")
    r = rows[0]
    assert len(ans) == 6 and r["stated_share"] == pytest.approx(1 / 3) and r["home_share"] == pytest.approx(1 / 3)
    assert (r["correct_orig"], r["correct_nomsg"]) == (pytest.approx(2 / 3), pytest.approx(1 / 3))
    assert (r["stated_contribution"], r["unstated_contribution"], r["away_contribution"]) == (1.0, 0.0, 0.5)
    assert {x["estimand"] for x in summ} >= {"away_contribution", "stated_correct_orig", "home_share"}
    assert run.score_e3(items, cache, {ep["meta"]["id"]: ep}) == ([], [], [])  # loc scoring ignores goal items


def test_effects_and_floor():
    e = run.e4_effects([("A", "up")] * 8, [("D", "left")] * 4, "A", "D")
    assert (e["tau"], e["delta"], e["floor_tau"], e["floor_delta"]) == (1.0, 1.0, 0.0, 0.0)
    noisy = [("A", "up"), ("A", "up"), ("D", "up"), None, ("A", "up"), ("D", "down"), ("D", "down"), "done"]
    e = run.e4_effects(noisy, noisy[:4], "A", "D")
    assert e["floor_tau"] == pytest.approx(0.25) and e["floor_delta"] == pytest.approx(0.25)  # halves differ by noise
    assert e["tau"] == pytest.approx(0.25 - 3 / 8) and e["unparseable_orig"] == 1 / 8 and e["unparseable_swap"] == 1 / 4


# ---------------------------------------------------------------- cache, cap, CLI

def test_cache_and_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setitem(client.PRICES, "dsv41", (1.0, 1.0))
    monkeypatch.setattr(client, "peak", lambda t: False)
    calls = []

    def fake(msgs):
        calls.append(msgs)
        return "cell: unknown", {"in": 300_000, "out": 200_000}, "stop"   # $0.50 each off-peak

    jobs = [(f"s{j}", i, [{"role": "user", "content": str(j)}]) for j in range(5) for i in range(2)]
    st = client.execute("dsv41", jobs + jobs[:3], fake, max_usd=1.2, workers=1)
    assert st["ok"] == 3 and len(calls) == 3 and st["skipped"] == 7      # the cap stops submission; duplicates once
    assert set(client.load_cache("dsv41")) == {("s0", 0), ("s0", 1), ("s1", 0)}
    client.execute("dsv41", jobs[:3], fake, max_usd=10, workers=1)
    assert len(calls) == 3                                               # cached: never paid twice


def test_prices():
    sat = datetime.datetime(2026, 10, 17, 7, tzinfo=datetime.timezone.utc)
    mon = datetime.datetime(2026, 10, 19, 7, tzinfo=datetime.timezone.utc)
    u = {"in": 1e6, "out": 1e6}
    assert client.usd("dsv41", u, sat) == pytest.approx(0.75) and client.usd("dsv41", u, mon) == pytest.approx(1.5)
    assert not client.peak(datetime.datetime(2026, 10, 19, 5, tzinfo=datetime.timezone.utc))
    assert client.usd("nemotron", u, mon) == pytest.approx(0.96)


def test_cli_refuses_potsdam(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="qwen38 refused"):
        run.main(["dev", "--model", "qwen38", "--task", "e3", "--dry-run"])


def test_run_scores_from_the_cache_only(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setattr(client, "BACKENDS", {})  # any backend use would fail
    ep = _e4_ep("A please leave (R1,C1)")
    A = grammar.analyse_all([ep])
    d = run.e4_decision(ep, A, ep["turns"][1])
    base = dict(eid=ep["meta"]["id"], ply=1, q=0, prompt_ply=1, O1="A", O2="D", c1=[0, 0], c2=[0, 3], seat="Player C")
    items = [dict(base, kind="move", arm="orig", k=8, sha="mo"), dict(base, kind="move", arm="swap", k=4, sha="ms")]
    with open(run.manifest_path("dev", "dsv41", "e4"), "w") as f:
        f.writelines(json.dumps(i) + "\n" for i in items)
    texts = {"mo": "message: m\nmove: A down", "ms": "message: m\nmove: D right"}
    for it in items:
        for i in range(it["k"]):
            client.store("dsv41", it["sha"], i, (texts[it["sha"]], {"in": 10, "out": 10}, "stop"))
    out = run.run("dev", [ep], A)
    r = out["probe_e4_decisions"][0]
    assert d["O1"] == "A" and (r["tau"], r["delta"], r["floor_tau"]) == (1.0, 1.0, 0.0)
    assert {x["estimand"] for x in out["probe_e4_summary"]} == {"tau", "delta", "floor_tau", "floor_delta"}
    assert out["probe_spend"][0]["cached"] == 12
