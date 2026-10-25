"""E5 destinations (extract/dest.py) on constructed episodes with a fake cache; no API call."""
import json

from analysis.messages import grammar
from analysis.messages.extract import client, dest, records
from analysis.messages.tests.test_alignment import make

START = {"A": (0, 0), "B": (3, 3)}
MSG = "Moving A right to (R1,C2), then A goes on to (R1,C4). B is at (R4,C4)."


def _ep(proto="freeform"):
    return make([(0, ("A", "right"), MSG, False),
                 (1, ("B", "up"), "ok", False),
                 (0, ("A", "right"), "going on", False),
                 (1, ("B", "up"), "ok", False),
                 (0, ("A", "right"), "A to (R1,C6) later.", False)], proto=proto, start=START)


def rec(L, c, span):
    return dict(letter=L, cell=c, span=span)


def test_eligible_freeform_own_letter_and_cell():
    ep = _ep()
    units = dest.eligible([ep], grammar.analyse_all([ep]))
    assert sorted(t["ply"] for ms in units.values() for _, t in ms) == [0, 4]   # "ok" / "going on" name no cell
    hy = _ep("hybrid")
    assert dest.eligible([hy], grammar.analyse_all([hy])) == {}                 # the PLAN carries it there
    other = make([(1, ("B", "up"), "A is at (R1,C1).", False)], start=START)    # only a teammate's letter
    assert dest.eligible([other], grammar.analyse_all([other])) == {}


def test_kinds_commitment_and_scores():
    ep = _ep()
    t0 = ep["turns"][0]
    now, later = rec("A", (0, 1), "Moving A right to (R1,C2)"), rec("A", (0, 3), "Moving A right to (R1,C2), then A goes on to (R1,C4)")
    assert dest.kind(ep, t0, now) == "now" and dest.kind(ep, t0, later) == "later"
    assert dest.kind(ep, t0, rec("A", (0, 0), "A")) == "already"
    assert dest.commitments(ep, t0, [now, later, rec("A", (0, 2), "A goes on")]) == [later]  # the route's end
    nxt = [t for t in ep["turns"] if t["seat"] == t0["seat"]][1:]
    s = dest.score(ep, t0, later, nxt)
    assert s == dict(saydo="progress", next_progress=True, reached=True)
    t4 = ep["turns"][4]
    assert dest.score(ep, t4, rec("A", (0, 5), "A to (R1,C6)"), []) == dict(saydo="progress", next_progress=None,
                                                                             reached=None)  # censored


def test_run_tables_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    ep = _ep()
    A = grammar.analyse_all([ep])
    units = dest.eligible([ep], A)
    msgs = [units[k][0][1]["msg"] for k in units]
    reply = [[i, "A", "(R1,C4)", "Moving A right to (R1,C2), then A goes on to (R1,C4)"] for i, m in enumerate(msgs, 1)
             if m == MSG]
    reply += [[i, "B", "(R4,C4)", "B is at (R4,C4)"] for i, m in enumerate(msgs, 1) if m == MSG]   # not the writer's
    reply += [[i, "A", "(R1,C6)", "A to (R1,C6)"] for i, m in enumerate(msgs, 1) if m != MSG]
    assert dest.run("dev", [ep], A)["dest_meta"][0]["annotated"] == 0        # nothing cached: no estimates
    client.store(dest.PROMPT, dest.MODEL, list(units), (json.dumps(reply), {}))
    out = dest.run("dev", [ep], A)
    meta = out["dest_meta"][0]
    assert meta["annotated"] == 2 and meta["reject_not_own"] == 1 and meta["validation"] in ("pending", "validated",
                                                                                              "exploratory (author check failed)")
    assert [(r["ply"], r["cell"], r["next_progress"], r["reached"]) for r in out["dest_records"]] == \
        [(0, "(R1,C4)", True, True), (4, "(R1,C6)", None, None)]
    e = out["dest_episodes"][0]
    # 4 messages (seat B's repeated "ok" dropped), 2 with a commitment
    assert e["commit_stated"] == 2 / 4 and e["next_progress"] == 1.0 and e["reached_n"] == 1
    assert {c["metric"] for c in out["dest_cells"]} >= {"commit_stated", "next_progress", "reached"}


def test_prompt_colon_free_and_example_passes():
    text = open(client.prompt_path(dest.PROMPT)).read()
    assert ":" not in text.split("\nANSWER\n")[0]
    example = dest.to_array(text.split("\nExample\n")[1].split("\n")[0])
    ok, bad = records.accept(example, ["I'll take D to (R2,C3) now.", "nothing", "Moving A up to (R2,C1), then (R1,C1)."],
                             "dev")
    assert not bad and sorted(ok) == [1, 3] and len(ok[3]) == 2


def test_schema_reply_becomes_e5_items():
    out = dest.to_array('{"items": [{"message": 2, "letter": "D", "cell": "(R2,C3)", "span": "D to (R2,C3)"}]}')
    assert json.loads(out) == [[2, "D", "(R2,C3)", "D to (R2,C3)"]]
    assert dest.to_array('{"items": []}') == "[]"
    assert dest.to_array("cut off [[1,") == "cut off [[1,"            # not repaired: parsing fails, the batch is re-called
