"""E5 pipeline on synthetic episodes with a fake backend (no API call is made)."""
import json
import re

from analysis.messages import grammar
from analysis.messages.extract import client, records, run

OWNER = {"A": "Player A", "D": "Player A", "B": "Player B", "E": "Player B", "C": "Player C", "F": "Player C"}
BOARD = {"A": (0, 7), "B": (0, 5), "C": (3, 4), "D": (1, 6), "E": (1, 5), "F": (2, 2)}


def make_ep(msgs, eid="dev|nemotron|3p|S1|freeform|01", tier="weak"):
    """One episode; msgs = [(seat, text)]; the board never changes."""
    turns = [dict(ply=i, seat=seat, obj=None, dir=None, done=False, msg=text, board=dict(BOARD), after=dict(BOARD),
                  is_msg=True, dup=False, slots=None, vis=[]) for i, (seat, text) in enumerate(msgs)]
    seats = ["Player A", "Player B", "Player C"]
    return dict(meta=dict(id=eid, arm="dev", model=eid.split("|")[1], tier=tier, proto="freeform"), seats=seats,
                owner=OWNER, objs={s: sorted(o for o in OWNER if OWNER[o] == s) for s in seats}, turns=turns)


MSG = "E is at (R2,C6) and F sits on R3C3. Player B, keep (R4,C5) clear. A good spot is (R1,C1)."


def why(item, msgs=(MSG,)):
    ok, bad = records.accept(json.dumps([item]), list(msgs), "dev")
    return bad[0][0] if bad else None


def test_accepts_and_canonicalises_cells():
    ok, bad = records.accept(json.dumps([[1, "E", "(R2,C6)", "E is at (R2,C6)"], [1, "F", "(R3,C3)", "F sits on R3C3"]]),
                             [MSG], "dev")
    assert not bad and [(r["letter"], r["cell"]) for r in ok[1]] == [("E", (1, 5)), ("F", (2, 2))]


def test_every_reject_reason():
    assert records.accept("no array here", [MSG], "dev")[1][0][0] == "parse"
    assert why([1, "E", "R2C6", "E is at (R2,C6)"]) == "schema"          # cell not written (Rr,Cc)
    assert why([1, "X", "(R2,C6)", "E is at (R2,C6)"]) == "schema"
    assert why(["1", "E", "(R2,C6)", "E is at (R2,C6)"]) == "schema"
    assert why([2, "E", "(R2,C6)", "E is at (R2,C6)"]) == "message"
    assert why([1, "E", "(R2,C6)", "E is at R2,C6"]) == "span_not_verbatim"
    assert why([1, "B", "(R4,C5)", "keep (R4,C5) clear"]) == "letter_not_in_span"
    assert why([1, "A", "(R1,C1)", "A good spot is (R1,C1)"]) == "letter_not_in_span"  # the article, not a letter
    assert why([1, "E", "(R3,C3)", "E is at (R2,C6)"]) == "cell_not_in_span"
    assert set(records.REASONS) == {"parse", "schema", "message", "span_not_verbatim", "letter_not_in_span",
                                    "cell_not_in_span"}


def test_batch_numbering_routes_records():
    msgs = ["nothing here", "D is at (R2,C7)."]
    ok, bad = records.accept(json.dumps([[2, "D", "(R2,C7)", "D is at (R2,C7)"], [1, "D", "(R2,C7)", "D is at (R2,C7)"]]),
                             msgs, "dev")
    assert list(ok) == [2] and bad[0][0] == "span_not_verbatim"  # a span from another message is rejected


def test_model_only_at_and_agreement():
    msg = "E is at (R2,C6). I think F sits on (R3,C3). D is at (R2,C7)."
    ep = make_ep([("Player A", msg)])
    A = grammar.analyse_all([ep])
    k = (ep["meta"]["id"], 0)
    assert records.gloc_pairs(A[k]) >= {("E", (1, 5)), ("D", (1, 6))}
    recs = {k: [dict(letter="E", cell=(1, 5), span="E is at (R2,C6)"),     # G-loc has it
                dict(letter="F", cell=(2, 2), span="F sits on (R3,C3)"),   # model only
                dict(letter="D", cell=(1, 6), span="D is at (R2,C7)")]}    # the writer's own object
    assert records.model_only_at([ep], A, recs) == {k + ("F", (2, 2))}
    row = records.agreement([ep], A, recs, {k})[0]
    assert (row["n_grammar"], row["n_model"], row["n_both"], row["model_only"]) == (1, 2, 1, 1)
    assert row["n_canonical"] == 1 and row["canonical_recall"] == 1.0
    assert records.truth(ep, ep["turns"][0], recs[k][1]) is True


def test_eligible_needs_teammate_letter_and_cell_and_dedupes():
    ep = make_ep([("Player A", "E is at (R2,C6)."), ("Player A", "D is at (R2,C7)."), ("Player A", "E is near me."),
                  ("Player A", "E is at (R2,C6)."), ("Player B", "E is at (R2,C6).")])
    units = run.eligible([ep], grammar.analyse_all([ep]))
    # own letter only (ply 1) and no cell (ply 2) are out; ply 3 repeats ply 0's input; Player B's E is its own
    assert sorted(sorted(t["ply"] for _, t in v) for v in units.values()) == [[0, 3]]


def test_prompts_same_answer_format_and_colon_free_prose():
    texts = {n: open(client.prompt_path(n)).read() for n in client.PROMPTS}
    tails = {t.split("\nANSWER\n", 1)[1] for t in texts.values()}
    assert len(tails) == 1  # same answer format, example and input template
    for n, t in texts.items():
        prose = t.split("\nANSWER\n")[0]
        assert ":" not in prose, n
    assert len({client.prompt_sha(n) for n in client.PROMPTS}) == 3


def test_prompt_example_passes_the_acceptance_rules():
    example = open(client.prompt_path("primary")).read().split("\nExample\n")[1].split("\n")[0]
    msgs = ["E is at (R2,C6) now.", "nothing", "Look, (R3,C3) is occupied by B."]
    ok, bad = records.accept(example, msgs, "dev")
    assert not bad and sorted(ok) == [1, 3]


def test_render_shows_only_messages_and_own_letters():
    ep = make_ep([("Player A", "first"), ("Player B", "E is at (R2,C6)")])
    system, user = client.render("primary", [(ep, ep["turns"][0]), (ep, ep["turns"][1])])
    assert "<<" not in user and "=== INPUT" not in system
    assert "### Message 1 (the writer's own objects are A, D)\nfirst" in user
    assert "### Message 2 (the writer's own objects are B, E)\nE is at (R2,C6)" in user
    for leak in ("nemotron", "weak", "strong", "success", "Player"):
        assert leak not in user


def test_fake_backend_batches_cache_and_dry_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setattr(client, "BATCH", 2)
    monkeypatch.setitem(run.PRICES, "fake-model", (1.0, 1.0))
    texts = ["E is at (R2,C6).", "F sits on (R3,C3).", "C is at (R4,C5).", "B is at (R1,C6)."]
    ep = make_ep([("Player A", x) for x in texts])
    A = grammar.analyse_all([ep])
    units = run.eligible([ep], A)
    calls = []

    def fake(prompt, model_id):
        calls.append(prompt[1])
        n = prompt[1].count("### Message")
        items = [[i, m.group(1), f"({m.group(2)})", m.group(0)] for i, block in
                 enumerate(prompt[1].split("### Message ")[1:], 1)
                 for m in [re.search(r"([A-F]) (?:is at|sits on) \((R\d,C\d)\)", block)]]
        return json.dumps(items), {"inputTokens": 300 * n, "outputTokens": 20 * n}

    run.extract("primary", "fake-model", fake, units, limit=2, workers=1)
    assert len(calls) == 1                     # the pilot: the first batch
    run.extract("primary", "fake-model", fake, units, workers=1)
    assert len(calls) == 2                     # the full run reuses the pilot's batch
    run.extract("primary", "fake-model", fake, units, workers=1)
    assert len(calls) == 2                     # nothing re-paid
    recs, rejects, _, annotated, n = run.load_records([ep], "primary", "fake-model", A, units)
    assert n == 4 and len(annotated) == 4 and not rejects
    assert recs[(ep["meta"]["id"], 1)][0]["cell"] == (2, 2)
    run.dry_run(["primary"], units, model="fake-model")
    assert "all batches cached" in capsys.readouterr().out
    run.dry_run(["primary", "alt1"], units)
    out = capsys.readouterr().out
    assert "gpt-5.6-luna" in out and "all wordings" in out
    tabs = run.report([ep], A, "primary", "fake-model")
    assert tabs["e5_model_only"] == [] or all(r["n"] >= 0 for r in tabs["e5_model_only"])
    assert tabs["e5_agreement"][0]["n_model"] == 4


def test_spending_cap_and_unpriced_model(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setattr(client, "BATCH", 1)
    ep = make_ep([("Player A", f"E is at (R{i},C6).") for i in range(1, 6)])
    units = run.eligible([ep], grammar.analyse_all([ep]))
    calls = []

    def fake(prompt, model_id):
        calls.append(1)
        return "[]", {"inputTokens": 1_000_000, "outputTokens": 0}  # $1 a batch at the test price

    import pytest
    with pytest.raises(SystemExit, match="no price"):
        run.extract("primary", "unpriced", fake, units, workers=1)
    monkeypatch.setitem(run.PRICES, "fake-model", (1.0, 1.0))
    state = run.extract("primary", "fake-model", fake, units, workers=1, max_usd=2.0)
    assert len(calls) == 2 and state["skipped"] == 3 and state["unanswered"] == 3
    # the CLI exits non-zero while batches are unanswered
    monkeypatch.setattr(run.corpus, "load", lambda scope: [ep])
    monkeypatch.setattr(run.client, "openai_api", lambda effort: fake)
    with pytest.raises(SystemExit, match="incomplete extraction"):
        run.main(["dev", "--backend", "openai", "--model", "fake-model", "--max-usd", "1", "--workers", "1"])
    assert len(calls) == 3
    run.main(["dev", "--backend", "openai", "--model", "fake-model", "--max-usd", "10"])  # completes: no exit
    assert len(calls) == 5


def test_unparseable_batch_is_retried_and_not_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setitem(run.PRICES, "fake-model", (1.0, 1.0))
    ep = make_ep([("Player A", "E is at (R2,C6).")])
    A = grammar.analyse_all([ep])
    units = run.eligible([ep], A)
    replies = iter(["oops, no array", '[[1, "E", "(R2,C6)", "E is at (R2,C6)"]]'])
    fake = lambda prompt, model_id: (next(replies), {"inputTokens": 10, "outputTokens": 10})
    run.extract("primary", "fake-model", fake, units, workers=1)
    assert run.load_records([ep], "primary", "fake-model", A, units)[3] == set()  # not annotated
    run.extract("primary", "fake-model", fake, units, workers=1)
    recs, rejects, _, annotated, _ = run.load_records([ep], "primary", "fake-model", A, units)
    assert annotated == {(ep["meta"]["id"], 0)} and recs[(ep["meta"]["id"], 0)][0]["letter"] == "E"


def test_wrapped_array_is_no_answer():
    assert records.parse('[[[1, "E", "(R2,C6)", "E is at (R2,C6)"]]]') is None
    assert records.parse('[[1, "E", "(R2,C6)", "E is at (R2,C6)"]]') == [[1, "E", "(R2,C6)", "E is at (R2,C6)"]]
    assert records.parse("[]") == []
