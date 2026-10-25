"""finish.py: E5 calls only what no usable batch answers, the partial and complete
test runs (guards and commands) and the comparison. No API call is made."""
import json
import os
import re

import pytest

from analysis.messages import corpus, finish, grammar
from analysis.messages.extract import client
from analysis.messages.extract import run as x
from analysis.messages.tests.test_extract import make_ep


def _eps():
    ds = make_ep([("Player A", "E is at (R2,C6)."), ("Player A", "F sits on (R3,C3)."), ("Player A", "C is at (R4,C5).")],
                 eid="test|dsv41|3p|S2|freeform|01", tier="strong")
    qw = make_ep([("Player A", "B is at (R1,C6)."), ("Player A", "E is at (R5,C6).")],
                 eid="test|qwen38|3p|S2|freeform|01", tier="strong")
    return [ds, qw]


def _fake(calls):
    def complete(prompt, model_id):
        calls.append(prompt[1])
        items = [[i, m.group(1), f"({m.group(2)})", m.group(0)] for i, block in
                 enumerate(prompt[1].split("### Message ")[1:], 1)
                 for m in [re.search(r"([A-F]) (?:is at|sits on) \((R\d,C\d)\)", block)]]
        return json.dumps(items), {"inputTokens": 10, "outputTokens": 10}
    return complete


def test_e5_partial_then_complete_never_replaces(tmp_path, monkeypatch):
    eps = _eps()
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setattr(client, "BATCH", 2)
    monkeypatch.setattr(corpus, "load", lambda scope, models=corpus.MODELS: [e for e in eps if e["meta"]["model"] in models])
    calls = []
    monkeypatch.setattr(client, "openai_api", lambda effort: _fake(calls))
    A = grammar.analyse_all(eps)
    ds_units = x.eligible(eps[:1], A)
    # a usable-elsewhere batch (another run's message beside one of DeepSeek's) does not count for this run
    client.store("primary", finish.E5_MODEL, [next(iter(ds_units)), "foreign-input"], ("[]", {}))
    finish.e5("test", ("dsv41",))
    assert len(calls) == 2 and all("B is at" not in c for c in calls)        # DeepSeek's 3 inputs, batches of 2
    first = {k: v[0]["text"] for k, v in client.load_cache("primary", finish.E5_MODEL).items() if k in ds_units}
    finish.e5("test", ("dsv41", "qwen38"))
    assert len(calls) == 3 and "B is at" in calls[-1] and "F sits" not in calls[-1]  # only Qwen's inputs
    after = {k: v[0]["text"] for k, v in client.load_cache("primary", finish.E5_MODEL).items() if k in ds_units}
    assert after == first                                                     # DeepSeek's answers kept
    recs, _, _, annotated, _ = x.load_records(eps, "primary", finish.E5_MODEL, A)
    assert len(annotated) == 5 and recs[("test|dsv41|3p|S2|freeform|01", 1)][0]["cell"] == (2, 2)
    finish.e5("test", ("dsv41", "qwen38"))
    assert len(calls) == 3                                                    # complete: nothing called


@pytest.fixture
def staged(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(finish, "sh", lambda *a: cmds.append(a[2:]))
    monkeypatch.setattr(corpus, "HERE", str(tmp_path))
    st = {(m, s): 0 for m in corpus.MODELS for s in ("dyad", "3p", "6p")}
    st.update({("dsv41", s): finish.PER_TREE for s in ("dyad", "3p", "6p")})
    st[("qwen38", "dyad")] = 113
    monkeypatch.setattr(finish, "status", lambda: dict(st))
    return cmds, st, tmp_path


def test_partial_test_run_commands_and_guards(staged):
    cmds, st, tmp = staged
    finish.test(("dsv41",))
    assert cmds == [("analysis.messages.finish", "--e5", "test", "dsv41", "primary"),
                    ("analysis.messages.finish", "--e5", "test", "dsv41", "dest"),
                    ("analysis.messages.probe.run", "test", "--model", "dsv41", "--task", "e3", "--max-usd", "30"),
                    ("analysis.messages.probe.run", "test", "--model", "dsv41", "--task", "e4", "--max-usd", "30"),
                    ("analysis.messages.run", "test", "--models", "dsv41", "--out", "test_dsv41"),
                    ("analysis.messages.pack", "test_dsv41")]
    os.makedirs(tmp / "out" / "test_dsv41")
    with pytest.raises(SystemExit, match="happens once"):
        finish.test(("dsv41",))
    with pytest.raises(SystemExit, match="qwen38"):
        finish.test()                                    # Qwen's final arm is incomplete
    with pytest.raises(SystemExit, match="nemotron"):
        finish.test(("nemotron",))


def test_complete_run_and_comparison(staged, capsys):
    cmds, st, tmp = staged
    st.update({("qwen38", s): finish.PER_TREE for s in ("dyad", "3p", "6p")})
    for d, rows in (("test_dsv41", ["dsv41,1"]), ("test", ["dsv41,1", "qwen38,2"])):
        os.makedirs(tmp / "out" / d)
        (tmp / "out" / d / "same.csv").write_text("model,v\n" + "\n".join(rows) + "\n")
        (tmp / "out" / d / "tier.csv").write_text(f"model,v\ndsv41,{len(rows)}\n")
    os.rename(tmp / "out" / "test", tmp / "out" / "full")
    finish.compare("full")
    assert "1 tables identical for ['dsv41'], 1 differ: tier" in capsys.readouterr().out
    finish.test()
    assert cmds[0] == ("analysis.messages.finish", "--e5", "test", ",".join(corpus.MODELS), "primary")
    assert cmds[4] == ("analysis.messages.run", "test") and cmds[5] == ("analysis.messages.pack", "test")


def test_run_main_models_and_out_dir(tmp_path, monkeypatch):
    from analysis.messages import run
    seen = []
    monkeypatch.setattr(corpus, "load", lambda scope, models=corpus.MODELS: seen.append((scope, models)) or [])
    monkeypatch.setattr(corpus, "HERE", str(tmp_path))
    monkeypatch.setattr(run, "OUT", str(tmp_path / "out"))
    from analysis.messages.probe import client as probe_client
    monkeypatch.setattr(probe_client, "CACHE", str(tmp_path / "cache"))
    # one module runs, so its tables share main's namespace with the output directory
    run.main(["run", "test", "--only", "run,tables", "--models", "dsv41", "--out", "test_dsv41"])
    assert seen == [("test", ("dsv41",))]
    assert (tmp_path / "out" / "test_dsv41" / "validity_evidence.csv").exists()
    assert (tmp_path / "out" / "test_dsv41" / "tex").is_dir() and not (tmp_path / "out" / "test").exists()


def test_e5_dest_uses_the_module_cache_label(tmp_path, monkeypatch):
    from analysis.messages.extract import dest
    seen = []

    class Stop(Exception):
        pass

    def fake_extract(name, model, *a, **k):
        seen.append((name, model))
        raise Stop

    monkeypatch.setattr(corpus, "load", lambda scope, models=corpus.MODELS: [])
    monkeypatch.setattr(client, "CACHE", str(tmp_path))
    monkeypatch.setattr(dest, "backend", lambda: None)
    monkeypatch.setattr(client, "openai_api", lambda effort: None)
    monkeypatch.setattr(dest, "eligible", lambda eps, A: {"k": []})
    monkeypatch.setattr(x, "eligible", lambda eps, A: {"k": []})
    monkeypatch.setattr(x, "extract", fake_extract)
    for args in (("dest",), ()):
        with pytest.raises(Stop):
            finish.e5("test", ("dsv41",), *args)
    assert seen == [("dest", dest.MODEL), ("primary", finish.E5_MODEL)]
