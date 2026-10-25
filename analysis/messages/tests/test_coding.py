"""§9.6 author check on synthetic pools and labels."""
import hashlib
import json
import re

import pytest

from analysis.messages import grammar
from analysis.messages.coding import importer, metrics, sampler
from analysis.messages.extract import records
from analysis.messages.tests.test_extract import make_ep


def pools_of(n_weak, n_strong, m_weak, m_strong):
    g = lambda tier, n: [(f"dev|m|3p|S1|freeform|{tier}", i, "E", (1, 5)) for i in range(n)]
    m = lambda tier, n: [(f"dev|m|3p|S1|hybrid|{tier}", i, "F", (2, 2)) for i in range(n)]
    return {"weak": g("w", n_weak), "strong": g("s", n_strong)}, {"weak": m("w", m_weak), "strong": m("s", m_strong)}


def test_draw_sizes_and_determinism():
    gp, mp = pools_of(0, 5000, 0, 30)  # the final arm: strong tier only
    a, b = sampler.draw(gp, mp), sampler.draw(gp, mp)
    assert a == b and a != sampler.draw(gp, mp, seed=1)
    n = {}
    for _, ex, tier in a:
        n[(ex, tier)] = n.get((ex, tier), 0) + 1
    assert n == {("grammar", "strong"): 140, ("model", "strong"): 30}
    assert set(x for x, ex, t in a if ex == "model") == set(mp["strong"])
    # shuffled into one list, not grouped by part
    assert [ex for _, ex, _ in a[:20]] != ["grammar"] * 20


def test_pools_from_episodes():
    msg = "E is at (R2,C6). I think F sits on (R3,C3). D is at (R2,C7)."
    weak = make_ep([("Player A", msg), ("Player A", msg)], tier="weak")
    weak["turns"][1]["dup"] = True
    strong = make_ep([("Player B", "A is at (R1,C8).")], eid="dev|dsv41|3p|S1|freeform|01", tier="strong")
    eps = [weak, strong]
    A = grammar.analyse_all(eps)
    recs = {(weak["meta"]["id"], 0): [dict(letter="F", cell=(2, 2), span="F sits on (R3,C3)")]}
    gp, mp = sampler.pools(eps, A, recs)
    assert gp == {"weak": [(weak["meta"]["id"], 0, "E", (1, 5))], "strong": [(strong["meta"]["id"], 0, "A", (0, 7))]}
    assert mp == {"weak": [(weak["meta"]["id"], 0, "F", (2, 2))], "strong": []}


def test_items_are_blind_and_tool_escapes(tmp_path):
    msg = "E is at (R2,C6) and F sits on (R3,C4) </script><b>x"
    ep = make_ep([("Player A", msg)], tier="weak")
    key0 = (ep["meta"]["id"], 0)
    picks = [(key0 + ("E", (1, 5)), "grammar", "weak"), (key0 + ("F", (2, 3)), "model", "weak")]
    items, key = sampler.build(picks, {key0: (ep, ep["turns"][0])})
    assert all(tuple(it) == sampler.ITEM_FIELDS for it in items)
    blob = json.dumps(items)
    for leak in ("grammar", "model", "weak", "strong", "truth", "nemotron", ep["meta"]["id"]):
        assert leak not in blob
    assert key["001"]["extractor"] == "grammar" and key["001"]["truth"] is True and key["002"]["truth"] is False
    assert [m[2] for m in items[0]["marks"]] == ["letter", "cell"]
    assert msg[slice(*items[0]["marks"][1][:2])] == "(R2,C6)"
    p = tmp_path / "items.json"
    p.write_text(json.dumps(items))
    sampler.write_tool(str(p), str(tmp_path / "tool.html"))
    html = (tmp_path / "tool.html").read_text()
    assert "__DATA__" not in html and html.count("</script>") == 2 and "\\u003c/script>" in html
    data = json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))
    assert data["items"] == items and data["sha"] == hashlib.sha256(p.read_bytes()).hexdigest()
    for k in ("tier", "extractor", "truth", "episode"):
        assert k not in html
    assert "saved = false" in html and "export before closing" in html  # storage failure is shown, not swallowed


def test_importer_join():
    items = [dict(id="001", text="t1", letter="E", cell="(R2,C6)", marks=[]), dict(id="002", text="t2", letter="F", cell="(R3,C3)", marks=[])]
    key = {i: dict(tier="weak", extractor="grammar", truth=True, episode="e", ply=0, letter="E", cell="(R2,C6)", proto="hybrid")
           for i in ("001", "002")}
    rows = importer.join(items, key, {"001": "yes", "002": "no"})
    assert [r["label"] for r in rows] == ["yes", "no"] and rows[0]["text"] == "t1"
    with pytest.raises(ValueError, match="missing 1"):
        importer.join(items, key, {"001": "yes"})
    with pytest.raises(ValueError, match="bad values"):
        importer.join(items, key, {"001": "yes", "002": "maybe"})


def coded(part, k_strong, n_strong, k_weak, n_weak):
    rows = []
    for tier, k, n in (("strong", k_strong, n_strong), ("weak", k_weak, n_weak)):
        rows += [dict(extractor=part, tier=tier, label="yes" if i < k else "no", truth="True", text="t", letter="E",
                      cell="(R2,C6)") for i in range(n)]
    return rows


def test_rule_boundaries():
    exact = metrics.part_result(coded("grammar", 9, 10, 0, 0))         # precision exactly .90
    assert exact["passed"] and exact["tiers"]["strong"]["precision"] == 0.9
    lo, hi = exact["tiers"]["strong"]["ci_lo"], exact["tiers"]["strong"]["ci_hi"]
    assert lo < 0.9 < hi
    assert not metrics.part_result(coded("grammar", 89, 100, 0, 0))["passed"]    # .89
    assert not metrics.part_result(coded("model", 0, 0, 10, 10))["passed"]       # an empty strong tier fails
    assert set(metrics.part_result(coded("model", 7, 7, 1, 1))["tiers"]) == {"strong"}


def test_decision_and_misreads():
    assert metrics.decide(True, True).startswith("both parts pass: D2 from the union")
    assert metrics.decide(True, False).startswith("only the grammar part passes: D2 from the grammar alone")
    for m in (True, False):
        assert metrics.MANIFEST in metrics.decide(False, m)
    rows = coded("grammar", 9, 10, 9, 10) + coded("model", 1, 2, 0, 1)
    res = metrics.evaluate(rows)
    assert res["parts"]["grammar"]["passed"] and not res["parts"]["model"]["passed"]
    mis = metrics.misreads(rows)
    assert len(mis["negative"]) == 2 and len(mis["positive"]) == 1


def test_model_only_feeds_model_part():
    # the model part draws from exactly the model-only set of records.model_only_at
    msg = "E is at (R2,C6). F sits on (R3,C3)."
    ep = make_ep([("Player A", msg)])
    A = grammar.analyse_all([ep])
    recs = {(ep["meta"]["id"], 0): [dict(letter="E", cell=(1, 5), span="E is at (R2,C6)"),
                                    dict(letter="F", cell=(2, 2), span="F sits on (R3,C3)")]}
    assert set(sampler.pools([ep], A, recs)[1]["weak"]) == records.model_only_at([ep], A, recs)
