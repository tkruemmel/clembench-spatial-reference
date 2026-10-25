"""Spec §7 vocabulary ranking on a tiny synthetic corpus."""
import math
from collections import Counter

import numpy as np

from analysis.messages import grammar, vocab

DOCS = [Counter(x=2, y=2), Counter(x=1, y=3), Counter(y=2, z=2), Counter(x=4)]


def test_rescale_equal_weight_preserves_total():
    docs = [Counter(a=1, b=1), Counter(a=3, b=5, c=4), Counter(c=6)]
    X, voc, N = vocab.rescale(docs)
    assert voc == ["a", "b", "c"] and N == 20
    assert np.allclose(np.asarray(X.sum(1)).ravel(), N / 3) and np.isclose(X.sum(), N)
    assert np.allclose(X.toarray()[1], np.array([3, 5, 4]) * (20 / 3) / 12)


def test_log_odds_and_variance_by_hand():
    # a = pooled count 4, a0 = N = 20, n_i = n_j = 10
    d, v, z = vocab.monroe(3.0, 1.0, 4.0, 10.0, 10.0, 20.0)
    assert math.isclose(d, math.log(7 / 23) - math.log(5 / 25))
    assert math.isclose(v, 1 / 7 + 1 / 5) and math.isclose(z, d / math.sqrt(v))
    # through rank(): every episode has 4 terms, so rescaling is the identity; groups {0,1} vs {2,3}
    X, voc, N = vocab.rescale(DOCS)
    r = vocab.rank(X, [1, 1, 0, 0], ["s"] * 4, N, B=50, min_support=1)
    i = voc.index("x")  # y_xi = 3, y_xj = 4, a_x = 7, n_i = n_j = 8, a0 = 16
    dx = math.log(10 / 14) - math.log(11 / 13)
    assert math.isclose(r["delta"][i], dx) and math.isclose(r["z"][i], dx / math.sqrt(1 / 10 + 1 / 11))
    assert (r["sup_a"][i], r["sup_b"][i]) == (2, 1)


def test_permutation_stays_within_strata():
    g = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0], bool)
    st = ["a", "a", "a", "b", "b", "c", "c", "c", "c"]
    G = vocab.perm_labels(g, st, B=300, seed=1)
    assert G[:3].all() and not G[3:5].any()           # one-sided strata never change
    assert (G[5:].sum(0) == 2).all()                   # stratum c keeps its two labels
    assert len({tuple(c) for c in G[5:].T}) == 6       # and visits all C(4,2) arrangements


def _episode(k, proto, suite, success, texts):
    m = dict(id=f"dev|qwen38|{suite}|S1|{proto}|{k:02d}", proto=proto, model="qwen38", tier="strong",
             suite=suite, level="S1", success=success)
    turns = [dict(ply=i, seat=f"Player {'AB'[i % 2]}", is_msg=True, msg=s) for i, s in enumerate(texts)]
    return dict(meta=m, turns=turns, objs={"Player A": ["A"], "Player B": ["B"]}, _haz=[])


def _corpus():
    words = ["moving A up to (R2,C3)", "B please clear row 4", "I am waiting at (R1,C1)", "A blocked, B go left",
             "path is clear now", "thanks B, moving A down"]
    eps = [_episode(k, p, "3p", k % 2, [words[(k + j) % 6] for j in range(4)] + ["hold position"] * (p == "hybrid"))
           for k in range(8) for p in ("freeform", "hybrid")]
    A = {(e["meta"]["id"], t["ply"]): grammar.analyse(t["msg"], "dev", e["meta"]["proto"]) for e in eps for t in e["turns"]}
    return eps, A


def test_windows():
    ep = _episode(0, "freeform", "3p", 0, ["a", "b", "c", "d"])
    ep["_haz"] = [dict(ply=3, case=True), dict(ply=1, case=False)]
    assert [t["ply"] for t in vocab.window_turns(ep, "prefatal")] == [0, 1, 2]
    assert [t["ply"] for t in vocab.window_turns(ep, "first")] == [0, 1]
    assert len(vocab.window_turns(ep, "all")) == 4


def test_deterministic_under_seed():
    X, _, N = vocab.rescale(DOCS * 3)
    g, st = [1, 1, 0, 0] * 3, ["s", "s", "t", "t"] * 3
    r1, r2 = (vocab.rank(X, g, st, N, B=200, seed=0, min_support=1) for _ in range(2))
    assert r1["threshold"] == r2["threshold"] and np.array_equal(r1["z_stud"], r2["z_stud"])
    eps, A = _corpus()
    o1, o2 = vocab.run("dev", eps, A, B=99), vocab.run("dev", eps, A, B=99)
    assert o1 == o2
    meta = {r["contrast"]: r for r in o1["vocab_meta"]}
    assert meta["freeform_vs_hybrid_prose"]["alpha0"] == meta["freeform_vs_hybrid_prose"]["tokens_N"] > 0
    assert meta["structured_vs_hybrid_slots"]["note"].startswith("skipped")
    rows = o1["vocab_freeform_vs_hybrid_prose"]
    assert rows and all(r["support_a"] + r["support_b"] >= vocab.MIN_SUPPORT for r in rows)
    assert {"<own>", "<other>", "<cell>"} & {r["term"] for r in o1["vocab_success_vs_failure_3p_prefatal"]}
