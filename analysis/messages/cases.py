"""Case selection (spec §10): typical, deviant and most-similar cases, each
"1 of N eligible", with replay-computed hoop tests and seeded quotes.

A feature is any per-episode flag {episode id: bool} a measure module
produces; `run` takes them by name. Selection never reads message text; the
quote is drawn by seed after the case is fixed.
"""
import collections
import random

from . import corpus

CELL = ("model", "proto", "suite", "level")
FACTORS = ("proto", "model", "suite")  # what a most-similar pair may vary


def quality(ep):
    """Episode Quality Score (thesis eq. benchscore), aborts scored 0 (spec §10)."""
    m = ep["meta"]
    if m["aborted"] or not m["success"]:
        return 0.0
    moves = sum(not t["done"] for t in ep["turns"])
    return min(100.0, 100.0 * ep["ins"]["optimal_moves"] / max(moves, 1))


def _cells(eps):
    by = collections.defaultdict(list)
    for ep in eps:
        by[tuple(ep["meta"][k] for k in CELL)].append(ep)
    return by


def select(eps, feature, seed=0):
    """Typical, deviant and most-similar cases for one feature.

    typical: feature present and success, drawn at random within the cell;
    deviant: feature present and failure where the cell succeeds >= 80 %, or
    feature present and success where it succeeds <= 20 %;
    most similar: the same board with exactly one factor varied and the
    outcome discordant, drawn at random among eligible pairs, with the
    probability of such a pair under no effect, p1(1-p2) + p2(1-p1)."""
    rng = random.Random(seed)
    cells, out = _cells(eps), []
    rate = {c: sum(e["meta"]["success"] for e in es) / len(es) for c, es in cells.items()}
    for c, es in sorted(cells.items()):
        has = [e for e in es if feature.get(e["meta"]["id"])]
        typ = [e for e in has if e["meta"]["success"]]
        dev = ([e for e in has if not e["meta"]["success"]] if rate[c] >= 0.8 else
               [e for e in has if e["meta"]["success"]] if rate[c] <= 0.2 else [])
        for kind, pool in (("typical", typ), ("deviant", dev)):
            if pool:
                e = rng.choice(sorted(pool, key=lambda x: x["meta"]["id"]))
                out.append(dict(kind=kind, episode=e["meta"]["id"], **dict(zip(CELL, c)), eligible=len(pool),
                                cell_size=len(es), feature_share=len(has) / len(es), cell_success=rate[c],
                                quality=quality(e)))
    boards = collections.defaultdict(list)  # a board is (level, game id)
    for e in eps:
        boards[(e["meta"]["level"], e["meta"]["board"])].append(e)
    cell = lambda e: tuple(e["meta"][k] for k in CELL)
    for f in FACTORS:
        pairs = []
        for es in boards.values():
            for e in es:
                if not feature.get(e["meta"]["id"]):
                    continue
                for e2 in es:
                    diff = [k for k, x, y in zip(CELL, cell(e), cell(e2)) if x != y]
                    dup = feature.get(e2["meta"]["id"]) and e2["meta"]["id"] < e["meta"]["id"]  # count a pair once
                    if diff == [f] and not dup and e["meta"]["success"] != e2["meta"]["success"]:
                        pairs.append((e, e2, rate[cell(e)], rate[cell(e2)]))
        if pairs:
            e, e2, p1, p2 = rng.choice(sorted(pairs, key=lambda x: (x[0]["meta"]["id"], x[1]["meta"]["id"])))
            out.append(dict(kind=f"most similar ({f})", episode=e["meta"]["id"], pair=e2["meta"]["id"],
                            eligible=len(pairs), p_discordant_null=p1 * (1 - p2) + p2 * (1 - p1),
                            note="one run per board and condition: a discordant pair is not evidence "
                                 "about the varied factor (Nielsen 2016, p. 579)"))
    return out


def hoop_tests(ep, info_visible):
    """Replay-computed tests for the episode's first fatal decision (spec §10):
    the information was visible (`info_visible(ep, hazard)` -> bool, from the
    failure module's features) and an alternative legal move existed. The
    rival explanations each test eliminates are named beside it."""
    fatal = [h for h in corpus.hazards(ep) if h["case"]]
    if not fatal:
        return None
    h = fatal[0]
    t = ep["turns"][h["ply"]]
    # a pinch has an alternative when it is free; a fork whenever any move is legal
    alt = h["free"] if h["kind"] == "pinch" else bool(corpus.legal(ep, t["board"], ep["objs"][t["seat"]]))
    vis = info_visible(ep, h)
    return dict(episode=ep["meta"]["id"], ply=h["ply"], kind=h["kind"], seat=h["seat"],
                hoop_visible=vis, hoop_alternative=alt,
                rival_no_alternative="eliminated" if alt else "stands",
                rival_not_in_view="eliminated" if vis else "stands")


def quote(ep, seed=0, before=None):
    """One message of the episode drawn by seed (before ply `before` if given)."""
    msgs = [t for t in ep["turns"] if t["is_msg"] and (before is None or t["ply"] < before)]
    if not msgs:
        return None
    t = random.Random(f"{seed}-{ep['meta']['id']}").choice(msgs)
    return dict(episode=ep["meta"]["id"], ply=t["ply"], seat=t["seat"], text=t["msg"])


def run(scope, eps, A, features=None, info_visible=None, seed=0):
    eps = [e for e in eps if e["meta"]["proto"] != "none"]
    rows = []
    for name, feat in sorted((features or {}).items()):
        for r in select(eps, feat, seed):
            r["feature"] = name
            r["quote"] = (quote(next(e for e in eps if e["meta"]["id"] == r["episode"]), seed) or {}).get("text")
            rows.append(r)
    out = dict(cases=rows)
    if info_visible is not None:
        out["case_hoop_tests"] = [h for e in eps if (h := hoop_tests(e, info_visible))]
    return out
