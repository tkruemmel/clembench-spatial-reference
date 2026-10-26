"""E3 listener interpretation probe, its goal variant E3g, and E4 referent swap (extension §3).

    python3 -m analysis.messages.probe.run dev --model dsv41 --task e3 --dry-run       # pool, calls, cost; no call
    python3 -m analysis.messages.probe.run test --model dsv41 --task e3g --dry-run
    python3 -m analysis.messages.probe.run dev --model dsv41 --task e3 --limit 10 --max-usd 2   # pilot
    python3 -m analysis.messages.probe.run dev --model dsv41 --task e3 --max-usd 15
    python3 -m analysis.messages.probe.run dev --model nemotron --task e4 --limit 10 --max-usd 1
    python3 -m analysis.messages.probe.run test --model dsv41 --task e3 --max-usd 25
    python3 -m analysis.messages.probe.run test --model qwen38 --task e3 --allow-potsdam  # after Qwen's campaign

The sample is seeded and fixed per (scope, model, task); `--limit N` calls the
first N decisions of its stratum-interleaved order, so a pilot's replies are
reused by the full run. Every run writes the whole sample's manifest
(cache/items_<scope>_<model>_<task>.jsonl), and `run(scope, eps, A)` scores
what is cached against the replay without calling anything.

Sample (E3): accepted, non-retry move decisions under a messaging protocol
with at least one teammate message in view; at most 2 decisions per episode,
stratified protocol x suite. Per decision, Q-loc about every teammate object
(k = 1) on two prompts: the original, and the same prompt with every message
entry removed. The difference in accuracy is what the channel adds; objects
are split by whether a visible message names them.
Sample (E3g): the decisions of E3, asked which cell is each teammate object's
target cell (k = 1), on the same two prompts. Teammate targets are never on the
board, so the no-message arm is the floor. Objects are split by whether their
owner stated the target in a visible message (G-target strict), and the objects
not yet on their target are scored apart, where the current cell does not answer.
Sample (E4): dyad and 3p structured/hybrid decisions whose latest teammate
REQUEST names a cell c1 holding the listener's O1 while its O2 stands on a
cell c2 that message does not name. Arms: the original prompt (k = 8) and
every G-cell form of c1 rewritten to c2 (k = 4). tau and delta compare them;
the floor is the same two statistics between the 4 + 4 halves of the original.
"""
import argparse
import collections
import fcntl
import gc
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np

from analysis.messages import corpus, grammar, stats
from analysis.messages.probe import client, edit

SEED = 0
REPORTED = ("dsv41",)  # models whose cached probes enter the tables; the others' caches stay unused
N = {"e3": 400, "e3g": 400, "e4": 150}
K = dict(loc=1, goal=1, move=8, move_swap=4)


def _rcells(A, ep, q):
    return [c for r, _, c, _ in A[(ep["meta"]["id"], q)]["cells"] if r == "request"]


def _named(A, ep, q, region=None):
    return {v for r, _, v, _ in A[(ep["meta"]["id"], q)]["letters"] if v in ep["owner"] and region in (None, r)}


def _teammate_msgs(ep, t):
    return [q for q in t["vis"] if ep["turns"][q]["seat"] != t["seat"] and ep["turns"][q]["is_msg"]]


# ---------------------------------------------------------------- sampling

def e3_decision(ep, A, t, seed=SEED):
    """The teammate objects to ask about at one decision (dict), or None if no
    teammate message is in view."""
    if ep["meta"]["proto"] == "none" or t["done"] or t["retry"] or not _teammate_msgs(ep, t):
        return None
    mates = sorted(o for o in ep["owner"] if ep["owner"][o] != t["seat"])
    seen = set().union(*(_named(A, ep, p) for p in t["vis"] if ep["turns"][p]["is_msg"]))
    return dict(eid=ep["meta"]["id"], ply=t["ply"], q=None, letters=mates, named=sorted(seen & set(mates)))


def e3g_decision(ep, A, t, seed=SEED):
    """E3's decision, plus the teammate objects whose owner stated the target in
    a visible message, and those already on their target."""
    d = e3_decision(ep, A, t, seed)
    if d is None:
        return None
    T, tg, eid = ep["turns"], ep["targets"], ep["meta"]["id"]
    d["stated"] = sorted(L for L in d["letters"] if any(
        T[p]["seat"] == ep["owner"][L] and grammar.target_mentions(A[(eid, p)], {L: tg[L]})[L]["stated"]
        for p in t["vis"] if T[p]["is_msg"]))
    d["home"] = sorted(L for L in d["letters"] if t["board"][L] == tg[L])
    return d


def e4_decision(ep, A, t, seed=SEED):
    m = ep["meta"]
    if m["suite"] not in ("dyad", "3p") or m["proto"] not in ("structured", "hybrid") or t["done"] or t["retry"]:
        return None
    T, own, b = ep["turns"], ep["objs"][t["seat"]], t["board"]
    reqs = [q for q in _teammate_msgs(ep, t) if T[q]["slots"] and not T[q]["slots"]["request_none"]]
    if not reqs:
        return None
    q = max(reqs)  # the latest teammate REQUEST
    named = {c for _, _, c, _ in A[(m["id"], q)]["cells"]}
    opts = [(o1, o2) for c1 in dict.fromkeys(_rcells(A, ep, q)) for o1 in own if b[o1] == c1
            for o2 in own if o2 != o1 and b[o2] not in named]
    if not opts:
        return None
    o1, o2 = random.Random(f"{seed}|{m['id']}|{t['ply']}").choice(opts)
    # another visible message naming c1 still points the listener at O1 after the swap
    elsewhere = any(b[o1] == c for p in t["vis"] if p != q and T[p]["is_msg"] for _, _, c, _ in A[(m["id"], p)]["cells"])
    return dict(eid=m["id"], ply=t["ply"], q=q, O1=o1, c1=b[o1], O2=o2, c2=b[o2], c1_elsewhere=elsewhere)


def sample(task, eps, A, n=None, seed=SEED):
    """Stratified (protocol x suite) seeded sample, at most 2 decisions per
    episode, one per episode first; returned interleaved across strata so that
    every prefix spreads over them. Also returns the eligible pool per stratum."""
    fn = dict(e3=e3_decision, e3g=e3g_decision, e4=e4_decision)[task]  # e3g keeps E3's pool, so its sample too
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    for ep in eps:
        for t in ep["turns"]:
            d = fn(ep, A, t, seed)
            if d:
                pool[(ep["meta"]["proto"], ep["meta"]["suite"])][d["eid"]].append(d)
    if not pool:
        return [], {}
    quota = math.ceil((n or N[task]) / len(pool))
    picks = {}
    for s in sorted(pool):
        eids = sorted(pool[s])
        random.Random(f"{seed}|{s}").shuffle(eids)
        for e in eids:
            random.Random(f"{seed}|{e}").shuffle(pool[s][e])
        picks[s] = [pool[s][e][r] for r in range(2) for e in eids if r < len(pool[s][e])][:quota]
    order = [picks[s][i] for i in range(quota) for s in sorted(picks) if i < len(picks[s])]
    return order, {"|".join(s): (sum(map(len, pool[s].values())), len(picks[s])) for s in sorted(pool)}


# ---------------------------------------------------------------- prompts and items

def logged_prompt(ep, call, files):
    """The logged prompt of ep's call index `call` and its logged usage, from
    player_<N>.requests.json (read once per episode and player)."""
    player = ep["calls"][call]["player"]
    if player not in files:
        p = os.path.join(corpus.REPO, ep["meta"]["path"], f"player_{player.split()[1]}.requests.json")
        files[player] = json.load(open(p))["calls"]
    mine = [i for i, c in enumerate(ep["calls"]) if c["player"] == player]
    calls = files[player]
    if len(calls) != len(mine):
        raise ValueError("requests.json and interactions.json disagree on the call count")
    c = calls[mine.index(call)]["call"]
    u = c["raw_response_obj"].get("usage") or {}
    return c["manipulated_prompt_obj"], dict(i=u.get("prompt_tokens", u.get("inputTokens", 0)),
                                             o=u.get("completion_tokens", u.get("outputTokens", 0)))


MSG_LINE = re.compile(r'^Player \w+(?: \(you\))? (?:says: |said ")', re.M)


def _item(d, ep, prompt_ply, usage, kind, arm, msgs, k, **kw):
    return dict(eid=ep["meta"]["id"], ply=d["ply"], q=d["q"], prompt_ply=prompt_ply, kind=kind, arm=arm, k=k,
                sha=edit.sha(msgs), in_tok=usage["i"], out_tok=usage["o"], **kw), msgs


def e3_items(ep, d, files, goal=False):
    T, arm = ep["turns"], ep["meta"]["arm"]
    t = T[d["ply"]]
    msgs, use = logged_prompt(ep, t["call"], files)
    label = lambda p: T[p]["seat"] + (" (you)" if T[p]["seat"] == t["seat"] else "")
    bare = edit.strip_messages(msgs, arm, [(label(p), T[p]["msg"]) for p in t["vis"]])
    if MSG_LINE.search(bare[-1]["content"].split("CURRENT BOARD:", 1)[0]):
        raise ValueError("a message entry is left in the no-message prompt")
    if goal:
        kind, q, kw = "goal", edit.q_goal, lambda L: dict(stated=L in d["stated"], home=L in d["home"])
    else:
        kind, q, kw = "loc", edit.q_loc, lambda L: dict(named=L in d["named"])
    return [_item(d, ep, d["ply"], use, kind, a, edit.ask(v, q(L)), K[kind], letter=L, **kw(L))
            for L in d["letters"] for a, v in (("orig", msgs), ("nomsg", bare))]


def e4_items(ep, d, files):
    T, arm, proto = ep["turns"], ep["meta"]["arm"], ep["meta"]["proto"]
    msgs, use = logged_prompt(ep, T[d["ply"]]["call"], files)
    sp, m = T[d["q"]]["seat"], T[d["q"]]["msg"]
    new, n = edit.swap_cell(m, arm, d["c1"], d["c2"])
    if not n or (proto in corpus.SCHEMA and not corpus.slots(new, proto)):
        raise ValueError("swap leaves the grammar or the schema")
    sw = edit.replace(msgs, arm, sp, m, new)
    kw = dict(O1=d["O1"], O2=d["O2"], c1=list(d["c1"]), c2=list(d["c2"]), seat=sp, c1_elsewhere=d["c1_elsewhere"])
    edit.ask(msgs, "")  # the response block is there
    return [_item(d, ep, d["ply"], use, "move", "orig", msgs, K["move"], **kw),
            _item(d, ep, d["ply"], use, "move", "swap", sw, K["move_swap"], **kw)]


def build(task, eps, A, decisions):
    """[(item, msgs)] in decision order, and the count of decisions skipped
    because an edit could not be made exactly. Logs are read per episode."""
    idx = {ep["meta"]["id"]: ep for ep in eps}
    rank = {(d["eid"], d["ply"]): r for r, d in enumerate(decisions)}
    by_ep = collections.defaultdict(list)
    for d in decisions:
        by_ep[d["eid"]].append(d)
    out, skipped = [], collections.Counter()
    for eid, ds in by_ep.items():
        files = {}
        for d in ds:
            try:
                its = e4_items(idx[eid], d, files) if task == "e4" else e3_items(idx[eid], d, files, task == "e3g")
            except ValueError as e:
                skipped[str(e)[:60]] += 1
                continue
            for it, msgs in its:
                it["rank"] = rank[(eid, d["ply"])]
                out.append((it, msgs))
    out.sort(key=lambda x: x[0]["rank"])
    return out, skipped


def manifest_path(scope, model, task):
    return os.path.join(client.CACHE, f"items_{scope}_{model}_{task}.jsonl")


# ---------------------------------------------------------------- dry run

def estimate(model, items, limit):
    """Calls and cost for what is not cached: input from the logged usage of the
    same call, output from the cached replies of the same kind when at least 5
    exist (the pilot), else the logged decision's output."""
    cache = client.load_cache(model)
    seen_out = collections.defaultdict(list)
    for it, _ in items:
        for i in range(it["k"]):
            r = cache.get((it["sha"], i))
            if r:
                seen_out[it["kind"]].append(r["usage"]["out"])
    tot = collections.Counter()
    for scope_lim, sel in (("limit", [x for x in items if limit is None or x[0]["rank"] < limit]), ("full", items)):
        calls = collections.Counter()
        tin = tout = done = 0
        jobs = set()
        for it, _ in sel:
            for i in range(it["k"]):
                if (it["sha"], i) in jobs:
                    continue
                jobs.add((it["sha"], i))
                if (it["sha"], i) in cache:
                    done += 1
                    continue
                calls[it["kind"]] += 1
                o = seen_out[it["kind"]]
                tin += it["in_tok"]
                tout += np.mean(o) if len(o) >= 5 else it["out_tok"]
        cost = client.usd(model, {"in": tin, "out": tout})
        tot[scope_lim] = cost
        print(f"  {scope_lim}: {len(jobs)} calls ({done} cached), to call {dict(calls)}; "
              f"input {tin / 1e6:.2f}M, output {tout / 1e6:.2f}M tokens -> ${cost:.2f} at the current price")
    print("  output per call observed (kind: n, mean): " +
          ", ".join(f"{k}: {len(v)}, {np.mean(v):.0f}" for k, v in sorted(seen_out.items()) if v))
    return tot


# ---------------------------------------------------------------- scoring (cache only)

def _ci(vals):
    """Mean of decision values with a level-stratified episode (board-cluster)
    bootstrap, spec §8: vals [(episode, level, value)]."""
    vals = [v for v in vals if v[2] is not None and not (isinstance(v[2], float) and math.isnan(v[2]))]
    if not vals:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0, episodes=0)
    eps = sorted({e for e, _, _ in vals})
    ix = {e: i for i, e in enumerate(eps)}
    S, Nn, lv = np.zeros(len(eps)), np.zeros(len(eps)), [None] * len(eps)
    for e, l, v in vals:
        S[ix[e]] += float(v)
        Nn[ix[e]] += 1
        lv[ix[e]] = l
    est = float(S.sum() / Nn.sum())
    if len(eps) < 2:
        return dict(est=est, lo=np.nan, hi=np.nan, n=len(vals), episodes=len(eps))
    _, lo, hi = stats.strat_bootstrap(list(range(len(eps))), lv, lambda u: S[u].sum() / Nn[u].sum(), B=10000, seed=0)
    return dict(est=est, lo=lo, hi=hi, n=len(vals), episodes=len(eps))


def _summ(rows, est_keys, group=("model", "suite")):
    """One row per (group, estimand); `all` pools the suites."""
    out = []
    keys = sorted({tuple(r[g] for g in group) for r in rows})
    for g in keys + sorted({(k[0], "all") for k in keys}):
        sel = [r for r in rows if r["model"] == g[0] and (g[1] == "all" or r["suite"] == g[1])]
        for name in est_keys:
            v = [(r["episode"], r["level"], r[name]) for r in sel if name in r]
            if v:
                out.append(dict(model=g[0], suite=g[1], estimand=name, **_ci(v)))
    return out


def _meta(ep):
    m = ep["meta"]
    return dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], proto=m["proto"],
                board=m["board"], episode=m["id"])


def score_e3(items, cache, idx, kind="loc"):
    """Per answer and per decision: accuracy on the original prompt and with
    every message removed, over all teammate objects and split by whether a
    visible message names the object (loc, Q-loc) or its owner stated its target
    (goal, E3g, which also scores the objects not yet home apart)."""
    nm = "named" if kind == "loc" else "stated"
    ans, dec = [], collections.defaultdict(dict)
    for it in items:
        ep, r = idx.get(it["eid"]), cache.get((it["sha"], 0))
        if ep is None or r is None or it["kind"] != kind or it["arm"] not in ("orig", "nomsg"):
            continue
        a = edit.parse_cell(r["text"])
        truth = ep["turns"][it["ply"]]["board"][it["letter"]] if kind == "loc" else ep["targets"][it["letter"]]
        row = dict(_meta(ep), ply=it["ply"], letter=it["letter"], **{nm: it[nm]}, arm=it["arm"],
                   answer=corpus.cell(a) if isinstance(a, tuple) else a or "unparseable", finish=r["finish"],
                   correct=a == truth, unknown=a == "unknown", unparseable=a is None)
        if kind == "goal":
            row["home"] = it["home"]
        ans.append(row)
        dec[(it["eid"], it["ply"])][(it["letter"], it["arm"])] = row
    rows = []
    for (eid, ply), d in dec.items():
        pairs = [(d[(L, "orig")], d[(L, "nomsg")]) for L in {L for L, _ in d} if (L, "orig") in d and (L, "nomsg") in d]
        if not pairs:
            continue
        r = dict(_meta(idx[eid]), ply=ply, objects=len(pairs), **{f"{nm}_share": float(np.mean([o[nm] for o, _ in pairs]))})
        splits = [("", pairs), (f"{nm}_", [x for x in pairs if x[0][nm]]), (f"un{nm}_", [x for x in pairs if not x[0][nm]])]
        if kind == "goal":
            r["home_share"] = float(np.mean([o["home"] for o, _ in pairs]))
            splits.append(("away_", [x for x in pairs if not x[0]["home"]]))
        for g, sel in splits:
            if sel:
                r[f"{g}correct_orig"] = float(np.mean([o["correct"] for o, _ in sel]))
                r[f"{g}correct_nomsg"] = float(np.mean([n["correct"] for _, n in sel]))
                r[f"{g}contribution"] = r[f"{g}correct_orig"] - r[f"{g}correct_nomsg"]
        r["unknown_orig"] = float(np.mean([o["unknown"] for o, _ in pairs]))
        r["unknown_nomsg"] = float(np.mean([n["unknown"] for _, n in pairs]))
        r["unparseable"] = float(np.mean([x["unparseable"] for p in pairs for x in p]))
        rows.append(r)
    groups = ["", f"{nm}_", f"un{nm}_"] + (["away_"] if kind == "goal" else [])
    est = [g + s for g in groups for s in ("correct_orig", "correct_nomsg", "contribution")] + \
        ["unknown_orig", "unknown_nomsg", "unparseable", f"{nm}_share"] + (["home_share"] if kind == "goal" else [])
    return ans, rows, _summ(rows, est)


def e4_effects(orig, swap, o1, o2):
    """Per-decision E4 outcomes from parsed moves ((obj, dir), 'done' or None).
    The floor is tau and delta between the two halves of the original samples,
    what the same statistics read when nothing is edited."""
    p = lambda xs, o: float(np.mean([isinstance(x, tuple) and x[0] == o for x in xs]))
    a, b = orig[:len(orig) // 2], orig[len(orig) // 2:]
    return dict(p_o1_orig=p(orig, o1), p_o1_swap=p(swap, o1), p_o2_orig=p(orig, o2), p_o2_swap=p(swap, o2),
                tau=p(swap, o2) - p(orig, o2), delta=p(orig, o1) - p(swap, o1),
                floor_tau=p(b, o2) - p(a, o2), floor_delta=p(a, o1) - p(b, o1),
                unparseable_orig=float(np.mean([x is None for x in orig])),
                unparseable_swap=float(np.mean([x is None for x in swap])))


def score_e4(items, cache, idx):
    dec = collections.defaultdict(dict)
    for it in items:
        rs = [cache.get((it["sha"], i)) for i in range(it["k"])]
        if it["kind"] == "move" and it["eid"] in idx and all(rs):
            dec[(it["eid"], it["ply"])][it["arm"]] = (it, [r["text"] for r in rs])
    rows = []
    for (eid, ply), d in dec.items():
        if set(d) != {"orig", "swap"}:
            continue
        it = d["orig"][0]
        o1, o2 = it["O1"], it["O2"]
        orig, swap = ([edit.parse_move(x) for x in d[a][1]] for a in ("orig", "swap"))
        rows.append(dict(_meta(idx[eid]), ply=ply, O1=o1, O2=o2, c1=corpus.cell(it["c1"]), c2=corpus.cell(it["c2"]),
                         c1_elsewhere=it.get("c1_elsewhere"),
                         logged_move=f"{idx[eid]['turns'][ply]['obj']} {idx[eid]['turns'][ply]['dir']}",
                         **e4_effects(orig, swap, o1, o2)))
    return rows, _summ(rows, ["tau", "delta", "floor_tau", "floor_delta"])


def run(scope, eps, A):
    """Tables from the cached replies of every manifest of this scope; no API call."""
    idx = {ep["meta"]["id"]: ep for ep in eps}
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(manifest_path(scope, "*", "*"))):
        model, task = os.path.basename(p)[:-6].split("_")[2:4]
        if model not in REPORTED:
            continue
        items = [json.loads(x) for x in open(p)]
        for it in items:
            for k in ("c1", "c2"):
                if k in it:
                    it[k] = tuple(it[k])
        cache = client.load_cache(model)
        if task in ("e3", "e3g"):
            ans, rows, summ = score_e3(items, cache, idx, "loc" if task == "e3" else "goal")
            out[f"probe_{task}_answers"] += ans
            out[f"probe_{task}_decisions"] += rows
            out[f"probe_{task}_summary"] += summ
        else:
            rows, summ = score_e4(items, cache, idx)
            out["probe_e4_decisions"] += rows
            out["probe_e4_summary"] += summ
        jobs = {(it["sha"], i) for it in items for i in range(it["k"])}
        got = [cache[j] for j in jobs if j in cache]
        out["probe_spend"].append(dict(model=model, task=task, jobs=len(jobs), cached=len(got),
                                       input_tokens=sum(r["usage"]["in"] for r in got),
                                       output_tokens=sum(r["usage"]["out"] for r in got),
                                       usd=sum(r["usd"] for r in got)))
    return dict(out)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scope", choices=("dev", "calib", "test"))
    ap.add_argument("--model", required=True, choices=sorted(client.BACKENDS))
    ap.add_argument("--task", required=True, choices=("e3", "e3g", "e4"))
    ap.add_argument("--limit", type=int, help="call only the first N decisions of the sample")
    ap.add_argument("--max-usd", type=float, default=5.0, help="stop submitting once this run has spent this much")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="print the summary tables from the cache")
    ap.add_argument("--allow-potsdam", action="store_true", help="qwen38 only, never while its campaign runs")
    a = ap.parse_args(argv)
    if a.model == "qwen38" and not a.allow_potsdam:
        raise SystemExit("qwen38 refused: the Potsdam GPU runs its campaign; pass --allow-potsdam afterwards")
    eps = corpus.load(a.scope, models=(a.model,))
    A = grammar.analyse_all(eps)
    if a.report:
        import pandas as pd
        for k, rows in run(a.scope, eps, A).items():
            if k.endswith(("summary", "spend")):
                print(f"\n== {k}\n{pd.DataFrame(rows).to_string(index=False, float_format='%.3f')}")
        return
    decisions, pool = sample(a.task, eps, A)
    items, skipped = build(a.task, eps, A, decisions)
    print(f"{a.scope} {a.model} {a.task}: eligible / sampled per stratum {pool}; {len(decisions)} decisions, "
          f"{len(items)} items, skipped {dict(skipped)}", flush=True)
    del eps, A
    gc.collect()
    if corpus._LOCK is not None:  # the episodes are gone; let other analysis jobs load while the API runs
        fcntl.flock(corpus._LOCK, fcntl.LOCK_UN)
    est = estimate(a.model, items, a.limit)
    if a.dry_run:
        return est
    os.makedirs(client.CACHE, exist_ok=True)
    with open(manifest_path(a.scope, a.model, a.task), "w") as f:
        for it, _ in items:
            f.write(json.dumps(it) + "\n")
    jobs = [(it["sha"], i, msgs) for it, msgs in items if a.limit is None or it["rank"] < a.limit for i in range(it["k"])]
    st = client.execute(a.model, jobs, client.BACKENDS[a.model](), a.max_usd)
    estimate(a.model, items, a.limit)
    return st


if __name__ == "__main__":
    main(sys.argv[1:])
