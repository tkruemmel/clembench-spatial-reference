"""The weak-tier and tier-gap numbers no other f6 script prints.

    ARM=updated python3 analysis/qualitative_analysis/f6_tiers/tiers/weak.py

Exploratory throughout, apart from D1 (confirmatory.py is the producer of
record; the McNemar here is the same test on the same pairs).  ARM picks the
team suites; solo is always the original campaign.  Reads tables_comm's
episode loader and the shared caches episodes/aborts.{ARM}.json, plus
claims.{ARM}.json from claims/claims.py (run that first).
"""
import collections
import json
import math
import os
import statistics as st
import sys

from scipy.stats import binomtest

ROOT = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, os.path.join(ROOT, "analysis/thesis"))
import tables_comm as T                                          # noqa: E402
import confirmatory as C                                         # noqa: E402

ARM = T.ARM
QA = os.path.join(ROOT, "analysis/qualitative_analysis")
W, S = "Nemotron", T.STRONG
TAG = {"nemotron": W, "qwen38": S[0], "dsv41": S[1]}
GAMES = ("solo",) + T.SUITES

opt = T.load_optima()
eps = T.all_episodes(opt)
idx = {(e["model"], e["suite"], e["level"], e["cond"], e["game_id"]): e for e in eps}
E = json.load(open(os.path.join(QA, f"episodes.{ARM}.json")))
A = json.load(open(os.path.join(QA, f"aborts.{ARM}.json")))
print(f"ARM={ARM}")

print("\n== 1. success / abort / % Played / Quality (A_k, played episodes) by model x game")
for m in (W,) + S:
    for g in GAMES + (None,):
        es = [e for e in eps if e["model"] == m and g in (None, e["suite"])]
        pl = [e["q"] for e in es if not e["ab"]]
        print(f"  {m:20} {g or 'all':5} n={len(es):3} succ={sum(e['success'] for e in es):3} "
              f"ab={sum(e['ab'] for e in es):3} played={100 * len(pl) / len(es):5.1f}% "
              f"QS={st.mean(pl) if pl else float('nan'):6.2f}")

print("\n== 2. Nemotron successes by game x level; every success off S1")
for g in GAMES:
    print(f"  {g:5}", {l: sum(e["success"] for e in eps if e["model"] == W and e["suite"] == g
                             and e["level"] == l) for l in T.ALL_LEVELS})
for e in eps:
    if e["model"] == W and e["success"] and e["level"] != "S1":
        k = opt[(e["suite"], e["level"], e["game_id"])]
        print(f"  off S1: {e['suite']} {e['level']} {e['cond']} #{e['game_id']}: "
              f"{round(k / e['eff'])} moves against A_k {k}, QS {e['q']:.1f}")

print("\n== 3. Nemotron team-suite success by protocol (of 180); exact McNemar over (suite, level, board)")
print("  ", {p: sum(e["success"] for e in eps if e["model"] == W and e["suite"] in T.SUITES
                and e["cond"] == p) for p in T.PROTOCOLS})
for a, b in (("freeform", "none"), ("hybrid", "none"), ("structured", "none"),
             ("freeform", "hybrid"), ("hybrid", "structured"), ("freeform", "structured")):
    for lv in (T.ALL_LEVELS, ("S1",)):
        prs = [(idx[(W, s, l, a, g)], idx[(W, s, l, b, g)]) for s in T.SUITES for l in lv
               for g in C.gids(eps, l)]
        r = C.mcnemar(prs)
        print(f"  {a} vs {b} ({'all levels' if len(lv) == 4 else 'S1 only'}): {r['note']} p={r['p']:.2g}")

print("\n== 4. Strict subset: Nemotron against each strong model, all 840 cells")
for m in S:
    prs = C.whole_set_pairs(eps, idx, m, W)
    so = sum(1 for x, y in prs if x["success"] and not y["success"])
    wo = sum(1 for x, y in prs if y["success"] and not x["success"])
    n = len(prs)
    po = sum(x["success"] == y["success"] for x, y in prs) / n
    px, py = sum(x["success"] for x, _ in prs) / n, sum(y["success"] for _, y in prs) / n
    pe = px * py + (1 - px) * (1 - py)
    # chance of a Nemotron-only win: the strong model's failure rate in the same (game, level)
    fr = {}
    for g in GAMES:
        for l in T.ALL_LEVELS:
            c = [e for e in eps if e["model"] == m and e["suite"] == g and e["level"] == l]
            fr[(g, l)] = 1 - sum(e["success"] for e in c) / len(c)
    ws = [y for _, y in prs if y["success"]]
    exp = sum(fr[(y["suite"], y["level"])] for y in ws)
    p0 = math.prod(1 - fr[(y["suite"], y["level"])] for y in ws)
    s1 = [e for e in eps if e["model"] == m and e["level"] == "S1"]
    s1f = sum(1 - e["success"] for e in s1)
    print(f"  {m}: n={n} strong-only {so} weak-only {wo} both {n - so - wo - sum(not x['success'] and not y['success'] for x, y in prs)} "
          f"kappa={(po - pe) / (1 - pe):.3f} McNemar p={binomtest(so, so + wo, .5).pvalue:.3g}; "
          f"expected weak-only {exp:.2f}, P(0)={p0:.2f} (per game x level); "
          f"S1-pooled: {len(ws)} x {s1f}/{len(s1)} = {len(ws) * s1f / len(s1):.2f}")
s1 = [e for e in eps if e["model"] in S and e["level"] == "S1"]
print(f"  strong S1 success pooled {sum(e['success'] for e in s1)}/{len(s1)}; per model",
      {m: f"{sum(e['success'] for e in s1 if e['model'] == m)}/{sum(e['model'] == m for e in s1)}" for m in S})

print("\n== 5. Team non-successes on boards the model solved alone (any solo condition)")
for m in (W,) + S:
    solved = {(e["level"], e["game_id"]) for e in eps if e["model"] == m and e["suite"] == "solo" and e["success"]}
    both = {k for k in solved if all(idx[(m, "solo", k[0], c, k[1])]["success"]
                                     for c in ("solo_silent", "solo_thinking"))}
    fails = [e for e in eps if e["model"] == m and e["suite"] in T.SUITES and not e["success"]]
    on = sum((e["level"], e["game_id"]) in solved for e in fails)
    print(f"  {m}: {on}/{len(fails)} = {100 * on / len(fails):.1f}%; boards solved alone {len(solved)} of 60 "
          f"{sorted(solved) if len(solved) < 10 else ''}; in both solo conditions {len(both)}")

print("\n== 6. Silent baseline: team-suite success under none, by level (of 45)")
for m in (W,) + S:
    c = [sum(e["success"] for e in eps if e["model"] == m and e["suite"] in T.SUITES
             and e["cond"] == "none" and e["level"] == l) for l in T.ALL_LEVELS]
    print(f"  {m:20}", " / ".join(f"{100 * k / 45:.1f}" for k in c), f"  ({sum(c)}/180)")

print("\n== 7. How episodes end (episodes cache), per model, all 840")
for mod in ("nemotron", "qwen38", "dsv41"):
    rs = [r for r in E if r["model"] == mod and r["outcome"] != "success"]
    kinds = collections.Counter()
    for r in rs:
        msg = (r["end_msg"] or "").lower()
        kinds[("abort: " if r["outcome"] == "abort" else "loss: ")
              + ("invalid moves" if "invalid move" in msg else "malformed hybrid" if "malformed" in msg
                 else "invalid responses" if "invalid response" in msg else "cycle" if "cycle" in msg
                 else "false done" if "declared done" in msg else "turn limit" if "turn" in msg or "budget" in msg
                 else msg[:40])] += 1
    print(f"  {TAG[mod]}: non-successes {len(rs)}", dict(kinds.most_common()))

print("\n== 8. Illegal and blocked moves by model x game")
print("     per-episode mean of scores.json Illegal / Blocked Move Rate | pooled per attempt (episodes cache)")
for mod in ("nemotron", "qwen38", "dsv41"):
    for g in GAMES:
        rs = [r for r in E if r["model"] == mod and r["suite"] == g]
        sc = [json.load(open(os.path.join(ROOT, r["path"], "scores.json")))["episode scores"] for r in rs]
        il = [s["Illegal Move Rate"] for s in sc if s["Illegal Move Rate"] == s["Illegal Move Rate"]]
        bl = [s["Blocked Move Rate"] for s in sc if s["Blocked Move Rate"] == s["Blocked Move Rate"]]
        mv, i, b = (sum(r[k] for r in rs) for k in ("moves", "illegal", "blocked"))
        print(f"  {TAG[mod]:20} {g:5} ep-mean illegal {100 * st.mean(il):5.1f}% blocked {100 * st.mean(bl):5.1f}% "
              f"(n={len(il)}) | per attempt illegal {100 * i / (mv + i + b):5.1f}% blocked {100 * b / (mv + i + b):5.1f}%")

print("\n== 9. Positional claims (claims.json, prose protocols)")
CL = json.load(open(os.path.join(QA, f"f6_tiers/claims/claims.{ARM}.json")))
CL = [c for c in CL if c["proto"] in ("freeform", "hybrid") and c["lt_raw"].isupper()]
for mod in ("nemotron", "qwen38", "dsv41"):
    tm = [c for c in CL if c["model"] == mod and not c["own"]]
    bad = [c for c in tm if not c["ok"]]
    print(f"  {TAG[mod]}: teammate claims {len(tm)}, right {sum(c['ok'] for c in tm)} "
          f"({100 * sum(c['ok'] for c in tm) / len(tm):.1f}%); wrong {len(bad)}, of which name a cell the object "
          f"used to occupy {sum(c['stale'] for c in bad)} ({100 * sum(c['stale'] for c in bad) / max(1, len(bad)):.0f}%)")
    for g in T.SUITES:
        for p in ("freeform", "hybrid"):
            o = [c for c in CL if c["model"] == mod and c["suite"] == g and c["proto"] == p and c["own"]]
            t = [c for c in CL if c["model"] == mod and c["suite"] == g and c["proto"] == p and not c["own"]]
            f = lambda x: f"{100 * sum(c['ok'] for c in x) / len(x):.1f}% of {len(x)}" if x else "--"
            print(f"     {g:4} {p:8} own {f(o):16} teammate {f(t)}")

print("\n== 10. Format aborts (malformed hybrid messages) by level, Nemotron")
fa = [a for a in A if a["model"] == "nemotron" and "malformed" in a["reason"].lower()]
print("  ", len(fa), dict(sorted(collections.Counter(a["level"] for a in fa).items())),
      " S2-S4:", sum(a["level"] != "S1" for a in fa))

print("\n== 11. Same-move traps: team-suite enclosure aborts on S4 #8 and S4 #13, by accepted moves")
for g, at in ((8, 23), (13, 12)):
    rs = [a for a in A if a["level"] == "S4" and a["suite"] != "solo" and f"instance_{g:05d}" in a["path"]]
    hit = [a for a in rs if a["moves"] == at and a["n_legal_own"] == 0]
    print(f"  S4 #{g}: {len(rs)} team-suite aborts; at exactly move {at} with no legal move: {len(hit)} runs, "
          f"models {sorted({a['model'] for a in hit})}, protocols {sorted({a['proto'] for a in hit})}; "
          f"all: {sorted((a['model'], a['proto'], a['moves']) for a in rs)}")
enc = [a for a in A if a["n_legal_own"] == 0]
print(f"  enclosure aborts (aborting player has no legal move): {len(enc)}",
      dict(collections.Counter(TAG[a["model"]] for a in enc)), dict(collections.Counter(a["suite"] for a in enc)))
