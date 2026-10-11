"""Strand A (Phase 2) helper: numbers the ledger thesis/numbers/A_core.md cites
that tables_comm.py / confirmatory.py do not print.  Read-only: it imports their
functions and reads analysis/qualitative_analysis/corpus.{ARM}.json; it writes nothing.

    ARM=updated  python3 analysis/thesis/ledger_extra.py > thesis/numbers/A_extra.updated.txt
    ARM=original python3 analysis/thesis/ledger_extra.py > thesis/numbers/A_extra.original.txt

Run from the repo root.  Every line is `key: value`; the ledger cites the key.
Permutation p-values are tables_comm.perm_p (sign-flip, 1e5 resamples, seed 0).
"""

import collections, glob, json, math, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as T
import confirmatory as C
from scipy.stats import binomtest, fisher_exact

ARM = os.environ.get("ARM", "updated")
opt = T.load_optima()
E = T.all_episodes(opt)
S = [e for e in E if e["model"] in T.STRONG]
NEM = [e for e in E if e["model"] == "Nemotron"]
P = lambda k, v: print(f"{k}: {v}")
nick = {"Qwen3.8": "qwen", "DeepSeek-V4.1-Flash": "ds", "Nemotron": "nem"}


def mcn(pairs):
    b = sum(1 for x, y in pairs if x["success"] and not y["success"])
    c = sum(1 for x, y in pairs if y["success"] and not x["success"])
    return b, c, (binomtest(b, b + c, .5).pvalue if b + c else float("nan"))


idx = {(e["model"], e["suite"], e["level"], e["cond"], e["game_id"]): e for e in E}

# 1. premise: team non-successes on solo-solved boards; base rate
for tier, models in (("strong", T.STRONG), ("nem", ("Nemotron",))):
    solved = {(e["model"], e["level"], e["game_id"]) for e in E
              if e["suite"] == "solo" and e["success"] and e["model"] in models}
    both = collections.Counter((e["model"], e["level"], e["game_id"]) for e in E
                               if e["suite"] == "solo" and e["success"] and e["model"] in models)
    P(f"{tier}.solo_boards_solved_once", f"{len(solved)}/{60*len(models)}")
    P(f"{tier}.solo_boards_solved_both", f"{sum(1 for v in both.values() if v == 2)}/{60*len(models)}")
    tot = on = 0
    for m in models:
        parts = []
        for s in T.SUITES:
            f = [e for e in E if e["model"] == m and e["suite"] == s and not e["success"]]
            o = sum(1 for e in f if (m, e["level"], e["game_id"]) in solved)
            parts.append(f"{s} {o}/{len(f)}")
            tot += len(f); on += o
        P(f"{tier}.{nick[m]}.team_fail_on_solo_solved", "; ".join(parts))
    P(f"{tier}.team_fail_on_solo_solved_total", f"{on}/{tot} ({100*on/tot:.1f}%)")
    n_team = sum(1 for e in E if e["model"] in models and e["suite"] != "solo")
    n_solo = sum(1 for e in E if e["model"] in models and e["suite"] == "solo")
    f_solo = sum(1 for e in E if e["model"] in models and e["suite"] == "solo" and not e["success"])
    P(f"{tier}.team_fail_rate", f"{tot}/{n_team} ({100*tot/n_team:.2f}%)")
    P(f"{tier}.solo_fail_rate", f"{f_solo}/{n_solo} ({100*f_solo/n_solo:.2f}%)")

# 2. S3 solo vs 6p (strong pooled)
for s in ("solo", "dyad", "3p", "6p"):
    c = T.cell([e for e in S if e["suite"] == s and e["level"] == "S3"])
    P(f"S3.{s}", f"{c['k']}/{c['n']} {c['rate']:.1f}% Q {c['q']:.2f} P {c['played']:.1f}")

# 3. dyad S2-S4 per-model cells and McNemar vs none (pooled)
for m in T.STRONG:
    for p in T.PROTOCOLS + ("solo_silent", "solo_thinking"):
        rows = [e for e in S if e["model"] == m and e["level"] != "S1" and e["cond"] == p
                and e["suite"] in ("dyad", "solo")]
        c = T.cell(rows)
        P(f"dyadS24.{nick[m]}.{p}", f"{c['k']}/{c['n']} {c['rate']:.1f}% eff {c['eff']:.3f} Q {c['q']:.2f}")
    solo = T.cell([e for e in S if e["model"] == m and e["level"] != "S1" and e["suite"] == "solo"])
    none = T.cell([e for e in S if e["model"] == m and e["level"] != "S1" and e["cond"] == "none" and e["suite"] == "dyad"])
    st_ = T.cell([e for e in S if e["model"] == m and e["level"] != "S1" and e["cond"] == "structured" and e["suite"] == "dyad"])
    g = solo["rate"] - none["rate"]
    P(f"dyadS24.{nick[m]}.structured_share_of_success_gap",
      f"gap {g:.1f}pp, structured recovers {st_['rate']-none['rate']:.1f}pp = {100*(st_['rate']-none['rate'])/g:.0f}%" if g else "gap 0")
    gq = solo["q"] - none["q"]
    P(f"dyadS24.{nick[m]}.structured_share_of_Q_gap", f"gap {gq:.2f}, structured {st_['q']-none['q']:.2f} = {100*(st_['q']-none['q'])/gq:.0f}%")
for a in ("freeform", "hybrid", "structured"):
    pr = C.dyad_pairs(E, idx, a, "none")
    b, c, p = mcn(pr)
    dq = [x["q"] - y["q"] for x, y in pr]
    P(f"dyadS24.mcnemar.{a}_vs_none", f"{b}:{c} p={p:.4g} n={len(pr)}; dQ {st.mean(dq):+.2f} perm p={T.perm_p(dq):.3g}")
pool = {k: T.cell([e for e in S if e["suite"] in ("dyad", "solo") and e["level"] != "S1" and (e["cond"] in k)])
        for k in (("solo_silent", "solo_thinking"), ("none",), ("structured",), ("freeform",), ("hybrid",))}
gq = pool[("solo_silent", "solo_thinking")]["q"] - pool[("none",)]["q"]
ge = pool[("solo_silent", "solo_thinking")]["eff"] - pool[("none",)]["eff"]
gs = pool[("solo_silent", "solo_thinking")]["rate"] - pool[("none",)]["rate"]
for p in ("structured", "freeform", "hybrid"):
    c = pool[(p,)]
    P(f"dyadS24.recovered.{p}", f"Q {100*(c['q']-pool[('none',)]['q'])/gq:.0f}% succ {100*(c['rate']-pool[('none',)]['rate'])/gs:.0f}% eff {100*(c['eff']-pool[('none',)]['eff'])/ge:.0f}%")

# dyad freeform vs hybrid QS paired (S1-S4 and S2-S4)
for lv, name in ((T.ALL_LEVELS, "S1-S4"), (T.LEVELS, "S2-S4")):
    pr = C.dyad_pairs(E, idx, "freeform", "hybrid", lv)
    d = [x["q"] - y["q"] for x, y in pr]
    fq = st.mean(x["q"] for x, _ in pr); hq = st.mean(y["q"] for _, y in pr)
    P(f"dyad.ff_vs_hy.Q.{name}", f"{fq:.2f} vs {hq:.2f} d {st.mean(d):+.2f} n={len(d)} perm p={T.perm_p(d):.3g}")

# 4. dyad per-level, pooled: solo (pooled), silent, none
for lv in T.ALL_LEVELS:
    sp = T.cell([e for e in S if e["suite"] == "solo" and e["level"] == lv])
    nn = T.cell([e for e in S if e["suite"] == "dyad" and e["cond"] == "none" and e["level"] == lv])
    pr = T.cell([e for e in S if e["suite"] == "dyad" and e["cond"] in ("freeform", "hybrid") and e["level"] == lv])
    P(f"dyad.{lv}.solo_pooled_vs_none", f"Q {sp['q']:.2f} vs {nn['q']:.2f} (gap {sp['q']-nn['q']:.2f}); succ {sp['k']}/{sp['n']} vs {nn['k']}/{nn['n']}")
    P(f"dyad.{lv}.prose", f"{pr['k']}/{pr['n']} Q {pr['q']:.2f}")

# 5. ladder by suite: success, per level, prose vs none, per model
L = T.ladder()
def succ(models, suites, protos, levels):
    v = [x for m in models for s in suites for p in protos for l in levels for x in L.get((m, s, l, p), [])]
    return sum(v), len(v)
for s in T.SUITES + ("all",):
    ss = T.SUITES if s == "all" else (s,)
    gaps = []
    for lv in T.ALL_LEVELS:
        kn, nn = succ(T.STRONG, ss, ("none",), (lv,))
        kp, np_ = succ(T.STRONG, ss, ("freeform", "hybrid"), (lv,))
        gaps.append(f"{lv} none {kn}/{nn}={100*kn/nn:.1f} prose {kp}/{np_}={100*kp/np_:.1f} gap {100*kp/np_-100*kn/nn:+.1f}")
    P(f"ladder.prose_vs_none.{s}", " | ".join(gaps))
for m in T.STRONG:
    P(f"ladder.S3none.{nick[m]}", ", ".join(f"{s} {succ((m,), (s,), ('none',), ('S3',))[0]}/15" for s in T.SUITES))
    for p in T.PROTOCOLS:
        P(f"ladder.succ.{nick[m]}.{p}", ", ".join(f"{s} {succ((m,), (s,), (p,), T.ALL_LEVELS)[0]}/60" for s in T.SUITES)
          + f", all {succ((m,), T.SUITES, (p,), T.ALL_LEVELS)[0]}/180")
for p in T.PROTOCOLS:
    P(f"ladder.succ.nem.{p}", ", ".join(f"{s} {succ(('Nemotron',), (s,), (p,), T.ALL_LEVELS)[0]}/60" for s in T.SUITES)
      + f", all {succ(('Nemotron',), T.SUITES, (p,), T.ALL_LEVELS)[0]}/180")
# success inversions: structured above a prose protocol, per (model, suite, level)
inv = []
for m in T.STRONG:
    for s in T.SUITES:
        for lv in T.ALL_LEVELS:
            k = {p: succ((m,), (s,), (p,), (lv,))[0] for p in T.PROTOCOLS}
            for q in ("freeform", "hybrid"):
                if k["structured"] > k[q]:
                    inv.append(f"{nick[m]} {s} {lv} str {k['structured']} > {q} {k[q]}")
P("ladder.success_inversions", f"{len(inv)}: " + "; ".join(inv))

# 6. multi-suite per model: prose/structured/none success paired McNemar, matched efficiency
for m in T.STRONG:
    def pairs_(a, b):
        return [(idx[(m, s, l, a, g)], idx[(m, s, l, b, g)]) for s in T.SUITES for l in T.ALL_LEVELS
                for g in C.gids(E, l) if (m, s, l, a, g) in idx and (m, s, l, b, g) in idx]
    for a, b in (("freeform", "structured"), ("hybrid", "structured"), ("structured", "none"), ("freeform", "hybrid")):
        pr = pairs_(a, b)
        bb, cc, p = mcn(pr)
        ka = sum(x["success"] for x, _ in pr); kb = sum(y["success"] for _, y in pr)
        d = [x["eff"] - y["eff"] for x, y in pr if x["success"] and y["success"]]
        P(f"multi.{nick[m]}.{a}_vs_{b}", f"succ {ka}/{len(pr)} vs {kb}/{len(pr)} McNemar {bb}:{cc} p={p:.3g}; eff d {st.mean(d):+.4f} n={len(d)} perm p={T.perm_p(d):.3g}")
    kp = sum(e["success"] for e in S if e["model"] == m and e["suite"] != "solo" and e["cond"] in ("freeform", "hybrid"))
    ks = sum(e["success"] for e in S if e["model"] == m and e["suite"] != "solo" and e["cond"] == "structured")
    P(f"multi.{nick[m]}.prose_vs_structured_fisher", f"{kp}/360 vs {ks}/180 Fisher p={fisher_exact([[kp, 360-kp], [ks, 180-ks]])[1]:.3g}")

# 7. prose losses (strong, freeform+hybrid, all three suites)
pl = [e for e in S if e["suite"] != "solo" and e["cond"] in ("freeform", "hybrid") and not e["success"]]
P("prose.losses", f"{len(pl)}/720; aborts {sum(e['ab'] for e in pl)}; by cell " +
  ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(f"{e['suite']} {e['level']}" for e in pl).items())))
P("prose.losses.by_model", ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(nick[e['model']] for e in pl).items())))
pa = [e for e in S if e["suite"] != "solo" and e["ab"]]
P("strong.aborts", f"{len(pa)}; by suite " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(e['suite'] for e in pa).items()))
  + "; by model " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(nick[e['model']] for e in pa).items()))
  + "; by protocol " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(e['cond'] for e in pa).items())))

# 8. team size per model per level paired Quality (A_k), and success pp per level pooled
for m in T.STRONG:
    for a, b in T.TEAMSIZE_STEPS:
        parts = []
        for lv in T.ALL_LEVELS:
            d = T.qpairs([e for e in S if e["model"] == m and e["suite"] == a and e["level"] == lv],
                         [e for e in S if e["model"] == m and e["suite"] == b and e["level"] == lv],
                         lambda r: (r["cond"], r["game_id"]))
            parts.append(f"{lv} {st.mean(d):+.2f} (n={len(d)}, p={T.perm_p(d):.2g})")
        P(f"teamsize.perlevel.{nick[m]}.{a}->{b}", "; ".join(parts))
for lv in T.ALL_LEVELS:
    cs = {s: T.cell([e for e in S if e["suite"] == s and e["level"] == lv]) for s in T.SUITES}
    P(f"teamsize.{lv}.dyad_to_6p", f"Q {cs['dyad']['q']-cs['6p']['q']:.2f}; succ {cs['dyad']['rate']-cs['6p']['rate']:.1f}pp; QJ {cs['dyad']['q_j']-cs['6p']['q_j']:.2f}")
for m in T.STRONG + (None,):
    ms = T.STRONG if m is None else (m,)
    out = []
    for s in T.SUITES:
        k = sum(e["success"] for e in S if e["model"] in ms and e["suite"] == s)
        n = sum(1 for e in S if e["model"] in ms and e["suite"] == s)
        out.append(f"{s} {k}/{n}={100*k/n:.2f}%")
    P(f"teamsize.success.{nick.get(m, 'pooled')}", ", ".join(out))
    for s in ("dyad", "6p"):
        f12 = sum(1 for e in S if e["model"] in ms and e["suite"] == s and e["level"] in ("S1", "S2") and not e["success"])
        f34 = sum(1 for e in S if e["model"] in ms and e["suite"] == s and e["level"] in ("S3", "S4") and not e["success"])
        P(f"teamsize.failures.{nick.get(m, 'pooled')}.{s}", f"S1+S2 {f12}, S3+S4 {f34}")
# two-stratum Haldane-corrected ORs (pooled), dyad vs 6p failure odds
for name, lvs in (("S1+S2", ("S1", "S2")), ("S3+S4", ("S3", "S4"))):
    a = sum(e["success"] for e in S if e["suite"] == "dyad" and e["level"] in lvs)
    na = sum(1 for e in S if e["suite"] == "dyad" and e["level"] in lvs)
    c = sum(e["success"] for e in S if e["suite"] == "6p" and e["level"] in lvs)
    nc = sum(1 for e in S if e["suite"] == "6p" and e["level"] in lvs)
    b, d = na - a, nc - c
    cells = (a, b, c, d) if 0 not in (a, b, c, d) else tuple(x + .5 for x in (a, b, c, d))
    orr = cells[0] * cells[3] / (cells[1] * cells[2])
    se = math.sqrt(sum(1 / x for x in cells))
    P(f"teamsize.OR.{name}", f"{orr:.2f} [{math.exp(math.log(orr)-1.96*se):.2f}, {math.exp(math.log(orr)+1.96*se):.2f}] dyad {a}/{na} 6p {c}/{nc}")
print("woolf two-strata pooled:", C.woolf([(n, *[sum(e['success'] for e in S if e['suite']==s and e['level'] in lv) if i==0 else sum(1 for e in S if e['suite']==s and e['level'] in lv and not e['success']) for s in ('dyad','6p') for i in (0,1)]) for n, lv in (("S1+S2", ("S1","S2")), ("S3+S4", ("S3","S4")))]))

# 9. rules share of the 6p inefficiency (successes; pooled S1-S4)
for s in T.SUITES:
    ok = [e for e in S if e["suite"] == s and e["success"]]
    ej = st.mean(e["eff_j"] for e in ok); ea = st.mean(e["eff"] for e in ok)
    P(f"rules_share.{s}", f"effJ {ej:.3f} effA {ea:.3f}; inefficiency on J {1-ej:.3f}, agents {1-ea:.3f}, rules {ea-ej:.3f} = {100*(ea-ej)/(1-ej):.0f}%")
for s in T.SUITES:
    r = [opt[(lv, g)] / opt[(s, lv, g)] for (lv, g) in {(e["level"], e["game_id"]) for e in S}]
    for lv in T.ALL_LEVELS:
        v = [opt[(lv, g)] / opt[(s, lv, g)] for g in C.gids(E, lv)]
        P(f"flawless.{s}.{lv}", f"{st.mean(v):.3f}")

# 10. efficiency ordering per model per suite (A_k, successes)
for m in T.STRONG:
    for s in ("solo",) + T.SUITES:
        effs = {lv: st.mean(e["eff"] for e in S if e["model"] == m and e["suite"] == s and e["level"] == lv and e["success"]) for lv in T.ALL_LEVELS}
        order = " > ".join(sorted(effs, key=lambda k: -effs[k]))
        qs = {lv: T.cell([e for e in S if e["model"] == m and e["suite"] == s and e["level"] == lv])["q"] for lv in T.ALL_LEVELS}
        P(f"order.{nick[m]}.{s}", f"eff {order} ({', '.join(f'{effs[l]:.3f}' for l in T.ALL_LEVELS)}); Q {' > '.join(sorted(qs, key=lambda k: -qs[k]))}")
for s in T.SUITES:
    P(f"S3vsS4.success.{s}", ", ".join(f"{lv} {T.cell([e for e in S if e['suite']==s and e['level']==lv])['rate']:.1f}" for lv in T.ALL_LEVELS))
allm = {lv: T.cell([e for e in S if e['suite'] != 'solo' and e['level'] == lv]) for lv in T.ALL_LEVELS}
P("S3vsS4.success.multi_pooled", ", ".join(f"{lv} {c['k']}/{c['n']}" for lv, c in allm.items()))

# 11. family C matched-board solo efficiency per level per model
bs = C.solo_boards(E)
for m in T.STRONG:
    P(f"C.matched.{nick[m]}", ", ".join(f"{lv} {st.mean(C.solo_eff(idx, m, l, g) for (l, g) in bs if l == lv):.3f} (n={sum(1 for (l, g) in bs if l == lv)})" for lv in T.ALL_LEVELS))

# 12. corpus: claim accuracies, seq/ack, none/think
rows = json.load(open(f"analysis/qualitative_analysis/corpus.{ARM}.json"))["rows"]
for scope in ("dyad", "all"):
    for m in ("qwen38", "dsv41", "nemotron"):
        for p in ("structured", "freeform", "hybrid", "prose", "any"):
            sel = [r for r in rows if r["model"] == m and (scope == "all" or r["suite"] == scope)
                   and (p == "any" or r["proto"] == p or (p == "prose" and r["proto"] in ("freeform", "hybrid")))]
            oo = sum(r["own_ok"] for r in sel); on = oo + sum(r["own_bad"] for r in sel)
            to = sum(r["oth_ok"] for r in sel); tn = to + sum(r["oth_bad"] for r in sel)
            P(f"claims.{scope}.{m}.{p}", f"own {oo}/{on}={100*oo/on if on else float('nan'):.1f}% other {to}/{tn}={100*to/tn if tn else float('nan'):.1f}%")
dy = [r for r in rows if r["suite"] == "dyad"]
for m in ("qwen38", "dsv41", "nemotron"):
    for p in ("structured", "freeform", "hybrid"):
        sel = [r for r in dy if r["model"] == m and r["proto"] == p]
        P(f"carries.{m}.{p}", f"n {len(sel)} seq {100*sum(r['seq'] for r in sel)/len(sel):.1f} ack {100*sum(r['ack'] for r in sel)/len(sel):.1f} tgt {100*sum(r['tgt_disclosed'] for r in sel)/len(sel):.1f} other {100*sum(r['names_other'] for r in sel)/len(sel):.1f} reqnone {100*sum(r['reqnone'] for r in sel)/len(sel):.1f} temp {sum(r['temp'] for r in sel)}")
P("corpus.none_msgs_any_suite", sum(1 for r in rows if r["proto"] == "none"))
P("corpus.think_markers", ", ".join(f"{k} {v}" for k, v in collections.Counter((r['model'], r['suite']) for r in rows if r['think']).items()) or "0")

# ---- part 2: Quality-Score-led versions ----


def c(**kw):
    return T.cell([e for e in S if all(e[k] in v for k, v in kw.items())])

# QS prose vs none by suite x level (pooled strong)
for s in T.SUITES + ("all",):
    ss = T.SUITES if s == "all" else (s,)
    out = []
    for lv in T.ALL_LEVELS:
        n = c(suite=ss, level=(lv,), cond=("none",)); p = c(suite=ss, level=(lv,), cond=("freeform", "hybrid"))
        out.append(f"{lv} none {n['q']:.2f}/{n['played']:.1f} prose {p['q']:.2f}/{p['played']:.1f} gap {p['q']-n['q']:+.2f}")
    P(f"Q.prose_vs_none.{s}", " | ".join(out))
# QS structured vs freeform/hybrid cells per (model, suite, level): inversions on QS
inv = []
for m in T.STRONG:
    for s in T.SUITES:
        for lv in T.ALL_LEVELS:
            q = {p: T.cell([e for e in S if e["model"] == m and e["suite"] == s and e["level"] == lv and e["cond"] == p])["q"] for p in T.PROTOCOLS}
            for x in ("freeform", "hybrid"):
                if q["structured"] > q[x]:
                    inv.append(f"{nick[m]} {s} {lv} str {q['structured']:.1f} > {x} {q[x]:.1f}")
P("Q.inversions", f"{len(inv)}: " + "; ".join(inv))
# prose QS lead over schema and over none per model (all suites), mean of the two prose protocols
for m in T.STRONG:
    def pairs_q(a, b):
        return [idx[(m, s, l, a, g)]["q"] - idx[(m, s, l, b, g)]["q"] for s in T.SUITES for l in T.ALL_LEVELS
                for g in C.gids(E, l) if (m, s, l, a, g) in idx and (m, s, l, b, g) in idx
                and idx[(m, s, l, a, g)]["q"] is not None and idx[(m, s, l, b, g)]["q"] is not None]
    for a, b in (("freeform", "none"), ("hybrid", "none")):
        d = pairs_q(a, b)
        P(f"Q.multi.{nick[m]}.{a}_vs_{b}", f"{st.mean(d):+.2f} n={len(d)} perm p={T.perm_p(d):.2g}")
    # per suite prose-vs-structured QS lead
    for s in T.SUITES:
        pr = T.cell([e for e in S if e["model"] == m and e["suite"] == s and e["cond"] in ("freeform", "hybrid")])
        sc = T.cell([e for e in S if e["model"] == m and e["suite"] == s and e["cond"] == "structured"])
        P(f"Q.lead.{nick[m]}.{s}", f"prose {pr['q']:.2f} vs structured {sc['q']:.2f} = {pr['q']-sc['q']:+.2f}")
# 3p->6p and dyad->3p pooled-model per-level Quality and success with played, for raw:491/558
for lv in T.ALL_LEVELS:
    cs = {s: c(suite=(s,), level=(lv,)) for s in ("solo",) + T.SUITES}
    P(f"ts.{lv}", ", ".join(f"{s} Q {x['q']:.2f}/{x['played']:.1f} succ {x['rate']:.1f} eff {x['eff']:.3f} effJ {x['eff_j']:.3f}" for s, x in cs.items()))
# QS of the solo-vs-6p S3 gap pooled; strong team QS overall vs solo
P("Q.team_vs_solo", f"solo {c(suite=('solo',))['q']:.2f}; teams {T.cell([e for e in S if e['suite']!='solo'])['q']:.2f}/{T.cell([e for e in S if e['suite']!='solo'])['played']:.2f}")
# flawless 6p at S1, ratio-of-sums vs mean-of-ratios
for lv in T.ALL_LEVELS:
    g = C.gids(E, lv)
    P(f"flawless6p.{lv}", f"mean J/A {st.mean(opt[(lv, x)]/opt[('6p', lv, x)] for x in g):.3f}; sumJ/sumA {sum(opt[(lv, x)] for x in g)/sum(opt[('6p', lv, x)] for x in g):.3f}")
# prose losses per suite as %; prose losses that are done-cuts not here (strand B)

# ---- part 3: stored-bound efficiency, solo McNemar ----

# stored Move Efficiency per episode
ME = {}
for m, prefix in T.ARMS.items():
    if m not in T.STRONG:
        continue
    for suite in ("solo", "dyad"):
        for f in sorted(glob.glob(os.path.join(T.ROOT, T.tree(prefix, suite), "**", "scores.json"), recursive=True)):
            exp = f.split(os.sep)[-3]
            if suite == "solo":
                if not exp.startswith("Solo_"):
                    continue
                _, lv, pr = exp.split("_")[:3]; cond = "solo_" + pr.lower()
            else:
                if not re.match(r"S\d_", exp):
                    continue
                lv, cond = exp.split("_")[:2]
            g = int(re.search(r"instance_(\d+)", f).group(1))
            s = json.load(open(f))["episode scores"]
            if s.get("Success"):
                ME[(m, suite, lv, cond, g)] = s.get("Move Efficiency")
bs = C.solo_boards(E)
for m in T.STRONG:
    print(f"C.matched.BFS.{m}:", ", ".join(
        f"{lv} {st.mean(st.mean(ME[(m, 'solo', l, c, g)] for c in ('solo_silent', 'solo_thinking')) for (l, g) in bs if l == lv):.3f}"
        for lv in T.ALL_LEVELS))
# dyad S2-S4 successes: A_k efficiency minus stored BFS efficiency
for suite in ("solo", "dyad"):
    d = [idx[k]["eff"] - v for k, v in ME.items() if k[1] == suite and k[2] != "S1" and v is not None]
    print(f"BFS_understatement.{suite}.S2-S4: mean {st.mean(d):.3f} n={len(d)}")
    for lv in ("S2", "S3", "S4"):
        d = [idx[k]["eff"] - v for k, v in ME.items() if k[1] == suite and k[2] == lv and v is not None]
        print(f"  {lv}: {st.mean(d):.3f}")
# solo silent vs thinking McNemar per model, S1-S4
for m in T.STRONG:
    pr = [(idx[(m, "solo", l, "solo_thinking", g)], idx[(m, "solo", l, "solo_silent", g)]) for l in T.ALL_LEVELS for g in C.gids(E, l)]
    b = sum(1 for x, y in pr if x["success"] and not y["success"]); c = sum(1 for x, y in pr if y["success"] and not x["success"])
    print(f"solo.mcnemar.thinking_vs_silent.{m}: {b}:{c} p={binomtest(b, b+c, .5).pvalue:.3g} n={len(pr)}")
