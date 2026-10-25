"""Accuracy and currency (spec §5.4), talk about finished teammates (§5.6),
owner-side correction (§5.7).

Spec: thesis/message_analysis_design.md §5.4, §5.6, §5.7, §8; extension §4
"Reporting" (grammar alone and union side by side).

D2, the one confirmatory outcome: teammate G-loc pairs (forward and reverse
rule) kept under the strict frame set, true iff the letter is on that cell at
B_t⁻; one proportion per episode over the episodes with at least one
statement; strong (qwen38 + dsv41) against weak (nemotron) by the unpaired
two-sided permutation test of analysis/thesis/confirmatory.py (`two_sample_p`).
The declared D2 ran the old CLAIM regex over the whole message text, so all
regions (prose, PLAN, REQUEST) are the primary scope and the prose / slot split
is reported beside it.

Statement-level rates count every message, repeats included, as the declared
D2 did: a repeat sent at a later ply is a new statement with its own truth.
Message-level rates (§5.6 opportunities) deduplicate exact repeats.
"""
import collections
import os
import re

import numpy as np
from scipy.stats import permutation_test

from analysis.messages import corpus, grammar, stats

STRONG, WEAK = ("qwen38", "dsv41"), ("nemotron",)
KEYS = ("model", "tier", "suite", "level", "proto", "board", "episode")
CELL = ("model", "suite", "proto")  # spec §8: a pooled cell is 60 boards
# §5.4 validation status. §9.6 sets `validation`: "pending" (not yet run),
# "failed" (grammar part fails), "grammar" (only the grammar part passes),
# "both" (grammar and model parts pass: D2 from the union).
D2_NAMES = {"validated": "accuracy of statements about teammates' objects",
            "manifest": "teammate letter–cell pairings in indicative frames, true at the moment of the message"}
FROM_FRAMES = ("past", "future", "negation", "question", "think")
SOURCE = ("restates owner's last visible cell", "states an updated true cell", "matches no visible source",
          "follows the MOVE BOARD from the owner's cell")  # the last only on the final arm


# The old scorer's claim regex (a letter, then "at (Rr,Cc)" within 24 characters).
CLAIM = re.compile(r"\b(?-i:([A-F]))\b[^.;,\n]{0,24}?\bat\s*\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?", re.I)


def ekeys(ep):
    m = ep["meta"]
    return dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], proto=m["proto"],
                board=m["board"], episode=m["id"])


def two_sample_p(x, y):
    """analysis/thesis/confirmatory.py two_sample_p, verbatim: two-sided
    permutation test on the difference of means, unpaired, 100,000
    resamples, seed 0. NaN below two episodes in a tier (scipy needs two)."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    return float(permutation_test(
        (np.asarray(x, float), np.asarray(y, float)),
        lambda a, b, axis: np.mean(a, axis=axis) - np.mean(b, axis=axis),
        permutation_type="independent", n_resamples=100_000, batch=5_000,  # batch bounds memory, same p
        alternative="two-sided", random_state=0, vectorized=True).pvalue)


def _ratio(k, n):
    return k / n if n else None


def cells(rows, value, by=CELL, **extra):
    """§8 cell estimates of an episode-level value over groups `by`."""
    g = collections.defaultdict(list)
    for r in rows:
        if r[value] is not None:
            g[tuple(r[k] for k in by)].append(r)
    return [dict(zip(by, k), value=value, **extra,
                 **stats.cell_estimate([r[value] for r in rs], [r["level"] for r in rs]))
            for k, rs in sorted(g.items())]


# ---------------------------------------------------------------- pairs

CANON = {"loc": (("at",), ("is", "at")), "frm": (("from",),)}


def clause_tokens(an, cid, P=grammar.DEFAULT):
    """Tokens of clause `cid` (ids are global over regions, as in grammar.analyse)."""
    base = 0
    for _, toks in an["tokens"]:
        cl = grammar.clauses(toks, P)
        if cid < base + len(cl):
            c0, c1, _ = cl[cid - base]
            return toks[c0:c1]
        base += len(cl)
    raise IndexError(cid)


def canonical(toks, kind, letter, cell):
    """§5.4 canonical forms, as contiguous tokens: `L at (R,C)`, `L is at
    (R,C)` (G-loc) and `L from (R,C)` (G-from)."""
    for i, t in enumerate(toks):
        if t.kind == "letter" and t.val == letter:
            for seq in CANON[kind]:
                j = i + 1 + len(seq)
                if grammar._seq_at(toks, i + 1, seq) and j < len(toks) and toks[j].kind == "cell" \
                        and toks[j].val == cell:
                    return True
    return False


def candidates(eps, A, P=grammar.DEFAULT):
    """Every letter–cell pair of every message about an object on the board:
    G-loc (`loc`), G-from (`frm`, any region) and PLAN-from (`plan`). One dict
    per pair: ep, i (turn index = ply), kind, region, rule, letter, cell,
    frames, kept (G-loc: P's frame set; G-from: no frame), own, true (teammate
    object: at B_t⁻; own object: at B_t⁻ or B_t⁺), canon (kept teammate pairs)."""
    out = []
    for ep in eps:
        for i, t in enumerate(ep["turns"]):
            if not t["is_msg"]:
                continue
            an = A[(ep["meta"]["id"], t["ply"])]
            for kind in ("loc", "frm"):
                for p in an[kind]:
                    if p["letter"] in ep["owner"]:
                        kept = grammar.kept(p["frames"], P["frames"]) if kind == "loc" else not p["frames"]
                        out.append(_pair(ep, i, kind, p["region"], p["rule"], p["letter"], p["cell"], p["frames"],
                                         kept, an, p["clause"], P))
            s = t["slots"]
            if s and s["plan_obj"] in ep["owner"]:
                out.append(_pair(ep, i, "plan", "plan", "plan", s["plan_obj"], s["plan_from"], set(), True,
                                 None, None, P))
    return out


def _pair(ep, i, kind, region, rule, L, c, frames, kept, an, cid, P):
    t = ep["turns"][i]
    own = ep["owner"][L] == t["seat"]
    canon = (kept and not own and rule == "fwd" and kind in CANON
             and canonical(clause_tokens(an, cid, P), kind, L, c))
    return dict(ep=ep, i=i, kind=kind, region=region, rule=rule, letter=L, cell=c, frames=frames, kept=kept,
                own=own, true=t["board"][L] == c or (own and t["after"][L] == c), canon=canon)


def frame_rows(cands, by, kind="loc"):
    """Share of candidate pairs removed by each exclusion frame (a pair can hit
    several), plus the share removed by the whole set, per group `by`."""
    names = grammar.FRAMES if kind == "loc" else FROM_FRAMES
    g = collections.defaultdict(list)
    for c in cands:
        g[tuple(c[k] if k in c else ekeys(c["ep"])[k] for k in by)].append(c)
    out = []
    for key, cs in sorted(g.items()):
        n = len(cs)
        sets = [("any (strict)", lambda f: not grammar.kept(f, "strict")),
                ("any (lenient)", lambda f: not grammar.kept(f, "lenient"))] if kind == "loc" else \
            [("any", bool)]
        for name, hit in [(f, lambda fs, f=f: f in fs) for f in names] + sets:
            out.append(dict(zip(by, key), kind=kind, frame=name, n=n,
                            removed=sum(hit(c["frames"]) for c in cs), share=sum(hit(c["frames"]) for c in cs) / n))
    return out


# ---------------------------------------------------------------- §5.4 own position

def own_rows(eps, cands):
    """Per episode and channel: PLAN-from (primary), prose G-loc, prose G-from;
    n own pairs, k true at B_t⁻ or B_t⁺. `plan_other` counts PLANs naming a
    letter the speaker does not own."""
    acc = collections.defaultdict(lambda: [0, 0])
    other = collections.Counter()
    for c in cands:
        eid = c["ep"]["meta"]["id"]
        if c["kind"] == "plan" and not c["own"]:
            other[eid] += 1
        if not (c["own"] and c["kept"]):
            continue
        ch = "plan_from" if c["kind"] == "plan" else f"prose_{c['kind']}" if c["region"] == "prose" else None
        if ch:
            acc[(eid, ch)][0] += 1
            acc[(eid, ch)][1] += c["true"]
    out = []
    for ep in eps:
        eid = ep["meta"]["id"]
        for ch in ("plan_from", "prose_loc", "prose_frm"):
            n, k = acc.get((eid, ch), (0, 0))
            out.append(dict(ekeys(ep), channel=ch, n=n, k=k, prop=_ratio(k, n),
                            plan_other=other[eid] if ch == "plan_from" else None))
    return out


# ---------------------------------------------------------------- §5.4 teammate statements, D2

def residual_teammate(ep, t, an):
    """Teammate letter–cell pairs within 7 tokens that G-loc did not pair
    (the §9.2 letter–cell audit restricted to teammates' letters), by region
    group (prose / slots)."""
    got = {(p["letter"], p["cell"]) for p in an["loc"]}
    n = collections.Counter()
    for region, toks in an["tokens"]:
        for i, x in enumerate(toks):
            if x.kind == "letter" and ep["owner"].get(x.val, t["seat"]) != t["seat"]:
                n["prose" if region == "prose" else "slots"] += sum(
                    toks[j].kind == "cell" and (x.val, toks[j].val) not in got
                    for j in range(max(0, i - 7), min(len(toks), i + 8)))
    return n


def old_claims(ep, t):
    """The old scorer on one message: CLAIM over the whole text, teammate
    letters on the board, true iff the letter is on the cell at B_t⁻."""
    out = []
    for L, r, c in CLAIM.findall(t["msg"]):
        L = L.upper()
        if L in ep["owner"] and ep["owner"][L] != t["seat"]:
            out.append(t["board"][L] == (int(r) - 1, int(c) - 1))
    return out


def teammate(cands):
    """The D2 indicator: kept G-loc pairs about a teammate's letter."""
    return [c for c in cands if c["kind"] == "loc" and c["kept"] and not c["own"]]


def d2_episodes(eps, A, cands, extra=None):
    """One row per messaging episode: statement counts (n) and true counts (k)
    for the primary scope (all regions) and beside it prose, slots (PLAN and
    REQUEST), canonical forms, the old scorer and, with `extra`, the union.
    `extra`: model-only records (episode id, ply, letter, cell) from E5,
    cell zero-based (r, c); records the grammar already has are ignored."""
    tm = teammate(cands)
    ver = collections.defaultdict(lambda: collections.Counter())
    seen = set()
    for c in tm:
        eid = c["ep"]["meta"]["id"]
        seen.add((eid, c["ep"]["turns"][c["i"]]["ply"], c["letter"], c["cell"]))
        for v in ("all", "prose" if c["region"] == "prose" else "slots", "union"):
            ver[eid][v + "_n"] += 1
            ver[eid][v + "_k"] += c["true"]
    for c in cands:  # canonical: G-loc `L (is) at`, G-from `L from`, about teammates
        if c["canon"]:
            ver[c["ep"]["meta"]["id"]]["canon_n"] += 1
            ver[c["ep"]["meta"]["id"]]["canon_k"] += c["true"]
    byid = {ep["meta"]["id"]: ep for ep in eps}
    for eid, ply, L, cell in sorted(set(extra or ()) - seen):
        ep = byid.get(eid)
        if ep is None or L not in ep["owner"]:
            continue
        t = ep["turns"][ply]
        if ep["owner"][L] != t["seat"]:
            ver[eid]["union_n"] += 1
            ver[eid]["union_k"] += t["board"][L] == tuple(cell)
    out = []
    for ep in eps:
        eid, v = ep["meta"]["id"], ver[ep["meta"]["id"]]
        resid = collections.Counter()
        for t in ep["turns"]:
            if t["is_msg"]:
                oc = old_claims(ep, t)
                v["old_n"] += len(oc)
                v["old_k"] += sum(oc)
                resid += residual_teammate(ep, t, A[(eid, t["ply"])])
        row = dict(ekeys(ep), n_true=v["all_k"], n_false=v["all_n"] - v["all_k"], n_stmt=v["all_n"],
                   resid=sum(resid.values()), resid_prose=resid["prose"], resid_slots=resid["slots"])
        for name in ("all", "prose", "slots", "canon", "old") + (("union",) if extra is not None else ()):
            row[f"{name}_n"], row[f"{name}_k"] = v[name + "_n"], v[name + "_k"]
            row[f"{name}_prop"] = _ratio(v[name + "_k"], v[name + "_n"])
        for g in ("prose", "slots"):
            row[f"{g}_true"], row[f"{g}_false"] = v[g + "_k"], v[g + "_n"] - v[g + "_k"]
        out.append(row)
    return out


def tier_contrast(rows, value, test=True):
    """D2's statistic on an episode-level value: strong mean, weak mean, the
    difference and two_sample_p (None if not `test`: accuracy contrasts
    beyond D2 and its sensitivities are descriptive), over episodes where
    the value exists."""
    x = [r[value] for r in rows if r["model"] in STRONG and r[value] is not None]
    y = [r[value] for r in rows if r["model"] in WEAK and r[value] is not None]
    res = dict(n_strong=len(x), n_weak=len(y), strong=float(np.mean(x)) if x else np.nan,
               weak=float(np.mean(y)) if y else np.nan, p=two_sample_p(x, y) if test else None)
    res["diff"] = res["strong"] - res["weak"]
    for tier, models in (("strong", STRONG), ("weak", WEAK)):
        rs = [r for r in rows if r["model"] in models and r[value] is not None]
        ci = stats.cell_estimate([r[value] for r in rs], [r["level"] for r in rs])
        res[f"{tier}_lo"], res[f"{tier}_hi"] = ci["lo"], ci["hi"]
    return res


def d2(eps, A, extra=None, validation="pending", P=grammar.DEFAULT, rows=None, scope="test"):
    """D2, grammar alone and (with `extra`) the union, each a dict with the
    outcome name, status, strong and weak means, n episodes, difference, p.
    `validation` is the §9.6 result (see D2_NAMES). Pass `rows` from
    d2_episodes to skip recomputing them."""
    assert validation in ("pending", "failed", "grammar", "both"), validation
    # without records the confirmatory union row would silently vanish
    assert validation != "both" or extra is not None, "§9.6 puts D2 on the union, but there are no E5 records"
    if rows is None:
        eps = [ep for ep in eps if ep["meta"]["proto"] != "none"]
        rows = d2_episodes(eps, A, candidates(eps, A, P), extra)
    name = D2_NAMES["validated" if validation in ("grammar", "both") else "manifest"]
    primary = "union" if validation == "both" else "grammar"
    out = {}
    for ver, col in (("grammar", "all_prop"), ("union", "union_prop")):
        if ver == "union" and extra is None:
            continue
        # only the test arm is confirmatory; the development arm is the exploratory experiment (spec §1.4)
        status = ("pending §9.6" if validation == "pending" else
                  ("confirmatory" if scope == "test" else "validated, development arm (exploratory)")
                  if ver == primary and validation in ("grammar", "both") else "exploratory")
        # a version is named as validated only if every extractor in it passed §9.6
        passed = validation == "both" or (validation == "grammar" and ver == "grammar")
        vname = D2_NAMES["validated" if passed else "manifest"]
        out[ver] = dict(version=ver, name=vname, status=status, primary=ver == primary,
                        **tier_contrast(rows, col))
    return out


def owner_cells(ep, cands_by_msg, i):
    """{letter: cells} the message at ply i pairs with its speaker's own letters
    (kept G-loc, G-from, PLAN-from)."""
    out = collections.defaultdict(set)
    for c in cands_by_msg.get((ep["meta"]["id"], i), ()):
        if c["own"] and c["kept"]:
            out[c["letter"]].add(c["cell"])
    return out


def source_class(ep, i, L, cell, true, cands_by_msg):
    """§5.4 source split, by rule. The owner's most recent visible statement
    about L is the owner's latest message in the speaker's visible context that
    pairs L with a cell; on the final arm the visible MOVE BOARD lines from
    that ply on are followed from each stated cell."""
    T, owner = ep["turns"], ep["owner"][L]
    stated, q = set(), None
    for v in sorted(T[i]["vis"], reverse=True):
        if T[v]["seat"] == owner and L in (oc := owner_cells(ep, cands_by_msg, v)):
            stated, q = oc[L], v
            break
    mb = set()
    if q is not None and ep["meta"]["arm"] == "final":
        for s in stated:
            for ply, own, src, dst in corpus.move_board(ep, i):
                if ply >= q and not own and src == s:
                    s = dst
            mb.add(s)
        mb -= stated
    if cell in stated:
        return SOURCE[0]
    if cell in mb:
        return SOURCE[3]
    return SOURCE[1] if true else SOURCE[2]


def moved_since(ep, i, L):
    """§5.4 freshness: has L moved since the speaker's previous turn (since the
    start if it has none)?"""
    T = ep["turns"]
    prev = max((k for k in range(i) if T[k]["seat"] == T[i]["seat"]), default=-1)
    return any(T[k]["obj"] == L for k in range(prev + 1, i))


def splits(tm, cands_by_msg):
    """Per tier and model: statements and true shares by source class and by
    freshness. Descriptive."""
    acc = collections.defaultdict(lambda: [0, 0, set()])
    for c in tm:
        ep, m = c["ep"], c["ep"]["meta"]
        src = source_class(ep, c["i"], c["letter"], c["cell"], c["true"], cands_by_msg)
        fresh = "L moved since S's previous turn" if moved_since(ep, c["i"], c["letter"]) else "L not moved"
        for split, val in (("source", src), ("freshness", fresh)):
            for model in (m["model"], "all"):
                a = acc[(m["tier"], model, split, val)]
                a[0] += 1
                a[1] += c["true"]
                a[2].add(m["id"])
    tot = collections.Counter()
    for (tier, model, split, _), a in acc.items():
        tot[(tier, model, split)] += a[0]
    return [dict(tier=tier, model=model, split=split, value=val, n=a[0], n_true=a[1], share_true=a[1] / a[0],
                 share_of_statements=a[0] / tot[(tier, model, split)], n_episodes=len(a[2]))
            for (tier, model, split, val), a in sorted(acc.items())]


def count_contrasts(rows):
    """Unconditional per-episode counts of true and false teammate statements,
    board-paired sign-flip. Tier: all regions, strong models averaged per board
    unit. Protocol (spec §5, voluntary content): freeform - hybrid on the
    prose part, hybrid - structured on the slots; both carry the §9.2 coverage
    bounds. Exploratory, Holm within this family."""
    from analysis.messages.measures import coverage
    unit = ("suite", "level", "proto", "board")
    avg = collections.defaultdict(list)
    for r in rows:
        avg[(r["tier"],) + tuple(r[u] for u in unit)].append(r)
    trows = [dict(zip(("tier",) + unit, k), **{v: float(np.mean([r[v] for r in rs])) for v in ("n_true", "n_false")})
             for k, rs in avg.items()]
    out = []
    for v in ("n_true", "n_false"):
        res = stats.paired_contrast(trows, v, "tier", "strong", "weak", unit=unit)
        out.append(dict(value=v, contrast=res["contrast"], est=res["mean"], p=res["p"], pairs=res["pairs"],
                        missing=res["missing"]))
        for g, a, b in (("prose", "freeform", "hybrid"), ("slots", "hybrid", "structured")):
            val = f"{g}_{v[2:]}"
            bd = coverage.bounds(rows, val, f"resid_{g}", "proto", a, b)
            out.append(dict(bd, value=val, est=bd["est_lower"], p=bd["p_lower"]))
    for r, h in zip(out, stats.holm([r["p"] for r in out])):
        r["p_holm"] = h
    return out


# ---------------------------------------------------------------- §5.6 finished teammates

def finished_rows(eps, A):
    """§5.6: per episode, region (REQUEST / prose) and whether the finished
    teammate's `done` was visible to the speaker: opportunities (message ×
    finished teammate, exact repeats dropped) and mentions (a cell held by
    that teammate's object at B_t⁻, or that teammate's letter)."""
    acc = collections.defaultdict(lambda: [0, 0])
    for ep in eps:
        for i, t in enumerate(ep["turns"]):
            if not t["is_msg"] or t["dup"]:
                continue
            fin = [s for s, q in ep["done_ply"].items() if q < t["ply"] and s != t["seat"]]
            if not fin:
                continue
            an, vis = A[(ep["meta"]["id"], t["ply"])], corpus.done_visible(ep, i)
            present = {r for r, _ in an["tokens"]}
            for s in fin:
                letters = set(ep["objs"][s])
                held = {t["board"][o] for o in letters}
                for region in ("request", "prose"):
                    if region not in present:
                        continue
                    hit = any(r == region and v in letters for r, _, v, _ in an["letters"]) or \
                        any(r == region and v in held for r, _, v, _ in an["cells"])
                    a = acc[(ep["meta"]["id"], region, s in vis)]
                    a[0] += 1
                    a[1] += hit
    out = []
    for ep in eps:
        for region in ("request", "prose"):
            for vis in (True, False):
                n, k = acc.get((ep["meta"]["id"], region, vis), (0, 0))
                if n:
                    out.append(dict(ekeys(ep), region=region, done_visible=vis, n=n, k=k, rate=k / n))
    return out


# ---------------------------------------------------------------- §5.7 owner-side correction

def correction_rows(eps, cands, cands_by_msg):
    """§5.7 (exploratory). Trigger: a teammate statement (D2 indicator) about
    T's letter L, at T's first decision that shows it. Outcome: T's next
    message states L at its true cell (own rule: B_t⁻ or B_t⁺), by any channel
    (kept G-loc, G-from, PLAN-from) and by prose G-loc / G-from only. Trigger
    false vs true (the baseline); stratum: L is T's PLAN or moved object then."""
    acc = collections.defaultdict(lambda: [0, 0, 0])
    expo = {}
    for c in {(id(c["ep"]), c["i"], c["letter"], c["cell"]): c for c in teammate(cands)}.values():
        ep, L = c["ep"], c["letter"]
        T, owner, eid = ep["turns"], ep["owner"][L], ep["meta"]["id"]
        if eid not in expo:
            expo[eid] = corpus.exposures(ep)
        e = expo[eid].get((c["i"], owner))
        j = next((k for k in range(e, len(T)) if T[k]["seat"] == owner and T[k]["is_msg"]), None) \
            if e is not None else None
        if j is None:
            continue
        stated = [x for x in cands_by_msg.get((eid, j), ()) if x["own"] and x["kept"] and x["letter"] == L
                  and x["true"] and (x["kind"] != "frm" or x["region"] == "prose")]
        sl = T[j]["slots"]
        stratum = "PLAN or moved object" if T[j]["obj"] == L or (sl and sl["plan_obj"] == L) else "other object"
        a = acc[(eid, "true" if c["true"] else "false", stratum)]
        a[0] += 1
        a[1] += bool(stated)
        a[2] += any(x["kind"] != "plan" and x["region"] == "prose" for x in stated)
    out = []
    for ep in eps:
        for trig in ("false", "true"):
            for stratum in ("PLAN or moved object", "other object"):
                n, k, kp = acc.get((ep["meta"]["id"], trig, stratum), (0, 0, 0))
                if n:
                    out.append(dict(ekeys(ep), trigger=trig, stratum=stratum, n=n, k_any=k, k_prose=kp,
                                    rate_any=k / n, rate_prose=kp / n,
                                    compliance_check=ep["meta"]["proto"] in ("structured", "hybrid")))
    return out


# ---------------------------------------------------------------- entry point

def run(scope, eps, A, extra=None, validation="pending", P=grammar.DEFAULT):
    """{table: rows}. On `calib` only the frame composition, `frame_*` (spec
    §2: the calibration set never gives estimates)."""
    eps = [ep for ep in eps if ep["meta"]["proto"] != "none"]
    cands = candidates(eps, A, P)
    tmc = [c for c in cands if c["kind"] == "loc" and not c["own"]]
    for c in tmc:
        c["region_group"] = "prose" if c["region"] == "prose" else "slots"
    own_prose = [c for c in cands if c["own"] and c["kind"] != "plan" and c["region"] == "prose"]
    out = dict(
        frame_d2=frame_rows(tmc, ("tier",)) + frame_rows(tmc, ("tier", "region_group")),
        frame_own=frame_rows([c for c in own_prose if c["kind"] == "loc"], ("tier",))
        + frame_rows([c for c in own_prose if c["kind"] == "frm"], ("tier",), kind="frm"))
    if scope == "calib":
        return out
    by_msg = collections.defaultdict(list)
    for c in cands:
        by_msg[(c["ep"]["meta"]["id"], c["i"])].append(c)
    own = own_rows(eps, cands)
    rows = d2_episodes(eps, A, cands, extra)
    res = d2(eps, A, extra, validation, P, rows=rows, scope=scope)
    sens = [dict(version=name, **tier_contrast(rows, col)) for name, col in
            (("prose region", "prose_prop"), ("slot regions (PLAN, REQUEST)", "slots_prop"),
             ("canonical forms", "canon_prop"), ("old scorer (qual_corpus CLAIM)", "old_prop"))]
    fin = finished_rows(eps, A)
    cor = correction_rows(eps, cands, by_msg)
    out.update(
        d2=list(res.values()),
        d2_sensitivity=sens,
        d2_episodes=rows,
        d2_cells=cells(rows, "all_prop"),
        d2_splits=splits(teammate(cands), by_msg),
        teammate_counts_cells=cells(rows, "n_true") + cells(rows, "n_false"),
        teammate_counts_contrasts=count_contrasts(rows),
        own_episodes=own,
        own_cells=[dict(r, channel=ch) for ch in ("plan_from", "prose_loc", "prose_frm")
                   for r in cells([r for r in own if r["channel"] == ch], "prop")],
        own_tier=[dict(channel=ch, **tier_contrast([r for r in own if r["channel"] == ch], "prop", test=False))
                  for ch in ("plan_from", "prose_loc", "prose_frm")],
        finished_episodes=fin,
        finished_cells=cells(fin, "rate", CELL + ("region", "done_visible")),
        correction_episodes=cor,
        correction_cells=cells(cor, "rate_any", ("model", "proto", "trigger", "stratum"))
        + cells(cor, "rate_prose", ("model", "proto", "trigger", "stratum")))
    return out
