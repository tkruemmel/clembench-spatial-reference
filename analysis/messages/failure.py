"""Failure analysis (spec §6): what was in view at fatal decisions.

Risk sets come from corpus.hazards, ported from averted.py (fork) and
seal_averted.py (pinch). A pinch enters only when it is free: the mover has a
legal move that seals nobody (seal_averted.py `safe`). corpus.hazards also
lists forced pinches, where every legal move seals; they hold no decision and
are left out here (see `free`).

Unit: a decision at which a hazard is live. A spell is a maximal run of a
player's consecutive own decisions at which the same hazard kind is live. The
primary row is the spell's first live decision, with case = the fatal action
taken at that decision; every live decision is a row of the secondary analysis.

Features come from the actor's visible context at the call (`turn["vis"]`).
The actor's message in the same response is never among them. The final
arm's MOVE BOARD names no letter and states no target, so it enters only c.
  a  a critical cell named (G-cell) in a visible teammate message
  b  a visible teammate message states the victim's target (G-target strict,
     the victim's owner speaking), or names a cell on the victim's remaining
     route that the action would cut off (`cut_cells`)
  c  the victim's owner resolvable by the frozen rules: test arm only, from
     §5.3's owners-resolvable rule; None here
  d  the actor's own last message before the spell's first live ply names a
     critical cell
  e  a victim letter paired with its current cell in a visible message
     (G-loc kept strict, G-from, PLAN-from)
"""
import collections

import numpy as np
from scipy import stats as sps

from . import corpus, grammar, stats

FEATURES = ("a", "b", "c", "d", "e")
FAMILY = ("a", "b", "d", "e")  # x {pinch}, Holm within a tier
MESSAGING = ("structured", "hybrid", "freeform")
GROUPS = {"prose": ("freeform", "hybrid"), "rigid": ("none", "structured")}  # the old scripts' split
B, SEED = 2000, 0
KEYS = ("model", "tier", "suite", "level", "proto", "board")


# ---------------------------------------------------------------- risk sets

def _done_before(ep, ply):
    return {s for s, q in ep["done_ply"].items() if q < ply}


def free(ep, h):
    """Does the pinch mover have a harmless legal move (corpus.hazards' `free`)?"""
    return h["free"]


def risk_set(ep):
    """Every live decision: all fork records, free pinch records."""
    return [h for h in corpus.hazards(ep) if h["kind"] == "fork" or free(ep, h)]


def spells(ep, recs):
    """{(seat, kind, ply): first ply of its spell}, a spell being a maximal run of
    the seat's consecutive own decisions at which that kind is live."""
    live = {(h["seat"], h["kind"], h["ply"]) for h in recs}
    out = {}
    for seat in ep["seats"]:
        own = [t["ply"] for t in ep["turns"] if t["seat"] == seat]
        for kind in ("fork", "pinch"):
            start = None
            for p in own:
                start = (p if start is None else start) if (seat, kind, p) in live else None
                if start is not None:
                    out[(seat, kind, p)] = start
    return out


def victim_objs(ep, h):
    """Fork victims are objects, pinch victims seats (all their objects)."""
    return sorted(h["victims"]) if h["kind"] == "fork" else sorted(o for s in h["victims"] for o in ep["objs"][s])


def cut_cells(ep, h):
    """Cells on a victim's remaining route that the fatal action would cut off.

    On a shortest route from the victim to its target with finished objects as
    walls, and unreachable from the victim once the action's cells are walls as
    well: the actor's objects frozen (fork), or each sealing destination (pinch)."""
    t = ep["turns"][h["ply"]]
    b = t["board"]
    done = _done_before(ep, h["ply"])
    fin = frozenset(b[o] for o in b if ep["owner"][o] in done)
    acts = ([fin | {b[o] for o in ep["objs"][t["seat"]]}] if h["kind"] == "fork"
            else [fin | {tuple(c)} for c in h["critical"]])
    out = set()
    for v in victim_objs(ep, h):
        tg = ep["targets"][v]
        dv, dt = corpus.dist_map(ep, b[v], fin), corpus.dist_map(ep, tg, fin)
        if tg not in dv:
            continue
        route = {c for c in dv if c in dt and dv[c] + dt[c] == dv[tg]}
        for blk in acts:
            out |= route - set(corpus.dist_map(ep, b[v], blk))
    return out


# ---------------------------------------------------------------- features

def owner_resolvable(ep, h):
    """Feature c, test arm only: §5.3's owners-resolvable rule supplies it. On the
    development arm seat letters reveal 6p owners, so it is undefined there."""
    return None


def _pairs(turn, an):
    """(letter, cell) pairings of a message: G-loc kept strict, G-from kept, PLAN-from."""
    out = {(p["letter"], p["cell"]) for p in an["loc"] if grammar.kept(p["frames"], "strict")}
    out |= {(p["letter"], p["cell"]) for p in an["frm"] if not p["frames"]}
    if turn["slots"]:
        out.add((turn["slots"]["plan_obj"], turn["slots"]["plan_from"]))
    return out


def features(ep, h, A, start):
    """Features a-e of decision h, whose spell starts at ply `start`. None under `none`."""
    if ep["meta"]["proto"] == "none":
        return dict.fromkeys(FEATURES)
    eid, T = ep["meta"]["id"], ep["turns"]
    t = T[h["ply"]]
    cells = lambda q: {c for _, _, c, _ in A[(eid, q)]["cells"]}
    crit = {tuple(c) for c in h["critical"]}
    vis = [q for q in t["vis"] if T[q]["is_msg"]]  # a visible `done` turn is an event, not a message
    mates = [q for q in vis if T[q]["seat"] != t["seat"]]
    vic, cut, tg = victim_objs(ep, h), cut_cells(ep, h), ep["targets"]
    stated = any(grammar.target_mentions(A[(eid, q)], {o: tg[o]})[o]["stated"]
                 for q in mates for o in vic if ep["owner"][o] == T[q]["seat"])
    own = [u["ply"] for u in T[:start] if u["seat"] == t["seat"] and u["is_msg"]]
    true = {(o, t["board"][o]) for o in vic}
    return dict(a=int(any(cells(q) & crit for q in mates)),
                b=int(stated or any(cells(q) & cut for q in mates)),
                c=owner_resolvable(ep, h),
                d=int(bool(own) and bool(cells(own[-1]) & crit)),
                e=int(any(_pairs(T[q], A[(eid, q)]) & true for q in vis)))


def info_visible(ep, h, A=None):
    """cases.py hoop test: any of a, b, e at decision h. Without A the episode's
    earlier messages are analysed here."""
    if ep["meta"]["proto"] == "none":
        return False
    m = ep["meta"]
    A = A or {(m["id"], t["ply"]): grammar.analyse(t["msg"], m["arm"], m["proto"], t["slots"])
              for t in ep["turns"][:h["ply"]] if t["is_msg"]}
    f = features(ep, h, A, h["ply"])
    return bool(f["a"] or f["b"] or f["e"])


def decision_rows(ep, A):
    """One row per live decision, with its spell, stage and features."""
    recs = risk_set(ep)
    sp = spells(ep, recs)
    fatal = corpus.first_fatal(ep)
    m, out = ep["meta"], []
    for h in recs:
        start = sp[(h["seat"], h["kind"], h["ply"])]
        r = {k: m[k] for k in KEYS}
        r.update(episode=m["id"], seat=h["seat"], kind=h["kind"], ply=h["ply"], spell=start, case=int(bool(h["case"])),
                 victims=";".join(map(str, h["victims"])), critical=";".join(corpus.cell(c) for c in h["critical"]),
                 n_active=sum(s != h["seat"] and s not in _done_before(ep, h["ply"]) for s in ep["seats"]),
                 post_fatal=int(fatal is not None and h["ply"] > fatal))
        r.update(features(ep, h, A, start))
        out.append(r)
    return out


def spell_rows(D):
    """The first live decision of each spell, with the spell's length and whether any of its decisions was the case."""
    by = collections.defaultdict(list)
    for r in D:
        by[(r["episode"], r["seat"], r["kind"], r["spell"])].append(r)
    out = []
    for rs in by.values():
        r = dict(rs[0], spell_len=len(rs), spell_taken=int(any(x["case"] for x in rs)))
        assert r["ply"] == r["spell"]
        out.append(r)
    return out


# ---------------------------------------------------------------- tests

def _strata(S):
    """Board x stage (ply tercile over these rows, number of active teammates)."""
    cuts = np.quantile([r["ply"] for r in S], [1 / 3, 2 / 3])
    keys = [(r["suite"], r["level"], r["board"], int(np.digitize(r["ply"], cuts)), r["n_active"]) for r in S]
    ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([ids[k] for k in keys])


def block_permuter(strata, episodes):
    """Permute outcomes within strata, moving each episode's rows in a stratum as one
    block among blocks of the same size (an episode-level permutation)."""
    blocks = collections.defaultdict(list)
    for i, k in enumerate(zip(strata, episodes)):
        blocks[k].append(i)
    groups = collections.defaultdict(list)
    for (s, _), idx in blocks.items():
        groups[(s, len(idx))].append(idx)
    groups = [np.array(g) for g in groups.values() if len(g) > 1]

    def permute(y, rng):
        out = y.copy()
        for g in groups:
            out[g.ravel()] = y[g[rng.permutation(len(g))].ravel()]
        return out
    return permute


def _reading(lo, hi):
    if np.isnan(lo):
        return "no informative stratum"
    return f"compatible with odds ratios from {lo:.3g} to {'infinity' if np.isinf(hi) else f'{hi:.3g}'}"


def feature_test(S, D, f, B=B, seed=SEED):
    """Primary (spell rows): stratified episode-level permutation of the MH score
    sum_k (a_k - E a_k), exact conditional OR over the same strata, minimum
    detectable OR. Secondary (decision rows): clogit stratified by episode."""
    y, x = np.array([r["case"] for r in S]), np.array([r[f] for r in S])
    st = _strata(S)
    ybar = np.bincount(st, y) / np.bincount(st)
    score, p = stats.perm_pvalue(lambda yy: float(x @ (yy - ybar[st])), y,
                                 block_permuter(st, [r["episode"] for r in S]), B=B, seed=seed)
    tabs = [(int((y * x)[st == k].sum()), int((y * (1 - x))[st == k].sum()),
             int(((1 - y) * x)[st == k].sum()), int(((1 - y) * (1 - x))[st == k].sum())) for k in np.unique(st)]
    ex = stats.exact_conditional_or(tabs)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    p0 = float(((1 - y) * x).sum() / n0) if n0 else np.nan
    mde = stats.mde_or(n1, n0, p0) if n1 and n0 else np.nan
    yd, xd = np.array([r["case"] for r in D]), np.array([r[f] for r in D])
    cl = stats.clogit(yd, xd, [r["episode"] for r in D])
    clp = float(2 * sps.norm.sf(abs(cl["beta"] / cl["se"]))) if np.isfinite(cl["se"]) else np.nan
    return dict(feature=f, spells=len(S), cases=n1, controls=n0,
                exposed_cases=int((y * x).sum()), exposed_controls=int(((1 - y) * x).sum()),
                strata=len(tabs), score=score, perm_p=p, B=B, seed=seed,
                exact_or=ex["or_"], exact_lo=ex["lo"], exact_hi=ex["hi"], exact_p=ex["p"],
                informative_strata=ex["n_informative"], mde_or_80=mde, p_exposed_controls=p0,
                reading=_reading(ex["lo"], ex["hi"]),
                decisions=len(D), decision_cases=int(yd.sum()), clogit_or=cl["or_"], clogit_lo=cl["ci_lo"],
                clogit_hi=cl["ci_hi"], clogit_p=clp, clogit_strata=cl["n_strata"], clogit_separated=cl["separated"])


def feature_tests(S, D, B=B, seed=SEED):
    """Family {a, b, d, e} x {pinch} per tier over messaging protocols, before the
    episode's first fatal decision; Holm within the tier. Strong tier is primary."""
    out = []
    for tier in ("strong", "weak"):
        sel = lambda rs: [r for r in rs if r["tier"] == tier and r["kind"] == "pinch"
                          and r["proto"] in MESSAGING and not r["post_fatal"]]
        s, d = sel(S), sel(D)
        if not s:
            continue
        rows = [dict(tier=tier, kind="pinch", primary=tier == "strong", **feature_test(s, d, f, B, seed))
                for f in FAMILY]
        for r, ph in zip(rows, stats.holm([r["perm_p"] for r in rows])):
            r["perm_p_holm"] = ph
        out += rows
    return out


# ---------------------------------------------------------------- descriptive

def descriptive(S, D):
    """Per kind x tier x protocol (and the old scripts' prose/rigid groups):
    spells, cases at the first live decision, spells ending in the case,
    decisions, decision cases, and feature counts among case and control spells.
    Forks are descriptive only; the `none` take-rate is descriptive because
    reaching the hazard is post-treatment."""
    out = []
    protos = {p: (p,) for p in corpus.PROTOS} | GROUPS
    for kind in ("fork", "pinch"):
        for tier in ("strong", "weak"):
            for name, ps in protos.items():
                s = [r for r in S if r["kind"] == kind and r["tier"] == tier and r["proto"] in ps]
                d = [r for r in D if r["kind"] == kind and r["tier"] == tier and r["proto"] in ps]
                if not d:
                    continue
                row = dict(kind=kind, tier=tier, proto=name, episodes=len({r["episode"] for r in s}),
                           spells=len(s), spell_cases=sum(r["case"] for r in s),
                           spells_taken=sum(r["spell_taken"] for r in s), take_rate=sum(r["case"] for r in s) / len(s),
                           decisions=len(d), decision_cases=sum(r["case"] for r in d),
                           decisions_declined=sum(1 - r["case"] for r in d),
                           post_fatal_decisions=sum(r["post_fatal"] for r in d))
                if "none" not in ps:
                    for f in FAMILY:
                        row[f"{f}_cases"] = sum(r[f] for r in s if r["case"])
                        row[f"{f}_controls"] = sum(r[f] for r in s if not r["case"])
                out.append(row)
    return out


def notes(S, D, forced):
    suites = collections.Counter(r["suite"] for r in D if r["kind"] == "pinch")
    return [dict(note=n) for n in (
        f"Pinches arise only at 6p: pinch decisions by suite {dict(suites)}.",
        "Forks are descriptive only (too few cases and controls to test).",
        "The none take-rate is descriptive: reaching the hazard is post-treatment.",
        "Feature c (owner resolvable) is test-arm only; spec 5.3's owners-resolvable rule supplies it; None here.",
        f"Forced pinches (every legal move seals a teammate) are not decisions and are left out: {forced}.",
        "Tests use messaging protocols and decisions up to the episode's first fatal decision (scope rule);"
        " the strong tier is primary, the weak tier is reported separately.",
        "Every p-value here is exploratory; Holm within the tier's family {a, b, d, e} x {pinch} is a multiplicity guard.")]


def run(scope, eps, A, B=B, seed=SEED):
    D = [r for ep in eps for r in decision_rows(ep, A)]
    S = spell_rows(D)
    forced = sum(1 for ep in eps for h in corpus.hazards(ep) if h["kind"] == "pinch" and not free(ep, h))
    return dict(failure_spells=S, failure_decisions=D, failure_tests=feature_tests(S, D, B, seed),
                failure_descriptive=descriptive(S, D), failure_notes=notes(S, D, forced))
