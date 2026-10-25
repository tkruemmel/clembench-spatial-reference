"""Spec §5.5: move alignment and acceptance evidence (associational).

    run(scope, eps, A) -> {table name: [row dicts]}

Opportunity. A teammate message names the listener's letter L
- with a direction d in one clause: a G-move pair (`grammar.analyse()["moves"]`)
  in any region, every frame except `negated` and `question` (kind `move`;
  the frame split is the `alignment_frames` table); or
- in a REQUEST slot by L's cell at B_t- (kind `vacate`: any move of L), or by
  the letter without a direction pair (kind `request_letter`, same outcome).
It is counted at the listener's first decision whose context shows the
message (`corpus.exposures`), only if the named move is legal there, and only
for messages before the episode's first fatal decision (§6 scope rule).

Primary rate: kind `move`, not an exact repeat (`dup`), not an echo (the
listener's own latest visible message already states (L, d): a G-move pair or
a PLAN whose displacement has a component along d).

Comparisons (`alignment_contrasts`, all by `stats.paired_contrast`, the
board-paired sign-flip over episode means with unit (model, suite, level,
board), i.e. clustered by board):
- board-matched silent baseline: the `none` episode of the same model, suite,
  level and board; moments at the start of an owner turn, before that
  episode's first fatal decision, with L on the same cell, the same occupancy
  of its 8 neighbours and of the cell two steps along d, and the same momentum
  (direction of L's last move, None before it first moves). Not reported below
  MIN_COVERAGE coverage;
- negative-control exposure: the same instruction written but not visible at
  the listener's decision (dev: its first decision after the message did not
  show it, a retry call or superseded; final: the first decision after it
  rolled off the board), with no visible message naming (L, d) there;
- negative-control outcome: the listener's move at its last decision before
  the message was sent;
- direction specificity: the rate for d against the other legal directions of
  L with the same goal-ward status that no visible message names.

Presenter-observable tier: executed (O1) and, at the presenter's next accepted
turn, L is bound in its view (a visible strict G-loc, a kept prose G-from or a
PLAN pairs L with a cell) and the board shows the move: L bound to its source
cell and that cell empty, or L bound to its current cell away from the source;
final arm also a MOVE BOARD line of the move with L bound to its source or
destination. Non-execution is "no evidence".

REQUEST exchange chain: per REQUEST slot and each distinct cell in it holding
teammate T's object h at B_t-: T's next turn E (moves off the cell) / P (PLAN
names h or PLAN-from is the cell, no vacating) / N / none; the presenter's next
message closes (moves into the cell, or no REQUEST for it) / re-presents (asks
again, cell occupied) / errs (asks again, cell empty) / none. Silent baseline:
h leaving the cell at matched moments (direction-free key).

Teammate-target cell left free: per listener decision (up to the first fatal
one) and legal move entering an unfilled teammate target; status `visible`
(stated, `grammar.target_mentions` strict, in an owner message in view),
`seen_before`, `unseen` (stated, never yet shown to the listener: the negative-
control exposure) or `never`; entering is split into `pass` (leaves at the next
own turn) and `park`.

Episode-level rows (`alignment_episodes`) feed `alignment_cells`
(`stats.cell_estimate`) and the contrasts; counts by model x suite x protocol
are `opportunity_counts`.
"""
import bisect
import collections
import math

import numpy as np

from analysis.messages import corpus, grammar, stats

DIRS = corpus.DIRS
N8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
EXCLUDED = {"negated", "question"}
MIN_COVERAGE = 0.30
STRATA = {
    "all": lambda r: True,
    "nonecho": lambda r: not r["echo"],
    "echo": lambda r: r["echo"],
    "nonecho_gw": lambda r: not r["echo"] and r["goalward"] is True,
    "nonecho_ngw": lambda r: not r["echo"] and r["goalward"] is False,
    "echo_gw": lambda r: r["echo"] and r["goalward"] is True,
    "echo_ngw": lambda r: r["echo"] and r["goalward"] is False,
}


# ---------------------------------------------------------------- small helpers

def _keys(ep):
    m = ep["meta"]
    return dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], proto=m["proto"],
                board=m["board"], episode=m["id"])


def _go(c, d, k=1):
    return (c[0] + k * DIRS[d][0], c[1] + k * DIRS[d][1])


def _occ8(occ, c):
    return tuple((c[0] + dr, c[1] + dc) in occ for dr, dc in N8)


def _near(ep, b, seat, c):
    """Teammates' objects within two cells (Manhattan) of c."""
    return sum(ep["owner"][o] != seat and abs(p[0] - c[0]) + abs(p[1] - c[1]) <= 2 for o, p in b.items())


def _home(ep, b):
    return sum(b[o] == ep["targets"][o] for o in b)


def _momentum(ep):
    """Per turn index: object -> direction of its last move before that turn."""
    out, last = [], {}
    for t in ep["turns"]:
        out.append(dict(last))
        if not t["done"]:
            last[t["obj"]] = t["dir"]
    return out


def _states(an, slots):
    """(letter, d) pairs a message states, by source: `text` = G-move pairs outside
    the negated and question frames, `plan` = PLAN directions (a component of
    PLAN-to minus PLAN-from)."""
    out = dict(text={(m["letter"], m["dir"]) for m in an["moves"] if m["frame"] not in EXCLUDED}, plan=set())
    if slots:
        fr, to = slots["plan_from"], slots["plan_to"]
        out["plan"] = {(slots["plan_obj"], d) for d, (dr, dc) in DIRS.items()
                       if dr * (to[0] - fr[0]) + dc * (to[1] - fr[1]) > 0}
    return out


def _moved(t, L, d):
    """Did turn t move L (in d; any direction when d is None)?"""
    return not t["done"] and t["obj"] == L and (d is None or t["dir"] == d)


def _legal(ep, b, L, d):
    lg = corpus.legal(ep, b, [L])
    return bool(lg) if d is None else (L, d) in lg


def _named(ep, an, t):
    """What message t names for listeners: [(listener, L, d, kind, region, frame)]; d None = vacate."""
    S, owner = t["seat"], ep["owner"]
    out = [(owner[m["letter"]], m["letter"], m["dir"], "move", m["region"], m["frame"])
           for m in an["moves"] if owner.get(m["letter"], S) != S]
    s = t["slots"]
    if s and not s["request_none"]:
        at = {c: o for o, c in t["board"].items()}
        req_move = {m["letter"] for m in an["moves"] if m["region"] == "request" and m["frame"] not in EXCLUDED}
        seen = set()
        for r, _, c, _ in an["cells"]:
            o = at.get(c)
            if r == "request" and o and owner[o] != S and o not in seen:
                seen.add(o)
                out.append((owner[o], o, None, "vacate", "request", None))
        for r, _, v, _ in an["letters"]:
            if r == "request" and owner.get(v, S) != S and v not in seen and v not in req_move:
                seen.add(v)
                out.append((owner[v], v, None, "request_letter", "request", None))
    return out


# ---------------------------------------------------------------- baseline index

def baseline_index(ep):
    """None episode -> {(obj, cell, occ8, momentum): [moment]} over the start of
    every owner turn before the first fatal decision. A moment is
    (ply, moved dir or None, two-step occupancy per DIRS order, objects home, teammates near)."""
    idx = collections.defaultdict(list)
    ff = corpus.first_fatal(ep)
    for t, mom in zip(ep["turns"], _momentum(ep)):
        if ff is not None and t["ply"] >= ff:
            break
        b = t["board"]
        occ, home = set(b.values()), _home(ep, b)
        for o in ep["objs"][t["seat"]]:
            c = b[o]
            ahead = tuple(_go(c, d, 2) in occ for d in DIRS)
            idx[(o, c, _occ8(occ, c), mom.get(o))].append(
                (t["ply"], t["dir"] if _moved(t, o, None) else None, ahead, home, _near(ep, b, t["seat"], c)))
    return idx


def match(bidx, b, L, mom, d):
    """Matched silent moments for L on board b with momentum mom; d None drops the two-step cell."""
    if bidx is None:
        return []
    occ, c = set(b.values()), b[L]
    recs = bidx.get((L, c, _occ8(occ, c), mom), [])
    if d is not None:
        k, want = list(DIRS).index(d), _go(c, d, 2) in occ
        recs = [r for r in recs if r[2][k] == want]
    return recs


def _base_fields(ep, recs, b, L, d, seat, ply):
    k = sum(r[1] == d if d is not None else r[1] is not None for r in recs)
    n = len(recs)
    return dict(base_n=n, base_k=k, base_rate=k / n if n else None, covered=n > 0,
                bal_ply=ply, bal_home=_home(ep, b), bal_near=_near(ep, b, seat, b[L]),
                base_ply=np.mean([r[0] for r in recs]) if n else None,
                base_home=np.mean([r[3] for r in recs]) if n else None,
                base_near=np.mean([r[4] for r in recs]) if n else None)


# ---------------------------------------------------------------- per-episode extraction

class _Ep:
    """Per-episode precomputation shared by the three extractors."""

    def __init__(self, ep, A, bidx):
        self.ep, self.A, self.bidx = ep, A, bidx
        self.T, self.eid, self.keys = ep["turns"], ep["meta"]["id"], _keys(ep)
        self.dec = collections.defaultdict(list)
        for i, t in enumerate(self.T):
            self.dec[t["seat"]].append(i)
        self.mom = _momentum(ep)
        self.expo = corpus.exposures(ep)
        self.ff = corpus.first_fatal(ep)
        self.vis = [set(t["vis"]) for t in self.T]
        self._vn = {}

    def an(self, p):
        return self.A.get((self.eid, p))

    def after(self, seat, i):
        """First turn index of seat after turn index i, or None."""
        d = self.dec[seat]
        k = bisect.bisect_right(d, i)
        return d[k] if k < len(d) else None

    def before(self, seat, i):
        d = self.dec[seat]
        k = bisect.bisect_left(d, i)
        return d[k - 1] if k else None

    def named_for(self, i):
        """(L, d) named for the decider of turn i by teammate messages in its view (d None: vacate kinds)."""
        if i not in self._vn:
            seat, out = self.T[i]["seat"], set()
            for p in self.T[i]["vis"]:
                if self.T[p]["seat"] != seat and self.an(p):
                    out |= {(L, d) for T, L, d, kind, _, fr in _named(self.ep, self.an(p), self.T[p])
                            if T == seat and fr not in EXCLUDED}
            self._vn[i] = out
        return self._vn[i]

    def echo(self, i, L, d):
        """Source ('text' before 'plan') by which the listener's own latest visible
        message states (L, d) (any move of L when d is None), or None."""
        own = [p for p in self.T[i]["vis"] if self.T[p]["seat"] == self.T[i]["seat"] and self.an(p)]
        if own:
            st = _states(self.an(max(own)), self.T[max(own)]["slots"])
            for src in ("text", "plan"):
                if (L, d) in st[src] if d is not None else any(x == L for x, _ in st[src]):
                    return src
        return None

    def bound(self, k, L):
        """Cells L is paired with in the messages shown at turn k."""
        out = set()
        for p in self.T[k]["vis"]:
            an = self.an(p)
            if not an:
                continue
            out |= {x["cell"] for x in an["loc"] if x["letter"] == L and grammar.kept(x["frames"], "strict")}
            out |= {x["cell"] for x in an["frm"] if x["letter"] == L and not x["frames"]}
            s = self.T[p]["slots"]
            if s and s["plan_obj"] == L:
                out |= {s["plan_from"], s["plan_to"]}
        return out

    def observable(self, presenter, i, L):
        """Presenter-observable execution of turn i's move of L (module doc)."""
        k = self.after(presenter, i)
        if k is None:
            return False
        t, b = self.T[i], self.T[k]["board"]
        bnd = self.bound(k, L)
        via_src = t["src"] in bnd and t["src"] not in set(b.values())
        via_cur = b[L] != t["src"] and b[L] in bnd
        via_mb = (self.ep["meta"]["arm"] == "final" and bool(bnd & {t["src"], t["dst"]})
                  and any(p == i for p, _, _, _ in corpus.move_board(self.ep, k)))
        return via_src or via_cur or via_mb


def opportunities(X):
    """Opportunity rows of one messaging episode (exposures and negative-control
    exposures) and the counts of what was filtered out."""
    ep, T, rows, cnt = X.ep, X.T, [], collections.Counter()
    arm = ep["meta"]["arm"]
    for q, tq in enumerate(T):
        an = X.an(q)
        if not tq["is_msg"] or an is None:
            continue
        groups = collections.OrderedDict()
        for lst, L, d, kind, region, frame in _named(ep, an, tq):
            g = groups.setdefault((lst, L, d, kind), dict(regions=set(), frames=set()))
            g["regions"].add(region)
            if frame:
                g["frames"].add(frame)
        for (lst, L, d, kind), g in groups.items():
            cnt["named"] += 1
            if kind == "move" and g["frames"] <= EXCLUDED:
                cnt["frame_excluded"] += 1
                continue
            if X.ff is not None and q >= X.ff:
                cnt["post_fatal"] += 1
                continue
            first = X.after(lst, q)
            if first is None:
                cnt["no_decision"] += 1
                continue
            events = []
            e = X.expo.get((q, lst))
            if e is not None:
                events.append((e, "visible"))
            if arm == "dev" and e != first:
                events.append((first, "nc_retry" if T[first]["retry"] else "nc_superseded"))
            elif arm == "final" and e is not None:
                i = e
                while i is not None and q in X.vis[i]:
                    i = X.after(lst, i)
                if i is not None:
                    events.append((i, "nc_rolled_off"))
            for i, how in events:
                b = T[i]["board"]
                if not _legal(ep, b, L, d):
                    cnt["illegal_" + how] += 1
                    continue
                if how != "visible" and (L, d) in X.named_for(i):
                    cnt["contaminated_" + how] += 1
                    continue
                rows.append(_opp_row(X, q, tq, lst, L, d, kind, g, i, how))
    return rows, cnt


def _opp_row(X, q, tq, lst, L, d, kind, g, i, how):
    ep, T = X.ep, X.T
    t, b = T[i], T[i]["board"]
    gw = corpus.goalward(ep, L, b[L], d) if d else None
    nxt, prev = X.after(lst, i), X.before(lst, i)
    o1 = _moved(t, L, d)
    r = dict(X.keys, msg_ply=q, speaker=tq["seat"], listener=lst, letter=L, dir=d, kind=kind,
             region="+".join(sorted(g["regions"])),
             src="prose" if g["regions"] == {"prose"} else "slots" if "prose" not in g["regions"] else "both",
             frames="+".join(sorted(g["frames"])), dup=tq["dup"], exposure=how, decision_ply=i,
             retry_decision=t["retry"], done_decision=t["done"], lag=i - q, goalward=gw,
             echo_src=X.echo(i, L, d),
             restated=prev is not None and (L, d) in X.named_for(prev),
             o1=o1, o2=o1 or (nxt is not None and _moved(T[nxt], L, d)))
    r["echo"] = r["echo_src"] is not None
    r.update(_base_fields(ep, match(X.bidx, b, L, X.mom[i].get(L), d), b, L, d, lst, i))
    # negative-control outcome: the listener's last decision before the message
    p = X.before(lst, q)
    r.update(nc_prev_ply=p, nc_prev_legal=p is not None and _legal(ep, T[p]["board"], L, d),
             nc_prev_o1=p is not None and _moved(T[p], L, d))
    # direction specificity: unnamed legal directions with the same goal-ward status
    if d is not None:
        alts = [a for o, a in corpus.legal(ep, b, [L]) if a != d and corpus.goalward(ep, L, b[L], a) == gw
                and (L, a) not in X.named_for(i)]
        r.update(alt_n=len(alts), alt_taken=any(_moved(t, L, a) for a in alts))
    else:
        r.update(alt_n=0, alt_taken=False)
    r["observable"] = o1 and X.observable(tq["seat"], i, L)
    r["evidence"] = "observable" if r["observable"] else "executed_unobservable" if o1 else "no_evidence"
    return r


def request_chain(X):
    """REQUEST exchange chain (spec §5.5): one row per REQUEST slot and named cell
    that holds a teammate's object at B_t-."""
    ep, T, rows = X.ep, X.T, []
    for q, tq in enumerate(T):
        s, an = tq["slots"], X.an(q)
        if not s or s["request_none"] or an is None or (X.ff is not None and q >= X.ff):
            continue
        at = {c: o for o, c in tq["board"].items()}
        S = tq["seat"]
        for c in dict.fromkeys(c for r, _, c, _ in an["cells"] if r == "request"):
            h = at.get(c)
            if not h or ep["owner"][h] == S:
                continue
            hs = ep["owner"][h]
            j = X.after(hs, q)
            if j is None:
                holder_next = "none"
            elif not T[j]["done"] and T[j]["src"] == c:
                holder_next = "E"
            elif T[j]["slots"] and (T[j]["slots"]["plan_obj"] == h or T[j]["slots"]["plan_from"] == c):
                holder_next = "P"
            else:
                holder_next = "N"
            k = X.after(S, q)
            if k is None or not T[k]["is_msg"]:
                presenter_next = "none"
            else:
                sk, ak = T[k]["slots"], X.an(k)
                again = bool(sk and not sk["request_none"] and ak
                             and any(r == "request" and x == c for r, _, x, _ in ak["cells"]))
                held = c in set(T[k]["board"].values())
                presenter_next = ("closes" if T[k]["dst"] == c or not again else
                                  "re-presents" if held else "errs")
            r = dict(X.keys, msg_ply=q, speaker=S, cell=corpus.cell(c), holder=h, holder_seat=hs,
                     holder_ply=j, holder_visible=j is not None and q in X.vis[j],
                     holder_can_vacate=j is not None and _legal(ep, T[j]["board"], h, None),
                     holder_next=holder_next, presenter_next_ply=k, presenter_next=presenter_next)
            if j is not None:
                r.update(_base_fields(ep, match(X.bidx, T[j]["board"], h, X.mom[j].get(h), None),
                                      T[j]["board"], h, None, hs, j))
            rows.append(r)
    return rows


def target_cells(X):
    """Teammate-target cell left free: one row per listener decision and legal
    move entering an unfilled teammate target, at decisions up to the first
    fatal one (so every visible message is pre-fatal)."""
    ep, T, rows = X.ep, X.T, []
    tg, owner = ep["targets"], ep["owner"]
    tgt_at = {c: v for v, c in tg.items()}
    stated = collections.defaultdict(list)  # object -> plies of its owner's messages stating its target
    for q, tq in enumerate(T):
        an = X.an(q)
        if tq["is_msg"] and an:
            for v, m in grammar.target_mentions(an, {v: tg[v] for v in ep["objs"][tq["seat"]]}).items():
                if m["stated"]:
                    stated[v].append(q)
    named_by = {}
    for i, t in enumerate(T):
        if X.ff is not None and i > X.ff:
            break
        seat, b = t["seat"], t["board"]
        for o, d in corpus.legal(ep, b, ep["objs"][seat]):
            dst = _go(b[o], d)
            v = tgt_at.get(dst)
            if v is None or owner[v] == seat:
                continue
            qs = [q for q in stated[v] if q < i]
            if any(q in X.vis[i] for q in qs):
                status = "visible"
            elif any(X.expo.get((q, seat), i) < i for q in qs):
                status = "seen_before"
            else:
                status = "unseen" if qs else "never"
            key = (i, v)
            if key not in named_by:
                named_by[key] = any(grammar.target_mentions(X.an(p), {v: tg[v]})[v]["named"]
                                    for p in t["vis"] if T[p]["seat"] == owner[v] and X.an(p))
            entered = _moved(t, o, d)
            nxt = X.after(seat, i)
            r = dict(X.keys, decision_ply=i, listener=seat, obj=o, dir=d, target_of=v, cell=corpus.cell(dst),
                     status=status, named_visible=named_by[key], goalward=corpus.goalward(ep, o, b[o], d),
                     entered=entered,
                     after=None if not entered else "pass" if nxt is not None and _moved(T[nxt], o, None) else "park")
            r.update(_base_fields(ep, match(X.bidx, b, o, X.mom[i].get(o), d), b, o, d, seat, i))
            p = X.before(seat, min(qs)) if status == "visible" else None
            if p is not None:
                pb = T[p]["board"]
                ins = [m for m in corpus.legal(ep, pb, ep["objs"][seat]) if _go(pb[m[0]], m[1]) == dst]
                r.update(nc_prev_ply=p, nc_prev_legal=bool(ins), nc_prev_o1=any(_moved(T[p], *m) for m in ins))
            else:
                r.update(nc_prev_ply=None, nc_prev_legal=False, nc_prev_o1=False)
            rows.append(r)
    return rows


# ---------------------------------------------------------------- summaries

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _smd(a, b):
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    s = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / s) if s else 0.0


def _group(rows, keys):
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def _pooled(keys, k):
    """Grouping-key dict with 'all' for the dimensions pooled over."""
    out = dict(model="all", tier="all", suite="all", level="all", proto="all")
    out.update(zip(keys, k))
    if out["model"] != "all":
        out["tier"] = corpus.TIER[out["model"]]
    return out


GROUPINGS = (("model", "suite", "proto"), ("tier", "suite", "proto"), ("tier", "proto"), ("tier",))


def summary(rows, outcome="o1"):
    """Counts and rates per grouping, kind, exposure, source and stratum."""
    out = []
    for keys in GROUPINGS:
        for k, rs in _group(rows, keys + ("kind", "exposure")).items():
            for src in ("all", "prose", "slots", "both"):
                rsrc = [r for r in rs if not r["dup"] and (src == "all" or r["src"] == src)]
                for name, f in STRATA.items():
                    sub = [r for r in rsrc if f(r)]
                    if not sub:
                        continue
                    cov = [r for r in sub if r["covered"]]
                    alt = [r for r in sub if r["alt_n"]]
                    out.append(dict(_pooled(keys, k[:len(keys)]), kind=k[-2], exposure=k[-1], src=src, stratum=name,
                                    n=len(sub), n_dup_excluded=sum(r["dup"] for r in rs if f(r)),
                                    o1=sum(r[outcome] for r in sub), o1_rate=_mean([r[outcome] for r in sub]),
                                    o2_rate=_mean([r["o2"] for r in sub]),
                                    observable_rate=_mean([r["observable"] for r in sub]),
                                    covered=len(cov), coverage=len(cov) / len(sub),
                                    o1_covered=_mean([r[outcome] for r in cov]),
                                    o1_uncovered=_mean([r[outcome] for r in sub if not r["covered"]]),
                                    base_rate_covered=_mean([r["base_rate"] for r in cov]),
                                    nc_prev_n=sum(r["nc_prev_legal"] for r in sub),
                                    nc_prev_rate=_mean([r["nc_prev_o1"] for r in sub if r["nc_prev_legal"]]),
                                    alt_n=len(alt), o1_alt_subset=_mean([r[outcome] for r in alt]),
                                    alt_rate=_mean([r["alt_taken"] / r["alt_n"] for r in alt]),
                                    restated_share=_mean([r["restated"] for r in sub]),
                                    echo_share=_mean([r["echo"] for r in rsrc]) if name == "all" else None))
    return out


def coverage(rows, outcome="o1"):
    """Primary-stratum coverage of the silent baseline per cell, O1 covered vs
    uncovered, and balance (opportunity vs mean of its matched moments)."""
    out = []
    prim = [r for r in rows if r["exposure"] == "visible" and not r["dup"] and not r["echo"]]
    for keys in GROUPINGS:
        for k, rs in _group(prim, keys + ("kind",)).items():
            cov = [r for r in rs if r["covered"]]
            row = dict(_pooled(keys, k[:len(keys)]), kind=k[-1], n=len(rs), covered=len(cov),
                       coverage=len(cov) / len(rs), contrast_reported=len(cov) / len(rs) >= MIN_COVERAGE,
                       o1_covered=_mean([r[outcome] for r in cov]),
                       o1_uncovered=_mean([r[outcome] for r in rs if not r["covered"]]),
                       base_rate=_mean([r["base_rate"] for r in cov]))
            for f in ("ply", "home", "near"):
                a, b = [r["bal_" + f] for r in cov], [r["base_" + f] for r in cov]
                row.update({f"bal_{f}_opp": _mean(a), f"bal_{f}_base": _mean(b), f"bal_{f}_smd": _smd(a, b)})
            out.append(row)
    return out


def _rate(rs, f):
    return _mean([f(r) for r in rs]) if rs else None


def episodes(eps, opp, tgt, chain):
    """One row per messaging episode: counts and rates of every stratum, the
    episode-level values that cells and contrasts are built from."""
    by = _group(opp, ("episode",))
    bt = _group(tgt, ("episode",))
    bc = _group(chain, ("episode",))
    out = []
    for ep in eps:
        if ep["meta"]["proto"] == "none":
            continue
        rs = [r for r in by.get((ep["meta"]["id"],), []) if not r["dup"]]
        vis = [r for r in rs if r["exposure"] == "visible"]
        mv = [r for r in vis if r["kind"] == "move"]
        prim = [r for r in mv if not r["echo"]]
        cov = [r for r in prim if r["covered"]]
        alt = [r for r in prim if r["alt_n"]]
        prev = [r for r in prim if r["nc_prev_legal"]]
        nc = [r for r in rs if r["exposure"] != "visible" and r["kind"] == "move" and not r["echo"]]
        vac = [r for r in vis if r["kind"] != "move" and not r["echo"]]
        tg = bt.get((ep["meta"]["id"],), [])
        row = dict(_keys(ep), success=ep["meta"]["success"],
                   n_opp=len(prim), n_o1=sum(r["o1"] for r in prim), o1=_rate(prim, lambda r: r["o1"]),
                   n_echo=sum(r["echo"] for r in mv), o1_echo=_rate([r for r in mv if r["echo"]], lambda r: r["o1"]),
                   o1_gw=_rate([r for r in prim if r["goalward"]], lambda r: r["o1"]),
                   o1_ngw=_rate([r for r in prim if not r["goalward"]], lambda r: r["o1"]),
                   o2=_rate(prim, lambda r: r["o2"]),
                   observable=_rate(prim, lambda r: r["observable"]), n_observable=sum(r["observable"] for r in prim),
                   n_covered=len(cov), obs_covered=_rate(cov, lambda r: r["o1"]),
                   base_covered=_rate(cov, lambda r: r["base_rate"]),
                   n_alt=len(alt), o1_altset=_rate(alt, lambda r: r["o1"]),
                   alt=_rate(alt, lambda r: r["alt_taken"] / r["alt_n"]),
                   n_prev=len(prev), o1_prevset=_rate(prev, lambda r: r["o1"]),
                   prev=_rate(prev, lambda r: r["nc_prev_o1"]),
                   n_nc=len(nc), o1_nc=_rate(nc, lambda r: r["o1"]),
                   n_vacate=len(vac), o1_vacate=_rate(vac, lambda r: r["o1"]),
                   n_request_chain=len(bc.get((ep["meta"]["id"],), [])))
        for st in ("visible", "seen_before", "unseen", "never"):
            s = [r for r in tg if r["status"] == st]
            row[f"tgt_n_{st}"] = len(s)
            row[f"tgt_entered_{st}"] = _rate(s, lambda r: r["entered"])
        tcov = [r for r in tg if r["status"] == "visible" and r["covered"]]
        row.update(tgt_obs_covered=_rate(tcov, lambda r: r["entered"]),
                   tgt_base_covered=_rate(tcov, lambda r: r["base_rate"]))
        out.append(row)
    return out


CELL_METRICS = ("o1", "o1_echo", "o1_gw", "o1_ngw", "observable", "o1_vacate", "o1_nc",
                "tgt_entered_visible", "tgt_entered_never")
# (name, observed column, comparison column, rows needed) for alignment_contrasts
CONTRASTS = (("silent_baseline", "obs_covered", "base_covered", "n_covered"),
             ("nc_exposure", "o1", "o1_nc", "n_nc"),
             ("nc_outcome_prev", "o1_prevset", "prev", "n_prev"),
             ("direction_specificity", "o1_altset", "alt", "n_alt"),
             ("target_silent_baseline", "tgt_obs_covered", "tgt_base_covered", "tgt_n_visible"))


def cells(ep_rows):
    """stats.cell_estimate of the episode-level metrics per model x suite x protocol,
    pooled over levels and per level."""
    out = []
    for (m, s, p), rs in sorted(_group(ep_rows, ("model", "suite", "proto")).items()):
        for lv in ["all"] + sorted({r["level"] for r in rs}):
            sub = [r for r in rs if lv == "all" or r["level"] == lv]
            for met in CELL_METRICS:
                vals = [r[met] for r in sub if r[met] is not None]
                if not vals:
                    continue
                est = stats.cell_estimate(vals, [r["level"] for r in sub if r[met] is not None])
                out.append(dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, level=lv, metric=met, **est))
    return out


def contrasts(ep_rows, cov_rows, tgt_cov):
    """Board-paired sign-flip (stats.paired_contrast) of each comparison per
    model x suite x protocol, over episodes, unit (model, suite, level, board)."""
    cov = {(r["model"], r["suite"], r["proto"]): r["coverage"] for r in cov_rows
           if r["kind"] == "move" and r["model"] != "all" and r["suite"] != "all"}
    out = []
    for (m, s, p), rs in sorted(_group(ep_rows, ("model", "suite", "proto")).items()):
        for name, a, b, need in CONTRASTS:
            c = cov.get((m, s, p)) if name == "silent_baseline" else tgt_cov.get((m, s, p)) \
                if name == "target_silent_baseline" else None
            row = dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, comparison=name,
                       method="board-paired sign-flip (stats.paired_contrast), unit model x suite x level x board",
                       coverage=c, episodes=sum(1 for r in rs if r[need]))
            if c is not None and c < MIN_COVERAGE:
                row.update(reported=False, note=f"coverage below {MIN_COVERAGE:.0%}")
                out.append(row)
                continue
            long = [dict(r, arm=arm, v=r[col] if r[need] else None) for r in rs for arm, col in (("a", a), ("b", b))]
            res = stats.paired_contrast(long, "v", "arm", "a", "b")
            row.update(reported=res["pairs"] >= 2, mean_observed=_mean([r[a] for r in rs if r[need]]),
                       mean_comparison=_mean([r[b] for r in rs if r[need] and r[b] is not None]),
                       diff=res["mean"], p=res["p"], exact=res["exact"], pairs=res["pairs"],
                       missing=res["missing"], note=res["note"])
            out.append(row)
    return out


def distribution(rows, unit_keys=(("episode",), ("episode", "listener"))):
    """Per-unit O1 rates of the primary stratum as distributions (Jaques et al. 2019: listening is sparse)."""
    prim = [r for r in rows if r["exposure"] == "visible" and not r["dup"] and r["kind"] == "move"]
    out = []
    for uk in unit_keys:
        for name in ("nonecho", "echo"):
            sub = [r for r in prim if STRATA[name](r)]
            per = {k: np.mean([r["o1"] for r in rs]) for k, rs in _group(sub, ("model", "suite", "proto") + uk).items()}
            for (m, s, p), vs in _group([dict(model=k[0], suite=k[1], proto=k[2], v=v) for k, v in per.items()],
                                        ("model", "suite", "proto")).items():
                v = np.array([x["v"] for x in vs])
                q = np.percentile(v, [10, 25, 50, 75, 90])
                out.append(dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, unit="+".join(uk), stratum=name,
                                n_units=v.size, mean=float(v.mean()), p10=q[0], p25=q[1], median=q[2], p75=q[3],
                                p90=q[4], share_zero=float((v == 0).mean()), share_one=float((v == 1).mean())))
    return out


def frames(eps, A):
    """Frame split of G-move pairs naming a teammate's letter (all frames, before
    exposure and legality) per model x suite x protocol and region."""
    c = collections.Counter()
    for ep in eps:
        if ep["meta"]["proto"] == "none":
            continue
        for t in ep["turns"]:
            an = A.get((ep["meta"]["id"], t["ply"]))
            if an:
                for lst, L, d, kind, region, frame in _named(ep, an, t):
                    if kind == "move":
                        c[(ep["meta"]["model"], ep["meta"]["suite"], ep["meta"]["proto"], region, frame)] += 1
    tot = collections.Counter()
    for (m, s, p, rg, f), n in c.items():
        tot[(m, s, p, rg)] += n
    return [dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, region=rg, frame=f, n=n, share=n / tot[(m, s, p, rg)],
                 included=f not in EXCLUDED) for (m, s, p, rg, f), n in sorted(c.items())]


def counts(eps, opp, cnts, chain, tgt):
    """Opportunity counts by model x suite x protocol (for the cross-arm report, §9.5)."""
    out = []
    by, bt, bc = _group(opp, ("model", "suite", "proto")), _group(tgt, ("model", "suite", "proto")), \
        _group(chain, ("model", "suite", "proto"))
    for k in sorted({(e["meta"]["model"], e["meta"]["suite"], e["meta"]["proto"]) for e in eps
                     if e["meta"]["proto"] != "none"}):
        rs = by.get(k, [])
        row = dict(model=k[0], tier=corpus.TIER[k[0]], suite=k[1], proto=k[2], **{f"n_{x}": n for x, n in cnts[k].items()})
        for exp in ("visible", "nc_retry", "nc_superseded", "nc_rolled_off"):
            for kind in ("move", "vacate", "request_letter"):
                row[f"opp_{exp}_{kind}"] = sum(r["exposure"] == exp and r["kind"] == kind and not r["dup"] for r in rs)
        mv = [r for r in rs if r["exposure"] == "visible" and r["kind"] == "move" and not r["dup"]]
        row.update(opp_dup=sum(r["dup"] for r in rs), opp_primary=sum(not r["echo"] for r in mv),
                   opp_echo=sum(r["echo"] for r in mv), opp_goalward=sum(bool(r["goalward"]) for r in mv),
                   opp_restated=sum(r["restated"] for r in mv), request_chain=len(bc.get(k, [])),
                   **{f"target_{st}": sum(r["status"] == st for r in bt.get(k, []))
                      for st in ("visible", "seen_before", "unseen", "never")})
        out.append(row)
    return out


def _summ_chain(chain):
    out = []
    for keys in GROUPINGS:
        for k, rs in _group(chain, keys).items():
            row = dict(_pooled(keys, k), n=len(rs))
            for f, vals in (("holder_next", ("E", "P", "N", "none")),
                            ("presenter_next", ("closes", "re-presents", "errs", "none"))):
                row.update({f"{f}_{v}": sum(r[f] == v for r in rs) for v in vals})
            cov = [r for r in rs if r.get("covered")]
            row.update(coverage=len(cov) / len(rs), E_rate_covered=_mean([r["holder_next"] == "E" for r in cov]),
                       base_rate_covered=_mean([r["base_rate"] for r in cov]),
                       E_rate_visible=_mean([r["holder_next"] == "E" for r in rs if r["holder_visible"]]))
            out.append(row)
    return out


def _summ_target(tgt):
    out = []
    for keys in GROUPINGS:
        for k, rs in _group(tgt, keys + ("status",)).items():
            cov = [r for r in rs if r["covered"]]
            ent = [r for r in rs if r["entered"]]
            prev = [r for r in rs if r["nc_prev_legal"]]
            out.append(dict(_pooled(keys, k[:-1]), status=k[-1], n=len(rs), entered=len(ent),
                            entered_rate=len(ent) / len(rs), pass_n=sum(r["after"] == "pass" for r in ent),
                            park_n=sum(r["after"] == "park" for r in ent),
                            entered_rate_named=_mean([r["entered"] for r in rs if r["named_visible"]]),
                            entered_rate_goalward=_mean([r["entered"] for r in rs if r["goalward"]]),
                            entered_rate_not_goalward=_mean([r["entered"] for r in rs if not r["goalward"]]),
                            covered=len(cov), coverage=len(cov) / len(rs),
                            contrast_reported=len(cov) / len(rs) >= MIN_COVERAGE,
                            entered_covered=_mean([r["entered"] for r in cov]),
                            base_rate_covered=_mean([r["base_rate"] for r in cov]),
                            nc_prev_n=len(prev), nc_prev_rate=_mean([r["nc_prev_o1"] for r in prev])))
    return out


# ---------------------------------------------------------------- entry point

def run(scope, eps, A):
    nones = {(e["meta"]["model"], e["meta"]["suite"], e["meta"]["level"], e["meta"]["board"]): e
             for e in eps if e["meta"]["proto"] == "none"}
    bidx = {}
    opp, chain, tgt, cnts = [], [], [], collections.defaultdict(collections.Counter)
    for ep in eps:
        m = ep["meta"]
        if m["proto"] == "none":
            continue
        k = (m["model"], m["suite"], m["level"], m["board"])
        if k not in bidx:
            bidx[k] = baseline_index(nones[k]) if k in nones else None
        X = _Ep(ep, A, bidx[k])
        rs, c = opportunities(X)
        opp += rs
        cnts[(m["model"], m["suite"], m["proto"])].update(c)
        chain += request_chain(X)
        tgt += target_cells(X)
    ep_rows = episodes(eps, opp, tgt, chain)
    cov = coverage(opp)
    tsum = _summ_target(tgt)
    tgt_cov = {(r["model"], r["suite"], r["proto"]): r["coverage"] for r in tsum
               if r["status"] == "visible" and "all" not in (r["model"], r["suite"], r["proto"])}
    return dict(
        alignment_opportunities=opp,
        alignment_episodes=ep_rows,
        alignment_summary=summary(opp),
        alignment_coverage=cov,
        alignment_cells=cells(ep_rows),
        alignment_contrasts=contrasts(ep_rows, cov, tgt_cov),
        alignment_distribution=distribution(opp),
        alignment_frames=frames(eps, A),
        alignment_request_chain=chain,
        alignment_request_chain_summary=_summ_chain(chain),
        alignment_target=tgt,
        alignment_target_summary=tsum,
        opportunity_counts=counts(eps, opp, cnts, chain, tgt),
    )
