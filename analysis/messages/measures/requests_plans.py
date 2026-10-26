"""Extension E1 (REQUEST) and E2 (PLAN): thesis/message_analysis_extension.md §2, §5.

    run(scope, eps, A) -> {table name: [row dicts]}

Structured and hybrid episodes, slots only (prose is out, extension §2 E2.1
and §5), except the hybrid shadow-request bound of §5. Notation as in the
extension: B_t- is turn["board"], d_W is BFS over non-wall cells, d_t is BFS
with every other object on that board as an obstacle, H = corpus.ROUNDS = 4.

Scope rule (spec §6): every row is a message (a ply for conflicts) before the
episode's first fatal decision (`corpus.first_fatal`); outcomes may fall after
it. Message-level rates drop exact repeats within a seat (`dup`); the
state-based measures (blocker spells, persistence, conflicts) use every message.

E1, per distinct G-cell c of a filled REQUEST, at B_t-
- class V (an active teammate's object), K (empty), F (a finished teammate's
  object), S (the speaker's own), W (wall or off-grid). A filled REQUEST with
  no G-cell is the class "—": `e2_plans.n_req_cells == 0`, with whether it
  names a letter or a direction.
- reference error `ref_err` = share F+S+W among the cells; `req_err` = share
  of the REQUESTs with a cell that name an F, S or W cell.
- relevance: c on a shortest d_W path PLAN-from -> PLAN-to of the same
  message; `relevant_own`: on an unfinished own object's shortest path to its
  target. Rates over V and K cells.
- blocker: an active teammate's object on c raises the speaker's d_t on its
  PLAN (PLAN object's cell -> PLAN-to): removing it lowers d_t, or it alone
  (own and finished objects kept) raises d_t over the no-teammate value; the
  second clause catches objects in series in one corridor. A spell is a
  maximal run of the speaker's consecutive messages with c a blocker; recall =
  share of spells whose messages include a REQUEST naming c. Hard: d_t = inf
  and finite without the active teammates' objects. `silent_blocked` = share
  of hard-blocked messages with REQUEST none.
- V compliance via `alignment.request_chain` (holder's next turn E/P/N, the
  presenter's next message closes / re-presents / errs, the board-matched
  silent baseline): `V_next` = E among rows whose REQUEST the holder sees at
  that turn and whose object can move; `V_H` = it moves within H holder turns.
- K compliance `K_kept`: no teammate object enters c before one of the
  requester's objects enters it or the requester's H-th next own turn.
  Censored windows (the episode ends first) and moot requests (the
  requester's own move enters c) are left out. Closure for K over the
  presenter's next message: closes (no REQUEST for c, or it moves into c),
  re-presents (asks again, not passed), errs (asks again after one of its
  objects entered c since it began asking for c).
- useful: the requester's d_t on that PLAN falls across one of the next two
  gaps between its own turns (only teammates move there); also computed for
  silent-blocked messages as a reference (`useful_silent`).
- baselines: (1) unnamed on-path cells, the cells on the PLAN's shortest d_W
  path (PLAN-from excluded) of class V or K at every non-repeat message, named
  in no REQUEST of that message or in view at the outcome decisions, same
  outcome rule, against the requested on-path cells; (2) the board-matched
  none episode, class V only (the spec §5.5 key of the holder object; K has
  no object to match); (3) negative-control exposure, final arm only: the
  holder's (V) or a teammate's (K) first decision after the REQUEST rolled off
  its board with no REQUEST in view naming c, against its first visible
  decision; outcome V the holder moves the object off c, K a move enters c.

E2, per PLAN slot
- say-do: `progress` (moved the PLAN object and lowered its d_W to PLAN-to),
  `same_no_progress`, `other_object`.
- PLAN-to validity, first that applies: wall (or off-grid), own (another own
  object), x (a teammate's object; `to_x_finished`), static (d_W = inf),
  dynamic (d_t = inf), valid. Goal: PLAN-to is the object's target.
  Waypoints: progress (d_W to the target falls), give_way (leaving the
  object's cell frees a cell on an active teammate object's shortest d_W path
  to its target that PLAN-to is not on), other; progress first, `frees` flags
  every waypoint that frees one.
- persistence: a run of a seat's consecutive messages with one (object,
  PLAN-to) ends reached (the object stands on PLAN-to after one of the run's
  turns) > stale (d_t = inf at two consecutive turns) > open (no later
  pre-fatal own message) > forced (a teammate raised d_t between two own turns
  of the run or before the revising turn, or holds PLAN-to then) > unforced.
- conflicts: per ply and pair of active seats that have sent a PLAN, on their
  latest PLANs (the latest message of a seat is in view of every teammate):
  vertex (same PLAN-to) or hold (one PLAN-to holds an object of the other
  seat that the other PLAN does not move). Incidence per pair-ply; a spell
  ends resolved (latency in plies), finished (a seat declared done) or
  censored (episode end or first fatal decision).
- uptake: at a listener's first decision showing a teammate's PLAN, every
  legal move of its objects into the PLAN's announced cells (the shortest d_W
  path minus PLAN-from, or PLAN-to alone when there is no path); `entered` =
  the move is taken (avoidance = 1 - entered). Against the board-matched silent
  rate of the same move (`alignment.match`) and, final arm, the first decision
  after the PLAN rolled off (cells still announced by a PLAN in view dropped).

Episode rows (`rp_episodes`) hold every metric as a mean over its units with
the unit count `<metric>_n`; `rp_cells` (stats.cell_estimate), `rp_baselines`
(observed - comparison per episode, board-paired sign-flip, silent baselines
gated at alignment.MIN_COVERAGE), `rp_proto` (hybrid - structured on the
slots), `rp_summary` (pooled descriptive rates per alignment.GROUPINGS),
`rp_coverage`, `rp_notes`. Every p-value is exploratory.
"""
import collections
import itertools
import math

from analysis.messages import corpus, grammar, stats
from analysis.messages.measures import alignment as al

H = corpus.ROUNDS
INF = math.inf
SLOTS = ("structured", "hybrid")
CLASSES = ("V", "K", "F", "S", "W")
SAYDO = ("progress", "same_no_progress", "other_object")
TO_CLASSES = ("wall", "own", "x", "static", "dynamic", "valid")
WAYPOINTS = ("progress", "give_way", "other")
RUN_ENDS = ("reached", "stale", "open", "forced", "unforced")
CLOSURE = ("closes", "re-presents", "errs")


# ---------------------------------------------------------------- board helpers

def _ok(ep, c):
    return 0 <= c[0] < ep["grid"] and 0 <= c[1] < ep["grid"] and c not in ep["walls"]


def _dw(ep, a, b):
    return corpus.dist_map(ep, b).get(a, INF) if _ok(ep, a) and _ok(ep, b) else INF


def _path(ep, a, b):
    """Cells on a shortest d_W path from a to b, both ends included (empty if none)."""
    if _dw(ep, a, b) == INF:
        return set()
    da, db = corpus.dist_map(ep, a), corpus.dist_map(ep, b)
    return {c for c in da if c in db and da[c] + db[c] == da[b]}


def _dt(ep, b, obj, to, blocked=None):
    """d_t of obj from its cell on board b to `to`; `blocked` defaults to the other objects' cells."""
    if blocked is None:
        blocked = {c for o, c in b.items() if o != obj}
    return corpus.dist_map(ep, b[obj], frozenset(blocked)).get(to, INF)


def _active(ep, seat, ply):
    """Has seat not declared done before ply?"""
    return ep["done_ply"].get(seat, INF) >= ply


def cell_class(ep, b, seat, c, ply):
    """E1 manifest class of cell c in seat's REQUEST at board b (ply's B_t-)."""
    if not _ok(ep, c):
        return "W"
    o = next((o for o, p in b.items() if p == c), None)
    if o is None:
        return "K"
    s = ep["owner"][o]
    return "S" if s == seat else "V" if _active(ep, s, ply) else "F"


def blockers(ep, b, obj, to, seat, ply):
    """(d_t, d_t without the active teammates' objects, blocker cells) of obj -> to on board b."""
    others = {c for o, c in b.items() if o != obj}
    tm = {c for o, c in b.items() if ep["owner"][o] != seat and _active(ep, ep["owner"][o], ply)}
    rest = others - tm
    d_all, d0 = _dt(ep, b, obj, to, others), _dt(ep, b, obj, to, rest)
    if d_all == d0:  # obstacles only raise d_t, so no single teammate object can matter
        return d_all, d0, set()
    return d_all, d0, {c for c in tm if _dt(ep, b, obj, to, others - {c}) < d_all
                       or _dt(ep, b, obj, to, rest | {c}) > d0}


# ---------------------------------------------------------------- per-episode context

class _X(al._Ep):
    """alignment's per-episode context plus REQUEST cells and PLAN cells in view."""

    def __init__(self, ep, A, bidx):
        super().__init__(ep, A, bidx)
        self._req, self._uc, self._vr = {}, {}, {}
        W = grammar.lists()
        self.directive = W["keep"] | W["put"]

    def req(self, p):
        """Distinct G-cells of message p's REQUEST slot (empty when none or no slots)."""
        if p not in self._req:
            s, an = self.T[p]["slots"], self.an(p)
            self._req[p] = [] if not s or s["request_none"] or an is None else \
                list(dict.fromkeys(c for r, _, c, _ in an["cells"] if r == "request"))
        return self._req[p]

    def ucells(self, p):
        """Cells message p's PLAN announces: its shortest d_W path minus PLAN-from, or PLAN-to alone."""
        if p not in self._uc:
            s = self.T[p]["slots"]
            if not s:
                self._uc[p] = set()
            else:
                path, to = _path(self.ep, s["plan_from"], s["plan_to"]), s["plan_to"]
                self._uc[p] = path - {s["plan_from"]} if path else {to} if _ok(self.ep, to) else set()
        return self._uc[p]

    def vis_req(self, i):
        if i not in self._vr:
            self._vr[i] = set().union(*(self.req(p) for p in self.T[i]["vis"]))
        return self._vr[i]

    def vis_plan(self, i, seat):
        return set().union(*(self.ucells(p) for p in self.T[i]["vis"] if self.T[p]["seat"] != seat))

    def decisions(self, q, seat):
        """[(decision, exposure)]: seat's first decision showing message q and, final
        arm, its first decision after q rolled off the board (as alignment.opportunities)."""
        e = self.expo.get((q, seat))
        if e is None:
            return []
        out = [(e, "visible")]
        if self.ep["meta"]["arm"] == "final":
            i = e
            while i is not None and q in self.vis[i]:
                i = self.after(seat, i)
            if i is not None:
                out.append((i, "nc_rolled_off"))
        return out

    def shadow(self, p):
        """Hybrid prose clauses with a keep/put verb naming a cell missing from REQUEST (extension §5)."""
        an = self.an(p)
        if self.ep["meta"]["proto"] != "hybrid" or an is None:
            return set()
        cells = {c for ci in an["clause_info"] if ci["region"] == "prose"
                 and self.directive & {w.lower() for w in ci["text"].split()} for c in ci["cells"]}
        return cells - set(self.req(p))


def plan_info(X, i):
    """PLAN of message i with its d_t, blockers and announced cells."""
    ep, t = X.ep, X.T[i]
    s, b, seat = t["slots"], t["board"], t["seat"]
    o, to = s["plan_obj"], s["plan_to"]
    f = dict(obj=o, fr=s["plan_from"], to=to, own=ep["owner"].get(o) == seat, req=X.req(i),
             filled=not s["request_none"], path=_path(ep, s["plan_from"], to), dt=None, d0=None,
             dt_plus=None, blockers=set(), hard=False, shadow=X.shadow(i))
    if f["own"]:
        f["dt"], f["d0"], f["blockers"] = blockers(ep, b, o, to, seat, i)
        f["hard"] = f["dt"] == INF and f["d0"] < INF
        f["dt_plus"] = _dt(ep, t["after"], o, to)
    return f


# ---------------------------------------------------------------- E1 outcomes

def vacates_within(X, h, hs, i):
    """Does seat hs move object h at one of its next H turns after turn i?"""
    j = i
    for _ in range(H):
        j = X.after(hs, j)
        if j is None:
            return False
        if X.T[j]["obj"] == h:
            return True
    return False


def keep_outcome(X, i, c):
    """K outcome of turn i's request to keep c clear: dict(kept, passed, censored, end), None if moot."""
    T, seat = X.T, X.T[i]["seat"]
    if T[i]["dst"] == c:
        return None
    own = 0
    for p in range(i + 1, len(T)):
        if T[p]["seat"] == seat:
            own += 1
            if T[p]["dst"] == c:
                return dict(kept=True, passed=True, censored=False, end=p)
            if own >= H:
                return dict(kept=True, passed=False, censored=False, end=p)
        elif T[p]["dst"] == c:
            return dict(kept=False, passed=False, censored=False, end=p + 1)
    return dict(kept=True, passed=False, censored=True, end=len(T))


def useful(X, i, f):
    """Does the speaker's d_t on message i's PLAN fall across one of the next two
    gaps between its own turns? None without a later own turn."""
    T, seat, o, to = X.T, X.T[i]["seat"], f["obj"], f["to"]
    prev, j = f["dt_plus"], i
    for k in range(2):
        j = X.after(seat, j)
        if j is None:
            return None if k == 0 else False
        if _dt(X.ep, T[j]["board"], o, to) < prev:
            return True
        if T[j]["done"]:
            return False
        prev = _dt(X.ep, T[j]["after"], o, to)
    return False


def _k_closure(X, i, c, r0):
    """Presenter's next message after a K request (r0: first of its consecutive REQUESTs naming c)."""
    T, seat = X.T, X.T[i]["seat"]
    k = X.after(seat, i)
    if k is None or not T[k]["is_msg"]:
        return "none"
    if T[k]["dst"] == c or c not in X.req(k):
        return "closes"
    return "errs" if any(T[p]["seat"] == seat and T[p]["dst"] == c for p in range(r0, k)) else "re-presents"


# ---------------------------------------------------------------- one episode

def episode(X):
    """Every row of one slot-protocol episode and its episode row."""
    ep, T, keys = X.ep, X.T, X.keys
    tg, owner = ep["targets"], ep["owner"]
    ff = len(T) if X.ff is None else X.ff
    msgs = [i for i in range(ff) if T[i]["slots"]]
    info = {i: plan_info(X, i) for i in msgs}
    chain = {(r["msg_ply"], r["cell"]): r for r in al.request_chain(X)}
    cells, decs, plans, base_v, base_k = [], [], [], [], []
    since, named_prev = {}, collections.defaultdict(set)
    for i in msgs:
        t, f = T[i], info[i]
        seat, b, o, to = t["seat"], t["board"], f["obj"], f["to"]
        for c in named_prev[seat] - set(f["req"]):
            since.pop((seat, c), None)
        for c in f["req"]:
            since.setdefault((seat, c), i)
        named_prev[seat] = set(f["req"])
        u = useful(X, i, f) if f["own"] and (f["filled"] or f["hard"]) else None
        an = X.an(i)
        # ---- E1: one row per REQUEST cell
        cls_of = {c: cell_class(ep, b, seat, c, i) for c in f["req"]}
        for c, cls in cls_of.items():
            r = dict(keys, ply=i, seat=seat, cell=corpus.cell(c), cls=cls, dup=t["dup"],
                     relevant=c in f["path"] if f["path"] else None,
                     relevant_own=any(c in _path(ep, b[x], tg[x]) for x in ep["objs"][seat] if b[x] != tg[x]),
                     useful=u, holder=None, V_next=None, V_H=None, presenter_next=None, base_n=None,
                     base_rate=None, covered=None, K_kept=None, K_passed=None, K_moot=None, K_censored=None)
            if cls in ("V", "F"):
                ch = chain[(i, corpus.cell(c))]
                r.update(holder=ch["holder"], holder_next=ch["holder_next"], holder_visible=ch["holder_visible"],
                         holder_can_vacate=ch["holder_can_vacate"], presenter_next=ch["presenter_next"])
                if cls == "V" and ch["holder_visible"] and ch["holder_can_vacate"]:
                    r.update(V_next=ch["holder_next"] == "E", V_H=vacates_within(X, ch["holder"], ch["holder_seat"], i),
                             base_n=ch["base_n"], base_rate=ch["base_rate"], covered=ch["covered"])
            elif cls == "K":
                k = keep_outcome(X, i, c)
                r.update(K_moot=k is None, presenter_next=_k_closure(X, i, c, since[(seat, c)]))
                if k:
                    r.update(K_passed=k["passed"], K_censored=k["censored"], K_kept=None if k["censored"] else k["kept"])
            cells.append(r)
            if t["dup"] or cls not in ("V", "K"):
                continue
            # ---- negative-control exposure: per-decision outcomes, visible and rolled off
            if cls == "V":
                h = r["holder"]
                for e, how in X.decisions(i, owner[h]):
                    be = T[e]["board"]
                    if be[h] == c and al._legal(ep, be, h, None) and (how == "visible" or c not in X.vis_req(e)):
                        decs.append(dict(keys, ply=i, cell=r["cell"], kind="V", listener=owner[h], exposure=how,
                                         decision_ply=e, outcome=T[e]["obj"] == h))
            else:
                for U in ep["seats"]:
                    if U == seat or not _active(ep, U, i):
                        continue
                    for e, how in X.decisions(i, U):
                        be = T[e]["board"]
                        ins = [m for m in corpus.legal(ep, be, ep["objs"][U]) if al._go(be[m[0]], m[1]) == c]
                        if ins and (how == "visible" or c not in X.vis_req(e)):
                            decs.append(dict(keys, ply=i, cell=r["cell"], kind="K", listener=U, exposure=how,
                                             decision_ply=e, outcome=(T[e]["obj"], T[e]["dir"]) in ins))
        # ---- E1 baseline: unnamed on-path cells of class V and K
        if not t["dup"]:
            for c in f["path"] - {f["fr"]} - set(f["req"]):
                cls = cell_class(ep, b, seat, c, i)
                if cls == "V":
                    h = next(x for x, p in b.items() if p == c)
                    j = X.after(owner[h], i)
                    if j is not None and i in X.vis[j] and c not in X.vis_req(j) and al._legal(ep, T[j]["board"], h, None):
                        base_v.append(T[j]["obj"] == h)
                elif cls == "K":
                    k = keep_outcome(X, i, c)
                    if k and not k["censored"] and not any(c in X.vis_req(p) for p in range(i + 1, k["end"])
                                                           if T[p]["seat"] != seat):
                        base_k.append(k["kept"])
        # ---- E2: one row per PLAN
        cur = b[o] if f["own"] else f["fr"]
        if t["obj"] == o:
            saydo = "progress" if _dw(ep, t["dst"], to) < _dw(ep, t["src"], to) else "same_no_progress"
        else:
            saydo = "other_object"
        at = next((x for x, p in b.items() if p == to), None)
        to_class = ("wall" if not _ok(ep, to) else
                    "own" if at is not None and at != o and owner[at] == seat else
                    "x" if at is not None and owner[at] != seat else
                    "static" if _dw(ep, cur, to) == INF else
                    "dynamic" if f["dt"] == INF else "valid")
        goal = o in tg and to == tg[o]
        wp, frees = None, None
        if f["own"] and not goal and to_class != "wall":
            frees = any(cur in P and to not in P for P in (
                _path(ep, b[v], tg[v]) for v in b if owner[v] != seat and _active(ep, owner[v], i) and b[v] != tg[v]))
            wp = "progress" if _dw(ep, to, tg[o]) < _dw(ep, cur, tg[o]) else "give_way" if frees else "other"
        plans.append(dict(keys, ply=i, seat=seat, dup=t["dup"], obj=o, own=f["own"], plan_from_ok=b.get(o) == f["fr"],
                          filled=f["filled"], n_req_cells=len(f["req"]),
                          req_err=any(v in ("F", "S", "W") for v in cls_of.values()),
                          names_letter=any(rg == "request" for rg, *_ in an["letters"]),
                          names_dir=any(rg == "request" for rg, *_ in an["dirs"]),
                          saydo=saydo, to_class=to_class, to_x_finished=to_class == "x" and not _active(ep, owner[at], i),
                          goal=goal, waypoint=wp, frees=frees, dt=None if f["dt"] in (None, INF) else f["dt"],
                          dt_inf=f["dt"] == INF, hard=f["hard"], n_blockers=len(f["blockers"]),
                          silent_blocked=f["hard"] and not f["filled"], useful=u, n_shadow=len(f["shadow"])))
    spells = _spells(X, msgs, info)
    runs = _runs(X, msgs, info)
    conf, pair_plies = _conflicts(X, ff)
    up = _uptake(X, msgs, info)
    row = _episode_row(X, msgs, ff, cells, decs, plans, base_v, base_k, spells, runs, conf, pair_plies, up)
    return dict(e1_request_cells=cells, e1_decisions=decs, e1_spells=spells, e2_plans=plans, e2_runs=runs,
                e2_conflicts=conf, e2_uptake=up), row


def _spells(X, msgs, info):
    """Blocker spells: maximal runs of a seat's consecutive messages with cell c a blocker."""
    out, open_ = [], {}
    for i in msgs:
        seat, f = X.T[i]["seat"], info[i]
        for key in [k for k in open_ if k[0] == seat and k[1] not in f["blockers"]]:
            out.append(open_.pop(key))
        for c in f["blockers"]:
            sp = open_.setdefault((seat, c), dict(X.keys, seat=seat, cell=corpus.cell(c), first_ply=i, n_turns=0,
                                                  hard=False, requested=False, first_request_ply=None, shadow=False))
            sp.update(last_ply=i, n_turns=sp["n_turns"] + 1, hard=sp["hard"] or f["dt"] == INF,
                      shadow=sp["shadow"] or c in f["shadow"])
            if c in f["req"] and not sp["requested"]:
                sp.update(requested=True, first_request_ply=i)
    return out + list(open_.values())


def _runs(X, msgs, info):
    """Plan episodes: runs of a seat's consecutive messages with the same (object, PLAN-to)."""
    ep, T, out = X.ep, X.T, []
    for seat in ep["seats"]:
        own = [i for i in msgs if T[i]["seat"] == seat]
        runs = [list(g) for _, g in itertools.groupby(own, key=lambda i: (info[i]["obj"], info[i]["to"]))]
        for n, run in enumerate(runs):
            f, nxt = info[run[0]], runs[n + 1][0] if n + 1 < len(runs) else None
            o, to = f["obj"], f["to"]
            r = dict(X.keys, seat=seat, obj=o, to=corpus.cell(to), first_ply=run[0], last_ply=run[-1],
                     n_turns=len(run), goal=to == ep["targets"].get(o), revised_at=nxt)
            if not f["own"]:
                out.append(dict(r, end="invalid_object"))
                continue
            pairs = list(zip(run, run[1:]))
            reached = any(T[k]["after"][o] == to for k in run)
            stale = any(info[a]["dt"] == INF and info[b]["dt"] == INF for a, b in pairs)
            raised = any(info[b]["dt"] > info[a]["dt_plus"] for a, b in pairs)
            held = False
            if nxt is not None:
                bn = T[nxt]["board"]
                raised = raised or _dt(ep, bn, o, to) > info[run[-1]]["dt_plus"]
                held = any(p == to and ep["owner"][x] != seat for x, p in bn.items())
            end = ("reached" if reached else "stale" if stale else "open" if nxt is None else
                   "forced" if raised or held else "unforced")
            out.append(dict(r, end=end, reached=reached, stale=stale, raised=raised, held=held))
    return out


def _conflicts(X, ff):
    """Conflict spells per seat pair, and per pair-ply (vertex, hold) flags."""
    ep, T = X.ep, X.T
    latest, open_, spells, flags = {}, {}, [], []
    for i in range(ff + 1):
        live = {}
        if i < ff:
            b = T[i]["board"]
            act = [s for s in ep["seats"] if s in latest and _active(ep, s, i)]
            for s1, s2 in itertools.combinations(act, 2):
                (o1, t1), (o2, t2) = latest[s1], latest[s2]
                vert = t1 == t2
                hold = any(p == t1 and ep["owner"][x] == s2 and x != o2 or p == t2 and ep["owner"][x] == s1 and x != o1
                           for x, p in b.items())
                flags.append((vert, hold))
                if vert or hold:
                    live[(s1, s2)] = {k for k, on in (("vertex", vert), ("hold", hold)) if on}
        for pair in [p for p in open_ if p not in live]:
            sp = open_.pop(pair)
            how = "censored" if i == ff else "resolved" if all(_active(ep, s, i) for s in pair) else "finished"
            spells.append(dict(sp, end=how, end_ply=i, latency=i - sp["start_ply"] if how == "resolved" else None,
                               kinds="+".join(sorted(sp.pop("_k")))))
        for pair, k in live.items():
            sp = open_.setdefault(pair, dict(X.keys, seats="+".join(pair), start_ply=i, _k=set()))
            sp["_k"] |= k
        if i < ff and T[i]["slots"]:
            latest[T[i]["seat"]] = (T[i]["slots"]["plan_obj"], T[i]["slots"]["plan_to"])
    return spells, flags


def _uptake(X, msgs, info):
    """Listener moves into a teammate PLAN's announced cells at its first decision showing
    the PLAN (and, final arm, after it rolled off)."""
    ep, T, out = X.ep, X.T, []
    for q in msgs:
        if T[q]["dup"] or not X.ucells(q):
            continue
        S = T[q]["seat"]
        for L in ep["seats"]:
            if L == S:
                continue
            for e, how in X.decisions(q, L):
                b = T[e]["board"]
                cs = X.ucells(q) if how == "visible" else X.ucells(q) - X.vis_plan(e, L)
                for o, d in corpus.legal(ep, b, ep["objs"][L]):
                    dst = al._go(b[o], d)
                    if dst not in cs:
                        continue
                    bf = al._base_fields(ep, al.match(X.bidx, b, o, X.mom[e].get(o), d), b, o, d, L, e)
                    out.append(dict(X.keys, msg_ply=q, speaker=S, listener=L, exposure=how, decision_ply=e, obj=o,
                                    dir=d, kind="to" if dst == info[q]["to"] else "path",
                                    goalward=corpus.goalward(ep, o, b[o], d),
                                    entered=not T[e]["done"] and (T[e]["obj"], T[e]["dir"]) == (o, d),
                                    base_n=bf["base_n"], base_rate=bf["base_rate"], covered=bf["covered"]))
    return out


# ---------------------------------------------------------------- episode metrics

def _put(row, name, vals):
    v = [float(x) for x in vals if x is not None]
    row[name + "_n"] = len(v)
    row[name] = sum(v) / len(v) if v else None


def _episode_row(X, msgs, ff, cells, decs, plans, base_v, base_k, spells, runs, conf, pair_plies, up):
    ep, T = X.ep, X.T
    row = dict(X.keys, success=ep["meta"]["success"], n_msgs=len(msgs),
               n_post_fatal=sum(1 for t in T[ff:] if t["slots"]))
    P = [r for r in plans if not r["dup"]]
    F = [r for r in P if r["filled"]]
    C = [r for r in cells if not r["dup"]]
    V, K = [r for r in C if r["cls"] == "V"], [r for r in C if r["cls"] == "K"]
    _put(row, "req_filled", [r["filled"] for r in P])
    _put(row, "req_nocell", [r["n_req_cells"] == 0 for r in F])
    _put(row, "req_err", [r["req_err"] for r in F if r["n_req_cells"]])
    for k in CLASSES:
        _put(row, "share_" + k, [r["cls"] == k for r in C])
    _put(row, "ref_err", [r["cls"] in ("F", "S", "W") for r in C])
    _put(row, "relevant", [r["relevant"] for r in V + K])
    _put(row, "relevant_own", [r["relevant_own"] for r in V + K])
    _put(row, "recall", [s["requested"] for s in spells])
    _put(row, "recall_hard", [s["requested"] for s in spells if s["hard"]])
    _put(row, "recall_or_shadow", [s["requested"] or s["shadow"] for s in spells] if ep["meta"]["proto"] == "hybrid" else [])
    _put(row, "silent_blocked", [r["silent_blocked"] for r in P if r["hard"]])
    _put(row, "V_next", [r["V_next"] for r in V])
    _put(row, "V_H", [r["V_H"] for r in V])
    _put(row, "V_next_onpath", [r["V_next"] for r in V if r["relevant"]])
    _put(row, "V_base_onpath", base_v)
    cov = [r for r in V if r["V_next"] is not None and r["covered"]]
    _put(row, "V_obs_covered", [r["V_next"] for r in cov])
    _put(row, "V_base_covered", [r["base_rate"] for r in cov])
    _put(row, "K_kept", [r["K_kept"] for r in K])
    _put(row, "K_kept_onpath", [r["K_kept"] for r in K if r["relevant"]])
    _put(row, "K_base_onpath", base_k)
    for cls, rs in (("V", V), ("K", K)):
        nx = [r["presenter_next"] for r in rs if r["presenter_next"] not in (None, "none")]
        for c in CLOSURE:
            _put(row, f"{cls}_{c}", [x == c for x in nx])
        for how in ("visible", "nc_rolled_off"):
            _put(row, f"{cls}_dec_{how}", [d["outcome"] for d in decs if d["kind"] == cls and d["exposure"] == how])
    _put(row, "useful_req", [r["useful"] for r in F if r["own"]])
    _put(row, "useful_V_complied", [r["useful"] for r in V if r["V_next"]])
    _put(row, "useful_silent", [r["useful"] for r in P if r["silent_blocked"]])
    _put(row, "shadow_msgs", [r["n_shadow"] > 0 for r in P] if ep["meta"]["proto"] == "hybrid" else [])
    for k in SAYDO:
        _put(row, "saydo_" + k, [r["saydo"] == k for r in P])
    for k in TO_CLASSES:
        _put(row, "to_" + k, [r["to_class"] == k for r in P])
    _put(row, "to_goal", [r["goal"] for r in P])
    W = [r for r in P if r["waypoint"]]
    for k in WAYPOINTS:
        _put(row, "wp_" + k, [r["waypoint"] == k for r in W])
    _put(row, "wp_frees", [r["frees"] for r in W])
    R = [r for r in runs if r["end"] != "invalid_object"]
    for k in RUN_ENDS:
        _put(row, "run_" + k, [r["end"] == k for r in R])
    _put(row, "run_len", [r["n_turns"] for r in R])
    _put(row, "conflict", [v or h for v, h in pair_plies])
    _put(row, "conflict_vertex", [v for v, _ in pair_plies])
    _put(row, "conflict_hold", [h for _, h in pair_plies])
    _put(row, "conf_resolved", [s["end"] == "resolved" for s in conf])
    _put(row, "conf_latency", [s["latency"] for s in conf if s["end"] == "resolved"])
    vis = [u for u in up if u["exposure"] == "visible"]
    _put(row, "up_enter", [u["entered"] for u in vis])
    _put(row, "up_enter_to", [u["entered"] for u in vis if u["kind"] == "to"])
    _put(row, "up_enter_path", [u["entered"] for u in vis if u["kind"] == "path"])
    _put(row, "up_enter_nc", [u["entered"] for u in up if u["exposure"] == "nc_rolled_off"])
    cv = [u for u in vis if u["covered"]]
    _put(row, "up_obs_covered", [u["entered"] for u in cv])
    _put(row, "up_base_covered", [u["base_rate"] for u in cv])
    return row


# ---------------------------------------------------------------- summaries and contrasts

def metrics(ep_rows):
    return [k[:-2] for k in ep_rows[0] if k.endswith("_n") and k[:-2] in ep_rows[0]] if ep_rows else []


def summary(ep_rows):
    """Pooled descriptive rates (unit-weighted over episodes) per alignment.GROUPINGS."""
    out, ms = [], metrics(ep_rows)
    for keys in al.GROUPINGS:
        for k, rs in sorted(al._group(ep_rows, keys).items()):
            row = dict(al._pooled(keys, k), n_episodes=len(rs))
            for m in ms:
                n = sum(r[m + "_n"] for r in rs)
                row[m] = sum(r[m] * r[m + "_n"] for r in rs if r[m + "_n"]) / n if n else None
                row[m + "_n"] = n
            out.append(row)
    return out


CELL_METRICS = ("req_filled", "ref_err", "relevant", "recall", "silent_blocked", "V_next", "V_H", "K_kept",
                "useful_req", "saydo_progress", "to_valid", "to_goal", "wp_give_way", "run_reached", "run_forced",
                "run_unforced", "run_stale", "conflict", "conf_latency", "up_enter")


def cells(ep_rows):
    """stats.cell_estimate of the headline episode metrics per model x suite x protocol, pooled and per level."""
    out = []
    for (m, s, p), rs in sorted(al._group(ep_rows, ("model", "suite", "proto")).items()):
        for lv in ["all"] + sorted({r["level"] for r in rs}):
            sub = [r for r in rs if lv == "all" or r["level"] == lv]
            for met in CELL_METRICS:
                vals = [(r[met], r["level"]) for r in sub if r[met] is not None]
                if vals:
                    est = stats.cell_estimate([v for v, _ in vals], [l for _, l in vals])
                    out.append(dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, level=lv, metric=met, **est))
    return out


def coverage(tabs):
    """Silent-baseline coverage per model x suite x protocol: V rows (holder sees the
    REQUEST and can move) and visible uptake moves with a matched moment."""
    out = []
    for name, rows in (("V", [r for r in tabs["e1_request_cells"] if r["V_next"] is not None and not r["dup"]]),
                       ("uptake", [r for r in tabs["e2_uptake"] if r["exposure"] == "visible"])):
        for (m, s, p), rs in sorted(al._group(rows, ("model", "suite", "proto")).items()):
            c = sum(bool(r["covered"]) for r in rs) / len(rs)
            out.append(dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, baseline=name, n=len(rs), coverage=c,
                            contrast_reported=c >= al.MIN_COVERAGE))
    return out


# (name, observed column, comparison column, coverage gate)
BASELINES = (("V_silent_baseline", "V_obs_covered", "V_base_covered", "V"),
             ("V_unnamed_onpath", "V_next_onpath", "V_base_onpath", None),
             ("K_unnamed_onpath", "K_kept_onpath", "K_base_onpath", None),
             ("V_nc_exposure", "V_dec_visible", "V_dec_nc_rolled_off", None),
             ("K_nc_exposure", "K_dec_visible", "K_dec_nc_rolled_off", None),
             ("useful_vs_silent", "useful_req", "useful_silent", None),
             ("uptake_silent_baseline", "up_obs_covered", "up_base_covered", "uptake"),
             ("uptake_nc_exposure", "up_enter", "up_enter_nc", None))


def _contrast(rows, a, b):
    long = [dict(r, arm=arm, v=r[col]) for r in rows for arm, col in (("a", a), ("b", b))]
    return stats.paired_contrast(long, "v", "arm", "a", "b")


def baselines(ep_rows, cov_rows):
    """Observed - comparison within episodes, board-paired sign-flip per model x suite x protocol."""
    cov = {(r["model"], r["suite"], r["proto"], r["baseline"]): r["coverage"] for r in cov_rows}
    out = []
    for (m, s, p), rs in sorted(al._group(ep_rows, ("model", "suite", "proto")).items()):
        for name, a, b, gate in BASELINES:
            c = cov.get((m, s, p, gate)) if gate else None
            both = [r for r in rs if r[a] is not None and r[b] is not None]
            row = dict(model=m, tier=corpus.TIER[m], suite=s, proto=p, comparison=name, observed=a, comparison_col=b,
                       method="board-paired sign-flip (stats.paired_contrast), unit model x suite x level x board",
                       coverage=c, episodes=len(both))
            if c is not None and c < al.MIN_COVERAGE:
                out.append(dict(row, reported=False, note=f"coverage below {al.MIN_COVERAGE:.0%}"))
                continue
            res = _contrast(both, a, b)
            out.append(dict(row, reported=res["pairs"] >= 2, mean_observed=al._mean([r[a] for r in both]),
                            mean_comparison=al._mean([r[b] for r in both]), diff=res["mean"], p=res["p"],
                            exact=res["exact"], pairs=res["pairs"], note=res["note"]))
    return out


PROTO_METRICS = ("req_filled", "req_nocell", "ref_err", "relevant", "recall", "silent_blocked", "V_next", "K_kept",
                 "useful_req", "saydo_progress", "to_valid", "to_goal", "wp_give_way", "wp_frees", "run_reached",
                 "run_forced", "run_unforced", "run_stale", "conflict", "up_enter")


def proto_contrasts(ep_rows):
    """Hybrid - structured on the slots per model x suite (and pooled over suites),
    board-paired sign-flip; Holm within each model x suite family."""
    out = []
    for m in sorted({r["model"] for r in ep_rows}):
        for s in sorted({r["suite"] for r in ep_rows if r["model"] == m}) + ["all"]:
            sub = [r for r in ep_rows if r["model"] == m and s in ("all", r["suite"])]
            fam = []
            for met in PROTO_METRICS:
                res = stats.paired_contrast(sub, met, "proto", "hybrid", "structured")
                fam.append(dict(model=m, tier=corpus.TIER[m], suite=s, metric=met, part="slots",
                                hybrid=al._mean([r[met] for r in sub if r["proto"] == "hybrid"]),
                                structured=al._mean([r[met] for r in sub if r["proto"] == "structured"]),
                                **{x: res[x] for x in ("contrast", "pairs", "missing", "mean", "stat", "p", "exact")},
                                prompt_differs=met in ("req_filled", "req_nocell", "ref_err", "relevant", "recall",
                                                       "silent_blocked", "V_next", "K_kept", "useful_req",
                                                       "wp_give_way", "wp_frees")))
            for r, h in zip(fam, stats.holm([r["p"] if r["pairs"] >= 2 else float("nan") for r in fam])):
                r["p_holm"] = h
            out += fam
    return out


NOTES = (
    "E1 and E2 use the PLAN and REQUEST slots of structured and hybrid episodes only (extension §2, §5); prose is "
    "out, except the hybrid shadow-request bound (prose clauses with a VerbNet keep-15.2 or put-9.1 member naming "
    "a cell missing from REQUEST).",
    "The prompts differ on REQUEST (extension §5). The structured prompt explains it (\"The REQUEST field lets you "
    "ask your teammate to clear a cell or move out of the way\", and asks the player to consider a temporary detour "
    "or a clearing move); the hybrid prompt does not and asks the player to acknowledge what the teammate might "
    "need and to ground its thoughts instead. Every hybrid - structured contrast on REQUEST content or give-way "
    "waypoints (rp_proto.prompt_differs) carries this instruction difference.",
    "Scope rule (spec §6): rows are messages, or plies for conflicts, before the episode's first fatal decision; "
    "outcomes may fall after it.",
    "Negative-control exposure (the REQUEST or PLAN rolled off the four-round board) exists on the final arm only; "
    "on the development arm those columns are empty and their contrasts unreported.",
    "The board-matched silent baseline (spec §5.5 key) exists for class V (the holder object) and for uptake moves; "
    "class K has no object to match, so its baseline is the unnamed on-path cells of the same episode. "
    "Silent-baseline contrasts are not reported below 30 % coverage.",
    "The none condition is differently prompted, so a difference from it is not a message effect alone; every "
    "comparison here is associational.",
    "Every p-value is exploratory; Holm within each model x suite hybrid - structured family is a multiplicity "
    "guard only.",
)


# ---------------------------------------------------------------- entry point

def run(scope, eps, A):
    nones = {(e["meta"]["model"], e["meta"]["suite"], e["meta"]["level"], e["meta"]["board"]): e
             for e in eps if e["meta"]["proto"] == "none"}
    bidx, ep_rows = {}, []
    tabs = collections.defaultdict(list)
    for ep in eps:
        m = ep["meta"]
        if m["proto"] not in SLOTS:
            continue
        k = (m["model"], m["suite"], m["level"], m["board"])
        if k not in bidx:
            bidx[k] = al.baseline_index(nones[k]) if k in nones else None
        rows, row = episode(_X(ep, A, bidx[k]))
        ep.pop("_dist", None)  # d_W maps from every cell: keep them per episode only
        for name, rs in rows.items():
            tabs[name] += rs
        ep_rows.append(row)
    for name in ("e1_request_cells", "e1_decisions", "e1_spells", "e2_plans", "e2_runs", "e2_conflicts", "e2_uptake"):
        tabs.setdefault(name, [])
    cov = coverage(tabs)
    return dict(tabs, rp_episodes=ep_rows, rp_summary=summary(ep_rows), rp_cells=cells(ep_rows), rp_coverage=cov,
                rp_baselines=baselines(ep_rows, cov), rp_proto=proto_contrasts(ep_rows),
                rp_notes=[dict(note=n) for n in NOTES])
