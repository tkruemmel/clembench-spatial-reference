"""Form, private information and reference (spec §5.1, §5.2, §5.3).

Spec: thesis/message_analysis_design.md §5 with §1, §2 and §8. `run(scope,
eps, A)` returns episode-level tables (the unit, §1.6) with their grouping
keys, `pr_cells` (cell estimates by model x suite x protocol x part, pooled
over levels, §8) and `pr_contrasts` (freeform - hybrid on the prose part, the
voluntary content of §5; slot measures are format compliance).

Parts (§5 slot tags): `slots` is the PLAN and REQUEST regions, `prose` the
free text, `all` both (for the curves). Speaker-side rates use the messages
minus exact repeats (`dup`); a listener's view holds everything shown.
`none` episodes carry no messages and are skipped.

Readings where the spec is silent:
- words: the master's own count, len(text.split()) per region (master.py l.133-153).
- slot presence: the game's PLAN regex found in the message (format compliance outside freeform).
- PLAN-to is the goal iff it equals the PLAN object's target, otherwise a waypoint.
- board classes are multi-label; walls and off-grid cells get none; chance is
  each class's share of the non-wall cells at B_t-.
- device = corridor cells plus entrance; the device line is the row or column they all share.
- the teammate-target class is invariant when targets are permuted among the
  teammate objects, so that baseline is letter-bound: clause pairs (teammate
  letter L, cell c), observed c == target(L), expected 1/k if c is one of the
  k teammate targets.
- private facts with a fixed count: per object, its owner's letter-cell binding
  and its target. Next moves form no fixed set and enter only "later move stated".
- ownership is hidden on the final arm and in the development dyad only (§5.2 table note).
- a fact is stated by its owner only: an own letter paired with its true cell
  (strict G-loc, G-from, PLAN-from) or a stated target (strict G-target); relays do not count.
- D(r) and V(r) count r in the owner's / decider's own turns (1-based, cumulative).
- own move described: a present-frame G-move pair about an own letter;
  consistent iff one equals the move made.
- later move: an own-letter G-move in a future or sequence frame, or a PLAN of
  an own object whose PLAN-to is not where the object stands after the move.
  kept1 / kept2: matched at the next / one of the next two own turns (PLAN:
  that object moved closer to PLAN-to, wall-only distance); no later own turn: unscored.
- cell named before entry counts entries strictly after the message's own turn
  (the same-turn move is shown with the message); lead time in own turns.
- the permutation baseline permutes message times among the speaker's own
  messages; its exact expectation is used.
- redundant: the fact was present in the listener's view at its last decision
  before the message (any part), or, on the final arm for ownership facts, the
  owner was resolvable there by the rules plus the MOVE BOARD.
- resolvable: listeners are teammates with an exposure (corpus.exposures);
  refs = distinct letters plus distinct cells of the part; "true cell" = true
  at B_t- or B_t+; "binds L to its current cell" = a visible pairing of L with
  its cell at the listener's decision.
- addressee: the listener's seat is named (G-seat) in the message; in the dyad the partner always is.
- owners resolvable: the dyad is left out (every X is the partner's); an object
  counts iff the rules name exactly its true owner; the prose rule is a
  present-frame G-move by the sender; the MOVE BOARD rule is in `owner_claims`.
"""
import bisect
import collections
import math

from analysis.messages import corpus, grammar, stats

PARTS = {"slots": ("plan", "request"), "prose": ("prose",), "all": ("prose", "plan", "request")}
PROTO_PARTS = {"structured": ("slots", "all"), "hybrid": ("slots", "prose", "all"), "freeform": ("prose", "all")}
CLASSES = ("own_obj", "mate_active", "mate_finished", "empty", "device", "own_target", "mate_target")
WINDOWS = (("W1", 1), ("W4", 4), ("Wall", math.inf))
VARIANTS = ("plan", "plan_prose", "plan_prose_mb")
R_MAX = 32          # curve rows r = 1..R_MAX own turns
R_SUM = (4, 16)     # curve points carried into pr_cells

# (table, column, baseline column) carried into pr_cells; parts slots/prose
# unless the table has its own (pr_form_msg: msg, pr_owners: the variants)
SUMMARY = (
    ("pr_form", "words", None), ("pr_form", "seat_named", None),
    ("pr_form", "plan_to_goal", None), ("pr_form", "request_none", None),
    ("pr_form_msg", "think", None), ("pr_form_msg", "dup", None), ("pr_form_msg", "retry", None),
    ("pr_private", "own_cell", None), ("pr_private", "target_stated", None), ("pr_private", "target_named", None),
    ("pr_private", "move_described", None), ("pr_private", "move_consistent", None),
    ("pr_private", "later_stated", None), ("pr_private", "later_kept1", None),
    ("pr_private", "entry_W1", "chance_W1"), ("pr_private", "entry_W4", "chance_W4"),
    ("pr_private", "entry_Wall", "chance_Wall"),
    ("pr_private", "redisc", None), ("pr_private", "redundant", None), ("pr_private", "v_mean", None),
    *(("pr_boardref", c, f"chance_{c}") for c in CLASSES),
    ("pr_boardref", "mate_tgt_letter", "mate_tgt_letter_perm"), ("pr_boardref", "device_named", None),
    ("pr_resolvable", "unres", None), ("pr_resolvable", "unres_addr", None),
    ("pr_owners", "owners", None),
)


def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _keys(ep):
    m = ep["meta"]
    return dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], proto=m["proto"],
                board=m["board"], episode=m["id"])


# ---------------------------------------------------------------- per message

def msg_info(ep, t, an):
    """Per part, the items of one message that the measures read."""
    own = set(ep["objs"][t["seat"]])
    sl, mode = t["slots"], grammar.DEFAULT["frames"]
    texts = grammar.regions(t["msg"], ep["meta"]["proto"], sl)
    out = {}
    for part, regs in PARTS.items():
        # letter-cell pairings: kept G-loc, kept G-from, PLAN-from (§4)
        pairs = [(p["letter"], p["cell"], "loc") for p in an["loc"] if p["region"] in regs and grammar.kept(p["frames"], mode)]
        pairs += [(p["letter"], p["cell"], "frm") for p in an["frm"] if p["region"] in regs and not p["frames"]]
        if sl and "plan" in regs:
            pairs.append((sl["plan_obj"], sl["plan_from"], "plan"))
        pairs = [p for p in pairs if p[0] in ep["owner"]]
        true = {(L, c) for L, c, _ in pairs if c in (t["board"][L], t["after"][L])}
        view = dict(cells=[x for x in an["cells"] if x[0] in regs],
                    clause_info=[ci for ci in an["clause_info"] if ci["region"] in regs])
        tm = grammar.target_mentions(view, {o: ep["targets"][o] for o in own})
        out[part] = dict(
            pairs=pairs, true=true, own_true={L for L, _ in true if L in own},
            stated={o for o, v in tm.items() if v["stated"]}, named={o for o, v in tm.items() if v["named"]},
            moves=[(m["letter"], m["dir"], m["frame"]) for m in an["moves"] if m["region"] in regs],
            cells=[x[2] for x in view["cells"]], lines=[x[2] for x in an["lines"] if x[0] in regs],
            letters=sorted({x[2] for x in an["letters"] if x[0] in regs}),
            seats={x[2] for x in an["seats"] if x[0] in regs},
            clauses=view["clause_info"],
            words=sum(len(s.split()) for r, s in texts if r in regs))
    return out


def _hidden(ep):
    """Is ownership a hidden fact? Final arm, and the development dyad (§5.2 table note)."""
    return ep["meta"]["arm"] == "final" or ep["meta"]["suite"] == "dyad"


def facts_of(ep, t, info, part):
    """Private facts the speaker states in this part (§5.2): ('own', o) for an
    own letter paired with its true cell where ownership is hidden, ('tgt', o)
    for a stated target."""
    p = info[part]
    return ({("own", o) for o in p["own_true"]} if _hidden(ep) else set()) | {("tgt", o) for o in p["stated"]}


def all_facts(ep):
    return [("own", o) for o in sorted(ep["owner"]) if _hidden(ep)] + [("tgt", o) for o in sorted(ep["owner"])]


# ---------------------------------------------------------------- board (§5.3)

def device(ep):
    """Device cells (corridor plus entrance) and the line they share, or (None, None)."""
    d = ep["device"]
    if not d.get("corridor"):
        return None, None
    cells = {tuple(c) for c in d["corridor"]} | ({tuple(d["entrance"])} if d.get("entrance") else set())
    rows, cols = {c[0] for c in cells}, {c[1] for c in cells}
    line = ("R", rows.pop()) if len(rows) == 1 else ("C", cols.pop()) if len(cols) == 1 else None
    return cells, line


def classes(ep, t, c, dev):
    """§5.3 board-reference classes of cell c at B_t- (multi-label; device None without a device)."""
    S, n = t["seat"], ep["grid"]
    on = 0 <= c[0] < n and 0 <= c[1] < n and c not in ep["walls"]
    occ = {p: o for o, p in t["board"].items()}
    o = occ.get(c)
    mate = o is not None and ep["owner"][o] != S
    fin = mate and ep["done_ply"].get(ep["owner"][o], math.inf) < t["ply"]
    tg = {p: ob for ob, p in ep["targets"].items()}
    return dict(own_obj=o is not None and not mate, mate_active=mate and not fin, mate_finished=fin,
                empty=on and o is None, device=None if dev is None else c in dev,
                own_target=c in tg and ep["owner"][tg[c]] == S, mate_target=c in tg and ep["owner"][tg[c]] != S)


def board_shares(ep, t, dev):
    """Chance baseline: each class's share of the non-wall cells at B_t-."""
    S, N = t["seat"], ep["grid"] ** 2 - len(ep["walls"])
    done = {s for s, q in ep["done_ply"].items() if q < t["ply"]}
    own = sum(ep["owner"][o] == S for o in t["board"])
    fin = sum(ep["owner"][o] in done and ep["owner"][o] != S for o in t["board"])
    return dict(own_obj=own / N, mate_active=(len(t["board"]) - own - fin) / N, mate_finished=fin / N,
                empty=(N - len(t["board"])) / N, device=None if dev is None else len(dev) / N,
                own_target=own / N, mate_target=(len(ep["targets"]) - own) / N)


# ---------------------------------------------------------------- owners (§5.3, test arm)

def mb_view(ep, idx, M):
    """Final arm: what the viewer of turn idx sees of its teammates, in rounds.

    history.py: both boards cover the turns since the viewer's 4th-last own
    turn; a round is one own turn and the teammates' turns after it, so the
    viewer's own lines split both boards into the same rounds. Round 0 holds
    the turns before the viewer's first own turn in the window. Per round:
    `moves` = the `X moved from src to dst` lines in play order as (src,
    object), the object being the replay's truth, never read by the rules
    except as the line's label; `msgs` = teammates' messages with sender,
    PLAN object, prose present-frame letters and stated source cells (PLAN-from,
    kept prose G-from)."""
    T = ep["turns"]
    viewer, vis = T[idx]["seat"], set(T[idx]["vis"])
    rounds = [dict(msgs=[], moves=[])]
    for t in T[corpus.window_start(ep, idx):idx]:
        if t["seat"] == viewer:
            rounds.append(dict(msgs=[], moves=[]))
            continue
        if not t["done"]:
            rounds[-1]["moves"].append((t["src"], t["obj"]))
        if t["ply"] in vis and t["ply"] in M:
            a = M[t["ply"]]["all"]
            rounds[-1]["msgs"].append(dict(
                seat=t["seat"], plan=t["slots"]["plan_obj"] if t["slots"] else None,
                prose={L for L, _, f in M[t["ply"]]["prose"]["moves"] if f == "present"},
                src={c for _, c, k in a["pairs"] if k in ("frm", "plan")}))
    return rounds


def owner_claims(rounds, n, variant):
    """Object -> seats the frozen rules name as its owner, from one view (§5.3).

    plan:          a PLAN names the sender's object (the prompt: "PLAN: move
                   <OBJECT>", and a player moves only its own objects).
    plan_prose:    plus a present-frame G-move in the sender's prose.
    plan_prose_mb: plus the MOVE BOARD. Moves name no player but keep play
                   order, and messages sit in their round (shuffled only
                   within it). (1) Direct link: a teammate's message whose
                   stated source cell is the src of exactly one X line of its
                   round owns that line's object. (2) Position link: turns
                   are round-robin, so in a round with n-1 X lines (every
                   teammate moved) the k-th line is always the same seat's.
                   A direct link in such a round fixes position k to its
                   sender when no other sender or position conflicts; every
                   k-th line of such rounds then names that sender's object.
    """
    claims = collections.defaultdict(set)
    for r in rounds:
        for m in r["msgs"]:
            if m["plan"]:
                claims[m["plan"]].add(m["seat"])
            if variant != "plan":
                for L in m["prose"]:
                    claims[L].add(m["seat"])
    if variant != "plan_prose_mb":
        return claims
    full = lambda j, r: j > 0 and len(r["moves"]) == n - 1
    pos = collections.defaultdict(set)
    for j, r in enumerate(rounds):
        for m in r["msgs"]:
            hits = [k for k, (src, _) in enumerate(r["moves"]) if src in m["src"]]
            if len(hits) == 1:
                claims[r["moves"][hits[0]][1]].add(m["seat"])
                if full(j, r):
                    pos[hits[0]].add(m["seat"])
    seat_pos = collections.Counter(s for ss in pos.values() for s in ss)
    for k, ss in pos.items():
        if len(ss) == 1 and seat_pos[next(iter(ss))] == 1:
            for j, r in enumerate(rounds):
                if full(j, r):
                    claims[r["moves"][k][1]] |= ss
    return claims


def resolved(ep, idx, M, variant, cache):
    """Teammates' objects of the viewer of idx whose owner the rules name uniquely and truly."""
    key = (idx, variant)
    if key not in cache:
        viewer = ep["turns"][idx]["seat"]
        cl = owner_claims(mb_view(ep, idx, M), ep["meta"]["n"], variant)
        cache[key] = {o for o, s in ep["owner"].items() if s != viewer and cl.get(o) == {s}}
    return cache[key]


# ---------------------------------------------------------------- §5.1 form

def form_rows(ep, A, M):
    k, T = _keys(ep), ep["turns"]
    msgs = [t for t in T if t["is_msg"]]
    nd = [t for t in msgs if not t["dup"]]
    eid = ep["meta"]["id"]
    rows = [dict(k, part="msg", n_msgs=len(msgs), retry=_mean(t["retry"] for t in msgs),
                 dup=_mean(t["dup"] for t in msgs), think=_mean(A[(eid, t["ply"])]["think"] for t in nd),
                 plan_present=_mean(corpus.PLAN.search(t["msg"]) is not None for t in nd))]
    dev, _ = device(ep)
    out = []
    for part in PROTO_PARTS[ep["meta"]["proto"]]:
        row = dict(k, part=part, n_msgs=len(nd), words=_mean(M[t["ply"]][part]["words"] for t in nd),
                   seat_named=_mean(bool(M[t["ply"]][part]["seats"]) for t in nd))
        if part == "slots":
            sl = [(t, t["slots"]) for t in nd]
            row.update(plan_own=_mean(s["plan_obj"] in ep["objs"][t["seat"]] for t, s in sl),
                       plan_moved=_mean(s["plan_obj"] == t["obj"] for t, s in sl),
                       plan_to_goal=_mean(ep["targets"].get(s["plan_obj"]) == s["plan_to"] for t, s in sl),
                       request_none=_mean(s["request_none"] for _, s in sl))
            req = [(t, [c[2] for c in A[(eid, t["ply"])]["cells"] if c[0] == "request"])
                   for t, s in sl if not s["request_none"]]
            row["req_cell"] = _mean(bool(cs) for _, cs in req)
            cls = [classes(ep, t, cs[0], dev) for t, cs in req if cs]  # class of the first cell
            row.update({f"req_{c}": _mean(x[c] for x in cls) for c in CLASSES})
        out.append(row)
    return rows, out


# ---------------------------------------------------------------- §5.2 private information

def _next_turns(ep):
    """ply -> the speaker's later own turns, in order."""
    by = collections.defaultdict(list)
    for t in ep["turns"]:
        by[t["seat"]].append(t)
    return {t["ply"]: by[t["seat"]][t["own_turn"] + 1:] for t in ep["turns"]}


def later_statements(ep, t, info, part):
    """§5.2 later move stated: [(kind, obj, dir or PLAN-to, frame)]."""
    own = set(ep["objs"][t["seat"]])
    out = [("move", L, d, f) for L, d, f in info[part]["moves"] if f in ("future", "sequence") and L in own]
    sl = t["slots"]
    if sl and "plan" in PARTS[part] and sl["plan_obj"] in own and t["after"][sl["plan_obj"]] != sl["plan_to"]:
        out.append(("plan", sl["plan_obj"], sl["plan_to"], None))
    return out


def kept(ep, st, nxt, j):
    """Is statement st matched at one of the next j own turns? None if there is none."""
    if not nxt:
        return None
    kind, o, x, _ = st
    for u in nxt[:j]:
        if u["done"] or u["obj"] != o:
            continue
        if kind == "move" and u["dir"] == x:
            return True
        if kind == "plan":
            dm = corpus.dist_map(ep, x)
            if u["src"] in dm and u["dst"] in dm and dm[u["dst"]] < dm[u["src"]]:
                return True
    return False


def entries(ep):
    """seat -> cell -> sorted own-turn indices at which the seat's object entered the cell."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for t in ep["turns"]:
        if not t["done"]:
            out[t["seat"]][t["dst"]].append(t["own_turn"])
    return out


def hit(ks, tau, W):
    """Lead time of the first entry strictly after own turn tau within W own turns, else None."""
    i = bisect.bisect_right(ks, tau)
    return ks[i] - tau if i < len(ks) and ks[i] - tau <= W else None


def private_rows(ep, A, M):
    k, T = _keys(ep), ep["turns"]
    nd = [t for t in T if t["is_msg"] and not t["dup"]]
    facts = all_facts(ep)
    nxt, ent, expo = _next_turns(ep), entries(ep), corpus.exposures(ep)
    hidden, final = _hidden(ep), ep["meta"]["arm"] == "final"
    prev, last = {}, {}  # ply -> {seat: that seat's last decision before ply}
    for t in T:
        prev[t["ply"]] = dict(last)
        last[t["seat"]] = t["ply"]
    times = collections.defaultdict(list)
    for t in nd:
        times[t["seat"]].append(t["own_turn"])
    F = {p: {q: facts_of(ep, T[q], M[q], p) for q in M} for p in PARTS}
    pres_cache, own_cache, rows, drows, vrows = {}, {}, [], [], []

    def present(idx, p):
        """Facts stated in the messages shown at decision idx."""
        if (idx, p) not in pres_cache:
            pres_cache[(idx, p)] = set().union(*(F[p][q] for q in T[idx]["vis"] if q in F[p]))
        return pres_cache[(idx, p)]

    for part in PROTO_PARTS[ep["meta"]["proto"]]:
        # disclosure: first own turn per fact; later statements are re-disclosures
        first, named, redisc = {}, {}, []
        for t in nd:
            for f in F[part][t["ply"]]:
                if f in first:
                    redisc.append(t["own_turn"])
                else:
                    first[f] = t["own_turn"]
            for o in M[t["ply"]][part]["named"]:
                named.setdefault(o, t["own_turn"])
        for r in range(1, R_MAX + 1):
            drows.append(dict(k, part=part, r=r, D=sum(v < r for v in first.values()) / len(facts),
                              redisc=sum(v < r for v in redisc)))
        # redundancy: per (statement, listener)
        red = []
        for t in nd:
            for f in F[part][t["ply"]]:
                for s in ep["seats"]:
                    if s == t["seat"] or (t["ply"], s) not in expo:
                        continue
                    q = prev[t["ply"]].get(s)
                    red.append(q is not None and (f in present(q, "all") or (
                        final and f[0] == "own" and f[1] in resolved(ep, q, M, "plan_prose_mb", own_cache))))
        # co-presence V at every decision
        byr, vs = collections.defaultdict(list), []
        for t in T:
            mates = [f for f in facts if ep["owner"][f[1]] != t["seat"]]
            v = len(present(t["ply"], part) & set(mates)) / len(mates)
            vs.append(v)
            byr[t["own_turn"] + 1].append(v)
        vrows += [dict(k, part=part, r=r, V=_mean(byr[r]), n_decisions=len(byr[r])) for r in range(1, R_MAX + 1) if byr[r]]
        # own move described, later move stated, cell named before entry
        desc, cons, later, k1, k2, fut = [], [], [], [], [], []
        pairs = []
        for t in nd:
            info = M[t["ply"]]
            own = set(ep["objs"][t["seat"]])
            pres = {(L, d) for L, d, f in info[part]["moves"] if f == "present" and L in own}
            desc.append(bool(pres))
            if pres:
                cons.append((t["obj"], t["dir"]) in pres)
            sts = later_statements(ep, t, info, part)
            later.append(bool(sts))
            for st in sts:
                k1.append(kept(ep, st, nxt[t["ply"]], 1))
                k2.append(kept(ep, st, nxt[t["ply"]], 2))
                if st[3]:
                    fut.append(st[3] == "future")
            pairs += [(t["seat"], t["own_turn"], c) for c in set(info[part]["cells"])]
        row = dict(k, part=part, n_msgs=len(nd), n_facts=len(facts),
                   own_cell=(sum(f[0] == "own" for f in first) / len(ep["owner"])) if hidden else None,
                   own_cell_first=_mean(v for f, v in first.items() if f[0] == "own") if hidden else None,
                   target_stated=sum(f[0] == "tgt" for f in first) / len(ep["owner"]),
                   target_stated_first=_mean(v for f, v in first.items() if f[0] == "tgt"),
                   target_named=len(named) / len(ep["owner"]), target_named_first=_mean(named.values()),
                   d_final=len(first) / len(facts), redisc=len(redisc), n_red=len(red), redundant=_mean(red),
                   v_mean=_mean(vs), move_described=_mean(desc), move_consistent=_mean(cons),
                   later_stated=_mean(later), n_later=len(k1), later_kept1=_mean(k1), later_kept2=_mean(k2),
                   later_future=_mean(fut), n_named_cells=len(pairs))
        frac = {}
        for name, W in WINDOWS:
            obs = [hit(ent[s][c], tau, W) for s, tau, c in pairs]
            for s, _, c in pairs:  # exact expectation of the within-speaker time permutation
                if (s, c, W) not in frac:
                    frac[(s, c, W)] = _mean(hit(ent[s][c], u, W) is not None for u in times[s])
            row[f"entry_{name}"] = _mean(x is not None for x in obs)
            row[f"chance_{name}"] = _mean(frac[(s, c, W)] for s, _, c in pairs)
            if W == math.inf:
                row["lead"] = _mean(obs)
        rows.append(row)
    return rows, drows, vrows


# ---------------------------------------------------------------- §5.3 reference

def reference_rows(ep, A, M):
    k, T = _keys(ep), ep["turns"]
    nd = [t for t in T if t["is_msg"] and not t["dup"]]
    dev, line = device(ep)
    expo = corpus.exposures(ep)
    pairset = {q: {(L, c) for L, c, _ in M[q]["all"]["pairs"]} for q in M}
    bound = {}  # decision ply -> (L, c) pairings in its visible messages
    br, rr = [], []
    for part in PROTO_PARTS[ep["meta"]["proto"]]:
        cls, ch, lt, named = [], [], [], []
        res, res_addr = [], []
        for t in nd:
            info, S = M[t["ply"]][part], t["seat"]
            sh = board_shares(ep, t, dev)
            for c in info["cells"]:
                cls.append(classes(ep, t, c, dev))
                ch.append(sh)
            mate_t = {ep["targets"][o] for o in ep["owner"] if ep["owner"][o] != S}
            kk = sum(ep["owner"][o] != S for o in ep["owner"])
            for ci in info["clauses"]:
                for L in set(ci["letters"]):
                    if ep["owner"][L] != S:
                        lt += [(c == ep["targets"][L], (c in mate_t) / kk) for c in ci["cells"]]
            if dev is not None:
                named.append(any(c in dev for c in info["cells"]) or (line is not None and line in info["lines"]))
            # listener-resolvable reference
            nref = len(info["letters"]) + len(set(info["cells"]))
            if not nref:
                continue
            true_l = {L for L, _ in info["true"]}
            for s in ep["seats"]:
                d = expo.get((t["ply"], s))
                if s == S or d is None:
                    continue
                if d not in bound:
                    bound[d] = set().union(*(pairset[q] for q in T[d]["vis"] if q in pairset))
                now = T[d]["board"]
                un = sum(not (ep["owner"][L] == s or L in true_l or (L, now[L]) in bound[d]) for L in info["letters"])
                res.append(un / nref)
                if ep["meta"]["suite"] == "dyad" or s in M[t["ply"]]["all"]["seats"]:
                    res_addr.append(un / nref)
        row = dict(k, part=part, n_cells=len(cls))
        for c in CLASSES:
            row[c] = _mean(x[c] for x in cls)
            row[f"chance_{c}"] = _mean(x[c] for x in ch)
        row.update(n_letter_cell=len(lt), mate_tgt_letter=_mean(o for o, _ in lt),
                   mate_tgt_letter_perm=_mean(p for _, p in lt), device_named=_mean(named))
        br.append(row)
        rr.append(dict(k, part=part, n_pairs=len(res), unres=_mean(res), n_pairs_addr=len(res_addr),
                       unres_addr=_mean(res_addr)))
    return br, rr


def owners_rows(ep, M):
    """§5.3 owners resolvable: test arm (final) and 3p/6p only."""
    if ep["meta"]["arm"] != "final" or ep["meta"]["suite"] == "dyad":
        return []
    k, cache, rows = _keys(ep), {}, []
    for v in VARIANTS:
        sh = []
        for t in ep["turns"]:
            mates = [o for o, s in ep["owner"].items() if s != t["seat"]]
            sh.append(len(resolved(ep, t["ply"], M, v, cache)) / len(mates))
        rows.append(dict(k, part=v, n_views=len(sh), owners=_mean(sh)))
    return rows


# ---------------------------------------------------------------- summaries (§8)

def cells(tables):
    """Cell estimates per SUMMARY measure by model x suite x protocol x part, pooled over levels."""
    out = []
    for name, col, base in SUMMARY:
        groups = collections.defaultdict(list)
        for r in tables[name]:
            if name in ("pr_form_msg", "pr_owners") or r["part"] in ("slots", "prose") or col == "v_mean":
                groups[(r["model"], r["suite"], r["proto"], r["part"])].append(r)
        for (m, s, p, part), rs in sorted(groups.items()):
            vals = [r.get(col) for r in rs]
            if all(v is None for v in vals):
                continue
            e = stats.cell_estimate(vals, [r["level"] for r in rs])
            out.append(dict(table=name, measure=col, model=m, suite=s, proto=p, part=part, **e,
                            baseline=_mean(r[base] for r in rs) if base else None))
    groups = collections.defaultdict(list)  # D(r) at R_SUM, both parts together
    for r in tables["pr_disclosure"]:
        if r["part"] == "all" and r["r"] in R_SUM:
            groups[(r["model"], r["suite"], r["proto"], r["r"])].append(r)
    for (m, s, p, rr), rs in sorted(groups.items()):
        e = stats.cell_estimate([r["D"] for r in rs], [r["level"] for r in rs])
        out.append(dict(table="pr_disclosure", measure=f"D({rr})", model=m, suite=s, proto=p, part="all", **e,
                        baseline=None))
    return out


def contrasts(tables):
    """Freeform - hybrid on the prose part per model x suite, board-paired sign-flip (§5, §8)."""
    out = []
    for name, col, _ in SUMMARY:
        rows = [r for r in tables[name] if r.get("part") == "prose" and col in r]
        for m, s in sorted({(r["model"], r["suite"]) for r in rows}):
            sub = [r for r in rows if r["model"] == m and r["suite"] == s]
            c = stats.paired_contrast(sub, col, "proto", "freeform", "hybrid")
            if c["pairs"]:
                out.append(dict(table=name, measure=col, model=m, suite=s, part="prose",
                                **{x: c[x] for x in ("contrast", "pairs", "missing", "n", "mean", "stat", "p", "exact")}))
    return out


def run(scope, eps, A):
    tables = {t: [] for t in ("pr_form", "pr_form_msg", "pr_private", "pr_disclosure", "pr_copresence",
                              "pr_boardref", "pr_resolvable", "pr_owners")}
    for ep in eps:
        if ep["meta"]["proto"] == "none":
            continue
        eid = ep["meta"]["id"]
        M = {t["ply"]: msg_info(ep, t, A[(eid, t["ply"])]) for t in ep["turns"] if t["is_msg"]}
        a, b = form_rows(ep, A, M)
        tables["pr_form_msg"] += a
        tables["pr_form"] += b
        a, b, c = private_rows(ep, A, M)
        tables["pr_private"] += a
        tables["pr_disclosure"] += b
        tables["pr_copresence"] += c
        a, b = reference_rows(ep, A, M)
        tables["pr_boardref"] += a
        tables["pr_resolvable"] += b
        tables["pr_owners"] += owners_rows(ep, M)
    tables["pr_cells"] = cells(tables)
    tables["pr_contrasts"] = contrasts(tables)
    return tables
