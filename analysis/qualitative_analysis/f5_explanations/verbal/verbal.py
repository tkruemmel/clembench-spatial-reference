"""V1-V3: what teammates' messages carry about goals, routes and options.

Read-only on the repo. Replays every prose (freeform+hybrid) multiplayer
episode, tags each broadcast sentence, and conditions on board state.
    python3 verbal.py            # build cache + tables
    python3 verbal.py --sample K # print samples for precision checks
"""
import sys
sys.dont_write_bytecode = True
import collections, json, os, random, re, string, pickle

QA = "/Users/tom/uni/clembench-spatial-reference/analysis/qualitative_analysis"
REPO = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, QA)
from seal_averted import legal, step, dist_map, spells, scan, MOVE, STEP  # noqa

OUT = os.path.dirname(os.path.abspath(__file__))
SPK = re.compile(r"^Player (\w+):\s*(.*)$", re.S)
SENT = re.compile(r"[^.!?\n;]+")
TIER = {"qwen38": "strong", "dsv41": "strong", "nemotron": "weak"}

# ---------------------------------------------------------------- lexicon
# W: a reference token; who it denotes is decided per sentence (teammate/self)
import itertools
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
_gid = itertools.count()
W = (r"(?P<who>(?-i:\b[A-FX]\b)|\bI(?:'m|’m|'d|’d|'ll|’ll)?(?![\w'’])|\bme\b|\bmy\b|\bmyself\b|"
     r"\byou\b|\byour\b|\bthem\b|\bhim\b|\bher\b|\bthey\b)")
WRX = re.compile(W, re.I)


def w():
    """W with a fresh group name, so one pattern can hold several."""
    return W.replace("(?P<who>", "(?P<who%d>" % next(_gid))
DET = r"(?:my|your|its|his|her|their|the|whose|(?-i:[A-F])(?:'s|’s))"
ADJ = r"(?:remaining |legal |valid |possible |available |free |open |other )"
MOBP = (r"(?:trapped|boxed[- ]in|pinned(?: in)?|immobil\w*|cornered|walled[- ]in|hemmed[- ]in|"
        r"sealed[- ]in|locked[- ]in|stranded|enclosed|"
        r"no " + ADJ + r"*(?:moves?|exits?|escape|way out|options?)\b|"
        r"nowhere (?:else )?to (?:go|move)|"
        # bare inability; "can't move X / the X / up" is about someone else's object or one direction
        r"(?:can(?:not|'t|’t| not)|could(?:n't|n’t| not)|unable to) move\b(?!\s+(?:\w+\s+)?(?:up|down|left|right)\b)"
        r"(?!\s+(?:X|the|that|those|these|your|their|them|it|its|any|other|others|teammates?|anyone|anything|(?-i:[A-F]))\b)|"
        + DET + r"\s+(?:only|sole|single)\s+" + ADJ + r"*(?:moves?|exits?|escape|way out|options?|square|cell|neighbou?r)\b|"
        + DET + r"\s+last\s+(?:free |open |empty |remaining )?(?:exit|escape|way out|square|cell|neighbou?r)\b|"
        r"(?:only|sole|single|one) " + ADJ + r"+(?:moves?|exits?|escape|way out|options?)\b|"
        r"only (?:move|exit|option)s? (?:possible|available)\b|"
        r"can only (?:move|go|step|exit)\b|"
        r"forced (?:up|down|left|right|to (?:move|step|go))\b)")
MOB_ANY = re.compile(MOBP, re.I)
MOB_VERB = re.compile(r"\b(?:trap(?:s|ping)?|box(?:es|ing)? in|strand(?:s|ing)?|immobiliz\w*|"
                      r"seal(?:s|ing)? in|wall(?:s|ing)? in|enclos(?:e|es|ing)|corner(?:s|ing)?)\s+" + w(), re.I)
STUCK_ANY = re.compile(r"\bstuck\b", re.I)

GOAL = re.compile(w() + r"(?:'s|’s)?\s+(?:own\s+|final\s+)?(?:goal|target|destination)s?\b"
                  r"|\b(?:goal|target|destination)s?\s+(?:of|for)\s+" + w(), re.I)
ROUTE = re.compile(
    w() + r"(?:'s|’s)?\s+(?:route|path|passage|way|approach|descent|ascent|crossing|journey|corridor|lane|transit)\b"
    r"|\b(?:for|let|letting|lets|allow|allowing|so|so that|until|once|after|when)\s+" + w() +
    r"\s+(?:to\s+|can\s+|could\s+|may\s+|will\s+|'ll\s+)?(?:pass|passes|reach|reaches|get|go|goes|follow|advance|"
    r"continue|descend|climb|come|enter|proceed|access|use|cross|head|through|move through|squeeze)\b"
    r"|\b(?:clear|open|free|vacate|keep\s+\S+\s+(?:clear|open|free))\b[^.;!?\n]{0,40}\bfor\s+" + w() +
    r"|" + w() + r"\s+(?:needs?|will need|must pass|has to pass|needs to pass)\b"
    r"|" + w() + r"\s+(?:can't|can’t|cannot|can not|won't be able to|couldn't|is unable to)\s+"
    r"(?:pass|get through|reach|get to|proceed|continue|advance|get past|go through)\b", re.I)
REQV = (r"(?:move|take|step|go|vacate|hold|wait|stay|keep|clear|shift|head|proceed|continue|follow|"
        r"come|use|avoid|don't|do not|let|sidestep|free|leave|make|back|retreat|slide)")
REQ = re.compile(r"\b(?:please|could you|can you|would you|kindly|need you to|I need (?:[A-F]|you) to)\b"
                 r"|(?:^|[,:]\s*)\s*(?-i:[A-F](?:\s*(?:and|/|&|,)\s*[A-F])*|X at \(?R\d+,?\s*C\d+\)?)\s*[,:]\s*(?:please\s+)?"
                 + REQV + r"\b", re.I)
# V3: holding in place / asking others to hold
NOTPLACE = (r"(?!\s+(?:clear|off|out|away|ahead|behind|below|above|low|high|left|right|east|west|north|south|"
            r"in (?!place\b|position\b)|outside|aside|close|near|nearby|adjacent|flexible|alert|ready|patient|focused|available|"
            r"in (?:my|the|your) (?:lane|row|column|area|zone|corner|half|region)|on (?:the |my )?(?:left|right|top|bottom|"
            r"lower|upper|row|column|track|course|R\d|C\d|path|route)))")
HOLDV = (r"(?:wait|waiting|hold|holding|stay|staying|remain|remaining|pause|pausing|sit tight|sitting tight|"
         r"stand still|standing still|idle|idling|park|parking|"
         r"pass(?= (?:this|my|a) turn)|skip(?= (?:this|my|a) turn))\b" + NOTPLACE)
HOLD_SELF = re.compile(
    r"(?P<who>(?-i:\bI\b|\b[A-F]\b))(?:'ll|’ll|'m|’m|'d|’d)?"
    r"(?:\s+(?:will|am|plan to|can|should|must|need to|intend to|would|shall|be|going to|'ll|have to|could|may|might|just|simply|now|then|briefly|temporarily|instead|still|also|first))*"
    r"\s+" + HOLDV, re.I)
HOLD_OTHER = re.compile(
    r"\b(?:please|could you|can you|would you|kindly|need you to)\s+(?:\w+\s+)?" + HOLDV +
    r"|(?:^|[,:]\s*)\s*(?:(?-i:[A-F](?:\s*(?:and|/|&|,)\s*[A-F])*)|you)\s*[,:]?\s*(?:please\s+)?" + HOLDV +
    r"|\b(?:(?-i:[A-F])|you)\s+(?:should|can|could|must|need to|may|will need to)\s+" + HOLDV, re.I)
HOLD_SELF2 = re.compile(r"(?:\band|\bthen|\bso|,)\s*(?:will|shall|can|'ll|’ll)\s+(?:just\s+|simply\s+)?" + HOLDV, re.I)
NOPASS = re.compile(r"\b(?:can(?:not|'t|’t| not)|could(?:n't|n’t| not)|unable to)\s+(?:just\s+)?(?:hold|wait|stay|pass|skip|remain|stand still|stay put)\b"
                    r"|\bmust move (?:every|each|this)\b|\b(?:have|has|need) to move (?:every|each)\b|\bno (?:pass|wait|stay|skip)(?:ing)? (?:option|action|move)\b"
                    r"|\b(?:passing|waiting|staying) (?:is not|isn't|isn’t) (?:allowed|an option|possible|a valid)"
                    r"|\bnot a valid (?:move|direction)|\bmust move exactly|\bforced to move\b|\bsince I must move\b", re.I)
FIRST = re.compile(r"\bI(?:'m|’m|'ll|’ll)?(?![\w'’])")
INPLACE = re.compile(r"(?:hold|holding|stay|staying|remain|remaining)\s+(?:on\s+|at\s+)?\(?R\d|\b(?:in place|here|put|still|where (?:I|it|you) (?:am|is|are)|in position|position|at \(?R\d|"
                     r"stationary|this turn|my turn|a turn|one turn|on (?:my|the|its) (?:goal|target))\b", re.I)


def who_kind(tok, own, mates):
    t = tok.strip()
    if t in mates or t == "X" or t.lower() in ("you", "your", "them", "him", "her", "they"):
        return "mate"
    if t in own or t.lower().rstrip("'’mdl") in ("i",) or t.lower() in ("me", "my", "myself"):
        return "self"
    return None


def kinds(rx, sent, own, mates):
    out = set()
    for m in rx.finditer(sent):
        for g, x in m.groupdict().items():
            if not (g.startswith("who") and x):
                continue
            k = who_kind(x, own, mates)
            if k:
                out.add(k)
    return out


def nearest(rx_pred, s, own, mates, back=45):
    """Subject of each predicate match: the nearest reference token before it."""
    out = set()
    for m in rx_pred.finditer(s):
        lead = re.match(DET + r"\s", m.group(), re.I)
        toks = list(WRX.finditer(s[max(0, m.start() - back):m.start() + (lead.end() if lead else 0)]))
        k = who_kind(toks[-1].group("who"), own, mates) if toks else None
        out.add(k or "none")
    return out


COORD = re.compile(r"R\s*(\d+)\s*[,.]?\s*C\s*(\d+)", re.I)


def tag_sentence(s, own, mates, mate_tgts=frozenset()):
    """Multi-label tags for one sentence from a speaker owning `own`."""
    t = {}
    mob = nearest(MOB_ANY, s, own, mates) | kinds(MOB_VERB, s, own, mates)
    stk = nearest(STUCK_ANY, s, own, mates)
    t["mob_mate"] = "mate" in mob
    t["mob_self"] = "self" in mob
    t["mob_none"] = mob == {"none"}
    t["stuck_mate"] = "mate" in stk
    t["stuck_self"] = "self" in stk
    t["goal_mate"] = "mate" in kinds(GOAL, s, own, mates)
    t["route_mate"] = "mate" in kinds(ROUTE, s, own, mates)
    t["req"] = bool(REQ.search(s))
    t["goalcell_mate"] = bool(mate_tgts) and any((int(r) - 1, int(c) - 1) in mate_tgts for r, c in COORD.findall(s)) \
        and bool(kinds(WRX, s, own, mates) & {"mate"})
    t["hold_self"] = "self" in kinds(HOLD_SELF, s, own, mates) or bool(
        HOLD_SELF2.search(s) and FIRST.search(s))
    t["hold_self_place"] = t["hold_self"] and bool(INPLACE.search(s))
    t["hold_other"] = bool(HOLD_OTHER.search(s))
    t["nopass"] = bool(NOPASS.search(s))
    return t


def cell_rx(cells):
    return re.compile("|".join(r"R\s*%d\s*[,.]?\s*C\s*%d\b" % (r + 1, c + 1) for r, c in cells), re.I)


# ---------------------------------------------------------------- replay
def replay(rec):
    d = os.path.join(REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
    groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
    L = string.ascii_uppercase
    owner = {o: L[i] for i, g in enumerate(groups) for o in g}
    mine = {L[i]: list(g) for i, g in enumerate(groups)}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    done, ply, last = set(), 0, 0
    states, movers, msgs = [], [], []
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            t, c = a.get("type"), str(a.get("content", ""))
            if t == "move":
                m = MOVE.match(c)
                if not m:
                    continue
                states.append((dict(pos), frozenset(done)))
                movers.append(owner[m.group(2)])
                last = ply
                pos = step(pos, m.group(2), m.group(3))
                ply += 1
            elif t == "player_done":
                done.add(c.split()[1])
                last = ply
            elif t == "message" and e.get("from") == "GM":
                m = SPK.match(c)
                if m and m.group(1) in mine:
                    msgs.append((last, m.group(1), m.group(2)))
    states.append((dict(pos), frozenset(done)))
    return dict(n=n, walls=walls, owner=owner, mine=mine, tgt=tgt, states=states,
                movers=movers, msgs=msgs, lvl=ins["spatial_level"])


def player_state(ep, k, q):
    """(n legal moves, blocked forward cells, exit cells) of player q at state k."""
    pos, done = ep["states"][k]
    n, walls = ep["n"], ep["walls"]
    occ = set(pos.values())
    lm = legal(pos, ep["mine"][q], walls, n, occ)
    exits = {step(pos, o, d)[o] for o, d in lm}
    blocked, any_open_fwd = set(), False
    where = {p: o for o, p in pos.items()}
    for o in ep["mine"][q]:
        if pos[o] == ep["tgt"][o]:
            continue
        dm = dist_map(walls, n, ep["tgt"][o])
        r0, c0 = pos[o]
        fwd = [(r0 + dr, c0 + dc) for dr, dc in STEP.values()
               if (r0 + dr, c0 + dc) in dm and dm[(r0 + dr, c0 + dc)] < dm.get(pos[o], 1e9)]
        if fwd and all(f in occ for f in fwd):
            bl = {f for f in fwd if ep["owner"][where[f]] != q}
            if bl:
                blocked |= bl
        elif fwd:
            any_open_fwd = True
    return len(lm), blocked, exits


def runs(flags):
    """Maximal runs of True in a list -> [(start, end)] inclusive."""
    out, s = [], None
    for i, f in enumerate(flags + [False]):
        if f and s is None:
            s = i
        elif not f and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def build():
    eps = json.load(open(os.path.join(QA, f"episodes.{ARM}.json")))
    sents, spl, epis = [], [], []
    for rec in eps:
        if rec["suite"] == "solo" or rec["proto"] not in ("freeform", "hybrid"):
            continue
        ep = replay(rec)
        tag_base = dict(model=rec["model"], tier=TIER[rec["model"]], suite=rec["suite"],
                        lvl=rec["level"], proto=rec["proto"], inst=rec["inst"], path=rec["path"],
                        outcome=rec["outcome"])
        letters = list(ep["mine"])
        # --- per-message sentence tags
        nm = collections.Counter()
        for mi, (k, spk, txt) in enumerate(ep["msgs"]):
            own = set(ep["mine"][spk])
            mates = set(ep["owner"]) - own
            pos, done = ep["states"][min(k, len(ep["states"]) - 1)]
            home = all(pos[o] == ep["tgt"][o] for o in own)
            agg = collections.Counter()
            ss = []
            for s in SENT.findall(txt):
                if len(s.strip()) < 3:
                    continue
                t = tag_sentence(s, own, mates, frozenset(ep["tgt"][o] for o in mates))
                ss.append((s.strip(), t))
                for kk, v in t.items():
                    agg[kk] += v
            sents.append(dict(tag_base, mi=mi, ply=k, spk=spk, home=home, text=txt[:1500],
                              tags={kk: v > 0 for kk, v in agg.items()},
                              hits={kk: [s for s, t in ss if t.get(kk)][:3] for kk in agg if agg[kk]}))
        # --- board-conditioned spells, per player
        K = len(ep["states"]) - 1  # last state index with a move after it is K-1
        for q in letters:
            prec, rblk, info = [], [], []
            for k in range(K + 1):
                pos, done = ep["states"][k]
                if q in done or (ep["lvl"] == "S4" and k <= 1):
                    prec.append(False); rblk.append(False); info.append(None)
                    continue
                nl, bl, ex = player_state(ep, k, q)
                info.append((nl, bl, ex))
                prec.append(nl == 1)
                rblk.append(nl >= 2 and bool(bl))
            for kind, flags in (("prec", prec), ("rblk", rblk)):
                for s, e in runs(flags):
                    # messages written while the writer saw the state (ply in [s, e]),
                    # plus the message accompanying the move that created it (s-1)
                    win = [(k, spk, txt) for k, spk, txt in ep["msgs"] if s - 1 <= k <= e]
                    cells = set()
                    for k in range(s, e + 1):
                        nl, bl, ex = info[k]
                        cells |= (ex if kind == "prec" else bl)
                    qcell = {ep["states"][k][0][o] for k in range(s, e + 1) for o in ep["mine"][q]}
                    nxt = info[e + 1] if e + 1 <= K else None
                    spl.append(dict(tag_base, player=q, kind=kind, start=s, end=e, length=e - s + 1,
                                    ply_cells={k: sorted(info[k][2] if kind == "prec" else info[k][1])
                                               for k in range(s, e + 1)},
                                    sealed=bool(nxt and nxt[0] == 0),
                                    nplayers=len(letters), cells=sorted(cells), qcells=sorted(qcell),
                                    win=win, mine=ep["mine"],
                                    q_moved=any(ep["movers"][k] == q for k in range(s, min(e + 1, len(ep["movers"]))))))
        epis.append(dict(tag_base, nmsg=len(ep["msgs"]), msgs=ep["msgs"], mine=ep["mine"],
                         pos=[s_[0] for s_ in ep["states"]], done=[s_[1] for s_ in ep["states"]],
                         movers=ep["movers"], tgt=ep["tgt"]))
    return sents, spl, epis


def load():
    f = os.path.join(OUT, f"cache.{ARM}.pkl")
    if not os.path.exists(f):
        pickle.dump(build(), open(f, "wb"))
    return pickle.load(open(f, "rb"))


if __name__ == "__main__":
    S, P, E = load()
    print(len(S), len(P), len(E))
