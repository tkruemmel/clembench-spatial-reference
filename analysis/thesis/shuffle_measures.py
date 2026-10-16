"""What would shuffling the final arm's message/move boards scramble, and what would it anonymize?

Replays the played updated-prompt arm (same boards and ownership, older display) and computes
  M1  within-round reply dependence of messages (what a within-round message shuffle scrambles)
  M2  content self-disclosure of the sender's own move (an attribution channel no shuffle removes)
  M3  structural attribution of teammate move lines under displays D0-D3 (ideal reasoner, brute force)
  M4  within-block causal chains among teammate moves (what a within-block move shuffle scrambles)

Usage:  python3 -I shuffle_measures.py REPO RESULT_DIR...      (RESULT_DIR relative to REPO)
        python3 -I shuffle_measures.py --check                  (self-check of the attribution engine)
"""
import collections, functools, glob, itertools, json, math, os, re, sys

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
MOVE = SPK = COORD = None  # taken from the repo's qual_corpus in main()
ROUNDS = 4                  # matrixgame_covered_shared/history.py ROUNDS; window = n * ROUNDS record entries

# ---------------------------------------------------------------- regexes (M1, M2)
L = r"(?-i:[A-F])"
LIST = rf"{L}(?:\s*(?:,|/|&|\band\b)\s*{L})*"
PLAYER = re.compile(rf"\bplayers?\s+((?-i:[A-F])(?:\s*(?:,|/|&|\band\b)\s*(?:players?\s+)?(?-i:[A-F])\b)*)\b", re.I)
VOC = [  # addressee patterns; strict = "Player Q" + these
    re.compile(rf"(?:^|[.!?;:—–]\s*|\n|\(\s*)@?({LIST})\s*[,:]\s", re.I),   # "B, please" / "C and E, your"
    re.compile(rf"@({L})\b"),                                                         # "@B"
    re.compile(rf"\b(?:thanks|thank you|thx|ty)\b,?\s+(?:players?\s+)?({L})\b", re.I),  # "thanks C"
    re.compile(rf",\s*({LIST})\s*(?:[.!?;]|$)", re.I),                                # "... your lane, C."
]
POSS = re.compile(rf"\b({L})['’]s\b")  # loose adds possessives, "C's row-3 lane"
LOOSE = VOC + [POSS]                   # audit_refs.py prints samples of each
YOU = re.compile(r"\byou(?:r|rs)?\b", re.I)  # dyad only: the one teammate
MARKER = re.compile(r"^\W*(as you (?:said|suggested|asked|requested|proposed)|thanks|thank you|ok(?:ay)?\b|agreed|sure\b"
                    r"|got it|understood|noted|sounds good|acknowledged)", re.I)
ROWCOL = re.compile(r"\brow\s*(\d+)\s*,?\s*(?:col(?:umn)?\.?\s*|c)(\d+)", re.I)


def refs(text, n, me, loose, you=True):
    out = set()
    for rx in [PLAYER] + VOC + ([POSS] if loose else []):
        for m in rx.finditer(text):
            out |= set(re.findall(r"[A-F]", m.group(1)))
    out = {ord(c) - 65 for c in out} & set(range(n)) - {me}
    if loose and you and n == 2 and YOU.search(text):
        out.add(1 - me)
    return out


def cells(text):
    return {(int(r) - 1, int(c) - 1) for r, c in COORD.findall(text) + ROWCOL.findall(text)}


# ---------------------------------------------------------------- episodes
def episode(d):
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    turns = []
    for t in it["turns"]:
        for e in t:
            a = e["action"]
            if e["from"] != "GM" or e["to"] != "GM":
                continue
            c = str(a["content"])
            if a["type"] == "move":
                p, o, dr = MOVE.match(c).groups()
                src = pos[o]
                own = {q: pos[q] for q in groups[ord(p) - 65]}
                pos[o] = dst = (src[0] + DIRS[dr][0], src[1] + DIRS[dr][1])
                turns.append(dict(p=ord(p) - 65, kind="move", obj=o, dir=dr, src=src, dst=dst, own=own, msg=None))
            elif a["type"] == "player_done":
                turns.append(dict(p=ord(c.split()[1]) - 65, kind="done", msg=None))
            elif a["type"] == "message":
                spk, txt = SPK.match(c).groups()
                assert turns and turns[-1]["p"] == ord(spk) - 65 and turns[-1]["msg"] is None, d
                turns[-1]["msg"] = txt
    r = 1
    for i, t in enumerate(turns):  # a round wraps when the seat index does not increase
        if i and t["p"] <= turns[i - 1]["p"]:
            r += 1
        t["r"] = r
    assert not turns or turns[0]["p"] == 0, d
    return dict(n=len(groups), proto=ins["comm_protocol"], turns=turns)


# ---------------------------------------------------------------- M3 engine
def gen(n, order, v, R, done, text):
    """Window (oldest first) of (seat, round, kind, has_message_line) under turn order `order`, the
    viewer v about to act in round R, done rounds `done`. None if a finished seat's done turn would
    lie in the future under this order."""
    pv = order.index(v)
    for s, dr in done.items():
        if dr > R or (dr == R and order.index(s) > pv):
            return None
    out, i, k = [], pv, R
    while len(out) < n * ROUNDS:
        i -= 1
        if i < 0:
            i, k = n - 1, k - 1
        if k < 1:
            break
        s = order[i]
        if s in done and k > done[s]:
            continue
        kind = "done" if done.get(s) == k else "move"
        out.append((s, k, kind, kind == "move" or text.get(s, False)))
    return out[::-1]


def signature(win, v, msgmode):
    moves = tuple("Y" if s == v else "X" for s, _, kind, _ in win if kind == "move")
    if msgmode == "none":
        return moves, None
    if msgmode == "chrono":     # named, chronological message board
        return moves, tuple(s for s, _, _, t in win if t)
    return moves, tuple(sorted((k, s) for s, k, _, t in win if t))  # round-labelled, shuffled within round


def block_seqs(win, v):
    bl = [[]]
    for s, _, kind, _ in win:
        if kind == "move":
            if s == v:
                bl.append([])
            else:
                bl[-1].append(s)
    return tuple(tuple(b) for b in bl)


@functools.lru_cache(maxsize=None)
def solve(n, v, R, done_items, text_items, msgmode, know):
    """Per block, the set of chronological owner sequences over all hypotheses consistent with the
    display; plus the consistent turn orders. Done timing (round of each done turn) is taken as known."""
    done, text = dict(done_items), dict(text_items)
    truth = gen(n, tuple(range(n)), v, R, done, text)
    tsig = signature(truth, v, msgmode)
    seqs, orders = None, []
    for o in ([tuple(range(n))] if know == "K" else itertools.permutations(range(n))):
        w = gen(n, o, v, R, done, text)
        if w is None or signature(w, v, msgmode) != tsig:
            continue
        orders.append(o)
        b = block_seqs(w, v)
        seqs = [set() for _ in b] if seqs is None else seqs
        for i, sq in enumerate(b):
            seqs[i].add(sq)
    return truth, seqs, frozenset(orders)


def feasible_positions(lines):
    """lines: [(src, dst)] in true order. Positions each line can take in a display order that is
    physically possible (no line enters a cell still held by an object that leaves it later)."""
    k = len(lines)
    before = [(i, j) for i in range(k) for j in range(i + 1, k) if lines[j][1] == lines[i][0]]
    pos = [set() for _ in range(k)]
    ok = 0
    for perm in itertools.permutations(range(k)):
        at = {x: q for q, x in enumerate(perm)}
        if all(at[i] < at[j] for i, j in before):
            ok += 1
            for x, q in at.items():
                pos[x].add(q)
    return pos, ok, bool(before)


def attrib(seqs, feas):
    """Attributable lines per block: a line is attributable iff every consistent owner sequence
    gives the same owner at every position the line can occupy."""
    a = 0
    for b, F in zip(seqs, feas):
        for Fp in F:
            a += len({sq[q] for sq in b for q in Fp}) == 1
    return a


# ---------------------------------------------------------------- main pass
def run(repo, roots):
    M1 = collections.defaultdict(collections.Counter)
    M2 = collections.defaultdict(collections.Counter)
    M3 = collections.defaultdict(collections.Counter)
    M4 = collections.defaultdict(collections.Counter)
    for root in roots:
        model = root.split("_")[1]
        suite = root.rsplit("_", 1)[1]
        for f in sorted(glob.glob(os.path.join(repo, root, "**", "interactions.json"), recursive=True)):
            ep = episode(os.path.dirname(f))
            n, proto, T = ep["n"], ep["proto"], ep["turns"]
            comm = proto != "none"
            g = (model, suite)
            # ---- M1 and M2
            latest = {}
            prevmsg = {}
            for i, t in enumerate(T):
                txt = t["msg"]
                if comm and txt and not (t["kind"] == "done" and txt.strip().lower() == "done"):
                    k = M1[g]
                    k["msgs"] += 1
                    k["first_in_round"] += not i or T[i - 1]["r"] != t["r"]
                    for tag, loose in (("s", False), ("l", True)):
                        rs = refs(txt, n, t["p"], loose)
                        cls = {q: ("none" if q not in latest else
                                   "same" if T[latest[q]]["r"] == t["r"] else
                                   "prev" if T[latest[q]]["r"] == t["r"] - 1 else "older") for q in rs}
                        k[f"{tag}_any"] += bool(rs)
                        k[f"{tag}_same"] += "same" in cls.values()
                        for c in cls.values():
                            k[f"{tag}_ref_{c}"] += 1
                    m = MARKER.match(txt)
                    if m:
                        k["mk"] += 1
                        head = re.split(r"[.;!?]", txt, maxsplit=1)[0][:80]
                        named = refs(head, n, t["p"], True, you=False)
                        prev = T[i - 1]["p"] if i else None
                        if named:
                            k["mk_named"] += 1
                            k["mk_named_prev"] += prev in named
                            k["mk_same"] += any(q in latest and T[latest[q]]["r"] == t["r"] for q in named)
                        else:
                            k["mk_same"] += bool(i) and T[i - 1]["r"] == t["r"]
                if comm and t["kind"] == "move" and txt:
                    k2 = M2[(model, suite, proto)]
                    cs = cells(txt)
                    k2["n"] += 1
                    k2["src_or_dst"] += t["src"] in cs or t["dst"] in cs
                    k2["dst"] += t["dst"] in cs
                    k2["own_any"] += bool(cs & (set(t["own"].values()) | {t["dst"]}))
                    k2["objdir"] += bool(re.search(rf"\b{t['obj']}\b", txt)) and bool(re.search(rf"\b{t['dir']}\b", txt, re.I))
                    pc = prevmsg.get(t["p"], set())
                    k2["src_or_dst_or_announced"] += t["src"] in cs or t["dst"] in cs or t["dst"] in pc
                if txt:
                    latest[t["p"]] = i
                    prevmsg[t["p"]] = cells(txt)
            # ---- M3: every turn is a view
            W = n * ROUNDS
            CAP = W + 1
            done_at = {}
            orders_by_viewer = collections.defaultdict(list)
            for i, t in enumerate(T):
                v, R = t["p"], t["r"]
                win = T[max(0, i - W):i]
                mates = [x for x in win if x["kind"] == "move" and x["p"] != v]
                done = {s: T[j]["r"] for s, j in done_at.items()}
                text = {s: bool(T[j]["msg"]) and comm and T[j]["msg"].strip().lower() != "done" for s, j in done_at.items()}
                Rk = min(R, CAP + 2)
                dk = tuple(sorted((s, Rk - min(R - dr, CAP + 1)) for s, dr in done.items()))
                tk = tuple(sorted(text.items()))
                # true blocks, with lines' cells for the physics variant
                blocks = [[]]
                for x in win:
                    if x["kind"] == "move":
                        if x["p"] == v:
                            blocks.append([])
                        else:
                            blocks[-1].append((x["src"], x["dst"]))
                fixed = [[{q} for q in range(len(b))] for b in blocks]
                free = [[set(range(len(b))) for _ in b] for b in blocks]
                phys = [feasible_positions(b)[0] for b in blocks]
                key3 = (model, suite, "comm" if comm else "none")
                k3 = M3[key3]
                steady = not done and R > ROUNDS  # full window, nobody finished
                k3["views"] += 1
                k3["lines"] += len(mates)
                k3["ss_lines"] += steady * len(mates)
                for know in ("K", "U"):
                    for mm in (("chrono", "round") if comm else ("none",)):
                        truth, seqs, orders = solve(n, v, Rk, dk, tk, mm, know)
                        if know == "K" and mm != "round":
                            assert [s for s, *_ in truth] == [x["p"] for x in win], (f, i)
                            assert all(len(s) == 1 for s in seqs)
                        tag = f"{mm}{know}"
                        k3[f"{tag}_chronomoves"] += attrib(seqs, fixed)
                        k3[f"{tag}_shuffled"] += attrib(seqs, free)
                        k3[f"{tag}_shuffledphys"] += attrib(seqs, phys)
                        if steady:
                            k3[f"ss_{tag}_chronomoves"] += attrib(seqs, fixed)
                            k3[f"ss_{tag}_shuffledphys"] += attrib(seqs, phys)
                        if know == "U" and mm == "round":
                            k3["roundU_orders"] += len(orders)
                            k3["roundU_unique"] += len(orders) == 1
                            orders_by_viewer[v].append(orders)
                if t["kind"] == "done":
                    done_at[v] = i
            for v, sets in orders_by_viewer.items():
                inter = functools.reduce(lambda a, b: a & b, sets)
                M3[(model, suite, "comm")]["pooled_viewers"] += 1
                M3[(model, suite, "comm")]["pooled_unique"] += len(inter) == 1
                M3[(model, suite, "comm")]["pooled_orders"] += len(inter)
            # ---- M4: newest block at each viewer turn (teammate moves since the viewer's previous turn)
            if n > 2:
                last = {}
                for i, t in enumerate(T):
                    v = t["p"]
                    if v in last:
                        lines = [(x["src"], x["dst"]) for x in T[last[v] + 1:i] if x["kind"] == "move"]
                        k4 = M4[(model, suite)]
                        k4["blocks"] += 1
                        if len(lines) >= 2:
                            pos, ok, chain = feasible_positions(lines)
                            k4["multi"] += 1
                            k4["chain"] += chain
                            k4["p_impossible"] += 1 - ok / math.factorial(len(lines))
                            k4["pinned_blocks"] += any(len(p) == 1 for p in pos)
                            k4["pinned_lines"] += sum(len(p) == 1 for p in pos)
                            k4["lines"] += len(lines)
                    last[v] = i
    return M1, M2, M3, M4


def pct(a, b):
    return f"{100 * a / b:5.1f}" if b else "  n/a"


def report(M1, M2, M3, M4):
    out = []
    P = out.append
    groups = sorted({g for g in M1} | {(m, s) for m, s, _ in M3}, key=lambda g: (g[0], ("dyad", "3p", "6p").index(g[1])))
    P("## M1 within-round reply dependence (comm protocols, all messages except bare 'done')\n")
    P("| model | suite | msgs | not first in round | strict: any ref | strict: ref to same-round msg | loose: any ref | loose: same-round | refs same-round (strict/loose) | marker msgs | marker plausibly answers same-round | marker named = previous speaker |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        k = M1[g]
        if not k["msgs"]:
            continue
        rs = sum(k[f"s_ref_{c}"] for c in ("same", "prev", "older", "none"))
        rl = sum(k[f"l_ref_{c}"] for c in ("same", "prev", "older", "none"))
        P(f"| {g[0]} | {g[1]} | {k['msgs']} | {pct(k['msgs'] - k['first_in_round'], k['msgs'])} | {pct(k['s_any'], k['msgs'])} | {pct(k['s_same'], k['msgs'])} | {pct(k['l_any'], k['msgs'])} | "
          f"{pct(k['l_same'], k['msgs'])} | {pct(k['s_ref_same'], rs)} / {pct(k['l_ref_same'], rl)} | {pct(k['mk'], k['msgs'])} | "
          f"{pct(k['mk_same'], k['mk'])} | {k['mk_named_prev']}/{k['mk_named']} |")
    P("\n## M2 content self-disclosure (move turns with a message)\n")
    P("| model | suite | protocol | n | states src or dst of this move | states dst | src/dst or dst announced in previous own msg | states any own-object cell | names object + direction |")
    P("|---|---|---|---|---|---|---|---|---|")
    for (m, s, p), k in sorted(M2.items(), key=lambda x: (x[0][0], ("dyad", "3p", "6p").index(x[0][1]), x[0][2])):
        P(f"| {m} | {s} | {p} | {k['n']} | {pct(k['src_or_dst'], k['n'])} | {pct(k['dst'], k['n'])} | {pct(k['src_or_dst_or_announced'], k['n'])} | "
          f"{pct(k['own_any'], k['n'])} | {pct(k['objdir'], k['n'])} |")
    P("\nPooled over the three message protocols:\n")
    P("| model | suite | n | src or dst | dst | any own cell | obj+dir |")
    P("|---|---|---|---|---|---|---|")
    for g in groups:
        k = sum((M2[(*g, p)] for p in ("freeform", "hybrid", "structured")), collections.Counter())
        if k["n"]:
            P(f"| {g[0]} | {g[1]} | {k['n']} | {pct(k['src_or_dst'], k['n'])} | {pct(k['dst'], k['n'])} | {pct(k['own_any'], k['n'])} | {pct(k['objdir'], k['n'])} |")
    P("\n## M3 structural attribution: % of teammate move lines in view attributable to one teammate\n")
    P("Message protocols (D0 = chrono msgs + chrono moves; D1 = msgs shuffled in round + chrono moves; "
      "D2 = chrono msgs + moves shuffled in block; D3 = both; K = turn order A,B,C.. known, U = display only; +phys = physically impossible orders excluded)\n")
    P("| model | suite | views | lines | D0 | D1 K | D1 U | D2 | D2+phys | D3 K | D3 K+phys | D3 U | D3 U+phys | D1U orders/view | D1U unique order (view / pooled per viewer) | steady-state lines | ss D1 U | ss D2+phys |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        k = M3[(*g, "comm")]
        if not k["views"]:
            continue
        Ln = k["lines"]
        P(f"| {g[0]} | {g[1]} | {k['views']} | {Ln} | {pct(k['chronoU_chronomoves'], Ln)} | {pct(k['roundK_chronomoves'], Ln)} | "
          f"{pct(k['roundU_chronomoves'], Ln)} | {pct(k['chronoU_shuffled'], Ln)} | {pct(k['chronoU_shuffledphys'], Ln)} | "
          f"{pct(k['roundK_shuffled'], Ln)} | {pct(k['roundK_shuffledphys'], Ln)} | {pct(k['roundU_shuffled'], Ln)} | "
          f"{pct(k['roundU_shuffledphys'], Ln)} | {k['roundU_orders'] / k['views']:.1f} | {pct(k['roundU_unique'], k['views'])} / {pct(k['pooled_unique'], k['pooled_viewers'])} | "
          f"{pct(k['ss_lines'], Ln)} | {pct(k['ss_roundU_chronomoves'], k['ss_lines'])} | {pct(k['ss_chronoU_shuffledphys'], k['ss_lines'])} |")
    P("\n'none' protocol (no message board, so D0 = D1 and D2 = D3):\n")
    P("| model | suite | views | lines | D0 K | D0 U | D2 K | D2 U | D2 K+phys |")
    P("|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        k = M3[(*g, "none")]
        if k["views"]:
            Ln = k["lines"]
            P(f"| {g[0]} | {g[1]} | {k['views']} | {Ln} | {pct(k['noneK_chronomoves'], Ln)} | {pct(k['noneU_chronomoves'], Ln)} | "
              f"{pct(k['noneK_shuffled'], Ln)} | {pct(k['noneU_shuffled'], Ln)} | {pct(k['noneK_shuffledphys'], Ln)} |")
    P("\n## M4 within-block causal chains (3p/6p, all protocols; newest block at each viewer turn)\n")
    P("| model | suite | blocks | blocks with >=2 lines | chain (of >=2) | P(random shuffle impossible) | >=1 line pinned by physics | lines pinned |")
    P("|---|---|---|---|---|---|---|---|")
    for g in groups:
        k = M4[g]
        if k["blocks"]:
            P(f"| {g[0]} | {g[1]} | {k['blocks']} | {pct(k['multi'], k['blocks'])} | {pct(k['chain'], k['multi'])} | "
              f"{pct(k['p_impossible'], k['multi'])} | {pct(k['pinned_blocks'], k['multi'])} | {pct(k['pinned_lines'], k['lines'])} |")
    return "\n".join(out)


def check():
    """Analytic cases: with no one done and a full window, D1U attributes the lines of a viewer's
    one-element segments only: 6p 2/30, 3p 1/3, dyad all; D0 and D1K all; D2 none beyond the dyad."""
    for n, want in ((6, 2 / 30), (3, 1 / 3), (2, 1.0)):
        a = tot = 0
        for v in range(n):
            _, seqs, _ = solve(n, v, 20, (), (), "round", "U")
            sizes = [len(next(iter(s))) for s in seqs]
            fixed = [[{q} for q in range(k)] for k in sizes]
            free = [[set(range(k)) for _ in range(k)] for k in sizes]
            a += attrib(seqs, fixed)
            tot += sum(sizes)
            _, s0, _ = solve(n, v, 20, (), (), "chrono", "U")
            assert attrib(s0, fixed) == sum(sizes)
            assert attrib(s0, free) == (sum(sizes) if n == 2 else 0)
            _, s1, _ = solve(n, v, 20, (), (), "round", "K")
            assert attrib(s1, fixed) == sum(sizes)
        assert abs(a / tot - want) < 1e-9, (n, a, tot)
    # a chain pins both lines: line 1 enters the cell line 0 vacated
    pos, ok, chain = feasible_positions([((0, 0), (0, 1)), ((1, 0), (0, 0))])
    assert chain and ok == 1 and pos == [{0}, {1}]
    # a 6p viewer whose successor E finished: E's done turn hides nothing positional; window still aligns
    truth, seqs, _ = solve(6, 3, 20, ((4, 18),), ((4, False),), "chrono", "K")
    assert [s for s, *_ in truth].count(4) == 3   # E in rounds 16, 17 (moves) and 18 (done) of the window
    print("check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check()
        sys.exit()
    repo, roots = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, os.path.join(repo, "analysis", "qualitative_analysis"))
    import qual_corpus as qc  # MOVE, SPK, COORD
    MOVE, SPK, COORD = qc.MOVE, qc.SPK, qc.COORD
    res = report(*run(repo, roots))
    print(res)
