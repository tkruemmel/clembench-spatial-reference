"""The counterfactual of S9.3.1: enclosure hazards that were *not* taken.

S9.3.1 shows 32 enclosure aborts: a player steps into the one free cell a
teammate had left, the teammate has no legal move, and with no pass action the
episode aborts. `seal_alt.py` shows the sealer had a harmless alternative in 30
of them and played the sealing move because it was the only one that made
progress. This asks the opposite question of the same corpus: how often did a
player stand in front of that move and decline it, and do its own messages say
why?

The decision point is the *pinch*: the moving player has a legal move that
would leave an active teammate with no legal move at all. The sharp subset is
the *costly pinch*, where the sealing move is the only one that takes the
mover nearer its own target -- the configuration all 30 free-choice aborts came
from, so declining there costs a tempo rather than costing nothing. The same
hazard seen from the victim's side is the *spell*: a stretch with no legal move,
which ends in a rescue or in the abort.

    python3 analysis/qualitative_analysis/seal_averted.py            # the counts
    python3 analysis/qualitative_analysis/seal_averted.py --list     # one line per pinch
    python3 analysis/qualitative_analysis/seal_averted.py --quotes   # transcript evidence
    python3 analysis/qualitative_analysis/seal_averted.py --spells   # from the victim's side
    python3 analysis/qualitative_analysis/seal_averted.py --proto none structured
    python3 analysis/qualitative_analysis/seal_averted.py --check    # re-derive the counts

The answer, in one line: unlike the cut hazard of S9.3.2, this one shows no
protocol effect at all -- in the strong tier, 43 of 73 free pinches declined
with prose against 62 of 106 without, 59% against 58%, p = 1 -- and in the
whole corpus exactly four pinches are preceded by a teammate spelling out that
the cell is its last square. Three of the four are Nemotron's and are taken
anyway; the one that is heeded is `dsv41 6p S4 freeform instance_00010`, where
the warning is issued, relayed by a third player and obeyed, and the episode is
won.
"""
import argparse
import collections
import json
import os
import re
import string

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
PROTOS = ("freeform", "hybrid")
# the two tiers S9.3 reports separately: Nemotron aborts on format, so its
# episodes end before most hazards can form and its arms are not comparable
TIERS = {"strong": ("qwen38", "dsv41"), "weak": ("nemotron",)}

_dist = {}


def dist_map(walls, n, target):
    """Shortest-path distance to `target` over walls; other objects ignored.

    Objects move, so a teammate in the way is not a reason to call a direction
    the wrong way to go. Same convention as seal_alt.dist_map.
    """
    key = (walls, n, target)
    if key not in _dist:
        seen, queue = {target: 0}, collections.deque([target])
        while queue:
            cell = queue.popleft()
            for dr, dc in STEP.values():
                nb = (cell[0] + dr, cell[1] + dc)
                if (0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls
                        and nb not in seen):
                    seen[nb] = seen[cell] + 1
                    queue.append(nb)
        _dist[key] = seen
    return _dist[key]


def legal(pos, objs, walls, n, occ=None):
    """Every (object, direction) those objects can legally play from `pos`."""
    occ = set(pos.values()) if occ is None else occ
    out = []
    for o in objs:
        r0, c0 = pos[o]
        for d, (dr, dc) in STEP.items():
            nb = (r0 + dr, c0 + dc)
            if (0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls
                    and nb not in occ):
                out.append((o, d))
    return out


def step(pos, obj, direction):
    p = dict(pos)
    p[obj] = (p[obj][0] + STEP[direction][0], p[obj][1] + STEP[direction][1])
    return p


def stream(it):
    """Messages, accepted moves and done declarations in recorded order."""
    out = []
    for turn in it["turns"]:
        for ev in turn:
            a = ev.get("action", {})
            t = a.get("type")
            if t == "get message":
                out.append(("msg", ev.get("from", ""), a.get("content", "")))
            elif t == "move":
                m = MOVE.match(a.get("content", ""))
                if m:
                    out.append(("move", m.group(2), m.group(3)))
            elif t == "player_done":
                out.append(("done", a["content"].split()[1], None))
    return out


def scan(rec):
    """One record per pinch: a ply where the mover could immobilise a teammate.

    `victims` are the teammates a single legal move of the mover would leave
    with no legal move at all; `alts` is everything else it could play. A done
    player is never a victim -- it will not be asked to move again.
    """
    d = os.path.join(REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
    groups = ins.get("player_objects") or [ins["player_a_objects"],
                                           ins["player_b_objects"]]
    letters = string.ascii_uppercase
    owner = {o: letters[i] for i, g in enumerate(groups) for o in g}
    mine = {letters[i]: list(g) for i, g in enumerate(groups)}
    seat = {"Player %d" % (i + 1): letters[i] for i in range(len(groups))}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}

    done, ply, out = set(), 0, []
    said = collections.defaultdict(list)

    def mobile(board):
        """Active players with at least one legal move on `board`."""
        occ = set(board.values())
        return {q for q in mine
                if q not in done and legal(board, mine[q], walls, n, occ)}

    for kind, a, b in stream(it):
        if kind == "msg":
            p = seat.get(a)
            if p:
                said[p].append((ply, b))
            continue
        if kind == "done":
            done.add(a)
            continue

        p = owner[a]
        moves = legal(pos, mine[p], walls, n)
        # only a teammate that can move *now* can be immobilised by this move;
        # one already stuck is a hazard someone else created
        free_now = mobile(pos) - {p}
        kills = {m: sorted(free_now - mobile(step(pos, *m))) for m in moves}
        seals = {m: v for m, v in kills.items() if v}
        if seals and (a, b) in kills:
            safe = [m for m in moves if not kills[m]]
            # "progress" is the reason seal_alt gives for every taken seal:
            # the move that shortens the mover's own distance to its target
            def gains(m):
                dm = dist_map(walls, n, tgt[m[0]])
                return dm[step(pos, *m)[m[0]]] < dm[pos[m[0]]]
            forward_seal = [m for m in seals if gains(m)]
            out.append(dict(
                meta=rec, ply=ply, player=p, played=(a, b), pos=pos,
                victims=sorted({v for vs in seals.values() for v in vs}),
                seals=sorted(seals), safe=safe,
                taken=bool(kills[(a, b)]),
                # the fork seal_alt characterises: the sealing move is the only
                # way forward, so declining it costs the mover a tempo
                costly=bool(forward_seal) and not any(gains(m) for m in safe),
                grid=n, walls=walls, mine=mine, owner=owner, targets=tgt,
                said=dict(said),
            ))
        pos = step(pos, a, b)
        ply += 1
    return out



BAD = re.compile(r"^Player ([A-F]):")
# Language that names a teammate's mobility rather than merely a route.
STUCK = re.compile(
    r"\b(?:stuck|trapped|boxed?[- ]?in|pinned|immobil\w*|"
    r"no (?:legal |valid )?moves?|nowhere to (?:go|move)|can(?:no|\u2019|')?t move|"
    r"cannot move|unable to move|has no moves?)\b", re.I)
GIVE = re.compile(
    r"\b(?:unblock\w*|free(?:ing|s|d)?\b|make (?:room|way|space)|"
    r"step(?:ping)? (?:aside|out of)|out of (?:the|your|his|her|their|its) way|"
    r"give (?:you|him|her|them|it|[A-F]) (?:room|space)|let (?:\w+ )?(?:move|out|escape|leave)|"
    r"so (?:you|[A-F]) can move|vacat\w*|clear(?:ing|s)? \(?R\d)\b", re.I)


def spells(rec):
    """Every stretch in which an active player had no legal move at all.

    This is the hazard S9.3.1 names, seen from the victim's side. A spell ends
    when somebody's move gives the player a square again (`freed`), when it
    declares done, or with the episode. `bad` are the invalid moves the stuck
    player itself was forced to emit, i.e. the proof that its own turn came
    round while it was still stuck.
    """
    d = os.path.join(REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
    groups = ins.get("player_objects") or [ins["player_a_objects"],
                                           ins["player_b_objects"]]
    letters = string.ascii_uppercase
    owner = {o: letters[i] for i, g in enumerate(groups) for o in g}
    mine = {letters[i]: list(g) for i, g in enumerate(groups)}
    seat = {"Player %d" % (i + 1): letters[i] for i in range(len(groups))}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}

    ev = []
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            t = a.get("type")
            if t == "get message":
                ev.append(("msg", seat.get(e.get("from", "")), a.get("content", "")))
            elif t == "move":
                m = MOVE.match(a.get("content", ""))
                if m:
                    ev.append(("move", m.group(2), m.group(3)))
            elif t == "player_done":
                ev.append(("done", a["content"].split()[1], None))
            elif t == "invalid_move":
                m = BAD.match(a.get("content", ""))
                ev.append(("bad", m.group(1) if m else None, a.get("content", "")))

    done, ply, open_, out = set(), 0, {}, []
    said = collections.defaultdict(list)

    def stuck():
        occ = set(pos.values())
        return {q for q in mine
                if q not in done and not legal(pos, mine[q], walls, n, occ)}

    for q in stuck():
        open_[q] = dict(player=q, start=0, bad=[], cause=None, cause_by=None)
    for kind, a, b in ev:
        if kind == "msg":
            if a:
                said[a].append((ply, b))
            continue
        if kind == "bad":
            if a in open_:
                open_[a]["bad"].append((ply, b))
            continue
        if kind == "done":
            done.add(a)
            if a in open_:
                out.append(dict(open_.pop(a), end="declared done", end_ply=ply,
                                freer=None, by=None))
            continue
        p = owner[a]
        pos = step(pos, a, b)
        ply += 1
        now = stuck()
        for q in list(open_):
            if q not in now:
                out.append(dict(open_.pop(q), end="freed", end_ply=ply,
                                freer=p, by=(a, b)))
        for q in now:
            if q not in open_:
                open_[q] = dict(player=q, start=ply, bad=[], cause=p, cause_by=(a, b))
    for q, r in open_.items():
        out.append(dict(r, end="episode ended", end_ply=ply, freer=None, by=None))
    for r in out:
        r.update(meta=rec, said=dict(said), plies=ply,
                 length=r["end_ply"] - r["start"],
                 own_turn=bool(r["bad"]),
                 opening=(r["start"] == 0))
    return out


def rows(models=None, protos=PROTOS, suites=("dyad", "3p", "6p")):
    eps = json.load(open(os.path.join(HERE, "episodes.json")))
    out, seen = [], 0
    for rec in eps:
        if rec["suite"] not in suites or rec["proto"] not in protos:
            continue
        if models and rec["model"] not in models:
            continue
        seen += 1
        out.extend(scan(rec))
    return out, seen

def relief(r):
    """The freeing player's own words at the rescue, and what they name."""
    if r["end"] != "freed":
        return []
    out = []
    for ply, text in r["said"].get(r["freer"], []):
        if r["end_ply"] - 1 <= ply <= r["end_ply"]:
            tags = [t for t, ok in (("names victim", re.search(r"\b%s\b" % r["player"], text)),
                                    ("immobility", STUCK.search(text)),
                                    ("gives way", GIVE.search(text)))
                    if ok]
            out.append((ply, tags, text))
    return out


CELL = "R\\s*%d\\s*[,.]?\\s*C\\s*%d\\b"
SENT = re.compile(r"[^.!?\n]+")


def warning(r, window=6):
    """Anyone naming the cell of a sealing move before it is offered.

    The canonical case is the victim itself: it can see that it is down to one
    square long before the teammate standing next to that square does.
    """
    cells = {step(r["pos"], *m)[m[0]] for m in r["seals"]}
    rx = re.compile("|".join(CELL % (c[0] + 1, c[1] + 1) for c in cells), re.I)
    out = []
    for q, msgs in r["said"].items():
        for ply, text in msgs:
            if not (r["ply"] - window <= ply <= r["ply"] and rx.search(text)):
                continue
            # the immobility claim has to be in the same breath as the cell,
            # or every long route description counts as a warning
            named = [x for x in SENT.findall(text) if rx.search(x) and STUCK.search(x)]
            tags = ([] if q not in r["victims"] else ["victim's own warning"]) + \
                   (["immobility"] if named else [])
            out.append((ply, q, tags, named[0] if named else text))
    return sorted(out)


def reason(r, window=1):
    """The mover's own words at a pinch, and what they name."""
    out = []
    for ply, text in r["said"].get(r["player"], []):
        if r["ply"] - window <= ply <= r["ply"] + 1:
            tags = [t for t, ok in
                    (("names victim", any(re.search(r"\b%s\b" % v, text) for v in r["victims"])),
                     ("immobility", STUCK.search(text)),
                     ("gives way", GIVE.search(text)))
                    if ok]
            out.append((ply, tags, text))
    return out


def selftest():
    """Every count this module reports, re-derived. Fails loudly if data moves."""
    ff, n_ff = rows(protos=("freeform", "hybrid"))
    ns, n_ns = rows(protos=("none", "structured"))
    assert (n_ff, n_ns) == (1080, 1080), (n_ff, n_ns)
    assert (len(ff), len(ns)) == (93, 110), (len(ff), len(ns))
    assert all(r["meta"]["suite"] == "6p" for r in ff + ns)      # one object per player
    free = lambda rs: [r for r in rs if r["safe"]]
    assert (len(free(ff)), len(free(ns))) == (91, 106)
    assert (sum(not r["taken"] for r in free(ff)),
            sum(not r["taken"] for r in free(ns))) == (45, 62)
    cost = lambda rs: [r for r in free(rs) if r["costly"]]
    assert (len(cost(ff)), len(cost(ns))) == (56, 72)
    assert (sum(not r["taken"] for r in cost(ff)),
            sum(not r["taken"] for r in cost(ns))) == (21, 32)

    # split by tier: pooling hides that the strong tier is flat to three
    # decimal places and that the whole pooled tilt is Nemotron's 16 of 18
    for tier, want in (("strong", ((73, 73, 43, 46, 20), (110, 106, 62, 72, 32))),
                       ("weak", ((20, 18, 2, 10, 1), (0, 0, 0, 0, 0)))):
        for rs, w in zip((ff, ns), want):
            d = [r for r in rs if r["meta"]["model"] in TIERS[tier]]
            fr = [r for r in d if r["safe"]]
            co = [r for r in fr if r["costly"]]
            got = (len(d), len(fr), sum(not r["taken"] for r in fr), len(co),
                   sum(not r["taken"] for r in co))
            assert got == w, (tier, got, w)

    sp = [s for rec in json.load(open(os.path.join(HERE, "episodes.json")))
          if rec["suite"] == "6p" for s in spells(rec)]
    assert len(sp) == 276, len(sp)
    op = [s for s in sp if s["opening"]]
    # every S4 6p board ships with player B already immobile, and nothing else does
    assert len(op) == 180 and {s["meta"]["level"] for s in op} == {"S4"}
    assert {s["player"] for s in op} == {"B"}
    assert len({(s["meta"]["model"], s["meta"]["proto"], s["meta"]["inst"])
                for s in op}) == 180
    # and it is relieved by whatever the first player plays: the 31 that are not
    # are the episodes in which no move was ever accepted
    assert sum(s["end"] == "freed" for s in op) == 149
    assert all(s["freer"] == "A" and s["end_ply"] == 1
               for s in op if s["end"] == "freed")
    assert all(s["meta"]["moves"] == 0 for s in op if s["end"] != "freed")
    mid = [s for s in sp if not s["opening"]]
    assert len(mid) == 96
    assert sum(s["end"] == "freed" for s in mid) == 51
    assert all(s["freer"] != s["cause"] for s in mid if s["end"] == "freed")
    # the hazard is fatal exactly when the victim's own turn arrives first
    # the opening hazard is relieved 120 of 120 in the strong tier, in both
    # arms; every failure to relieve it is a Nemotron episode with no moves
    st = [s for s in sp if s["meta"]["model"] in TIERS["strong"]]
    wk = [s for s in sp if s["meta"]["model"] in TIERS["weak"]]
    assert (len(st), len(wk)) == (198, 78)
    assert [(sum(x["opening"] for x in t),
             sum(x["opening"] and x["end"] == "freed" for x in t))
            for t in (st, wk)] == [(120, 120), (60, 29)]
    assert [(sum(x["end"] == "freed" for x in t if not x["opening"] and group(x) == g),
             sum(1 for x in t if not x["opening"] and group(x) == g))
            for t in (st, wk) for g in ("prose", "rigid")] \
        == [(19, 30), (25, 48), (7, 18), (0, 0)]
    assert not any(s["own_turn"] for s in sp if s["end"] == "freed")
    assert all(s["meta"]["outcome"] in ("abort", "lose")
               for s in sp if s["own_turn"])

    # the language: of the 45 declines, nearly all name the teammate, but the
    # naming is route courtesy -- only the cell-specific immobility warning is
    # evidence that anybody computed the hazard, and there are four of those
    named = lambda rs: [r for r in rs if r["safe"]
                        and any("names victim" in t for _, t, _ in reason(r))]
    assert (len(named([r for r in ff if not r["taken"]])),
            len(named([r for r in ns if not r["taken"]]))) == (42, 2)
    warn = lambda rs: [r for r in rs if r["safe"]
                       and any("immobility" in t for _, _, t, _ in warning(r))]
    assert not warn(ns)
    got = sorted((r["meta"]["model"], r["meta"]["level"], r["meta"]["proto"],
                  r["meta"]["inst"], r["ply"], r["taken"], r["meta"]["outcome"])
                 for r in warn(ff))
    assert got == [
        ("dsv41", "S4", "freeform", "instance_00010", 8, False, "success"),
        ("nemotron", "S3", "freeform", "instance_00013", 4, True, "lose"),
        ("nemotron", "S3", "freeform", "instance_00015", 20, True, "abort"),
        ("nemotron", "S4", "freeform", "instance_00012", 31, True, "abort"),
    ], got

    # and the 32 enclosure aborts S9.3.1 already reports are all in here, as
    # pinches that were taken: same event, counted from the other side
    import seal_alt
    taken = {(r["meta"]["path"], r["ply"]) for r in ff + ns if r["taken"]}
    assert all((c["meta"]["path"], c["ply"] - 1) in taken for c in seal_alt.cases())
    print("selftest ok")


def group(r):
    return "prose" if r["meta"]["proto"] in ("freeform", "hybrid") else "rigid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", nargs="*", default=list(PROTOS))
    ap.add_argument("--model")
    ap.add_argument("--tier", choices=sorted(TIERS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quotes", action="store_true")
    ap.add_argument("--spells", action="store_true",
                    help="the hazard from the victim's side")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return selftest()
    models = TIERS[args.tier] if args.tier else (
        (args.model,) if args.model else None)

    if args.spells:
        sp = [s for rec in json.load(open(os.path.join(HERE, "episodes.json")))
              if rec["suite"] == "6p" for s in spells(rec)
              if (not models or rec["model"] in models)]
        op = [s for s in sp if s["opening"]]
        mid = [s for s in sp if not s["opening"]]
        print("immobile spells (6p only: below six players a teammate still has")
        print("another object to move): %d in %d episodes"
              % (len(sp), len({s["meta"]["path"] for s in sp})))
        print()
        print("  opening hazard -- every S4 board starts with B walled in: %d" % len(op))
        print("    relieved by the first accepted move : %d" % sum(s["end"] == "freed" for s in op))
        print("    never relieved                      : %d (all of them episodes"
              " in which no move was ever accepted)" % sum(s["end"] != "freed" for s in op))
        print()
        print("  formed in play: %d" % len(mid))
        for g in ("prose", "rigid"):
            sel = [s for s in mid if group(s) == g]
            print("    %-6s freed %2d of %2d   victim's own turn came first %2d"
                  % (g, sum(s["end"] == "freed" for s in sel), len(sel),
                     sum(s["own_turn"] for s in sel)))
        print("    the player who caused the spell is never the one who ends it: %d of %d"
              % (sum(s["freer"] != s["cause"] for s in mid if s["end"] == "freed"),
                 sum(s["end"] == "freed" for s in mid)))
        print("    a spell that reaches the victim's own turn is fatal: %d of %d"
              % (sum(s["meta"]["outcome"] != "success" for s in sp if s["own_turn"]),
                 sum(s["own_turn"] for s in sp)))
        if args.list or args.quotes:
            print()
            for s in sorted((x for x in mid if x["end"] == "freed"),
                            key=lambda x: -x["length"]):
                ev = relief(s)
                if args.quotes and not any("names victim" in t for _, t, _ in ev):
                    continue
                m = s["meta"]
                print("%-8s %-3s %-9s %-14s P%s stuck %d-%d (%d plies) freed by P%s %s  %s"
                      % (m["model"], m["level"], m["proto"], m["inst"], s["player"],
                         s["start"], s["end_ply"], s["length"], s["freer"],
                         "".join(s["by"] or ()), m["outcome"]))
                if args.quotes:
                    for ply, tags, text in ev:
                        print("      [ply %d %s] %s"
                              % (ply, "+".join(tags), " ".join(text.split())[:330]))
        return

    data, seen = rows(models=models, protos=tuple(args.proto))
    free = [r for r in data if r["safe"]]
    cost = [r for r in free if r["costly"]]
    print("%s, dyad/3p/6p, %s" % (" + ".join(args.proto),
          args.tier and "%s tier (%s)" % (args.tier, ", ".join(models))
          or args.model or "all three models"))
    print("  episodes scanned                    : %d" % seen)
    print("  pinches (a legal move of the mover would")
    print("  leave an active teammate with no move): %d in %d episodes"
          % (len(data), len({r["meta"]["path"] for r in data})))
    print("    no harmless alternative existed    : %d" % sum(not r["safe"] for r in data))
    print("    a choice existed                   : %d" % len(free))
    print("      sealing move played              : %d" % sum(r["taken"] for r in free))
    print("      declined                         : %d" % sum(not r["taken"] for r in free))
    print("    and the sealing move was the only one")
    print("    that made progress (the S9.3.1 fork) : %d" % len(cost))
    print("      taken                            : %d" % sum(r["taken"] for r in cost))
    print("      declined                         : %d" % sum(not r["taken"] for r in cost))
    print()
    dec = [r for r in free if not r["taken"]]
    named = [r for r in dec if any("names victim" in t for _, t, _ in reason(r))]
    print("  of the %d declines, the mover's message names the teammate it could"
          " have stranded: %d" % (len(dec), len(named)))
    print("    and uses give-way language        : %d"
          % sum(1 for r in dec if any({"names victim", "gives way"} <= set(t)
                                      for _, t, _ in reason(r))))
    print("    and names immobility outright     : %d"
          % sum(1 for r in dec if any({"names victim", "immobility"} <= set(t)
                                      for _, t, _ in reason(r))))
    warned = [r for r in dec if warning(r)]
    print("  declines where somebody named the fatal cell in the six plies before")
    print("  it was offered: %d, of which the victim itself warned: %d"
          % (len(warned),
             sum(1 for r in warned
                 if any("victim's own warning" in t for _, _, t, _ in warning(r)))))
    print("    and the warning says the victim would be stuck: %d"
          % sum(1 for r in warned
                if any("immobility" in t for _, _, t, _ in warning(r))))
    print("  by model : %s" % dict(collections.Counter(r["meta"]["model"] for r in dec)))
    print("  by proto : %s" % dict(collections.Counter(r["meta"]["proto"] for r in dec)))
    print("  by level : %s" % dict(collections.Counter(r["meta"]["level"] for r in dec)))

    if args.list or args.quotes:
        print()
        show = named if args.quotes else data
        for r in sorted(show, key=lambda x: (x["meta"]["model"], x["meta"]["level"],
                                             x["meta"]["proto"], x["meta"]["inst"], x["ply"])):
            m = r["meta"]
            print("%-8s %-3s %-9s %-14s ply %-4d P%s would strand %s by %s; played %s %s%s  %s"
                  % (m["model"], m["level"], m["proto"], m["inst"], r["ply"], r["player"],
                     "".join(r["victims"]), "/".join("%s %s" % s for s in r["seals"]),
                     "%s %s" % r["played"], "TAKEN" if r["taken"] else "declined",
                     " (costly)" if r["costly"] else "", m["outcome"]))
            if args.quotes:
                for ply, q, tags, text in warning(r):
                    print("      warn  [ply %d P%s %s] %s"
                          % (ply, q, "+".join(tags) or "names the cell",
                             " ".join(text.split())[:330]))
                for ply, tags, text in reason(r):
                    print("      mover [ply %d %s] %s"
                          % (ply, "+".join(tags), " ".join(text.split())[:330]))


if __name__ == "__main__":
    main()
