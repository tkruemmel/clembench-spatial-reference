"""The counterfactual of S9.3.2: cut hazards that were *not* taken.

S9.3.2 shows 101 permanent cuts, every one formed by a `done` declared by a
player standing on its own target. This asks the opposite question of the same
corpus: how often did a player reach that exact position -- every object home,
`done` still in hand, and declaring it would sever a teammate's last route --
and *not* declare? And where it held off, do its own messages say why?

The decision point is the eligible instant: all of the player's objects are on
their targets, so the game invites the declaration. A hazard is a victim
teammate whose route exists now and would not exist if this player froze.

    python3 analysis/qualitative_analysis/averted.py            # the counts
    python3 analysis/qualitative_analysis/averted.py --list     # one line per case
    python3 analysis/qualitative_analysis/averted.py --quotes   # with transcript evidence
    python3 analysis/qualitative_analysis/averted.py --proto none structured
    python3 analysis/qualitative_analysis/averted.py --check    # re-derive the counts

The answer, in one line: the prose protocols reach the fork seven times less
often (12 of 2705 done-eligible players against 89 of 2446), and they are the
only ones that ever step off it -- 4 give-ways against 0.
"""
import argparse
import collections
import json
import os
import re
import string
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DIRS = tuple(STEP.values())
PROTOS = ("freeform", "hybrid")
# the two tiers S9.3 reports separately; Nemotron rarely gets a player's
# objects home at all, so it barely reaches this fork in either arm
TIERS = {"strong": ("qwen38", "dsv41"), "weak": ("nemotron",)}

# Language that names the hazard rather than merely coordinating a route.
# DONE_HOLD: the player ties its own declaration to a condition or defers it.
# TRAP: the player names the consequence -- shutting a teammate in or out.
DONE_HOLD = re.compile(
    r"\b(?:not|won'?t|will not|shan'?t|no[t]? yet|before|until|unless|hold(?:ing)? off|"
    r"wait\w*|delay\w*|defer\w*|refrain\w*|keep\w*)\b[^.;!?\n]{0,90}\bdone\b"
    r"|\bdone\b[^.;!?\n]{0,90}\b(?:only after|only once|not until|but not yet|"
    r"after (?:you|he|she|they|[A-F]\b)|once .{0,45}(?:clear|pass|through|home|place)|"
    r"last|later)\b", re.I)
TRAP = re.compile(
    r"\b(?:trap\w*|seal\w*|strand\w*|lock(?:ed|ing|s)? (?:in|out|away)|shut (?:in|out)|"
    r"cut off|cutting off|box(?:ed|ing)? in|wall(?:ed|ing)? (?:in|off)|"
    r"permanent\w*|irreversib\w*|freez\w*|frozen|no way out|"
    r"unreachable|can'?t reach|cannot reach|block(?:ed|ing)? (?:it |them |[A-F] )?"
    r"(?:in|forever|permanently))\b", re.I)


def label(n, walls, blocked):
    """Component id per free cell, with `blocked` treated as wall."""
    comp, cid = {}, 0
    for r in range(n):
        for c in range(n):
            if (r, c) in walls or (r, c) in blocked or (r, c) in comp:
                continue
            stack, cid = [(r, c)], cid + 1
            comp[(r, c)] = cid
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS:
                    nb = (cr + dr, cc + dc)
                    if (0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls
                            and nb not in blocked and nb not in comp):
                        comp[nb] = cid
                        stack.append(nb)
    return comp


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
    """One record per (player, episode) that ever stood at a hazardous decision.

    Records the whole eligible stretch: how long the player sat with every
    object home and `done` in hand, whether declaring would have cut a teammate
    at any of those instants, and how the standoff ended.
    """
    d = os.path.join(REPO, rec["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    groups = ins.get("player_objects") or [ins["player_a_objects"],
                                           ins["player_b_objects"]]
    letters = string.ascii_uppercase
    owner = {o: letters[i] for i, g in enumerate(groups) for o in g}
    mine = {letters[i]: list(g) for i, g in enumerate(groups)}
    seat = {"Player %d" % (i + 1): letters[i] for i in range(len(groups))}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    device = ins.get("device") or {}
    entrance = tuple(device["entrance"]) if device.get("entrance") else None
    corridor = {tuple(c) for c in device.get("corridor", [])}

    done, ply = set(), 0
    said = collections.defaultdict(list)
    book = {}          # player -> record, created on its first eligible instant
    live = {}          # player -> victims live at the last sweep

    def hazard(p):
        """(victims, door objects of p) if p froze now; doors reopen a route."""
        frozen = {pos[o] for o in owner if owner[o] in done}
        add = {pos[o] for o in mine[p]}
        before, after = label(n, walls, frozen), label(n, walls, frozen | add)
        vic = [o for o in tgt
               if owner[o] != p and owner[o] not in done and pos[o] != tgt[o]
               and before.get(pos[o]) and before.get(pos[o]) == before.get(tgt[o])
               and after.get(pos[o]) != after.get(tgt[o])]
        doors = []
        if vic:
            for o in mine[p]:
                lab = label(n, walls, (frozen | add) - {pos[o]})
                if any(lab.get(pos[v]) and lab.get(pos[v]) == lab.get(tgt[v])
                       for v in vic):
                    doors.append(o)
        return vic, doors

    def sweep():
        for p in mine:
            if p in done or not all(pos[o] == tgt[o] for o in mine[p]):
                live.pop(p, None)
                continue
            vic, doors = hazard(p)
            live[p] = vic
            r = book.setdefault(p, dict(
                player=p, elig_from=ply, elig_plies=0, haz_plies=0,
                first_haz=None, last_haz=None, victims=set(), doors=set(),
                cells=set(), vacated=[], declared_at=None, cut=False))
            r["elig_plies"] += 1
            if vic:
                r["haz_plies"] += 1
                r["last_haz"] = ply
                if r["first_haz"] is None:
                    r["first_haz"] = ply
                r["victims"].update(vic)
                r["doors"].update(doors)
                r["cells"].update(pos[o] for o in doors)

    sweep()
    for kind, a, b in stream(it):
        if kind == "msg":
            p = seat.get(a)
            if p:
                said[p].append((ply, b))
            continue
        if kind == "move":
            p = owner[a]
            # a door object stepping off its own target while the hazard is
            # live is the unambiguous give-way: the player un-finishes itself
            if live.get(p) and a in book.get(p, {}).get("doors", ()) \
                    and pos[a] == tgt[a]:
                book[p]["vacated"].append((ply, a))
            dr, dc = STEP[b]
            pos = dict(pos)
            pos[a] = (pos[a][0] + dr, pos[a][1] + dc)
            ply += 1
        else:
            if a in book:
                book[a]["declared_at"] = ply
                book[a]["cut"] = bool(live.get(a))
            done.add(a)
        sweep()

    out = []
    home_end = {o for o in tgt if pos[o] == tgt[o]}
    for p, r in book.items():
        r.update(meta=rec, grid=n, walls=walls, owner=owner, mine=mine,
                 targets=tgt, entrance=entrance, corridor=corridor,
                 said=said[p], plies=ply, end_pos=pos,
                 on_entrance=(entrance is not None and entrance in r["cells"]),
                 victims_home=sorted(v for v in r["victims"] if v in home_end),
                 still_live=bool(live.get(p)),
                 waited=(0 if r["first_haz"] is None else
                         (r["declared_at"] if r["declared_at"] is not None else ply)
                         - r["first_haz"]))
        out.append(r)
    return out


def rows(models=None, protos=PROTOS, suites=("dyad", "3p", "6p"), all_eligible=False):
    """Hazardous decision points; with `all_eligible`, every eligible player.

    The eligible players are the fair denominator: a player only faces the
    question at all once every one of its objects is home and the game is
    inviting it to declare.
    """
    eps = json.load(open(os.path.join(HERE, f"episodes.{ARM}.json")))
    out, seen = [], 0
    for rec in eps:
        if rec["suite"] not in suites or rec["proto"] not in protos:
            continue
        if models and rec["model"] not in models:
            continue
        seen += 1
        out.extend(r for r in scan(rec) if all_eligible or r["haz_plies"])
    return out, seen


def evidence(run, window=8):
    """The holder's own words around the hazard, and which family they name."""
    lo, hi = run["first_haz"] - window, run["last_haz"] + window
    hits = []
    for ply, text in run["said"]:
        if lo <= ply <= hi:
            tags = [t for t, rx in (("done", DONE_HOLD), ("trap", TRAP))
                    if rx.search(text)]
            if tags:
                hits.append((ply, "+".join(tags), text))
    return hits


def classify(r):
    if r["cut"]:
        return "cut"
    if r["vacated"]:
        return "gave way"
    if r["declared_at"] is not None:
        return "held, then declared safely"
    if r["still_live"]:
        return "held to the end, hazard still live"
    return "held, never declared"


def selftest():
    """The counts S9.3 quotes, re-derived. Fails loudly if the corpus moves."""
    ff, n_ff = rows(protos=("freeform", "hybrid"), all_eligible=True)
    ns, n_ns = rows(protos=("none", "structured"), all_eligible=True)
    assert (n_ff, n_ns) == (1080, 1080), (n_ff, n_ns)
    assert (len(ff), len(ns)) == (2705, 2446), (len(ff), len(ns))
    hz = lambda rs: [r for r in rs if r["haz_plies"]]
    assert (len(hz(ff)), len(hz(ns))) == (12, 89), (len(hz(ff)), len(hz(ns)))
    assert (sum(r["cut"] for r in hz(ff)),
            sum(r["cut"] for r in hz(ns))) == (8, 87)
    gave = lambda rs: sum(classify(r) == "gave way" for r in hz(rs))
    assert (gave(ff), gave(ns)) == (4, 0), (gave(ff), gave(ns))
    # by tier: the strong tier carries the whole effect, and the weak tier
    # barely reaches the fork in either arm (113 and 81 eligible players)
    for tier, want in (("strong", ((2705 - 113, 9, 6, 3), (2446 - 81, 89, 87, 0))),
                       ("weak", ((113, 3, 2, 1), (81, 0, 0, 0)))):
        for rs, w in zip((ff, ns), want):
            d = [r for r in rs if r["meta"]["model"] in TIERS[tier]]
            got = (len(d), len(hz(d)), sum(r["cut"] for r in hz(d)), gave(d))
            assert got == w, (tier, got, w)
    # the two non-cuts outside the prose protocols are aborts, not aversions
    assert all(r["meta"]["outcome"] == "abort"
               for r in hz(ns) if not r["cut"])
    # the two give-ways that mirror the figures of S9.3.2 sit on the same
    # boards as the cuts they mirror
    import itertools
    for a, b in (("S3_freeform_6p_grid/instance_00014",
                  "S3_hybrid_6p_grid/instance_00014"),
                 ("S4_none_6p_grid/instance_00003",
                  "S4_hybrid_6p_grid/instance_00003")):
        got = [json.load(open(os.path.join(REPO, r["meta"]["path"], "instance.json")))
               for r in itertools.chain(ff, ns)
               for tail in (a, b) if r["meta"]["path"].endswith(tail)]
        keys = ("grid_size", "walls", "start_positions", "target_positions",
                "player_objects", "device")
        assert all(g[k] == got[0][k] for g in got for k in keys), (a, b)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quotes", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--tier", choices=sorted(TIERS))
    ap.add_argument("--proto", nargs="*", default=list(PROTOS))
    ap.add_argument("--check", action="store_true", help="re-derive the quoted counts")
    args = ap.parse_args()
    if args.check:
        return selftest()
    models = TIERS[args.tier] if args.tier else (
        (args.model,) if args.model else None)
    data, seen = rows(models=models, protos=tuple(args.proto))

    eps = {r["meta"]["path"] for r in data}
    print("freeform + hybrid, dyad/3p/6p, %s"
          % (args.tier and "%s tier (%s)" % (args.tier, ", ".join(models))
             or args.model or "all three models"))
    print("  episodes scanned                        : %d" % seen)
    print("  episodes where a player stood done-eligible")
    print("  with a teammate-cutting declaration available: %d (%d player-cases)"
          % (len(eps), len(data)))
    print()
    kinds = collections.Counter(classify(r) for r in data)
    for k, v in kinds.most_common():
        print("    %-34s %4d" % (k, v))
    print()
    av = [r for r in data if not r["cut"]]
    print("  averted: %d of %d hazardous decision points" % (len(av), len(data)))
    print("    hazard cell was the corridor entrance : %d" % sum(r["on_entrance"] for r in av))
    print("    every threatened teammate got home    : %d"
          % sum(len(r["victims_home"]) == len(r["victims"]) for r in av))
    print("    plies the player sat on it: median %d, max %d"
          % (sorted(r["waited"] for r in av)[len(av) // 2],
             max(r["waited"] for r in av)))
    print()
    print("  by model : %s" % dict(collections.Counter(r["meta"]["model"] for r in av)))
    print("  by proto : %s" % dict(collections.Counter(r["meta"]["proto"] for r in av)))
    print("  by suite : %s" % dict(collections.Counter(r["meta"]["suite"] for r in av)))
    print("  by level : %s" % dict(collections.Counter(r["meta"]["level"] for r in av)))
    print("  outcome  : %s" % dict(collections.Counter(r["meta"]["outcome"] for r in av)))
    print()
    withev = [r for r in av if evidence(r)]
    print("  averted AND the holder names the hazard in its own messages: %d"
          " spells in %d episodes" % (len(withev), len({r["meta"]["path"] for r in withev})))
    print("    defers its own `done`        : %d"
          % sum(1 for r in withev if any("done" in t for _, t, _ in evidence(r))))
    print("    names trapping/sealing       : %d"
          % sum(1 for r in withev if any("trap" in t for _, t, _ in evidence(r))))
    print("    gave way (stepped off target): %d" % sum(1 for r in withev if r["vacated"]))

    if args.list or args.quotes:
        print()
        show = withev if args.quotes else data
        for r in sorted(show, key=lambda x: (x["meta"]["model"], x["meta"]["suite"],
                                             x["meta"]["level"], x["meta"]["inst"])):
            m = r["meta"]
            print("%-8s %-3s %-3s %-9s %-14s P%s doors %s victims %s(home %s) "
                  "haz %d-%d/%d elig, waited %d %s -> %-26s ep=%s"
                  % (m["model"], m["suite"], m["level"], m["proto"], m["inst"],
                     r["player"], "".join(sorted(r["doors"])),
                     "".join(sorted(r["victims"])), "".join(r["victims_home"]) or "-",
                     r["first_haz"], r["last_haz"], r["elig_plies"], r["waited"],
                     "ENTRANCE" if r["on_entrance"] else "        ",
                     classify(r), m["outcome"]))
            if args.quotes:
                for ply, tags, text in evidence(r)[:3]:
                    print("      [ply %d %s] %s"
                          % (ply, tags, " ".join(text.split())[:320]))


if __name__ == "__main__":
    main()
