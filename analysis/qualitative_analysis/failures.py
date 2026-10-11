"""The numbers S9.3 (enclosure, `done` cut, both counterfactuals) and S9.4 (board
S3 #10) state that no other script prints, on one arm.

seal_alt.py, cut.py, seal_averted.py and averted.py replay the four analyses;
this module re-uses their replays and derives what they do not print: the
residual prose failures, the victim's one-exit spell, the 3p/dyad enclosures,
the pinch and fork tables by tier, `done`-cut rates by protocol and team size,
the give-ways and whether they were announced, the invitation chain under a
written coding rule, the S4 opening, and the boards S3 #10 and S4 #8.

    ARM=updated python3 analysis/qualitative_analysis/failures.py           # the numbers
    ARM=updated python3 analysis/qualitative_analysis/failures.py --tests   # the p-values
    ARM=updated python3 analysis/qualitative_analysis/failures.py --check   # assert the pins

Strong tier = Qwen3.8 + DeepSeek; prose = freeform + hybrid; rigid = none +
structured. Every p-value here is exploratory (none is one of the nineteen
contrasts): two-sided Fisher exact. Where the unit is a decision instant, the
instants of one episode are not independent, so each such test is repeated on
episodes.
"""
import argparse
import collections
import json
import math
import os
import re
import statistics as st
import sys

from scipy.stats import fisher_exact

import averted as av
import cut
import seal_alt as sa
import seal_averted as sv

ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
STRONG = ("qwen38", "dsv41")
PROSE, RIGID = ("freeform", "hybrid"), ("none", "structured")
ALL4 = PROSE + RIGID
SUITES = ("dyad", "3p", "6p")
ARMS = {"prose": PROSE, "rigid": RIGID}


def arm_of(proto):
    return "prose" if proto in PROSE else "rigid"


def tier_of(model):
    return "strong" if model in STRONG else "weak"


# --- the invitation chain (S9.3.1): a written coding rule --------------------
# A victim's message "invites" the seal if a sentence that names the sealing
# cell (a) addresses the mover with a movement verb ("D, take R8C5", "C can
# occupy", "F wait (R3,C5)"), (b) frees/clears/opens/vacates something "for"
# the mover, or (c) calls a cell free/clear/open/yours while naming the mover.
# Calibrated on the original arm, where it reproduces the hand reading pinned
# in figures/invitation.py exactly (10 invited, 2 not, of 12).
VERBS = (r"(?:take|use|occupy|enter|go|move|step|climb|ascend|advance|proceed|"
         r"continue|follow|rise|drop|descend|head|slide|shift|come)\b")
OFFER_TO = (r"\b{m}\b[,:]?\s*(?i:please\s+|can\s+|may\s+|could\s+|should\s+|then\s+)?"
            r"(?i:" + VERBS + r"|(?:hold|wait)\s+(?:at\s+|in\s+|on\s+)?\(?R\s*\d)")
FREED_FOR = r"(?i:\b(?:free|clear|open|vacat)\w*\b)[^;]{{0,40}}?(?i:\bfor\s+){m}\b"
IS_FREE = r"(?i:\bis\s+(?:now\s+)?(?:free|clear|open|yours|available)\b)"
HAND_INVITED = {   # figures/invitation.py's hand reading, original arm
    ("qwen38", "S3", "instance_00010", "freeform", 56),
    ("qwen38", "S3", "instance_00010", "hybrid", 82),
    ("qwen38", "S4", "instance_00008", "freeform", 17),
    ("qwen38", "S4", "instance_00008", "freeform", 21),
    ("qwen38", "S4", "instance_00011", "freeform", 19),
    ("dsv41", "S3", "instance_00006", "freeform", 24),
    ("dsv41", "S3", "instance_00010", "freeform", 47),
    ("dsv41", "S3", "instance_00010", "freeform", 53),
    ("dsv41", "S4", "instance_00005", "freeform", 33),
    ("dsv41", "S4", "instance_00008", "hybrid", 22),
}


def invites(text, cell_rx, mover):
    for s in sv.SENT.findall(text):
        if not cell_rx.search(s):
            continue
        m = re.escape(mover)
        if (re.search(OFFER_TO.format(m=m), s) or re.search(FREED_FOR.format(m=m), s)
                or (re.search(IS_FREE, s) and re.search(r"\b%s\b" % m, s))):
            return True
    return False


def invitation_chain(pinches):
    """30 -> named -> failing -> invited, as in figures/invitation.py, plus the
    rule applied to every named seal. `named`: a victim's message in the six
    plies before the seal names the sealing cell and the mover."""
    taken = [r for r in pinches if r["meta"]["model"] in STRONG
             and r["meta"]["proto"] in PROSE and r["taken"] and r["safe"]]
    out = []
    for r in taken:
        cell = sv.step(r["pos"], *r["played"])[r["played"][0]]
        rx = re.compile(sv.CELL % (cell[0] + 1, cell[1] + 1), re.I)
        msgs = [t for q in r["victims"] for p, t in r["said"].get(q, [])
                if r["ply"] - 6 <= p <= r["ply"] and rx.search(t)
                and re.search(r"\b%s\b" % r["player"], t)]
        if msgs:
            m = r["meta"]
            out.append(dict(key=(m["model"], m["level"], m["inst"], m["proto"], r["ply"]),
                            fail=m["outcome"] != "success",
                            invited=any(invites(t, rx, r["player"]) for t in msgs)))
    return taken, out


# --- give-ways (S9.3.5): announced before the step-off? ----------------------
# The holder's own message, from the turn it settled on the door (first
# hazardous ply - 1) up to the ply before it steps off, names a threatened
# teammate and states a future step ("I'll step up to (R1,C5) just before you
# enter", "I can temporarily step to R2C6").
FUTURE = re.compile(r"\b(?:I['\u2019]?ll|I will|I can|can|will)\b[^.;!?\n]{0,80}?"
                    r"\b(?:step|move|drop|slide|shift|vacate|clear|yield|return)\b", re.I)


def announced(r):
    v = r["vacated"][0][0]
    return [p for p, t in r["said"] if r["first_haz"] - 1 <= p < v and FUTURE.search(t)
            and any(re.search(r"\b%s\b" % x, t) for x in r["victims"])]


# --- loading -----------------------------------------------------------------
def load():
    E = json.load(open(os.path.join(HERE, f"episodes.{ARM}.json")))
    return dict(
        E=E,
        enc=sa.cases(),
        cuts=cut.cases(),
        pinches=sv.rows(protos=ALL4)[0],
        spells=[s for rec in E if rec["suite"] == "6p" for s in sv.spells(rec)],
        elig=av.rows(protos=ALL4, all_eligible=True)[0],
    )


def med(x):
    return st.median(x) if x else None


def numbers(D):
    """Every derived number, keyed by a stable name. Values are ints, tuples or
    short dicts so --check can pin them exactly."""
    E, N = D["E"], collections.OrderedDict()
    strong_multi = [e for e in E if e["model"] in STRONG and e["suite"] in SUITES]
    enc_paths = {c["meta"]["path"] for c in D["enc"]}
    cut_paths = {r["meta"]["path"] for r in D["cuts"]}

    # residual failures under prose (raw:473, disc:63)
    def kind(e):
        if e["path"] in enc_paths:
            return "enclosure abort"
        if e["path"] in cut_paths:
            return "cycle loss, permanent cut"
        if "cycle" in (e["end_msg"] or ""):
            return "cycle loss, no cut"
        return e["outcome"] + ": " + (e["end_msg"] or "")[:40]
    for a, protos in ARMS.items():
        eps = [e for e in strong_multi if e["proto"] in protos]
        bad = [e for e in eps if e["outcome"] != "success"]
        N[f"residual.{a}.nonsuccess_of"] = (len(bad), len(eps))
        N[f"residual.{a}.by_cell"] = dict(sorted(collections.Counter(
            f"{e['suite']} {e['level']}" for e in bad).items()))
        N[f"residual.{a}.by_kind"] = dict(sorted(collections.Counter(map(kind, bad)).items()))
        N[f"residual.{a}.6p_S3S4"] = sum(e["suite"] == "6p" and e["level"] in ("S3", "S4")
                                        for e in bad)
        N[f"residual.{a}.6p_S3S4_by_kind"] = dict(sorted(collections.Counter(
            kind(e) for e in bad if e["suite"] == "6p" and e["level"] in ("S3", "S4")).items()))

    # S9.3.1: the enclosure aborts
    enc = D["enc"]
    N["enc.aborts_strong_by_suite"] = dict(sorted(collections.Counter(
        e["suite"] for e in strong_multi if e["outcome"] == "abort").items()))
    N["enc.n"] = len(enc)
    N["enc.harmless_alt"] = sum(bool(c["harmless"]) for c in enc)
    N["enc.sealed_by_teammate"] = sum(c["sealer"] != c["meta"]["who"] for c in enc)
    N["enc.by_proto"] = dict(sorted(collections.Counter(c["meta"]["proto"] for c in enc).items()))
    N["enc.by_model"] = dict(sorted(collections.Counter(c["meta"]["model"] for c in enc).items()))
    since, spell = [], []
    for c in enc:
        pos, hist = dict(c["start"]), []
        for o, d in c["moves"][:c["ply"] - 1]:
            hist.append(len(sa.legal_moves(pos, [c["victim"]], c["walls"], c["grid"])))
            pos = sa.step(pos, o, d)
        hist.append(len(sa.legal_moves(pos, [c["victim"]], c["walls"], c["grid"])))
        assert hist[-1] == 1  # one exit left just before the seal, in every case
        since.append(len(hist) - 1 - hist.index(1))
        k = 0
        while k < len(hist) and hist[-1 - k] == 1:
            k += 1
        spell.append(k - 1)
    N["enc.victim_one_exit_unbroken_moves"] = (med(spell), round(st.mean(spell), 2), max(spell))
    N["enc.victim_moves_since_first_one_exit"] = (med(since), round(st.mean(since), 1),
                                                  min(since), max(since))
    keys, g = sa.greedy_rows(enc)
    N["enc.greedy"] = {k: dict(sorted(g[k].items())) for k in keys}
    man = []
    for c in enc:
        if c["harmless"]:
            o, d = c["played"]
            t, p, q = c["targets"][o], c["pre"][o], sa.step(c["pre"], o, d)[o]
            if abs(q[0] - t[0]) + abs(q[1] - t[1]) >= abs(p[0] - t[0]) + abs(p[1] - t[1]):
                man.append(sa.progress(c, c["pre"], o, d))
    N["enc.manhattan_retreats(n, of_which_wall_detours)"] = (len(man), sum(man))

    # the same enclosure below six players (raw:569): after some accepted
    # move, an off-target object has no legal move
    below = collections.Counter()
    for e in strong_multi:
        if e["suite"] == "6p":
            continue
        d = os.path.join(REPO, e["path"])
        ins = json.load(open(os.path.join(d, "instance.json")))
        it = json.load(open(os.path.join(d, "interactions.json")))
        n, walls = ins["grid_size"], frozenset(tuple(w) for w in ins["walls"])
        pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
        tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
        hit = False
        for k, a, b in sv.stream(it):
            if k == "move":
                pos = sv.step(pos, a, b)
                hit = hit or any(pos[o] != tgt[o] and not sv.legal(pos, [o], walls, n)
                                 for o in pos)
        if hit:
            below[(e["model"], e["suite"])] += 1
            below[(e["model"], e["suite"], e["outcome"])] += 1
    N["enc.below6p_offtarget_enclosed_episodes"] = {
        f"{m} {s}": below[(m, s)] for m in STRONG for s in ("dyad", "3p")}
    N["enc.below6p_of_those_aborted"] = sum(v for k, v in below.items()
                                            if len(k) == 3 and k[2] == "abort")

    # S9.3.5: pinches, by tier and arm
    P = D["pinches"]
    N["pinch.total_all_models"] = len(P)
    N["pinch.suites"] = sorted({r["meta"]["suite"] for r in P})
    for t in ("strong", "weak"):
        for a, protos in ARMS.items():
            d = [r for r in P if tier_of(r["meta"]["model"]) == t and r["meta"]["proto"] in protos]
            fr = [r for r in d if r["safe"]]
            co = [r for r in fr if r["costly"]]
            N[f"pinch.{t}.{a}"] = dict(
                pinches=len(d), episodes=len({r["meta"]["path"] for r in d}),
                no_alt=len(d) - len(fr), free=len(fr),
                declined=sum(not r["taken"] for r in fr), taken=sum(r["taken"] for r in fr),
                costly=len(co), costly_declined=sum(not r["taken"] for r in co))
    for a, protos in ARMS.items():
        N[f"pinch.strong.{a}.enclosure_aborts"] = sum(c["meta"]["proto"] in protos for c in enc)
    taken = {(r["meta"]["path"], r["ply"]) for r in P if r["taken"]}
    N["pinch.aborts_recovered_as_taken"] = sum((c["meta"]["path"], c["ply"] - 1) in taken
                                               for c in enc)
    dec = [r for r in P if r["meta"]["model"] in STRONG and r["meta"]["proto"] in PROSE
           and r["safe"] and not r["taken"]]
    N["pinch.strong.prose.declines_naming_victim"] = (
        sum(any("names victim" in t for _, t, _ in sv.reason(r)) for r in dec), len(dec))
    warn = []
    for r in P:
        if r["safe"] and any("immobility" in t for _, _, t, _ in sv.warning(r)):
            m = r["meta"]
            warn.append((m["model"], m["suite"], m["level"], m["proto"], m["inst"],
                         r["ply"], "taken" if r["taken"] else "declined", m["outcome"]))
    N["pinch.immobility_warnings"] = sorted(warn)

    # spells: the S4 opening and the hazard from the victim's side
    S = D["spells"]
    op = [s for s in S if s["opening"]]
    mid = [s for s in S if not s["opening"]]
    N["spell.total"] = len(S)
    N["spell.opening(n, levels, players)"] = (len(op), sorted({s["meta"]["level"] for s in op}),
                                             sorted({s["player"] for s in op}))
    for t in ("strong", "weak"):
        o = [s for s in op if tier_of(s["meta"]["model"]) == t]
        N[f"spell.opening.{t}.relieved_of"] = (sum(s["end"] == "freed" for s in o), len(o))
    for a in ARMS:
        o = [s for s in op if arm_of(s["meta"]["proto"]) == a]
        N[f"spell.opening.pooled.{a}.relieved_of"] = (sum(s["end"] == "freed" for s in o), len(o))
    for t in ("strong", "weak"):
        for a in ARMS:
            m_ = [s for s in mid if tier_of(s["meta"]["model"]) == t
                  and arm_of(s["meta"]["proto"]) == a]
            N[f"spell.mid.{t}.{a}.relieved_of"] = (sum(s["end"] == "freed" for s in m_), len(m_))
    fr = [s for s in mid if s["end"] == "freed"]
    N["spell.mid.freer_not_cause_of"] = (sum(s["freer"] != s["cause"] for s in fr), len(fr))
    N["spell.own_turn_fatal_of"] = (sum(s["meta"]["outcome"] != "success" for s in S if s["own_turn"]),
                                    sum(s["own_turn"] for s in S))

    # the invitation chain
    tk, named = invitation_chain(P)
    fail = [x for x in named if x["fail"]]
    N["invite.chain(taken, named, failing, invited)"] = (
        len(tk), len(named), len(fail), sum(x["invited"] for x in fail))
    N["invite.invited_among_named"] = (sum(x["invited"] for x in named), len(named))
    N["invite.failing"] = sorted((x["key"], x["invited"]) for x in fail)

    # S9.3.2: the permanent cut
    C = D["cuts"]
    cyc = [e for e in strong_multi if e["suite"] in ("3p", "6p") and e["outcome"] == "lose"]
    cyc_c = [e for e in cyc if "cycle" in (e["end_msg"] or "")]
    N["cut.strong_3p6p_losses(all, cycle)"] = (len(cyc), len(cyc_c))
    N["cut.cycle_losses_with_cut"] = (len(cut_paths), len(cyc_c))
    N["cut.by_model_suite"] = {f"{m} {s}": (len({e['path'] for e in cyc_c if e['model'] == m
                                                  and e['suite'] == s} & cut_paths),
                                            sum(e["model"] == m and e["suite"] == s for e in cyc_c))
                               for m in STRONG for s in ("3p", "6p")}
    first = collections.defaultdict(int)
    for r in C:
        first[r["meta"]["path"]] = max(first[r["meta"]["path"]], r["tail"])
    tails = list(first.values())
    N["cut.moves_after_first_cut(median, mean, max)"] = (med(tails), round(st.mean(tails), 1),
                                                        max(tails))
    N["cut.episodes_cut_more_than_10_moves_before_end"] = sum(t > 10 for t in tails)
    geo = 0
    for e in cyc_c:
        d = os.path.join(REPO, e["path"])
        ins = json.load(open(os.path.join(d, "instance.json")))
        it = json.load(open(os.path.join(d, "interactions.json")))
        n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
        pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
        tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
        for k, a, b in sv.stream(it):
            if k == "move":
                pos = sv.step(pos, a, b)
        geo += sum(pos[o] != tgt[o] and cut.path(pos[o], tgt[o], n, walls, set()) is None
                   for o in pos)
    N["cut.unfinished_objects_unreachable_by_walls_alone"] = geo
    N["cut.objects"] = len(C)
    N["cut.event_done_of"] = (sum(1 for _ in C), len(C))   # cut.cases asserts each is a done
    N["cut.sealer_on_own_target"] = sum(r["pre"][r["sealer"]] == r["targets"][r["sealer"]] for r in C)
    N["cut.single_object_door"] = sum(bool(r["culprits"]) for r in C)
    N["cut.shape"] = dict(sorted(collections.Counter(r["kind"] for r in C).items()))
    N["cut.shape_by_level"] = dict(sorted(collections.Counter(
        f"{r['meta']['level']} {r['kind']}" for r in C).items()))
    ch = [r for r in C if r["kind"] in cut.CHOKEPOINTS]
    N["cut.chokepoint(objects, episodes)"] = (len(ch), len({r["meta"]["path"] for r in ch}))
    N["cut.mincut_by_shape"] = {k: sorted({r["mincut"] for r in C if r["kind"] == k})
                                for k in sorted({r["kind"] for r in C})}
    N["cut.by_protocol_suite_episodes"] = {
        f"{p} {s}": len({r["meta"]["path"] for r in C if r["meta"]["proto"] == p
                         and r["meta"]["suite"] == s}) for p in ALL4 for s in ("3p", "6p")}
    N["cut.episodes_per_suite_protocol"] = sum(1 for e in strong_multi if e["suite"] == "6p"
                                               and e["proto"] == "freeform")

    # S9.3.5: the fork (done-eligible instants), by tier and arm
    EL = D["elig"]
    for t in ("strong", "weak"):
        for a, protos in ARMS.items():
            d = [r for r in EL if tier_of(r["meta"]["model"]) == t and r["meta"]["proto"] in protos]
            hz = [r for r in d if r["haz_plies"]]
            N[f"fork.{t}.{a}"] = dict(
                eligible=len(d), eligible_episodes=len({r["meta"]["path"] for r in d}),
                forks=len(hz), fork_episodes=len({r["meta"]["path"] for r in hz}),
                **{k.replace(" ", "_").replace(",", ""): v
                   for k, v in sorted(collections.Counter(av.classify(r) for r in hz).items())})
    sd = [r for r in EL if r["meta"]["model"] in STRONG]
    N["fork.strong.by_protocol_suite(eligible, forks, cut, gave_way)"] = {
        f"{p} {s}": (sum(1 for r in sd if r["meta"]["proto"] == p and r["meta"]["suite"] == s),
                     sum(1 for r in sd if r["meta"]["proto"] == p and r["meta"]["suite"] == s
                         and r["haz_plies"]),
                     sum(1 for r in sd if r["meta"]["proto"] == p and r["meta"]["suite"] == s
                         and r["haz_plies"] and av.classify(r) == "cut"),
                     sum(1 for r in sd if r["meta"]["proto"] == p and r["meta"]["suite"] == s
                         and r["haz_plies"] and av.classify(r) == "gave way"))
        for p in ALL4 for s in SUITES}
    gw = []
    for r in sd:
        if r["haz_plies"] and av.classify(r) == "gave way":
            m, ann = r["meta"], announced(r)
            gw.append((m["model"], m["suite"], m["level"], m["proto"], m["inst"], m["outcome"],
                       r["vacated"][0][0] - ann[0] if ann else None))
    N["fork.strong.give_ways(model, suite, level, proto, inst, outcome, announced_plies_ahead)"] = sorted(gw)
    # how much of the prose/rigid ratio of declarations into the fork comes
    # from reaching the fork at all: share of the log ratio, P(cut) = P(fork) x P(cut | fork)
    pr, ri = N["fork.strong.prose"], N["fork.strong.rigid"]
    reach = math.log((ri["forks"] / ri["eligible"]) / (pr["forks"] / pr["eligible"]))
    take = math.log((ri["cut"] / ri["forks"]) / (pr["cut"] / pr["forks"]))
    N["fork.strong.rigid_over_prose(reach, declare_given_fork, share_from_reach)"] = (
        round(math.exp(reach), 2), round(math.exp(take), 2), round(reach / (reach + take), 2))
    N["fork.cut_sealer_was_done_eligible"] = N["cut.sealer_on_own_target"]

    # S9.4: board S3 #10, and board S4 #8 (raw:936)
    b10 = [e for e in E if e["level"] == "S3" and e["inst"] == "instance_00010"
           and e["model"] in STRONG]
    N["s3_10.success_by_suite"] = {s: (sum(e["outcome"] == "success" for e in b10 if e["suite"] == s),
                                       sum(e["suite"] == s for e in b10))
                                   for s in ("solo",) + SUITES}
    N["s3_10.6p"] = sorted((e["model"], e["proto"], kind(e) if e["outcome"] != "success"
                            else "success", e["moves"]) for e in b10 if e["suite"] == "6p")
    sf = {json.loads(l)["game_id"]: json.loads(l)["stuckfree"] for l in open(os.path.join(
        REPO, "analysis/efficiency_baseline/attainable_6p.jsonl")) if '"S3"' in l}
    N["s3_10.stuckfree_6p"] = sf[10]
    b8 = [e for e in E if e["level"] == "S4" and e["inst"] == "instance_00008"
          and e["model"] in STRONG]
    rows = {}
    for e in b8:
        it = json.load(open(os.path.join(REPO, e["path"], "interactions.json")))
        a_moves = sum(1 for k, a, _ in sv.stream(it) if k == "move" and a == "A")
        rows.setdefault(e["suite"], []).append((a_moves, e["moves"], e["outcome"]))
    # A starts beside the dead end it must finish on, so it has to leave the
    # corridor and come back: 9 moves at least
    N["s4_8.A_moves_range_by_outcome"] = {
        f"{s} {o}": (min(r[0] for r in v if r[2] == o), max(r[0] for r in v if r[2] == o))
        for s, v in sorted(rows.items()) for o in sorted({r[2] for r in v})}
    N["s4_8.length_range_by_outcome"] = {
        f"{s} {o}": (min(r[1] for r in v if r[2] == o), max(r[1] for r in v if r[2] == o))
        for s, v in sorted(rows.items()) for o in sorted({r[2] for r in v})}
    N["s4_8.A_max_excess_over_9"] = max(r[0] for v in rows.values() for r in v) - 9
    N["s4_8.outcomes"] = {s: dict(sorted(collections.Counter(r[2] for r in v).items()))
                          for s, v in sorted(rows.items())}
    # S9.5: six-player efficiency of successes on the attainable 6p optimum and
    # on the free-order joint optimum, per level (tables_comm.all_episodes)
    sys.path.insert(0, os.path.join(REPO, "analysis", "thesis"))
    import tables_comm as tc
    TC = tc.all_episodes(tc.load_optima())
    # board S3 #10, Quality Score (played episodes) and % Played, strong tier
    for s_ in ("solo",) + SUITES:
        r = [e for e in TC if e["suite"] == s_ and e["level"] == "S3" and e["game_id"] == 10
             and e["model"] in ("Qwen3.8", "DeepSeek-V4.1-Flash")]
        q = [e["q"] for e in r if e["q"] is not None]
        N[f"s3_10.{s_}.QS_played"] = (round(st.mean(q), 1) if q else None,
                                      f"{len(q)}/{len(r)}")
    for m in ("Qwen3.8", "DeepSeek-V4.1-Flash"):
        r = [e for e in TC if e["model"] == m and e["suite"] == "6p" and e["success"]]
        for f in ("eff", "eff_j"):
            by = {lv: round(st.mean(e[f] for e in r if e["level"] == lv), 3)
                  for lv in ("S1", "S2", "S3", "S4")}
            by["all"] = round(st.mean(e[f] for e in r), 3)
            N[f"eff6p.{m}.{f}"] = by
    N["s4_8.6p_abort_moves"] = sorted(r[1] for r in rows.get("6p", []) if r[2] == "abort")
    N["s4_8.solo_A_moves"] = sorted(r[0] for r in rows.get("solo", []))
    return N


# The episodes the worked example and the figures of S9.3-S9.4 are drawn from
# (rewrite plan, Phase 4). Move numbers are 1-based; "after move k" = k moves
# accepted.
NAMED = (("qwen38", "6p", "S3", "hybrid", "instance_00010"),
         ("qwen38", "6p", "S3", "freeform", "instance_00010"),
         ("dsv41", "6p", "S3", "freeform", "instance_00010"),
         ("dsv41", "6p", "S3", "freeform", "instance_00014"),
         ("dsv41", "6p", "S3", "hybrid", "instance_00014"),
         ("qwen38", "6p", "S4", "hybrid", "instance_00005"),
         ("dsv41", "6p", "S4", "freeform", "instance_00008"),
         ("dsv41", "6p", "S3", "structured", "instance_00002"),
         ("dsv41", "6p", "S3", "freeform", "instance_00002"))


def episodes(D):
    """One line of facts per named episode: outcome and length, every `done`
    (after which move), the seal of an enclosure abort, the cut, the give-way."""
    key = lambda m: (m["model"], m["suite"], m["level"], m["proto"], m["inst"])
    out = {}
    for e in D["E"]:
        k = key(e)
        if k not in NAMED:
            continue
        f = dict(outcome=e["outcome"], moves=e["moves"])
        it = json.load(open(os.path.join(REPO, e["path"], "interactions.json")))
        ply, dn = 0, {}
        for kk, a, _ in sv.stream(it):
            if kk == "move":
                ply += 1
            elif kk == "done":
                dn[a] = ply
        f["done_after_move"] = dn
        for c in D["enc"]:
            if key(c["meta"]) == k:
                o = c["played"][0]
                dm = sa.dist_map(c["walls"], c["grid"], c["targets"][o])
                f["seal"] = dict(move=c["ply"], victim=c["victim"], sealer=o,
                                 harmless=[f"{x} {d}" for x, d in c["harmless"]],
                                 sealer_dist_to_target_after=dm[sa.step(c["pre"], *c["played"])[o]])
        for r in D["cuts"]:
            if key(r["meta"]) == k:
                f.setdefault("cuts", []).append(dict(after_move=r["ply"], victim=r["victim"],
                                                     sealer=r["sealer"], kind=r["kind"],
                                                     moves_after=r["tail"]))
        for r in D["elig"]:
            if key(r["meta"]) == k and r["haz_plies"]:
                f.setdefault("forks", []).append(dict(
                    player=r["player"], on_door_after_move=r["first_haz"],
                    last_hazard_after_move=r["last_haz"], outcome=av.classify(r),
                    stepped_off_at_move=(r["vacated"][0][0] + 1 if r["vacated"] else None)))
        out[k] = f
    return out


def tests(N):
    """The exploratory p-values S9.3 quotes or needs, two-sided Fisher exact."""
    out = []

    def f(name, a, n1, b, n2, unit):
        p = fisher_exact([[a, n1 - a], [b, n2 - b]])[1]
        out.append((name, a, n1, b, n2, p, unit))

    for t in ("strong",):
        pr, ri = N[f"pinch.{t}.prose"], N[f"pinch.{t}.rigid"]
        f("pinch declined | free pinch (prose vs rigid)", pr["declined"], pr["free"],
          ri["declined"], ri["free"], "pinch (instants)")
        f("costly pinch declined (prose vs rigid)", pr["costly_declined"], pr["costly"],
          ri["costly_declined"], ri["costly"], "pinch (instants)")
        f("taken free pinch -> enclosure abort (prose vs rigid)",
          N["pinch.strong.prose.enclosure_aborts"], pr["taken"],
          N["pinch.strong.rigid.enclosure_aborts"], ri["taken"], "taken pinch")
        n_ep = 2 * N["cut.episodes_per_suite_protocol"]   # 6p: every pinch is at 6p
        f("6p episodes with an enclosure abort (prose vs rigid)",
          N["pinch.strong.prose.enclosure_aborts"], n_ep,
          N["pinch.strong.rigid.enclosure_aborts"], n_ep, "episode")
        f("6p episodes with a pinch (prose vs rigid)", pr["episodes"], n_ep, ri["episodes"], n_ep,
          "episode")
    pw = [N[f"pinch.{t}.{a}"] for t in ("strong", "weak") for a in ARMS]
    f("POOLED tiers: pinch declined | free pinch (prose vs rigid)",
      pw[0]["declined"] + pw[2]["declined"], pw[0]["free"] + pw[2]["free"],
      pw[1]["declined"] + pw[3]["declined"], pw[1]["free"] + pw[3]["free"], "pinch (instants)")
    a, b = N["spell.mid.strong.prose.relieved_of"], N["spell.mid.strong.rigid.relieved_of"]
    f("mid-game spell relieved (prose vs rigid)", *a, *b, "spell")
    a, b = N["spell.opening.pooled.prose.relieved_of"], N["spell.opening.pooled.rigid.relieved_of"]
    f("POOLED tiers: S4 opening spell relieved (prose vs rigid)", *a, *b, "episode")
    for t in ("strong", "weak"):
        pr, ri = N[f"fork.{t}.prose"], N[f"fork.{t}.rigid"]
        f(f"{t}: fork reached | done-eligible player (prose vs rigid)", pr["forks"],
          pr["eligible"], ri["forks"], ri["eligible"], "player-episode")
        if t == "weak":
            continue
        f("fork reached | episode with an eligible player (prose vs rigid)",
          pr["fork_episodes"], pr["eligible_episodes"], ri["fork_episodes"],
          ri["eligible_episodes"], "episode")
        f("declared into the fork | done-eligible player (prose vs rigid)", pr.get("cut", 0),
          pr["eligible"], ri.get("cut", 0), ri["eligible"], "player-episode")
        f("declared | fork reached (prose vs rigid)", pr.get("cut", 0), pr["forks"],
          ri.get("cut", 0), ri["forks"], "fork")
        f("gave way | fork reached (prose vs rigid)", pr.get("gave_way", 0), pr["forks"],
          ri.get("gave_way", 0), ri["forks"], "fork")
    cp = N["cut.by_protocol_suite_episodes"]
    n_ep = 2 * N["cut.episodes_per_suite_protocol"] * 2   # 2 protocols x (3p, 6p)
    f("3p/6p episodes ending in a permanent cut (prose vs rigid)",
      sum(v for k, v in cp.items() if k.split()[0] in PROSE), n_ep,
      sum(v for k, v in cp.items() if k.split()[0] in RIGID), n_ep, "episode")
    return out


# Pinned values, per arm. On the original arm they are the values the old text
# states, where it states one; everything else is what this module returned.
PINS = {
    "original": {
        "residual.prose.nonsuccess_of": (14, 720),
        "enc.n": 32, "enc.harmless_alt": 30, "enc.sealed_by_teammate": 32,
        "enc.by_proto": {"freeform": 6, "hybrid": 3, "none": 14, "structured": 9},
        "enc.victim_one_exit_unbroken_moves": (1.0, 1.19, 4),
        "enc.victim_moves_since_first_one_exit": (14.5, 18.1, 0, 73),
        "enc.manhattan_retreats(n, of_which_wall_detours)": (3, 3),
        "enc.below6p_offtarget_enclosed_episodes": {
            "qwen38 dyad": 34, "qwen38 3p": 40, "dsv41 dyad": 44, "dsv41 3p": 46},
        "enc.below6p_of_those_aborted": 0,
        "pinch.total_all_models": 203, "pinch.suites": ["6p"],
        "pinch.strong.prose": dict(pinches=73, episodes=48, no_alt=0, free=73, declined=43,
                                   taken=30, costly=46, costly_declined=20),
        "pinch.strong.rigid": dict(pinches=110, episodes=58, no_alt=4, free=106, declined=62,
                                   taken=44, costly=72, costly_declined=32),
        "pinch.weak.prose": dict(pinches=20, episodes=12, no_alt=2, free=18, declined=2,
                                 taken=16, costly=10, costly_declined=1),
        "pinch.strong.prose.enclosure_aborts": 9, "pinch.strong.rigid.enclosure_aborts": 23,
        "pinch.aborts_recovered_as_taken": 32,
        "pinch.strong.prose.declines_naming_victim": (42, 43),
        "spell.total": 276, "spell.opening(n, levels, players)": (180, ["S4"], ["B"]),
        "spell.opening.strong.relieved_of": (120, 120),
        "spell.opening.pooled.prose.relieved_of": (85, 90),
        "spell.opening.pooled.rigid.relieved_of": (64, 90),
        "spell.mid.strong.prose.relieved_of": (19, 30),
        "spell.mid.strong.rigid.relieved_of": (25, 48),
        "spell.mid.freer_not_cause_of": (51, 51), "spell.own_turn_fatal_of": (39, 39),
        "invite.chain(taken, named, failing, invited)": (30, 29, 12, 10),
        "cut.cycle_losses_with_cut": (78, 91),
        "cut.by_model_suite": {"qwen38 3p": (17, 25), "qwen38 6p": (21, 24),
                               "dsv41 3p": (18, 20), "dsv41 6p": (22, 22)},
        "cut.moves_after_first_cut(median, mean, max)": (17.0, 22.1, 96),
        "cut.episodes_cut_more_than_10_moves_before_end": 52,
        "cut.unfinished_objects_unreachable_by_walls_alone": 0,
        "cut.objects": 101, "cut.sealer_on_own_target": 101,
        "cut.chokepoint(objects, episodes)": (97, 74),
        "fork.strong.prose": dict(eligible=2592, eligible_episodes=718, forks=9,
                                  fork_episodes=9, cut=6, gave_way=3),
        "fork.strong.rigid": dict(eligible=2365, eligible_episodes=701, forks=89,
                                  fork_episodes=87, cut=87, gave_way=None),
        "fork.strong.give_ways(model, suite, level, proto, inst, outcome, announced_plies_ahead)": [
            ("qwen38", "6p", "S3", "freeform", "instance_00007", "success", 4),
            ("qwen38", "6p", "S3", "hybrid", "instance_00014", "success", None),
            ("qwen38", "6p", "S4", "hybrid", "instance_00003", "success", None)],
        "enc.greedy": {"Qwen3.8": {"costly": 10, "free": 2, "n": 12, "nearer": 12, "noprog": 0},
                       "DeepSeek": {"costly": 15, "free": 3, "n": 18, "nearer": 18, "noprog": 0},
                       "all": {"costly": 25, "free": 5, "n": 30, "nearer": 30, "noprog": 0}},
        "s3_10.success_by_suite": {"solo": (4, 4), "dyad": (8, 8), "3p": (5, 8), "6p": (0, 8)},
        "s3_10.stuckfree_6p": 62,
        "s4_8.solo_A_moves": [9, 9, 9, 9],
        "s4_8.6p_abort_moves": [23, 23, 23, 23],
    },
    "updated": {
        "residual.prose.nonsuccess_of": (26, 720),
        "enc.n": 30, "enc.harmless_alt": 30, "enc.sealed_by_teammate": 30,
        "enc.by_proto": {"freeform": 5, "hybrid": 6, "none": 12, "structured": 7},
        "enc.victim_one_exit_unbroken_moves": (1.0, 1.1, 4),
        "enc.victim_moves_since_first_one_exit": (16.0, 15.8, 0, 40),
        "enc.manhattan_retreats(n, of_which_wall_detours)": (3, 1),
        "enc.below6p_offtarget_enclosed_episodes": {
            "qwen38 dyad": 29, "qwen38 3p": 37, "dsv41 dyad": 33, "dsv41 3p": 50},
        "enc.below6p_of_those_aborted": 0,
        "pinch.total_all_models": 243, "pinch.suites": ["6p"],
        "pinch.strong.prose": dict(pinches=91, episodes=52, no_alt=3, free=88, declined=58,
                                   taken=30, costly=41, costly_declined=16),
        "pinch.strong.rigid": dict(pinches=139, episodes=62, no_alt=6, free=133, declined=77,
                                   taken=56, costly=87, costly_declined=39),
        "pinch.weak.prose": dict(pinches=12, episodes=8, no_alt=2, free=10, declined=5,
                                 taken=5, costly=5, costly_declined=1),
        "pinch.strong.prose.enclosure_aborts": 11, "pinch.strong.rigid.enclosure_aborts": 19,
        "pinch.aborts_recovered_as_taken": 30,
        "pinch.strong.prose.declines_naming_victim": (55, 58),
        "invite.chain(taken, named, failing, invited)": (30, 22, 8, 6),
        "cut.cycle_losses_with_cut": (88, 110),
        "cut.by_model_suite": {"qwen38 3p": (15, 22), "qwen38 6p": (31, 34),
                               "dsv41 3p": (20, 29), "dsv41 6p": (22, 25)},
        "cut.moves_after_first_cut(median, mean, max)": (19.0, 22.6, 87),
        "cut.episodes_cut_more_than_10_moves_before_end": 61,
        "cut.unfinished_objects_unreachable_by_walls_alone": 0,
        "cut.objects": 120, "cut.sealer_on_own_target": 120, "cut.single_object_door": 119,
        "cut.chokepoint(objects, episodes)": (115, 83),
        "fork.strong.prose": dict(eligible=2558, eligible_episodes=714, forks=22,
                                  fork_episodes=22, cut=13, gave_way=8),
        "fork.strong.rigid": dict(eligible=2381, eligible_episodes=708, forks=97,
                                  fork_episodes=93, cut=91, gave_way=3),
        "spell.total": 282, "spell.opening(n, levels, players)": (180, ["S4"], ["B"]),
        "spell.opening.strong.relieved_of": (120, 120),
        "spell.opening.pooled.prose.relieved_of": (86, 90),
        "spell.opening.pooled.rigid.relieved_of": (63, 90),
        "spell.mid.strong.prose.relieved_of": (22, 33),
        "spell.mid.strong.rigid.relieved_of": (43, 62),
        "spell.mid.freer_not_cause_of": (69, 69), "spell.own_turn_fatal_of": (32, 32),
        "enc.greedy": {"Qwen3.8": {"costly": 12, "free": 1, "further": 1, "n": 13,
                                   "nearer": 12, "noprog": 1},
                       "DeepSeek": {"costly": 13, "free": 4, "further": 1, "n": 17,
                                    "nearer": 16, "noprog": 0},
                       "all": {"costly": 25, "free": 5, "further": 2, "n": 30,
                               "nearer": 28, "noprog": 1}},
        "s4_8.outcomes": {"3p": {"success": 8}, "6p": {"abort": 5, "success": 3},
                          "dyad": {"lose": 1, "success": 7}, "solo": {"success": 4}},
        "s3_10.success_by_suite": {"solo": (4, 4), "dyad": (8, 8), "3p": (5, 8), "6p": (1, 8)},
        "s3_10.stuckfree_6p": 62,
        "s4_8.solo_A_moves": [9, 9, 9, 9],
        "s4_8.6p_abort_moves": [23, 23, 23, 23, 24],
    },
}


def check(N):
    """Every pinned value of this arm; None in a pinned dict means 'not pinned'."""
    bad = []
    for k, want in PINS[ARM].items():
        got = N[k]
        if isinstance(want, dict) and isinstance(got, dict):
            got = {kk: (got.get(kk) if vv is not None else None) for kk, vv in want.items()}
        if got != want:
            bad.append((k, want, got))
    tk, named = invitation_chain(LOADED["pinches"])
    if ARM == "original":   # the coding rule reproduces the hand reading
        coded = {x["key"] for x in named if x["fail"] and x["invited"]}
        if coded != HAND_INVITED:
            bad.append(("invite.rule_vs_hand", HAND_INVITED, coded))
    for k, want, got in bad:
        print("MISMATCH %s\n   pinned %r\n   got    %r" % (k, want, got))
    assert not bad, "%d pinned values moved" % len(bad)
    print("failures.py check ok (%s arm, %d pins)" % (ARM, len(PINS[ARM])))


LOADED = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", action="store_true", help="the exploratory p-values")
    ap.add_argument("--check", action="store_true", help="assert the pinned values")
    args = ap.parse_args()
    LOADED.update(load())
    N = numbers(LOADED)
    if args.check:
        return check(N)
    if args.tests:
        print("exploratory, two-sided Fisher exact, strong tier unless marked (%s arm)" % ARM)
        for name, a, n1, b, n2, p, unit in tests(N):
            print("  %-62s %4d/%-5d vs %4d/%-5d p = %.3g   [unit: %s]"
                  % (name, a, n1, b, n2, p, unit))
        return
    print("%s arm" % ARM)
    for k, v in N.items():
        print("%-58s %s" % (k, v))
    print()
    for k, f in sorted(episodes(LOADED).items()):
        print(" ".join(k), f)


if __name__ == "__main__":
    main()
