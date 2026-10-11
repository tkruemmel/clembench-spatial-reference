"""Timing in the shared REQUEST slot (discussion ch., REQUEST paragraphs; todo A2).

Rebuilds the lost reqtime.py / seqfollow.py.  Scope, from todo A2 and the
f2_section.tex header: strong-tier dyads (Qwen3.8, DeepSeek-V4.1), grid, the
two protocols with a REQUEST slot (structured, hybrid), every message replayed
against the board at the moment it was written.  The thesis ranges span the
two models.  Messages come from qual_corpus.extract(); the board state from a
replay of the same events (extract() keeps it local).

Definitions, pre-registered 2026-10-11 before any output was looked at
-----------------------------------------------------------------------
message      every broadcast row extract() yields for the cell (done messages
             included), the denominator of "their messages".
empty        qual_corpus REQNONE ("REQUEST: none"), the feature behind the
             thesis's 56/66 % and 23-24 %.
slot         the text after the first "REQUEST:" to the end of the message
             (the enforced schema ends every message with it).
filled       a slot exists and the message is not empty.
timed        filled, and the slot matches TIMING: defer (later, yet, for now,
             next turn/move, in N turns), condition (once, when, whenever, as
             soon as, until, till, unless, while, during), wait (wait*, hold,
             pause, stay put/there/in place), order (after, afterwards, before,
             then, first, meanwhile).  "when you can" counts, as A2's lexicon
             caveat says the original did; "if" does not (both prompts' example
             request ends "if you can", a politeness echo, not timing).
blocked      a teammate object lengthens or cuts the path of one of the
             speaker's unplaced objects to its target: BFS over walls + the
             speaker's other objects vs. over walls + all other objects; blocked
             iff the first exists and the second is longer or missing (a
             teammate on the target cuts it).  Board = extract()'s snapshot,
             taken at the speaker's 'get message'.  unblocked = not blocked.
right after  the message immediately preceding in the episode is the teammate's;
             messages without one (first message, partner done) are left out
             of claim 3.
announced    the teammate's message contains qual_corpus SEQ: in its prose
             (text before the first "PLAN:") under hybrid; in the whole message
             under structured, which has no prose (the corpus `seq` feature,
             the "1 %" the thesis cites).
stuck        = blocked ("structured requests mostly come once a player is
             already stuck": share of filled structured requests made blocked).
ratio        timed % after an announcement / timed % after none, unblocked
             hybrid; also an episode-stratified Mantel-Haenszel odds ratio.

The one revision (claim 3 only, made after the pre-registered run missed)
-------------------------------------------------------------------------
The thesis splits "unblocked moments" into two groups, teammate announced a
sequence / teammate announced none, with no third; the pre-registered
exclusion of messages without a teammate message right before is not in the
spec.  Revised ("rev"): every unblocked message, split by the teammate's most
recent message; no teammate message, or a `done`, counts as none.  Printed
beside the pre-registered split ("pre"), with the moments the revision adds
after a teammate's `done` ("done"), where no request can be acted on.

Usage:  ARM=original python3 request_timing.py [--check]
"""
import argparse, collections, glob, json, os, re, string, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "efficiency_baseline"))
from qual_corpus import ARM, ARMS, DIRS, MODEL_LABEL, MOVE, REQNONE, SEQ, SPK, STRONG, extract, tree
from instance_audit import bfs_dist

PROTOS = ("structured", "hybrid")
SLOT = re.compile(r"REQUEST\s*:\s*(.*)\Z", re.I | re.S)
TIMING = re.compile(
    r"\b(?:later|yet|for now|next (?:turn|move|step|round)"
    r"|in (?:\d+|one|two|three|a few|a couple of) (?:turns?|moves?)"
    r"|once|when|whenever|as soon as|until|till|unless|while|during"
    r"|wait\w*|hold|pause|stay (?:put|there|in place)"
    r"|after|afterwards|before|then|first|meanwhile)\b", re.I)


def blocked_map(repo):
    """{(model, lvl, proto, gid): [blocked per message]} from the same event walk extract() does."""
    out = {}
    for model in STRONG:
        pat = os.path.join(repo, tree(ARMS[model], "dyad"), "*", "*", "*", "instance_*", "instance.json")
        for f in sorted(glob.glob(pat)):
            ins = json.load(open(f))
            if ins["comm_protocol"] not in PROTOS:
                continue
            it = json.load(open(os.path.join(os.path.dirname(f), "interactions.json")))
            n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
            pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
            tgt = {o: tuple(q) for o, q in ins["target_positions"].items()}
            groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
            owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
            snap, flags = dict(pos), []
            for t in it["turns"]:
                for e in t:
                    a = e.get("action", {})
                    ty, c = a.get("type"), str(a.get("content", ""))
                    if ty == "get message":
                        snap = dict(pos)
                    elif ty == "move" and (m := MOVE.match(c)):
                        _, o, dr = m.groups()
                        pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
                    elif ty == "message" and (m := SPK.match(c)):
                        spk, b = m.group(1), False
                        for o in (o for o in tgt if owner[o] == spk and snap[o] != tgt[o]):
                            own = walls | {snap[x] for x in tgt if owner[x] == spk and x != o}
                            d0 = bfs_dist(snap[o], tgt[o], n, own)
                            d1 = bfs_dist(snap[o], tgt[o], n, own | {snap[x] for x in tgt if owner[x] != spk})
                            b = b or (d0 is not None and (d1 is None or d1 > d0))
                        flags.append(b)
            key = (model, ins["spatial_level"], ins["comm_protocol"], ins["game_id"])
            assert key not in out, key
            out[key] = flags
    return out


def annotate(repo):
    rows, _ = extract(repo)
    rows = [r for r in rows if r["suite"] == "dyad" and r["model"] in STRONG and r["proto"] in PROTOS]
    bl = blocked_map(repo)
    ep = collections.defaultdict(list)
    for r in rows:
        ep[(r["model"], r["lvl"], r["proto"], r["gid"])].append(r)
    for k, msgs in ep.items():
        msgs.sort(key=lambda r: r["midx"])
        assert len(msgs) == len(bl[k]), k
        prev, last = None, {}
        for r, b in zip(msgs, bl[k]):
            assert len(r["text"]) < 1200, "extract() truncated a message"
            s = SLOT.search(r["text"])
            r["blocked"] = b
            r["filled"] = bool(s) and not r["reqnone"]
            r["timed"] = r["filled"] and bool(TIMING.search(s.group(1)))
            said = re.split(r"PLAN\s*:", r["text"], 1, flags=re.I)[0] if r["proto"] == "hybrid" else r["text"]
            r["announces"] = bool(SEQ.search(said))
            r["after"] = None if prev is None or prev["spk"] == r["spk"] else prev["announces"]
            mate = next((x for s_, x in last.items() if s_ != r["spk"]), None)
            r["after_last"] = bool(mate and mate["announces"])
            r["mate_done"] = bool(mate and mate["text"].strip().lower() == "done")
            r["ep"] = k
            prev = last[r["spk"]] = r
    return rows


def pct(sel, key):
    return 100.0 * sum(bool(r[key]) for r in sel) / len(sel) if sel else float("nan")


def mh_or(sel, x):
    """Episode-stratified Mantel-Haenszel OR of timed on exposure field x."""
    num = den = 0.0
    strata = collections.defaultdict(list)
    for r in sel:
        strata[r["ep"]].append(r)
    for s in strata.values():
        n = len(s)
        a = sum(r[x] and r["timed"] for r in s); b = sum(r[x] and not r["timed"] for r in s)
        c = sum(not r[x] and r["timed"] for r in s); d = sum(not r[x] and not r["timed"] for r in s)
        num += a * d / n; den += b * c / n
    return num / den if den else float("inf")


def split(sel, x, tag):
    ys, no = [r for r in sel if r[x]], [r for r in sel if not r[x]]
    v = {f"{tag}_seq": pct(ys, "timed"), f"{tag}_none": pct(no, "timed"),
         f"{tag}_n_seq": len(ys), f"{tag}_n_none": len(no), f"{tag}_or": mh_or(sel, x)}
    v[f"{tag}_ratio"] = v[f"{tag}_seq"] / v[f"{tag}_none"] if v[f"{tag}_none"] else float("inf")
    return v


def report(rows):
    res = {}
    print(f"ARM={ARM}   strong-tier dyads, grid   ({len(rows):,} messages)\n")
    print(f"{'model':<15}{'protocol':<12}{'msgs':>6}{'empty':>8}{'timed':>8}{'t/filled':>9}"
          f"{'unblk':>7}{'t|unblk':>8}{'filled blk':>11}")
    for m in STRONG:
        for p in PROTOS:
            sel = [r for r in rows if r["model"] == m and r["proto"] == p]
            fil = [r for r in sel if r["filled"]]
            ub = [r for r in sel if not r["blocked"]]
            v = dict(empty=pct(sel, "reqnone"), timed=pct(sel, "timed"), timed_of_filled=pct(fil, "timed"),
                     timed_unblocked=pct(ub, "timed"), filled_blocked=pct(fil, "blocked"))
            res[f"{m}/{p}"] = v
            print(f"{MODEL_LABEL[m]:<15}{p:<12}{len(sel):>6}{v['empty']:>7.1f}%{v['timed']:>7.1f}%"
                  f"{v['timed_of_filled']:>8.1f}%{len(ub):>7}{v['timed_unblocked']:>7.1f}%{v['filled_blocked']:>10.1f}%")
    # pre: pre-registered, right after an active teammate message.
    # rev: the one revision, every unblocked message split by the teammate's most
    #      recent message (none if there is none, or it was `done`).
    # done: the moments rev adds that follow a teammate's `done`.
    print("\nclaim 3: unblocked messages, timed % after a teammate's announced sequence vs after none")
    print(f"{'model':<15}{'protocol':<12}{'mate seq%':>10}{'def':>5}{'n seq':>7}{'timed':>8}{'n none':>8}"
          f"{'timed':>8}{'ratio':>7}{'MH OR':>7}")
    for m in STRONG:
        for p in PROTOS:
            ub = [r for r in rows if r["model"] == m and r["proto"] == p and not r["blocked"]]
            v = res[f"{m}/{p}"]
            v["mate_announces"] = pct([r for r in rows if r["model"] == m and r["proto"] == p], "announces")
            v.update(split([r for r in ub if r["after"] is not None], "after", "pre"))
            v.update(split(ub, "after_last", "rev"))
            dn = [r for r in ub if r["mate_done"]]
            v["done_n"], v["done_timed"] = len(dn), pct(dn, "timed")
            for i, t in enumerate(("pre", "rev")):
                head = f"{MODEL_LABEL[m]:<15}{p:<12}{v['mate_announces']:>9.1f}%" if i == 0 else " " * 37
                print(f"{head}{t:>5}{v[t + '_n_seq']:>7}{v[t + '_seq']:>7.1f}%{v[t + '_n_none']:>8}"
                      f"{v[t + '_none']:>7.1f}%{v[t + '_ratio']:>7.2f}{v[t + '_or']:>7.2f}")
            print(f"{'':37}{'done':>5}{'':15}{v['done_n']:>8}{v['done_timed']:>7.1f}%")
    return res


# Headline values per arm, from the 2026-10-11 runs; the updated arm's are the
# final data's (re-run and confirmed in Phase 2, plus the hybrid MH odds ratio
# behind the surviving association claim).  Checked to 0.05 in the value's unit.
CHECK = {
    "original": {"qwen38/structured/empty": 56.65, "qwen38/structured/timed": 3.59, "qwen38/structured/timed_of_filled": 8.58, "qwen38/structured/timed_unblocked": 2.88, "qwen38/structured/filled_blocked": 71.15, "qwen38/structured/mate_announces": 1.36, "qwen38/structured/pre_seq": 54.55, "qwen38/structured/pre_none": 2.82, "qwen38/structured/pre_ratio": 19.38, "qwen38/structured/rev_none": 2.31, "qwen38/structured/rev_ratio": 23.62,
                 "qwen38/hybrid/empty": 24.36, "qwen38/hybrid/timed": 12.88, "qwen38/hybrid/timed_of_filled": 17.44, "qwen38/hybrid/timed_unblocked": 9.9, "qwen38/hybrid/filled_blocked": 47.8, "qwen38/hybrid/mate_announces": 78.38, "qwen38/hybrid/pre_seq": 14.67, "qwen38/hybrid/pre_none": 4.9, "qwen38/hybrid/pre_ratio": 2.99, "qwen38/hybrid/rev_none": 2.42, "qwen38/hybrid/rev_ratio": 6.06,
                 "dsv41/structured/empty": 66.06, "dsv41/structured/timed": 2.98, "dsv41/structured/timed_of_filled": 9.23, "dsv41/structured/timed_unblocked": 1.52, "dsv41/structured/filled_blocked": 78.82, "dsv41/structured/mate_announces": 0.94, "dsv41/structured/pre_seq": 0.0, "dsv41/structured/pre_none": 1.96, "dsv41/structured/pre_ratio": 0.0, "dsv41/structured/rev_none": 1.52, "dsv41/structured/rev_ratio": 0.0,
                 "dsv41/hybrid/empty": 23.07, "dsv41/hybrid/timed": 24.92, "dsv41/hybrid/timed_of_filled": 33.17, "dsv41/hybrid/timed_unblocked": 16.02, "dsv41/hybrid/filled_blocked": 44.15, "dsv41/hybrid/mate_announces": 74.92, "dsv41/hybrid/pre_seq": 23.05, "dsv41/hybrid/pre_none": 13.23, "dsv41/hybrid/pre_ratio": 1.74, "dsv41/hybrid/rev_none": 6.9, "dsv41/hybrid/rev_ratio": 3.34,
                 "qwen38/hybrid/pre_or": 2.99, "dsv41/hybrid/pre_or": 1.90},
    "updated": {"qwen38/structured/empty": 60.53, "qwen38/structured/timed": 3.63, "qwen38/structured/timed_of_filled": 9.61, "qwen38/structured/timed_unblocked": 3.04, "qwen38/structured/filled_blocked": 69.92, "qwen38/structured/mate_announces": 1.58, "qwen38/structured/pre_seq": 23.81, "qwen38/structured/pre_none": 3.55, "qwen38/structured/pre_ratio": 6.71, "qwen38/structured/rev_none": 2.82, "qwen38/structured/rev_ratio": 8.44,
                "qwen38/hybrid/empty": 24.61, "qwen38/hybrid/timed": 14.4, "qwen38/hybrid/timed_of_filled": 19.55, "qwen38/hybrid/timed_unblocked": 10.68, "qwen38/hybrid/filled_blocked": 48.78, "qwen38/hybrid/mate_announces": 79.27, "qwen38/hybrid/pre_seq": 15.8, "qwen38/hybrid/pre_none": 6.76, "qwen38/hybrid/pre_ratio": 2.34, "qwen38/hybrid/rev_none": 3.03, "qwen38/hybrid/rev_ratio": 5.21,
                "dsv41/structured/empty": 64.48, "dsv41/structured/timed": 3.08, "dsv41/structured/timed_of_filled": 9.12, "dsv41/structured/timed_unblocked": 2.02, "dsv41/structured/filled_blocked": 76.62, "dsv41/structured/mate_announces": 1.26, "dsv41/structured/pre_seq": 28.57, "dsv41/structured/pre_none": 2.18, "dsv41/structured/pre_ratio": 13.13, "dsv41/structured/rev_none": 1.73, "dsv41/structured/rev_ratio": 16.54,
                "dsv41/hybrid/empty": 22.55, "dsv41/hybrid/timed": 22.94, "dsv41/hybrid/timed_of_filled": 30.32, "dsv41/hybrid/timed_unblocked": 15.05, "dsv41/hybrid/filled_blocked": 48.71, "dsv41/hybrid/mate_announces": 70.35, "dsv41/hybrid/pre_seq": 22.49, "dsv41/hybrid/pre_none": 11.69, "dsv41/hybrid/pre_ratio": 1.92, "dsv41/hybrid/rev_none": 6.54, "dsv41/hybrid/rev_ratio": 3.44,
                "qwen38/hybrid/pre_or": 2.21, "dsv41/hybrid/pre_or": 2.01},
}


def check(res):
    bad = 0
    for k, want in CHECK[ARM].items():
        cell, key = k.rsplit("/", 1)
        got = res[cell][key]
        ok = abs(got - want) <= 0.05
        bad += not ok
        print(f"  {k:<36}{want:>8.2f}{got:>8.2f}   {'ok' if ok else 'DIFF'}")
    assert CHECK[ARM], f"no check values for ARM={ARM}"
    assert not bad, f"{bad} values differ"
    print(f"  all {len(CHECK[ARM])} values reproduced")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(HERE)))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = report(annotate(a.repo))
    if a.check:
        check(res)
