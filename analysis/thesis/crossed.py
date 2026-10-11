#!/usr/bin/env python3
"""The crossed-device probe (decision D5): does a level's difficulty follow its
corridor device, or the dependency chain the device sits on?

    python3 analysis/thesis/crossed.py

DeepSeek-V4.1-Flash only, dyad and 6p, updated prompt, A_k budget.  S3x moves
S3's threading device onto the shallow pair (E<-F); S4x makes S4's swap the
head link of the deep chain (A<-B<-C<-D).  Crossed board i keeps the corridor
of released board i and nothing else (matrixgame_covered_crossed_shared/boards.py).
S3/S4 come from results_dsv41_grid_updated_prompt_n15_{dyad,6p}.

Prints, in order:
  1. motivation: DeepSeek's S1-S4 Quality / % Played / success and failure
     anatomy at dyad and 6p (the numbers the section opens on);
  2. instances: optima, interaction gap and the door facts, S3 v S3x, S4 v S4x;
  3. outcomes: Quality with % Played, success, efficiency on A_k, per suite x
     protocol, per suite and pooled;
  4. anatomy: enclosure aborts, cycle losses, `done` cuts and where they sit.

Pairing.  Crossed board i shares only its corridor cells with released board i;
placement and carving come from an independent stream, so a pair is not the
same puzzle.  The tests are therefore unpaired: Fisher exact on success and on
failure counts, confirmatory.two_sample_p (100,000 resamples, seed 0) on
Quality and efficiency.  Section 2 prints the between-index correlations that
pairing would need, and section 3 an index-paired sign-flip (tables_comm.perm_p)
as a sensitivity.  Everything here is exploratory; nothing is Holm-corrected.

Definitions are the main results': Quality is tables_comm.quality on the
suite's attainable optimum; an enclosure is an abort whose aborting player had
no legal move for its own objects (forced.py); a cut is a cycle loss with an
object whose last route a `done` severed (cut.py, here also run on the dyad).
Self-check (runs every time): the loader reproduces tables_comm.all_episodes,
aborts.updated.json and cut.cases() on the updated-arm DeepSeek episodes.
"""
import collections
import glob
import json
import os
import statistics as st
import string
import sys

from scipy.stats import fisher_exact, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EB = os.path.join(ROOT, "analysis/efficiency_baseline")
sys.path.insert(0, EB)
import tables_comm as tc                        # noqa: E402  (puts qualitative_analysis on the path)
from confirmatory import two_sample_p          # noqa: E402
import cut                                     # noqa: E402  (import-safe; scan.py/forced.py are not)
import instance_audit as ia                    # noqa: E402

assert tc.ARM == "updated", "the crossed arm exists with the updated prompt only"
SUITES = ("dyad", "6p")
TREE = {"ref": "results_dsv41_grid_updated_prompt_n15_{}",
        "crossed": "results_dsv41_grid_crossed_n15_{}"}
PAIRS = (("S3", "S3x"), ("S4", "S4x"))
PROTOS = tc.PROTOCOLS
DS = "DeepSeek-V4.1-Flash"


def optima():
    """tables_comm's optima plus the crossed boards' J and A_k (all exact)."""
    opt = tc.load_optima()
    for name, key, val in [("joint_optimum_crossed", None, "opt")] + [
            (f"attainable_{s}_crossed", s, "stuckfree") for s in SUITES]:
        for ln in open(os.path.join(EB, name + ".jsonl")):
            d = json.loads(ln)
            assert d["status"] == "exact", (name, d)
            opt[(key, d["level"], d["game_id"]) if key else (d["level"], d["game_id"])] = d[val]
    return opt


def enclosed(ins, it, owner):
    """forced.py's test: at the abort, the author of the last invalid move has
    no legal move for any of its own objects."""
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    who = None
    for t in it["turns"]:
        for e in t:
            a = e.get("action", {})
            ty, c = a.get("type", ""), a.get("content", "")
            if ty == "move":
                m = cut.MOVE.match(c)
                if m:
                    pos = cut.step(pos, m.group(2), m.group(3))
            elif ty == "invalid_move":
                who = c.split(":")[0].replace("Player", "").strip() or "A"
            elif ty == "abort":
                objs = [o for o in owner if owner[o] == who] or list(owner)
                occ = set(pos.values())
                return not any(
                    0 <= pos[o][0] + dr < n and 0 <= pos[o][1] + dc < n
                    and (pos[o][0] + dr, pos[o][1] + dc) not in walls | occ
                    for o in objs for dr, dc in cut.DIRS.values())
    return False


def cuts(ins, it, owner, where):
    """cut.cases()'s replay for one cycle loss, any team size: (victim, shape)
    for every object whose last route a `done` severed."""
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    dev = ins.get("device") or {}
    corridor = {tuple(c) for c in dev.get("corridor", [])}
    entrance = tuple(dev["entrance"]) if dev.get("entrance") else None
    pos, done, sealed = {o: tuple(p) for o, p in ins["start_positions"].items()}, set(), {}
    for kind, payload in cut._events(it):
        before, before_done = dict(pos), frozenset(done)
        if kind == "move":
            pos = cut.step(pos, *payload)
        else:
            done.add(payload)
        frozen = {pos[o] for o in owner if owner[o] in done}
        for o in tgt:
            if o in sealed or owner[o] in done or pos[o] == tgt[o]:
                continue
            if cut.path(pos[o], tgt[o], n, walls, frozen - {pos[o]}) is None:
                sealed[o] = (kind, payload, before, before_done)
    frozen_end = {pos[o] for o in owner if owner[o] in done}
    out = []
    for v in tgt:
        if owner[v] in done or pos[v] == tgt[v] or cut.path(
                pos[v], tgt[v], n, walls, frozen_end - {pos[v]}) is not None:
            continue
        kind, payload, pre, pre_done = sealed[v]
        assert kind == "done", (where, v, kind)      # no cut is made by a move
        frozen_pre = {pre[o] for o in owner if owner[o] in pre_done}
        frozen_post = frozen_pre | {pre[o] for o in owner if owner[o] == payload}
        route = cut.path(pre[v], tgt[v], n, walls, frozen_pre - {pre[v]})
        seals = [pre[o] for o in owner if owner[o] == payload and pre[o] in route]
        assert seals, (where, v)
        out.append((v, cut._classify(pre[v], tgt[v], seals, entrance, corridor,
                                     n, walls, frozen_post)))
    return out


def load(opt):
    rows, boards = [], {}
    for arm, pat in TREE.items():
        for suite in SUITES:
            for f in sorted(glob.glob(os.path.join(ROOT, pat.format(suite), "*", "*", "*",
                                                   "instance_*", "scores.json"))):
                d = os.path.dirname(f)
                ins = json.load(open(os.path.join(d, "instance.json")))
                it = json.load(open(os.path.join(d, "interactions.json")))
                s = json.load(open(f))["episode scores"]
                level, gid = ins["spatial_level"], ins["game_id"]
                prev = boards.setdefault((level, gid), ins)
                assert ia.board_key(prev) == ia.board_key(ins), (level, gid)
                groups = ins.get("player_objects") or [ins["player_a_objects"],
                                                       ins["player_b_objects"]]
                owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
                ok, ab = int(s["Success"]), int(s["Aborted"])
                ends = [e["action"]["content"] for t in it["turns"] for e in t
                        if e.get("action", {}).get("type") in ("lose", "abort", "success")]
                cycle = not ok and not ab and "cycle" in ends[-1]
                eff = opt[(suite, level, gid)] / s["Move Count"] if ok else None
                rows.append(dict(
                    arm=arm, suite=suite, level=level, cond=ins["comm_protocol"],
                    game_id=gid, success=ok, ab=ab, eff=eff, q=tc.quality(s, eff),
                    eff_j=opt[(level, gid)] / s["Move Count"] if ok else None,
                    enclosure=bool(ab) and enclosed(ins, it, owner),
                    abort_reason=ends[-1] if ab else None, cycle=cycle,
                    cuts=cuts(ins, it, owner, d) if cycle else [],
                    path=os.path.relpath(d, ROOT)))
    return rows, boards


def selfcheck(rows, opt):
    ref = {(r["suite"], r["level"], r["cond"], r["game_id"]): r for r in rows if r["arm"] == "ref"}
    them = [r for r in tc.all_episodes(opt, games=SUITES) if r["model"] == DS]
    assert len(them) == len(ref) == 480, (len(them), len(ref))
    for r in them:
        mine = ref[(r["suite"], r["level"], r["cond"], r["game_id"])]
        assert all(mine[k] == r[k] for k in ("success", "ab", "q", "eff")), r
    shared = os.path.join(ROOT, "analysis/qualitative_analysis/aborts.updated.json")
    enc = {a["path"] for a in json.load(open(shared))
           if a["model"] == "dsv41" and a["suite"] in SUITES and a["n_legal_own"] == 0}
    assert enc == {r["path"] for r in ref.values() if r["enclosure"]}, "enclosures differ"
    theirs = sorted((c["meta"]["path"], c["victim"], c["kind"]) for c in cut.cases(("dsv41",))
                    if c["meta"]["suite"] == "6p")
    mine = sorted((r["path"], v, k) for r in ref.values() if r["suite"] == "6p" for v, k in r["cuts"])
    assert theirs == mine, "cuts differ"
    print(f"self-check: {len(them)} episodes match all_episodes; {len(enc)} enclosures match "
          f"aborts.updated.json; {len(mine)} 6p cut objects match cut.cases()\n")


# ---------------------------------------------------------------- printing
def summ(rs):
    c = tc.cell(rs)
    q0 = st.mean([r["q"] if r["q"] is not None else 0.0 for r in rs]) if rs else float("nan")
    return (f"QS {c['q']:5.1f} / {c['played']:5.1f}% (ab=0: {q0:5.1f})  "
            f"succ {c['k']:3d}/{c['n']:<3d} {c['rate']:5.1f}%  eff {c['eff']:.3f} (n={c['eff_n']})")


def anatomy(rs):
    n_cut = sum(bool(r["cuts"]) for r in rs)
    return dict(lose=sum(not r["success"] for r in rs), ab=sum(r["ab"] for r in rs),
                enc=sum(r["enclosure"] for r in rs), cyc=sum(r["cycle"] for r in rs),
                cut=n_cut, obj=sum(len(r["cuts"]) for r in rs),
                shapes=collections.Counter(k for r in rs for _, k in r["cuts"]))


def fisher(k1, n1, k2, n2):
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1]


def motivation(rows):
    print("1. MOTIVATION  DeepSeek, updated arm, by level (QS / % Played; aborts excluded from QS)")
    for suite in SUITES:
        for lv in ("S1", "S2", "S3", "S4"):
            rs = [r for r in rows if r["arm"] == "ref" and r["suite"] == suite and r["level"] == lv]
            a = anatomy(rs)
            print(f"  {suite:4s} {lv}  {summ(rs)}  non-succ {a['lose']:2d}: enclosure {a['enc']:2d}, "
                  f"other abort {a['ab'] - a['enc']}, cycle {a['cyc']:2d} (cut {a['cut']:2d} eps, "
                  f"{a['obj']} objs) {dict(a['shapes'])}")
    print()


def instances(boards):
    print("2. INSTANCES  (means over 15 boards; gap = J - h0; door = target on the corridor entrance)")
    jo, _ = ia.load_solver_outputs()
    for ln in open(os.path.join(EB, "joint_optimum_crossed.jsonl")):
        d = json.loads(ln)
        jo[(d["level"], d["game_id"])] = d
    att = {}
    for s in SUITES:
        for x in ("", "_crossed"):
            for ln in open(os.path.join(EB, f"attainable_{s}{x}.jsonl")):
                d = json.loads(ln)
                att[(s, d["level"], d["game_id"])] = d["stuckfree"]
    per = {}
    for lv in ("S3", "S3x", "S4", "S4x"):
        rs = []
        for gid in range(1, 16):
            ins = boards[(lv, gid)]
            r = ia.audit_board(lv, gid, ins, jo, {})
            dev, tgt = ins["device"], ins["target_positions"]
            r.update(a_dyad=att[("dyad", lv, gid)], a_6p=att[("6p", lv, gid)],
                     door=sum(t == dev["entrance"] for t in tgt.values()),
                     in_corr=sum(t in dev["corridor"] for t in tgt.values()))
            r["rules_6p"] = r["a_6p"] - r["opt"]
            rs.append(r)
        per[lv] = rs
        m = lambda k: st.mean(x[k] for x in rs)  # noqa: E731
        print(f"  {lv:4s} h0 {m('h0'):5.1f}  J {m('opt'):5.1f}  gap {m('gap'):4.1f} "
              f"({100 * m('gap') / m('h0'):4.1f}%)  A_dyad {m('a_dyad'):5.1f}  A_6p {m('a_6p'):5.1f}  "
              f"A_6p-J {m('rules_6p'):4.1f}  swap {sum(x['swap'] > 0 for x in rs):2d}/15  "
              f"door {sum(x['door'] for x in rs):2d}/15  targets in corridor {m('in_corr'):.1f}  "
              f"mob {m('mob_mean'):4.2f}  zero@6p {sum(x['zero_6p'] for x in rs)}  "
              f"forced@6p {sum(x['forced_6p'] for x in rs)}")
    for a, b in PAIRS:
        for k in ("opt", "gap", "a_6p"):
            rho, p = spearmanr([x[k] for x in per[a]], [x[k] for x in per[b]])
            print(f"  index pairing {a}/{b}: Spearman rho on {k:5s} = {rho:+.2f} (p = {p:.2f}, n = 15)")
    print()
    return per


def outcomes(rows):
    print("3. OUTCOMES  ref vs crossed; unpaired p: QS/eff two_sample_p (1e5, seed 0), succ Fisher;"
          " paired p: index sign-flip perm_p on QS")
    for a, b in PAIRS:
        for suite in SUITES + ("pooled",):
            for proto in PROTOS + ("all",):
                sel = lambda lv: [r for r in rows if r["level"] == lv  # noqa: E731
                                  and (suite == "pooled" or r["suite"] == suite)
                                  and (proto == "all" or r["cond"] == proto)]
                ra, rb = sel(a), sel(b)
                qa = [r["q"] for r in ra if r["q"] is not None]
                qb = [r["q"] for r in rb if r["q"] is not None]
                ea = [r["eff"] for r in ra if r["success"]]
                eb = [r["eff"] for r in rb if r["success"]]
                ka, kb = sum(r["success"] for r in ra), sum(r["success"] for r in rb)
                key = lambda r: (r["suite"], r["cond"], r["game_id"])  # noqa: E731
                paired = tc.qpairs(rb, ra, key)
                print(f"  {a}->{b} {suite:6s} {proto:10s}  {a}: {summ(ra)}")
                print(f"  {'':18s} {'':10s}  {b}: {summ(rb)}")
                print(f"  {'':18s} {'':10s}  dQS {st.mean(qb) - st.mean(qa):+5.1f} p={two_sample_p(qb, qa):.2g}"
                      f" (n {len(qb)}/{len(qa)}; paired n {len(paired)} p={tc.perm_p(paired):.2g})  "
                      f"dSucc {100 * (kb / len(rb) - ka / len(ra)):+5.1f}pp p={fisher(kb, len(rb), ka, len(ra)):.2g}  "
                      f"dEff {st.mean(eb) - st.mean(ea):+.3f} p={two_sample_p(eb, ea):.2g} (n {len(eb)}/{len(ea)})")
    print()


def failures(rows):
    print("4. ANATOMY  episodes; Fisher on episodes with the event, ref vs crossed")
    for a, b in PAIRS:
        for suite in SUITES + ("pooled",):
            sel = lambda lv: [r for r in rows if r["level"] == lv  # noqa: E731
                              and (suite == "pooled" or r["suite"] == suite)]
            ra, rb = sel(a), sel(b)
            A, B = anatomy(ra), anatomy(rb)
            ps = {k: fisher(B[k], len(rb), A[k], len(ra)) for k in ("lose", "enc", "cyc", "cut")}
            print(f"  {a}->{b} {suite:6s} non-succ {A['lose']}->{B['lose']} (p={ps['lose']:.2g})  "
                  f"enclosure {A['enc']}->{B['enc']} (p={ps['enc']:.2g})  other abort "
                  f"{A['ab'] - A['enc']}->{B['ab'] - B['enc']}  cycle {A['cyc']}->{B['cyc']} "
                  f"(p={ps['cyc']:.2g})  cut eps {A['cut']}->{B['cut']} (p={ps['cut']:.2g}) "
                  f"objs {A['obj']}->{B['obj']}")
            for lv, X in ((a, A), (b, B)):
                if X["shapes"]:
                    print(f"  {'':18s} {lv} cut shapes: {dict(X['shapes'])}")
    enc = [r for r in rows if r["arm"] == "crossed" and r["enclosure"]]
    print("  crossed enclosures:", [(r["suite"], r["level"], r["cond"], r["game_id"],
                                    r["abort_reason"]) for r in enc])
    oth = [r for r in rows if r["arm"] == "crossed" and r["ab"] and not r["enclosure"]]
    print("  crossed non-enclosure aborts:", [(r["suite"], r["level"], r["cond"], r["game_id"],
                                              r["abort_reason"]) for r in oth])


def main():
    opt = optima()
    rows, boards = load(opt)
    selfcheck(rows, opt)
    motivation(rows)
    instances(boards)
    outcomes([r for r in rows if r["level"] in ("S3", "S3x", "S4", "S4x")])
    failures(rows)


if __name__ == "__main__":
    main()
