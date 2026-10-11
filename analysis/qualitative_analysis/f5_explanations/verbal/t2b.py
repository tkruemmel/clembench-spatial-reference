"""Task 2, per ply: for every (ply k, active player T) classify T's state
   prec  = exactly one legal move
   rblk  = >=2 legal moves, every walls-only forward cell of an unfinished object
           occupied by a teammate's object
   free  = >=2 legal moves, not route-blocked (control)
and ask whether any OTHER player's message in the preceding full round
(plies k-R+1..k, R = active players) mentions T with
   mob   : a mobility predicate (trapped / only move / can't move ...)
   route : route / goal vocabulary (route_mate or goal_mate tag)
   exit  : one of T's free neighbour cells at ply k (cell-level, symmetric)
   block : one of T's blocking cells at ply k (rblk only)
S4 plies 0-1 and messages written at S4 plies 0-1 are excluded.
"""
import sys
sys.dont_write_bytecode = True
import collections, json, os, re, random
from verbal import load, tag_sentence, SENT, cell_rx, player_state
from t2 import tref, walls_of

S, P, E = load()


def main(suites=("6p",)):
    out = collections.defaultdict(collections.Counter)
    for ep in E:
        if ep["suite"] not in suites:
            continue
        w, n = walls_of(ep["path"])
        owner = {o: p for p, objs in ep["mine"].items() for o in objs}
        epx = dict(n=n, walls=w, mine=ep["mine"], tgt=ep["tgt"], owner=owner,
                   states=list(zip(ep["pos"], ep["done"])))
        s4 = ep["lvl"] == "S4"
        # pre-tag every sentence once
        tagged = []
        for j, spk, txt in ep["msgs"]:
            if s4 and j <= 1:
                continue
            own = set(ep["mine"][spk]); mates = set(ep["tgt"]) - own
            tagged.append((j, spk, [(s, tag_sentence(s, own, mates)) for s in SENT.findall(txt)]))
        for k in range(len(ep["pos"]) - 1):
            if s4 and k <= 1:
                continue
            done = ep["done"][k]
            R = len(ep["mine"]) - len(done)
            for T in ep["mine"]:
                if T in done:
                    continue
                nl, bl, ex = player_state(epx, k, T)
                if nl == 0:
                    continue
                st = "prec" if nl == 1 else ("rblk" if bl else "free")
                key = (ep["tier"], ep["suite"], st)
                c = out[key]
                c["n"] += 1
                exr, blr = cell_rx(ex), (cell_rx(bl) if bl else None)
                # per-cell baselines: one random free neighbour, one random occupied neighbour
                rnd = random.Random(f"{ep['path']}|{k}|{T}")
                pos = ep["pos"][k]
                occ = set(pos.values())
                nbo = sorted({(p[0] + dr, p[1] + dc) for o in ep["mine"][T] for p in [pos[o]]
                              for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} & occ - {pos[o] for o in ep["mine"][T]})
                one_ex = cell_rx([rnd.choice(sorted(ex))])
                one_oc = cell_rx([rnd.choice(nbo)]) if nbo else None
                c["has_oc"] += bool(nbo)
                hit = collections.Counter()
                for j, spk, sents in tagged:
                    if spk == T or not (k - R + 1 <= j <= k):
                        continue
                    rx = tref(ep, T, j, k, ep["suite"] == "dyad")
                    for s, t in sents:
                        if not rx.search(s):
                            continue
                        hit["any_ref"] = 1
                        hit["mob"] |= t["mob_mate"]
                        hit["route"] |= t["route_mate"] or t["goal_mate"]
                        hit["exit"] |= bool(exr.search(s))
                        if blr:
                            hit["block"] |= bool(blr.search(s))
                        hit["one_exit"] |= bool(one_ex.search(s))
                        if one_oc:
                            hit["one_occ"] |= bool(one_oc.search(s))
                for kk, v in hit.items():
                    c[kk] += v
    for key in sorted(out):
        c = out[key]
        n = c["n"]
        print(f"{key[0]:6} {key[1]:4} {key[2]:4} T-plies={n:6}  T named at all {100*c['any_ref']/n:5.1f}%  "
              f"T+mob {c['mob']:4} ({100*c['mob']/n:.2f}%)  T+route/goal {c['route']:5} ({100*c['route']/n:.1f}%)  "
              f"T+exit cell {c['exit']:5} ({100*c['exit']/n:.1f}%)  T+block cell {c['block']:5} ({100*c['block']/n:.1f}%)  "
              f"T+1 random exit {100*c['one_exit']/n:.1f}%  T+1 random occupied nbr {100*c['one_occ']/max(c['has_oc'],1):.1f}% (of {c['has_oc']})")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("6p",))
