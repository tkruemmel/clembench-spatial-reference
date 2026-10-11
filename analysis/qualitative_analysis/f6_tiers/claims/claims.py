"""Per-claim re-extraction mirroring qual_corpus.extract's claim loop exactly,
but logging every claim (letter case, matched span, context, move index).
Read-only on the repo."""
import sys, os, json, glob, string, collections, re
sys.path.insert(0, "/Users/tom/uni/clembench-spatial-reference/analysis/qualitative_analysis")
import qual_corpus as Q
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

REPO = "/Users/tom/uni/clembench-spatial-reference"
out = []
for model, pref in Q.ARMS.items():
    for suite in Q.SUITES:
        for f in sorted(glob.glob(os.path.join(REPO, Q.tree(pref, suite), "*", "*", "*", "instance_*", "interactions.json"))):
            d = os.path.dirname(f)
            ins = json.load(open(os.path.join(d, "instance.json")))
            es = json.load(open(os.path.join(d, "scores.json")))["episode scores"]
            pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
            tgt = {o: tuple(q) for o, q in ins["target_positions"].items()}
            groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
            owner = {o: string.ascii_uppercase[i] for i, objs in enumerate(groups) for o in objs}
            it = json.load(open(f))
            snapshot = dict(pos); nmoves = 0; nmsg = 0; hist = {o: {q} for o, q in pos.items()}
            for t in it["turns"]:
                for e in t:
                    a = e.get("action", {}); ty, c = a.get("type"), str(a.get("content", ""))
                    if ty == "get message":
                        snapshot = dict(pos)
                    elif ty == "move":
                        m = Q.MOVE.match(c)
                        if m:
                            _, o, dr = m.groups(); dy, dx = Q.DIRS[dr]
                            pos[o] = (pos[o][0] + dy, pos[o][1] + dx); nmoves += 1; hist[o].add(pos[o])
                    elif ty == "message":
                        m = Q.SPK.match(c)
                        if not m:
                            continue
                        spk, txt = m.group(1), m.group(2); nmsg += 1
                        for mm in Q.CLAIM.finditer(txt):
                            lt_raw, r_, c_ = mm.groups(); lt = lt_raw.upper()
                            p = (int(r_) - 1, int(c_) - 1)
                            if lt not in snapshot:
                                continue
                            pre = txt[max(0, mm.start() - 40):mm.start()]
                            out.append(dict(model=model, suite=suite, proto=ins["comm_protocol"], lvl=ins["spatial_level"],
                                            gid=ins["game_id"], ab=bool(es.get("Aborted")), su=bool(es.get("Success")),
                                            spk=spk, lt_raw=lt_raw, own=owner.get(lt) == spk, ok=snapshot[lt] == p,
                                            is_tgt=tgt[lt] == p, stale=(p in hist[lt] and snapshot[lt] != p), occ=next((o for o, q in snapshot.items() if q == p and o != lt), None), occ_own=(owner.get(next((o for o, q in snapshot.items() if q == p and o != lt), '?')) == spk), post=pos[lt] == p, start=tuple(ins['start_positions'][lt]) == p, nmoves=nmoves, nmsg=nmsg,
                                            span=mm.group(0), pre=pre))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"claims.{ARM}.json"), "w"))
print(len(out))
