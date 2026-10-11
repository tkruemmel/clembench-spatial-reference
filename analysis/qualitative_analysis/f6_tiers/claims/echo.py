"""For each teammate claim (lowercase-letter matches dropped): was the same
(letter, cell) asserted in an earlier message of the episode, and by whom?"""
import sys, os, json, glob, string, collections
sys.path.insert(0, "/Users/tom/uni/clembench-spatial-reference/analysis/qualitative_analysis")
import qual_corpus as Q
REPO = "/Users/tom/uni/clembench-spatial-reference"
res = collections.defaultdict(collections.Counter)
for model, pref in Q.ARMS.items():
    for suite in Q.SUITES:
        for f in sorted(glob.glob(os.path.join(REPO, Q.tree(pref, suite), "*", "*", "*", "instance_*", "interactions.json"))):
            d = os.path.dirname(f); ins = json.load(open(os.path.join(d, "instance.json")))
            if ins["comm_protocol"] not in ("freeform", "hybrid"): continue
            pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
            groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
            owner = {o: string.ascii_uppercase[i] for i, objs in enumerate(groups) for o in objs}
            said = {}  # (letter, cell) -> set of speakers who said it earlier
            snap = dict(pos)
            for t in json.load(open(f))["turns"]:
                for e in t:
                    a = e.get("action", {}); ty, c = a.get("type"), str(a.get("content", ""))
                    if ty == "get message": snap = dict(pos)
                    elif ty == "move":
                        m = Q.MOVE.match(c)
                        if m:
                            _, o, dr = m.groups(); dy, dx = Q.DIRS[dr]; pos[o] = (pos[o][0] + dy, pos[o][1] + dx)
                    elif ty == "message":
                        m = Q.SPK.match(c)
                        if not m: continue
                        spk, txt = m.group(1), m.group(2); new = []
                        for lt_raw, r_, c_ in Q.CLAIM.findall(txt):
                            if lt_raw.islower(): continue
                            lt = lt_raw.upper(); p = (int(r_) - 1, int(c_) - 1)
                            if lt not in snap: continue
                            new.append((lt, p))
                            if owner.get(lt) == spk: continue
                            ok = snap[lt] == p; prev = said.get((lt, p), set())
                            src = "echo of owner" if owner[lt] in prev else ("echo of other" if prev else "not said before")
                            res[model][(src, ok)] += 1
                        for k in new: said.setdefault(k, set()).add(spk)
for m, c in res.items():
    tot = sum(c.values())
    print(m, tot)
    for src in ("echo of owner", "echo of other", "not said before"):
        n = c[(src, True)] + c[(src, False)]
        if n: print(f"   {src:16} {n:5} ({100*n/tot:4.0f}% of claims)  accuracy {100*c[(src, True)]/n:5.1f}%")
