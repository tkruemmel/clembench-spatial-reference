"""Item 5: Nemotron solo S1 (no walls), silent vs thinking: how episodes end, what the illegal moves are,
and the full rejection chains (response -> GM verdict -> next response)."""
import json, collections, os
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attempts.json")))
A, META = J["attempts"], J["meta"]
for cond in ("silent", "thinking"):
    X = [a for a in A if a["model"] == "nemotron" and a["suite"] == "solo" and a["level"] == "S1" and a["proto"] == cond]
    eps = sorted({a["ep"] for a in X})
    print(f"== S1 solo {cond}: {len(eps)} episodes")
    print("   endings:", collections.Counter(f"{META[e]['outcome']}: {META[e]['end_msg']}" for e in eps))
    acc = [a for a in X if a["verdict"] == "move"]; rej = [a for a in X if a["verdict"] == "invalid_move"]
    print(f"   accepted moves {len(acc)}, rejected {len(rej)}; causes {dict(collections.Counter(a['status'] for a in rej))}")
    print(f"   rejected greedy {sum(a['greedy'] for a in rej)}/{len(rej)}; same object had another legal greedy dir {sum(a['legal_greedy_same'] for a in rej)}/{len(rej)}; any legal greedy alt {sum(a['legal_greedy_any'] for a in rej)}/{len(rej)}")
    print(f"   moves per episode before end: {sorted(sum(1 for a in acc if a['ep']==e) for e in eps)}")
    occ_tgt = sum(1 for a in rej if a["status"] == "occ_own" and [a["pos"][0] + {"up": -1, "down": 1}.get(a["move"][1], 0), a["pos"][1] + {"left": -1, "right": 1}.get(a["move"][1], 0)] == a["tgt"])
    print(f"   occupied rejections into the mover's own target cell: {occ_tgt}/{sum(a['status']=='occ_own' for a in rej)}")
    # identical-repeat chains ending in abort
    for e in eps:
        L = [a for a in X if a["ep"] == e]
        if META[e]["outcome"] != "abort": continue
        tail = L[-3:]
        fmt = lambda a: f"{a['move'][0]} {a['move'][1]} @R{a['pos'][0]+1}C{a['pos'][1]+1} [{a['status']}{',greedy' if a['greedy'] else ''}]" if 'pos' in a else a['verdict']
        print(f"   {e.split('/')[-1]}: last 3 -> " + " | ".join(fmt(a) for a in tail))

print("== undo rate (accepted move reverses the same object's previous accepted move) and greedy share, solo by level/cond ==")
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
for m in ("nemotron", "qwen38", "dsv41"):
    for lv in ("S1", "S2", "S3", "S4"):
        row = []
        for cond in ("silent", "thinking"):
            acc = [a for a in A if a["model"] == m and a["suite"] == "solo" and a["level"] == lv and a["proto"] == cond and a["verdict"] == "move"]
            last, und = {}, 0
            for a in acc:
                k = (a["ep"], a["move"][0]); und += last.get(k) == OPP[a["move"][1]]; last[k] = a["move"][1]
            row.append(f"{cond}: undo {100*und/max(len(acc),1):4.1f}% greedy {100*sum(a['greedy'] for a in acc)/max(len(acc),1):4.1f}% (n={len(acc)})")
        print(f"   {m:8s} {lv}  " + " | ".join(row))
