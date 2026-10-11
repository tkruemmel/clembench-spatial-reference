import sys, os, json, collections, statistics as st
ROOT = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, os.path.join(ROOT, "analysis/thesis"))
import tables_comm as tc
A = json.load(open(os.path.join(ROOT, "analysis/qualitative_analysis/aborts.json")))
def nmoves(p):
    n = 0
    for t in json.load(open(os.path.join(ROOT, p, "interactions.json")))["turns"]:
        for e in t:
            if e.get("action", {}).get("type") == "move": n += 1
    return n
def start_home(p):
    inst = json.load(open(os.path.join(ROOT, p, "instance.json")))
    return sum(tuple(inst["start_positions"][o]) == tuple(inst["target_positions"][o]) for o in inst["objects"])
by = collections.defaultdict(list)
mism = 0
for a in A:
    m = nmoves(a["path"]); h, _ = tc.objects_home(os.path.join(ROOT, a["path"]))
    if m != a["moves"]: mism += 1
    by[a["model"]].append(dict(moves=m, home=h, reason=a["reason"], suite=a["suite"], sh=start_home(a["path"]), path=a["path"]))
print("moves field mismatches vs recount:", mism)
for mod, v in by.items():
    mv = [x["moves"] for x in v]
    print(f"\n{mod}: aborts {len(v)}  reasons {collections.Counter(x['reason'] for x in v)}")
    print("  objects home at start (any >0):", sum(1 for x in v if x["sh"]))
    print("  accepted moves: zero", sum(1 for x in mv if x == 0), " median", st.median(mv), " mean %.1f" % st.mean(mv),
          " <=5", sum(1 for x in mv if x <= 5), " >=10", sum(1 for x in mv if x >= 10), " >=20", sum(1 for x in mv if x >= 20))
    z = [x for x in v if x["home"] == 0]
    print("  0/6 home:", len(z), " of which zero moves", sum(1 for x in z if x["moves"] == 0),
          " median moves among 0/6", st.median([x["moves"] for x in z]))
    print("  home dist", sorted(collections.Counter(x["home"] for x in v).items()))
    print("  by suite zero-move:", collections.Counter(x["suite"] for x in v if x["moves"] == 0), " totals", collections.Counter(x["suite"] for x in v))
