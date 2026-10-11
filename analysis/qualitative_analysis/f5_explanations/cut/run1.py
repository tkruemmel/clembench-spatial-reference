import collections, json, pickle
from order import *
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
rows=[]
for rec in episodes(levels=("S3",)):
    ins, ev = load(rec)
    for h, v, kind in genuine(ins):
        r = door_order(rec, ins, ev, h, v)
        r.update(rec=rec, h=h, v=v, kind=kind)
        rows.append(r)
pickle.dump(rows, open(f"s3rows.{ARM}.pkl","wb"))
for kind in ("door","occ-door"):
    print("=== kind", kind)
    for suite in ("3p","6p","both"):
        print("--", suite)
        for pr in PROTOS:
            rs=[r for r in rows if r["kind"]==kind and r["rec"]["proto"]==pr and (suite=="both" or r["rec"]["suite"]==suite)]
            c=collections.Counter(r["order"] for r in rs)
            reach=[r for r in rs if r["reach"] is not None]
            aft=collections.Counter(r["after"][0] for r in reach)
            elig_unpassed=sum(1 for r in rs if r["elig"] and not r["elig"][1])
            print("%-10s n=%2d  v first %2d  h first %2d  h never %2d | reach %2d %s | v home end %d | success %d"%(pr,len(rs),c["v first"],c["h first"],c["h never arrived"],len(reach),dict(aft),sum(r["v_home_end"] for r in rs),sum(r["rec"]["outcome"]=="success" for r in rs)))
# summary lines for the ledger (kind door, 3p+6p, episodes in which the holder arrived)
D = [r for r in rows if r["kind"] == "door"]
arr = lambda prs: [r for r in D if r["rec"]["proto"] in prs and r["order"] != "h never arrived"]
print("door, B arrived, v first: " + "  ".join("%s %d/%d" % (pr, sum(r["order"] == "v first" for r in arr((pr,))), len(arr((pr,)))) for pr in PROTOS))
for lab, prs in (("prose", ("freeform", "hybrid")), ("rigid", ("none", "structured"))):
    rs = [r for r in D if r["rec"]["proto"] in prs and r["reach"] is not None]
    c = collections.Counter(r["after"][0] for r in rs)
    print("door, at the fork (%s): %d, declared at once (cut) %d, gave way %d" % (lab, len(rs), c["cut"], c["gave way"]))
