import collections, json, pickle
from order import *
rows=[]
for rec in episodes(levels=("S3",)):
    ins, ev = load(rec)
    for h, v, kind in genuine(ins):
        r = door_order(rec, ins, ev, h, v)
        r.update(rec=rec, h=h, v=v, kind=kind)
        rows.append(r)
pickle.dump(rows, open("s3rows.pkl","wb"))
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
