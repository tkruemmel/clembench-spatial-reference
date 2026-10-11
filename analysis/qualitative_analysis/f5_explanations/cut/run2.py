import collections, json, pickle, statistics as st
from order import *
rows=pickle.load(open("s3rows.pkl","rb"))
rows=[r for r in rows if r["kind"]=="door"]
# per board greedy order
seen={}
for r in rows:
    k=r["rec"]["inst"]
    if k in seen: continue
    ins,_=load(r["rec"]); B=Board(ins)
    E=B.tgt["B"]
    seen[k]=(B.dist(B.start["B"],E,set()), B.dist(B.start["D"],E,set())+1, B.dist(B.start["D"],B.tgt["D"],set()))
print("board: dist B->E, dist D->pass, dist D->target")
for k in sorted(seen): print(k, seen[k], "B earlier" if seen[k][0]<seen[k][1] else "")
def med(x): return st.median(x) if x else None
print()
for suite in ("3p","6p"):
  for pr in PROTOS:
    rs=[r for r in rows if r["rec"]["proto"]==pr and r["rec"]["suite"]==suite]
    fa=[r["first_arr"] for r in rs if r["first_arr"] is not None]
    fp=[r["first_pass"] for r in rs if r["first_pass"] is not None]
    ex=[r["arr_moves_h"]-r["dmin_h"] for r in rs if r["first_arr"] is not None]
    print("%s %-10s B first-arrival ply med %s | D pass ply med %s | B excess moves before arrival med %s mean %.1f | B-move share of plies before arr %.3f"%(suite,pr,med(fa),med(fp),med(ex),st.mean(ex),
        st.mean([r["arr_moves_h"]/max(1,r["first_arr"]) for r in rs if r["first_arr"]])))
# order by board greedy class
print()
for pr in PROTOS:
    for cls in (True,False):
        rs=[r for r in rows if r["rec"]["proto"]==pr and (seen[r["rec"]["inst"]][0]<seen[r["rec"]["inst"]][1])==cls]
        c=collections.Counter(r["order"] for r in rs)
        print(pr, "B-greedy-first" if cls else "D-greedy-first", len(rs), dict(c))
