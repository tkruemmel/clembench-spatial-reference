import collections, json, pickle, statistics as st
from order import *
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
rows=[r for r in pickle.load(open(f"s3rows.{ARM}.pkl","rb")) if r["kind"]=="door"]
G={}
for r in rows:
    k=r["rec"]["inst"]
    if k not in G:
        ins,_=load(r["rec"]); B=Board(ins); E=B.tgt["B"]
        G[k]=B.dist(B.start["B"],E,set())<B.dist(B.start["D"],E,set())+1
# D's moves before pass, B moves before arrival, per protocol, on B-greedy-first boards
def dmoves(rec, upto, obj):
    ins,ev=load(rec); n=0; ply=0
    for e in ev:
        if e[0]=="move":
            if ply>=upto: break
            ply+=1
            if e[1]==obj: n+=1
    return n
for cls in (True,False):
  print("B-greedy-first boards" if cls else "D-greedy-first boards")
  for pr in PROTOS:
    rs=[r for r in rows if r["rec"]["proto"]==pr and G[r["rec"]["inst"]]==cls]
    both=[r for r in rs if r["first_arr"] is not None and r["first_pass"] is not None]
    exB=[r["arr_moves_h"]-r["dmin_h"] for r in rs if r["first_arr"] is not None]
    dD=[dmoves(r["rec"],r["first_pass"],"D") for r in rs if r["first_pass"] is not None]
    # B moves made by the time D passed (or B arrival if earlier)
    bAtPass=[dmoves(r["rec"],r["first_pass"],"B") for r in rs if r["first_pass"] is not None]
    print("  %-10s n=%d  B excess moves before arr: med %.1f mean %.1f | D moves to pass med %.1f | B moves by D's pass med %.1f | B arr ply - D pass ply med %s"%(
        pr,len(rs),st.median(exB),st.mean(exB),st.median(dD),st.median(bAtPass),st.median([r["first_arr"]-r["first_pass"] for r in both])))
