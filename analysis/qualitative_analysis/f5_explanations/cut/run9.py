import collections, pickle, statistics as st
from order import *
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
rows=pickle.load(open(f"s3ord.{ARM}.pkl","rb"))
def has(r,key): return any(x[0]<=r["win_end"] for x in r[key])
def target_rev(r):
    pass
import msgs
tg=pickle.load(open(f"s3tagged.{ARM}.pkl","rb"))
tmap={(t["rec"]["path"]):t["hits"] for t in tg}
print("structured / prose: v-first rate by presence of a door-order statement in the window (episodes where B arrived)")
for prs in (("structured",),("freeform","hybrid")):
    for flag in (True,False):
        rs=[r for r in rows if r["rec"]["proto"] in prs and r["order"]!="h never arrived" and has(r,"ord")==flag]
        print("  %-22s order-stmt=%-5s v first %d/%d"%("+".join(prs),flag,sum(r["order"]=="v first" for r in rs),len(rs)))
arrived=lambda prs:[r for r in rows if r["rec"]["proto"] in prs and r["order"]!="h never arrived"]
P_,S_=arrived(("freeform","hybrid")),arrived(("structured",))
a,b=sum(has(r,"ord") for r in P_),sum(has(r,"ord") for r in S_)
print("door-order statement made (B arrived): prose %d/%d structured %d/%d Fisher p=%.2g"%(a,len(P_),b,len(S_),fisher(a,len(P_)-a,b,len(S_)-b)))
c=sum(r["order"]=="v first" for r in P_ if has(r,"ord")); d=sum(r["order"]=="v first" for r in S_ if has(r,"ord"))
print("v first given a statement: prose %d/%d structured %d/%d Fisher p=%.2g"%(c,a,d,b,fisher(c,a-c,d,b-d)))
print("structured 2x2: order stmt x D-target revealed (d)")
for o in (True,False):
    for d in (True,False):
        rs=[r for r in rows if r["rec"]["proto"]=="structured" and r["order"]!="h never arrived" and has(r,"ord")==o and bool(tmap[r["rec"]["path"]].get("d_v_target"))==d]
        print("  order=%s targetrev=%s v first %d/%d"%(o,d,sum(r["order"]=="v first" for r in rs),len(rs)))
sa=[r for r in rows if r["rec"]["proto"]=="structured" and r["order"]!="h never arrived"]
print("structured: D's target disclosed before min(D pass, B arrival) %d/%d"%(sum(bool(tmap[r["rec"]["path"]].get("d_v_target")) for r in sa),len(sa)))
print("structured: window length (plies) by order")
for o in ("v first","h first"):
    rs=[r for r in rows if r["rec"]["proto"]=="structured" and r["order"]==o]
    print("  ",o,"median win",st.median([r["win_end"] for r in rs]),"| order-stmt rate among win>=20:",sum(has(r,"ord") for r in rs if r["win_end"]>=20),"/",sum(1 for r in rs if r["win_end"]>=20))
# greedy class within structured
G={}
for r in rows:
    k=r["rec"]["inst"]
    if k not in G:
        ins,_=load(r["rec"]); B=Board(ins); E=B.tgt["B"]
        G[k]=B.dist(B.start["B"],E,set())<B.dist(B.start["D"],E,set())+1
print("structured, B-greedy-first boards only: v-first by order stmt")
for flag in (True,False):
    rs=[r for r in rows if r["rec"]["proto"]=="structured" and r["order"]!="h never arrived" and G[r["rec"]["inst"]] and has(r,"ord")==flag]
    print("  order-stmt=%s v first %d/%d"%(flag,sum(r["order"]=="v first" for r in rs),len(rs)))
print("first door-order statement ply relative to B's first arrival / D pass (prose & structured, v-first)")
for prs in (("structured",),("freeform","hybrid")):
    rs=[r for r in rows if r["rec"]["proto"] in prs and r["order"]=="v first" and has(r,"ord")]
    lead=[r["first_pass"]-min(x[0] for x in r["ord"]) for r in rs]
    print("  ",prs,"plies from first statement to D pass: median",st.median(lead),"min",min(lead))
