import collections, pickle, re
from order import *
from msgs import SENT, ref
from run8 import ORD2
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
W=pickle.load(open(f"waitS3.{ARM}.pkl","rb"))
key={(w["rec"]["path"],w["ply"]):w for w in W if w["h"]=="B" and w["pend"] is True}
rows=[r for r in pickle.load(open(f"s3rows.{ARM}.pkl","rb")) if r["kind"]=="door"]
def is_order(text, spk, B, rxE):
    rv=ref("D"); vo=B.owner["D"]
    for s in SENT.findall(text):
        mv = rv.search(s) or (spk == vo and vo == "D" and re.search(r"\b(I|I'll|I’ll|me|my)\b", s))
        if mv and rxE.search(s) and ORD2.search(s): return True
    return False
res=collections.defaultdict(collections.Counter); arr=collections.defaultdict(collections.Counter)
for r in rows:
    rec=r["rec"]
    if rec["proto"]=="none": continue
    ins,ev=load(rec); B=Board(ins); E=B.tgt["B"]; rxE=re.compile(cell_rx(E),re.I)
    last={}; ever=False
    for e,ply,pos,done in replay(ins,ev):
        if e[0]=="msg":
            last[e[1]]=e[2]; ever = ever or is_order(e[2],e[1],B,rxE); continue
        if e[0]!="move": continue
        vis=any(is_order(t,s,B,rxE) for s,t in last.items())
        g="prose" if rec["proto"] in ("freeform","hybrid") else "structured"
        w=key.get((rec["path"],ply))
        if w and w["rec"]["suite"]=="6p":
            res[(g,vis)]["wait" if w["kind"]!="progress" else "progress"]+=1
        if e[1]=="B" and r["first_arr"]==ply+1:
            arr[g][("visible" if vis else ("said earlier, not visible" if ever else "never said"), r["order"])]+=1
print("6p B near-home decisions with D pending: wait rate by whether a door-order statement is in the current context (players' last messages)")
for k in sorted(res): print("  ",k,dict(res[k]),"wait share %.2f"%(res[k]["wait"]/sum(res[k].values())))
print("At B's first arrival on the door (3p+6p): door-order statement status in context")
for g in arr:
    for k in sorted(arr[g]): print("  ",g,k,arr[g][k])
for g in ("prose", "structured"):
    t, f = sum(res[(g, True)].values()), sum(res[(g, False)].values())
    print("6p statement in view (%s): %d/%d decisions; waited with %d/%d, without %d/%d" % (g, t, t + f, res[(g, True)]["wait"], t, res[(g, False)]["wait"], f))
