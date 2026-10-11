import collections, pickle, re
from order import *
from msgs import SENT, ref
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
rows=pickle.load(open(f"s3ord.{ARM}.pkl","rb"))
has=lambda r: any(x[0]<=r["win_end"] for x in r["ord"])
out=collections.defaultdict(collections.Counter)
ex=[]
for r in rows:
    rec=r["rec"]
    if rec["proto"]=="none" or r["order"]=="h never arrived": continue
    ins,ev=load(rec); B=Board(ins); E=B.tgt["B"]; rxE=re.compile(cell_rx(E),re.I); vo=B.owner["D"]; rv=ref("D")
    named=False
    for e,ply,pos,done in replay(ins,ev):
        if e[0]=="msg" and ply<=r["win_end"]:
            for s in SENT.findall(e[2]):
                if rxE.search(s) and (e[1]==vo or rv.search(s)) and not re.search(r"PLAN:\s*move\s+[A-F]\s+from\s+\(R\d,\s*C\d\)\s+to\s+%s\s*$"%cell_rx(E),s.strip()):
                    named=True
                    if rec["proto"]=="structured" and not has(r) and len(ex)<12: ex.append((rec["model"],rec["suite"],rec["inst"],ply,e[1],r["order"],s.strip()[:160]))
    g="prose" if rec["proto"]!="structured" else "structured"
    out[(g, "order-stmt" if has(r) else ("door named by/for D, no order" if named else "neither"))][r["order"]]+=1
for k in sorted(out): print(k, dict(out[k]))
for x in ex: print(x)
