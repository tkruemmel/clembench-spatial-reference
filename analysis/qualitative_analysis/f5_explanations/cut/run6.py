import collections, pickle
from order import *
import os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
W,A=pickle.load(open(f"dyn.{ARM}.pkl","rb"))
print("Dynamic holder waiting (h 1-2 from target, free progress step), by level; per-decision; 6p h-nonprog only meaningful; 3p shows h-nonprog+other")
for lev in ("S1","S2","S3","S4"):
  for suite in ("3p","6p"):
    for pend in (True,False):
      line=[]
      for pr in PROTOS:
        ws=[w for w in W if w["rec"]["level"]==lev and w["rec"]["suite"]==suite and w["pend"]==pend and w["rec"]["proto"]==pr]
        c=collections.Counter(w["kind"] for w in ws)
        line.append("%s %d/%d"%(pr[:4],c["h non-progress"],len(ws)))
      print(lev,suite,"pending" if pend else "no-pend"," | ".join(line))
print()
print("Arrivals on own target that land on a live door (dynamic): eps with >=1 / eps; arrivals")
for lev in ("S1","S2","S3","S4","all"):
  for suite in ("3p","6p"):
    line=[]
    for pr in PROTOS:
      a=[x for x in A if (lev=="all" or x["rec"]["level"]==lev) and x["rec"]["suite"]==suite and x["rec"]["proto"]==pr]
      hz=[x for x in a if x["vic"]]
      hze=[x for x in hz if x["elig"]]
      line.append("%s %d/%d (elig %d)"%(pr[:4],len(hz),len(a),len(hze)))
    print(lev,suite," | ".join(line))
