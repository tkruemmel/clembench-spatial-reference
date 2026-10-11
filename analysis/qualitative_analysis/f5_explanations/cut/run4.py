import collections, json, pickle
from order import *
W=[]
for rec in episodes(levels=("S3",)):
    ins, ev = load(rec)
    sp=collections.defaultdict(list)
    for h,v,k in genuine(ins):
        if k=="door": sp[h].append(v)
    for w in waiting(rec, ins, ev, static_pairs=dict(sp)):
        w["rec"]=rec; W.append(w)
pickle.dump(W,open("waitS3.pkl","wb"))
def tab(sel, title):
    print(title)
    for suite in ("3p","6p"):
        for pr in PROTOS:
            ws=[w for w in W if sel(w) and w["rec"]["proto"]==pr and w["rec"]["suite"]==suite]
            c=collections.Counter(w["kind"] for w in ws)
            n=len(ws); eps=len({w["rec"]["path"] for w in ws})
            print("  %s %-10s decisions %3d (eps %2d)  progress %3d  h-nonprog %3d  other-obj %3d  -> not-progress %.2f"%(suite,pr,n,eps,c["progress"],c["h non-progress"],c["other object"],(n-c["progress"])/n if n else float('nan')))
tab(lambda w: w["h"]=="B" and w["pend"] is True, "S3 holder B, D pending (D not inside corridor)")
tab(lambda w: w["h"]=="B" and w["pend"] is False, "S3 holder B, D already passed")
tab(lambda w: w["pend"] is None, "S3 non-holders (all other objects)")
