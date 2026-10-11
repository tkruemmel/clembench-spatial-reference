import collections, pickle
from order import *
P=("freeform","hybrid"); R=("none","structured")
rows=pickle.load(open("s3ord.pkl","rb"))
arr=lambda rs:[r for r in rs if r["order"]!="h never arrived"]
def cnt(prs, f, base=arr, extra=lambda r: True):
    rs=[r for r in base(rows) if r["rec"]["proto"] in prs and extra(r)]
    return sum(f(r) for r in rs), len(rs)
def show(name,a,b):
    print("%-55s prose %d/%d  rigid %d/%d  p=%.2g"%(name,a[0],a[1],b[0],b[1],fisher(a[0],a[1]-a[0],b[0],b[1]-b[0])))
vf=lambda r:r["order"]=="v first"
show("S3 v first (B arrived)",cnt(P,vf),cnt(R,vf))
for s in ("3p","6p"):
    show("  %s"%s,cnt(P,vf,extra=lambda r:r["rec"]["suite"]==s),cnt(R,vf,extra=lambda r:r["rec"]["suite"]==s))
G={}
for r in rows:
    k=r["rec"]["inst"]
    if k not in G:
        ins,_=load(r["rec"]); B=Board(ins); E=B.tgt["B"]
        G[k]=B.dist(B.start["B"],E,set())<B.dist(B.start["D"],E,set())+1
show("  boards where greedy order = B first",cnt(P,vf,extra=lambda r:G[r["rec"]["inst"]]),cnt(R,vf,extra=lambda r:G[r["rec"]["inst"]]))
show("  boards where greedy order = D first",cnt(P,vf,extra=lambda r:not G[r["rec"]["inst"]]),cnt(R,vf,extra=lambda r:not G[r["rec"]["inst"]]))
show("S3 reach (B's player done-eligible, D not passed), all eps",cnt(P,lambda r:r["reach"] is not None,base=lambda x:x),cnt(R,lambda r:r["reach"] is not None,base=lambda x:x))
hf=lambda x:[r for r in x if r["order"]=="h first"]
show("  reach | B first",cnt(P,lambda r:r["reach"] is not None,base=hf),cnt(R,lambda r:r["reach"] is not None,base=hf))
for s in ("3p","6p"):
    show("  reach | B first, %s"%s,cnt(P,lambda r:r["reach"] is not None,base=hf,extra=lambda r:r["rec"]["suite"]==s),cnt(R,lambda r:r["reach"] is not None,base=hf,extra=lambda r:r["rec"]["suite"]==s))
# per-episode waiting
W=pickle.load(open("waitS3.pkl","rb"))
def epwait(prs, suite, pend, kinds=("h non-progress",)):
    eps=collections.defaultdict(list)
    for w in W:
        if w["h"]=="B" and w["pend"] is pend and w["rec"]["proto"] in prs and w["rec"]["suite"]==suite:
            eps[w["rec"]["path"]].append(w["kind"] in kinds)
    return sum(any(v) for v in eps.values()), len(eps)
for s,k in (("6p",("h non-progress",)),("3p",("h non-progress",)),("3p",("h non-progress","other object"))):
    show("S3 B waits near home (eps>=1 wait), D pending %s %s"%(s,"/".join(x[:5] for x in k)),epwait(P,s,True,k),epwait(R,s,True,k))
    show("    same, D already passed",epwait(P,s,False,k),epwait(R,s,False,k))
# dynamic S2/S4 6p per-episode
Wd,A=pickle.load(open("dyn.pkl","rb"))
for lev in ("S2","S4"):
    for suite in ("6p",):
        def ep(prs,pend):
            eps=collections.defaultdict(list)
            for w in Wd:
                if w["rec"]["level"]==lev and w["rec"]["suite"]==suite and w["pend"]==pend and w["rec"]["proto"] in prs:
                    eps[w["rec"]["path"]].append(w["kind"]=="h non-progress")
            return sum(any(v) for v in eps.values()), len(eps)
        show("%s %s dyn holder waits (eps), pending"%(lev,suite),ep(P,True),ep(R,True))
# arrivals on a live door, all levels
for lev in ("S2","S3","S4"):
    a=lambda prs:[x for x in A if x["rec"]["level"]==lev and x["rec"]["proto"] in prs]
    show("%s arrivals landing on a live door (dynamic)"%lev,(sum(1 for x in a(P) if x["vic"]),len(a(P))),(sum(1 for x in a(R) if x["vic"]),len(a(R))))
a=lambda prs:[x for x in A if x["rec"]["proto"] in prs]
show("all arrivals landing on a live door",(sum(1 for x in a(P) if x["vic"]),len(a(P))),(sum(1 for x in a(R) if x["vic"]),len(a(R))))
show("  ... that also made the player done-eligible",(sum(1 for x in a(P) if x["vic"] and x["elig"]),len(a(P))),(sum(1 for x in a(R) if x["vic"] and x["elig"]),len(a(R))))
# structured
S=lambda f,extra: cnt(("structured",),f,extra=extra)
has=lambda r: any(x[0]<=r["win_end"] for x in r["ord"])
a1=S(vf,has); a2=S(vf,lambda r:not has(r))
print("structured: v first with door-order stmt %d/%d vs without %d/%d p=%.2g"%(a1+a2[:0]+a2 if False else (a1[0],a1[1],a2[0],a2[1],fisher(a1[0],a1[1]-a1[0],a2[0],a2[1]-a2[0]))))
n=cnt(("none",),vf)
print("structured without order stmt (D target revealed 8/21) vs none %d/%d p=%.2g"%(n[0],n[1],fisher(8,13,n[0],n[1]-n[0])))
print("none vs structured v first: %d/%d vs %d/%d p=%.2g"%(n[0],n[1],*cnt(("structured",),vf),fisher(n[0],n[1]-n[0],cnt(("structured",),vf)[0],cnt(("structured",),vf)[1]-cnt(("structured",),vf)[0])))
