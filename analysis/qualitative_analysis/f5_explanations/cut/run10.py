import collections, pickle, re, random
from order import *
from msgs import ref
W=pickle.load(open("waitS3.pkl","rb"))
WAITRX=re.compile(r"\b(until|till|once|after|before|first|wait\w*|hold\w*|let\b|clear\w*|open|free|room|pass\w*|cross\w*|through)\b",re.I)
# message accompanying each decision: the msg by the mover right after the move
cache={}
def msg_at(rec, ply, spk):
    k=rec["path"]
    if k not in cache:
        ins,ev=load(rec); m={}
        for e,p,pos,done in replay(ins,ev):
            if e[0]=="msg": m.setdefault((p,e[1]),e[2])
        cache[k]=(ins,m)
    return cache[k][1].get((ply+1,spk),"")
random.seed(3)
res=collections.defaultdict(collections.Counter); samples=collections.defaultdict(list)
for w in W:
    if w["h"]!="B" or w["pend"] is not True: continue
    pr=w["rec"]["proto"]
    if pr=="none": continue
    ins=load(w["rec"])[0] if w["rec"]["path"] not in cache else cache[w["rec"]["path"]][0]
    m=msg_at(w["rec"],w["ply"],w["p"])
    B=Board(ins); E=B.tgt["B"]
    rxE=re.compile(cell_rx(E),re.I)
    dref=bool(ref("D").search(m)) or (w["rec"]["suite"]=="6p" and False)
    tag="D+door+wait" if dref and rxE.search(m) and WAITRX.search(m) else ("D mentioned" if dref else "no D")
    kind="wait" if w["kind"]!="progress" else "progress"
    res[(pr in ("freeform","hybrid"), w["rec"]["suite"], kind)][tag]+=1
    if kind=="wait": samples[pr].append((w["rec"]["model"],w["rec"]["suite"],w["rec"]["inst"],w["ply"],w["kind"],m))
for k in sorted(res): print(k, dict(res[k]), sum(res[k].values()))
for pr in ("structured","freeform","hybrid"):
    print("=====",pr,len(samples[pr]))
    for s in random.sample(samples[pr],min(8,len(samples[pr]))): print(s[:5],"|",s[5][:300].replace("\n"," "))
