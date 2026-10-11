import collections, pickle, re, random
from order import *
from msgs import SENT, ref, PASS
ORD2 = re.compile(r"\b(until|till|once|after|before)\b[^.;!?]{0,60}\b(pass\w*|clear\w*|cross\w*|through|enter\w*|reach\w*|descend\w*|climb\w*|vacate\w*|in|inside|done|placed|home)\b"
                  r"|\bfirst\b|\blet\b[^.;!?]{0,25}\bpass|\b(wait|hold)\w*\b", re.I)
def door_order_sents(rec, ins, ev, h, v, upto=None):
    B = Board(ins); E = B.tgt[h]
    rxE = re.compile(cell_rx(E), re.I); rv = ref(v); vo = B.owner[v]
    out = []
    for e, ply, pos, done in replay(ins, ev):
        if e[0] != "msg" or (upto is not None and ply > upto): continue
        spk, text = e[1], e[2]
        for s in SENT.findall(text):
            mv = rv.search(s) or (spk == vo and vo == v and re.search(r"\b(I|I'll|I’ll|me|my)\b", s))
            if mv and rxE.search(s) and ORD2.search(s):
                role = "holder" if spk == B.owner[h] else "victim" if spk == vo else "third"
                out.append((ply, spk, role, s.strip()))
    return out
def route_sents(rec, ins, ev, h, v, upto=None):
    B = Board(ins); E = B.tgt[h]
    rxE = re.compile(cell_rx(E), re.I); rv = ref(v); vo = B.owner[v]
    out = []
    for e, ply, pos, done in replay(ins, ev):
        if e[0] != "msg" or (upto is not None and ply > upto): continue
        spk, text = e[1], e[2]
        if re.match(r"\s*PLAN:", text) and spk == vo:
            pass
        for s in SENT.findall(text):
            mv = rv.search(s) or (spk == vo and vo == v and re.search(r"\b(I|I'll|I’ll|me|my)\b", s))
            if mv and rxE.search(s) and PASS.search(s) and not ORD2.search(s):
                out.append((ply, spk, s.strip()))
    return out
if __name__ == "__main__":
    rows=[r for r in pickle.load(open("s3rows.pkl","rb")) if r["kind"]=="door"]
    for r in rows:
        ins, ev = load(r["rec"])
        r["ord"] = door_order_sents(r["rec"], ins, ev, r["h"], r["v"])
        r["route"] = route_sents(r["rec"], ins, ev, r["h"], r["v"])
    pickle.dump(rows, open("s3ord.pkl","wb"))
    print("Door-order statement (D-ref + door cell + order/wait word, same sentence) in the window before min(D pass, B arrival)")
    for pr in PROTOS[1:]:
        for o in ("v first","h first"):
            rs=[r for r in rows if r["rec"]["proto"]==pr and r["order"]==o]
            w=[r for r in rs if any(p<=r["win_end"] for p,*_ in r["ord"])]
            first_roles=collections.Counter(min((x for x in r["ord"] if x[0]<=r["win_end"]))[2] for r in w)
            rt=[r for r in rs if any(p<=r["win_end"] for p,*_ in r["route"])]
            print("  %-10s %-8s n=%2d  door-order %2d  first said by %s | route-only (no order word) %2d"%(pr,o,len(rs),len(w),dict(first_roles),len(rt)))
    random.seed(5)
    allh=[(r["rec"]["proto"],r["rec"]["model"],r["rec"]["suite"],r["rec"]["inst"],x) for r in rows for x in r["ord"] if x[0]<=r["win_end"]]
    print(len(allh))
    for x in random.sample(allh,40): print(x[0],x[1],x[2],x[3],x[4][0],x[4][1],x[4][2],"|",x[4][3][:200])
