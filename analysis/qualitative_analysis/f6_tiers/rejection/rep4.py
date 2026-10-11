import json, re, collections
REPO = "/Users/tom/uni/clembench-spatial-reference"
A = json.load(open(f"{REPO}/analysis/qualitative_analysis/aborts.json"))
MOVE = re.compile(r"move:\s*\**\s*([A-Za-z])(?:@\S+)?\s+(up|down|left|right)", re.I)
CANT = re.compile(r"Cannot move ([A-FX]) (up|down|left|right)")
VERD = {"move", "invalid_move", "invalid_format", "malformed_message", "player_done", "invalid_done"}
def pm(r):
    if r is None: return None
    m = MOVE.findall(r)
    return (m[-1][0].upper(), m[-1][1].lower()) if m else None
V = collections.Counter(); rows=[]
for a in A:
    it = json.load(open(f"{REPO}/{a['path']}/interactions.json"))
    ev = [e for t in it["turns"] for e in t]
    pairs, resp = [], None
    for e in ev:
        ty = e["action"]["type"]; c = e["action"].get("content", "")
        if ty == "get message": resp = c
        elif ty in VERD: pairs.append((ty, c, resp)); resp = None
    t = "W" if a["model"] == "nemotron" else "S"
    # rejected move (invalid_move) followed immediately by the same parsed move (any verdict)
    va = any(pairs[i][0]=="invalid_move" and pm(pairs[i][2]) and pm(pairs[i][2])==pm(pairs[i+1][2]) for i in range(len(pairs)-1))
    # both invalid_move, same parsed move, anywhere in episode
    vb = any(pairs[i][0]==pairs[i+1][0]=="invalid_move" and pm(pairs[i][2]) and pm(pairs[i][2])==pm(pairs[i+1][2]) for i in range(len(pairs)-1))
    # both invalid_move, same GM 'Cannot move' content, anywhere
    def g(p):
        m=CANT.search(p[1]); return m.groups() if m else p[1].split(':',1)[-1]
    vc = any(pairs[i][0]==pairs[i+1][0]=="invalid_move" and g(pairs[i])==g(pairs[i+1]) for i in range(len(pairs)-1))
    # full GM invalid content identical anywhere
    vd = any(pairs[i][0]==pairs[i+1][0]=="invalid_move" and pairs[i][1]==pairs[i+1][1] for i in range(len(pairs)-1))
    for k,v in (("a_rej_then_same_any",va),("b_both_inv_parsed",vb),("c_both_inv_gm",vc),("d_both_inv_gm_full",vd)):
        V[(k,t,v)]+=1
    rows.append((a,va,vb,vc,vd))
for k in sorted(set(x[0] for x in V)):
    print(k, "S", V[(k,"S",True)], "W", V[(k,"W",True)], "total", V[(k,"S",True)]+V[(k,"W",True)])
print()
E=collections.Counter()
for a,*_ in rows:
    it = json.load(open(f"{REPO}/{a['path']}/interactions.json"))
    ev = [e for t in it["turns"] for e in t]
    pairs, resp = [], None
    for e in ev:
        ty = e["action"]["type"]; c = e["action"].get("content", "")
        if ty == "get message": resp = c
        elif ty in VERD: pairs.append((ty, c, resp)); resp = None
    t = "W" if a["model"] == "nemotron" else "S"
    F=lambda p: p[0] not in ("move","player_done")
    ve = any(F(pairs[i]) and F(pairs[i+1]) and pairs[i][:2]==pairs[i+1][:2] for i in range(len(pairs)-1))
    run=[]
    for p in reversed(pairs):
        if not F(p): break
        run.append(p)
    run=run[::-1]
    vf = any(run[i][:2]==run[i+1][:2] for i in range(len(run)-1))
    E[("e_any_fail_same_gm_episode",t,ve)]+=1; E[("f_final_run_same_gm_anytype",t,vf)]+=1
    E[("types_final",t,tuple(p[0] for p in run))]+=1
for k in sorted(set(x[0] for x in E if not x[0].startswith('types'))):
    print(k, "S", E[(k,"S",True)], "W", E[(k,"W",True)], "total", E[(k,"S",True)]+E[(k,"W",True)])
for k,v in sorted(E.items(), key=lambda kv:-kv[1]):
    if k[0]=='types_final': print(k[1:],v)
