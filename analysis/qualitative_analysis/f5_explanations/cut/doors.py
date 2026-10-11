import json, os, collections, string
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
DIRS=((-1,0),(1,0),(0,-1),(0,1))
def comp(n, walls, blocked):
    lab={};cid=0
    for r in range(n):
        for c in range(n):
            if (r,c) in walls or (r,c) in blocked or (r,c) in lab: continue
            cid+=1; st=[(r,c)]; lab[(r,c)]=cid
            while st:
                a,b=st.pop()
                for dr,dc in DIRS:
                    q=(a+dr,b+dc)
                    if 0<=q[0]<n and 0<=q[1]<n and q not in walls and q not in blocked and q not in lab:
                        lab[q]=cid; st.append(q)
    return lab
def doors(ins):
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    groups=ins['player_objects']
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    s={o:tuple(p) for o,p in ins['start_positions'].items()}
    t={o:tuple(p) for o,p in ins['target_positions'].items()}
    out=[]
    for h in t:
        lab=comp(n,walls,{t[h]})
        for v in t:
            if v==h: continue
            if lab.get(s[v]) is None or lab.get(s[v])!=lab.get(t[v]):
                out.append((h,v,owner[h]!=owner[v]))
    return out
if __name__=="__main__":
    eps=json.load(open(REPO+f'/analysis/qualitative_analysis/episodes.{ARM}.json'))
    seen={}
    for e in eps:
        if e['suite'] not in ('3p','6p') or e['model']!='qwen38' or e['proto']!='none': continue
        ins=json.load(open(os.path.join(REPO,e['path'],'instance.json')))
        seen[(e['suite'],e['level'],e['inst'])]=(doors(ins),ins.get('device'),ins['player_objects'])
    for k in sorted(seen): print(k, seen[k][0], seen[k][1], seen[k][2])

def genuine(ins):
    """(h, v) with t_h a cut cell between v's start (or v's exits from t_h) and t_v."""
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    s={o:tuple(p) for o,p in ins['start_positions'].items()}
    t={o:tuple(p) for o,p in ins['target_positions'].items()}
    out=[]
    for h in t:
        lab=comp(n,walls,{t[h]})
        for v in t:
            if v==h: continue
            if s[v]!=t[h]:
                if lab.get(s[v])!=lab.get(t[v]): out.append((h,v,'door'))
            else:
                nb=[(t[h][0]+a,t[h][1]+b) for a,b in DIRS]
                nb=[q for q in nb if q in lab]
                if any(lab[q]!=lab.get(t[v]) for q in nb): out.append((h,v,'occ-door'))
    return out
