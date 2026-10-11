"""The 39 forced 6p aborts: what was the hemming-in made of, and were the neighbours 'done'?"""
import json, os, re, collections, string
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
A=[a for a in json.load(open(f'aborts.{ARM}.json')) if a['n_legal_own']==0]
comp=collections.Counter(); donewall=0; rows=[]
for a in A:
    d=os.path.join(REPO,a['path'])
    ins=json.load(open(os.path.join(d,'instance.json')))
    it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    groups=ins['player_objects']
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    done=set()
    for t in it['turns']:
        for e in t:
            ac=e.get('action',{}); ty=ac.get('type',''); c=ac.get('content','')
            if ty=='move':
                m=MOVE.match(c)
                if m: o,dr=m.group(2),m.group(3); pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
            elif ty=='player_done':
                done.add(c.split()[1])
            elif ty=='abort': break
    obj=groups[string.ascii_uppercase.index(a['who'])][0]
    r0,c0=pos[obj]; inv={v:k for k,v in pos.items()}
    nb=[]
    for name,(dr,dc) in DIRS.items():
        nr,nc=r0+dr,c0+dc
        if not(0<=nr<n and 0<=nc<n): nb.append('edge')
        elif (nr,nc) in walls: nb.append('wall')
        elif (nr,nc) in inv.values() or (nr,nc) in set(pos.values()):
            o2=inv[(nr,nc)]; nb.append('obj%s%s'%(o2,'(done)' if owner[o2] in done else ''))
        else: nb.append('FREE')
    comp.update(x.split('(')[0] if x.startswith('obj') else x for x in nb)
    if any('(done)' in x for x in nb): donewall+=1
    rows.append((a['model'],a['level'],a['proto'],a['inst'],a['who'],obj,f"R{r0+1}C{c0+1}",
                 'target' if pos[obj]==tgt[obj] else f"dist{abs(pos[obj][0]-tgt[obj][0])+abs(pos[obj][1]-tgt[obj][1])}",
                 nb, a['moves'], sorted(done)))
print("neighbour composition of the enclosed object (4 sides x 39 cases):")
print(" ", collections.Counter({k:v for k,v in comp.items()}))
print("cases where at least one enclosing neighbour belongs to a player who had declared DONE:", donewall, "/", len(A))
print()
for x in sorted(rows):
    print(f"{x[0]:9s} {x[1]} {x[2]:10s} {x[3]} player {x[4]} obj {x[5]} at {x[6]} ({x[7]}) walls={x[8]} moves={x[9]} done={x[10]}")
