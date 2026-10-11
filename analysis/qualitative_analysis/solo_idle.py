"""In solo there is always something productive to do. Were the abandoned objects actually
movable toward their targets at the moment the player started cycling?"""
import json, os, re, collections
from collections import deque
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
R=json.load(open(f'episodes.{ARM}.json'))
def bfs(start,goal,n,blocked):
    if start==goal: return 0
    seen={start}; q=deque([(start,0)])
    while q:
        (r,c),dd=q.popleft()
        for dr,dc in DIRS.values():
            nr,nc=r+dr,c+dc
            if 0<=nr<n and 0<=nc<n and (nr,nc) not in blocked and (nr,nc) not in seen:
                if (nr,nc)==goal: return dd+1
                seen.add((nr,nc)); q.append(((nr,nc),dd+1))
    return None
out=[]
for r in [x for x in R if x['suite']=='solo' and x['outcome']=='lose']:
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}; tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m: o,dr=m.group(2),m.group(3); pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
    # final state: for each unfinished object, is there a free path to its target?
    occ=set(pos.values())
    info=[]
    for o in sorted(tgt):
        if pos[o]==tgt[o]: continue
        blocked=walls|(occ-{pos[o]})
        dist=bfs(pos[o],tgt[o],n,blocked)
        firststep=any(0<=pos[o][0]+dr<n and 0<=pos[o][1]+dc<n and (pos[o][0]+dr,pos[o][1]+dc) not in blocked
                      for dr,dc in DIRS.values())
        info.append((o,dist,firststep))
    out.append((r['model'],r['level'],r['proto'],r['inst'],r['moves'],r['max_turns'],info))
strong=[x for x in out if x[0]!='nemotron']
print("=== strong-tier solo losses: was the abandoned work actually doable? ===")
for x in strong:
    tot=sum(d for _,d,_ in x[6] if d is not None)
    print(f"{x[0]:7s} {x[1]} {x[2]:8s} {x[3]}  stopped at {x[4]}/{x[5]} moves; unfinished: "
          + ", ".join(f"{o} needs {d if d is not None else 'no path'} more move(s){'' if s else ' [boxed in]'}" for o,d,s in x[6])
          + f"  | total remaining work {tot} moves, budget left {x[5]-x[4]}")
print()
allsolo=out
ok=[x for x in allsolo if all(d is not None and s for _,d,s in x[6])]
print(f"solo losses where EVERY unfinished object still had a clear path to its target: {len(ok)}/{len(allsolo)}")
print(f"  strong tier: {sum(1 for x in ok if x[0]!='nemotron')}/{len(strong)}")
print(f"  median remaining work vs remaining budget: "
      f"{sorted(sum(d for _,d,_ in x[6] if d is not None) for x in allsolo)[len(allsolo)//2]} moves needed vs "
      f"{sorted(x[5]-x[4] for x in allsolo)[len(allsolo)//2]} available")
