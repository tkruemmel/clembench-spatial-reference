"""At each reversal move: was a strictly better move available to the same player?
Better = a legal move of one of the player's OTHER unplaced objects that reduces that
object's BFS distance to its target (walls + all other objects as blockers)."""
import json, os, re, collections, string
from collections import deque
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
OPP={"up":"down","down":"up","left":"right","right":"left"}
def bfs(src,goal,n,blocked):
    if src==goal: return 0
    seen={src}; q=deque([(src,0)])
    while q:
        (r,c),dd=q.popleft()
        for dr,dc in DIRS.values():
            t=(r+dr,c+dc)
            if 0<=t[0]<n and 0<=t[1]<n and t not in blocked and t not in seen:
                if t==goal: return dd+1
                seen.add(t); q.append((t,dd+1))
    return None
R=json.load(open('episodes.json'))
cnt=collections.Counter(); spec=[]
for r in R:
    if r['model']=='nemotron': continue
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    groups=ins.get('player_objects') or ([ins['player_a_objects'],ins['player_b_objects']] if 'player_a_objects' in ins else [ins['objects']])
    mine={string.ascii_uppercase[i]:list(g) for i,g in enumerate(groups)}
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    seq=[]
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m: seq.append((m.group(1) or 'A',m.group(2),m.group(3)))
    lastdir={}
    for k,(who,o,dr) in enumerate(seq):
        rev = lastdir.get(o) is not None and OPP[lastdir[o]]==dr
        if rev:
            occ=set(pos.values())
            better=[]
            for x in mine[who]:
                if x==o or pos[x]==tgt[x]: continue
                blocked=walls|(occ-{pos[x]})
                d0=bfs(pos[x],tgt[x],n,blocked)
                if d0 is None: continue
                for nm,(ddr,ddc) in DIRS.items():
                    t2=(pos[x][0]+ddr,pos[x][1]+ddc)
                    if not(0<=t2[0]<n and 0<=t2[1]<n) or t2 in blocked: continue
                    d1=bfs(t2,tgt[x],n,walls|(occ-{pos[x]}))
                    if d1 is not None and d1<d0: better.append((x,nm)); break
            cnt[(r['suite'], bool(better))]+=1
            if better and len(spec)<40 and r['suite'] in ('dyad','solo'):
                spec.append((r['model'],r['suite'],r['level'],r['proto'],r['inst'],r['outcome'],k+1,who,o,dr,better[:3],r['path']))
        lastdir[o]=dr
        pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
json.dump(spec,open('dominated_spec.json','w'))
print("reversal moves, strong tier: was a distance-reducing move available for one of the same player's other objects?")
for s in ('solo','dyad','3p','6p'):
    y,nn=cnt[(s,True)],cnt[(s,False)]
    if y+nn: print(f"  {s:4s}: {y:5d} of {y+nn:5d} reversals were strictly dominated ({100*y/(y+nn):.0f}%)")
