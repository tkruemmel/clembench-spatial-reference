"""For each enclosure: whose move closed the last escape route, and what did the trapped
player do on the move before it lost its second-to-last exit?"""
import json, os, re, collections, string
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
A=[a for a in json.load(open('aborts.json')) if a['n_legal_own']==0]
who_closed=collections.Counter(); rows=[]
for a in A:
    d=os.path.join(REPO,a['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    groups=ins['player_objects']
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    victim=groups[string.ascii_uppercase.index(a['who'])][0]
    def exits(o):
        occ=set(pos.values()); r0,c0=pos[o]
        return [k for k,(dr,dc) in DIRS.items()
                if 0<=r0+dr<n and 0<=c0+dc<n and (r0+dr,c0+dc) not in walls and (r0+dr,c0+dc) not in occ]
    hist=[]
    for t in it['turns']:
        for e in t:
            ac=e.get('action',{})
            if ac.get('type')=='move':
                m=MOVE.match(ac.get('content',''))
                if not m: continue
                o,dr=m.group(2),m.group(3)
                before=len(exits(victim))
                pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
                after=len(exits(victim))
                hist.append((o,owner[o],dr,before,after))
    # the move that took the victim from >0 exits to 0
    closer=None
    for i,(o,ow,dr,b,af) in enumerate(hist):
        if b>0 and af==0: closer=(i,o,ow,dr)
    if closer:
        i,o,ow,dr=closer
        kind=("the trapped player itself, moving its own object into the pocket" if o==victim
              else ("a teammate sealing the last exit"))
        who_closed[kind]+=1
        rows.append((a['model'],a['level'],a['proto'],a['inst'],a['who'],victim,kind,
                     [f"{h[0]}{h[2][0]}" for h in hist[max(0,i-4):i+1]], i+1, len(hist)))
    else:
        who_closed['enclosed from the start / never had an exit']+=1
print("who closed the last exit of the object that then had no legal move (n=%d):"%len(A))
for k,v in who_closed.most_common(): print(f"  {v:3d}  {k}")
print()
print("specimens (last 5 moves up to the sealing move; lowercase = direction initial):")
for r in sorted(rows)[:14]:
    print(f"  {r[0]:9s} {r[1]} {r[2]:10s} {r[3]} victim {r[5]} (player {r[4]}) | {r[6]}")
    print(f"      sealed at move {r[8]} of {r[9]}: {' '.join(r[7])}")
