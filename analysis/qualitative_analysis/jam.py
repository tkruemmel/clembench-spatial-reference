"""How often is some player's object completely hemmed in, and does the episode survive it?"""
import json, os, re, collections, string
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
R=json.load(open('episodes.json'))
res=[]
for r in [x for x in R if x['suite']=='6p']:
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json')))
    it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    def stuck():
        occ=set(pos.values()); out=[]
        for o,(r0,c0) in pos.items():
            if not any(0<=r0+dr<n and 0<=c0+dc<n and (r0+dr,c0+dc) not in walls and (r0+dr,c0+dc) not in occ
                       for dr,dc in DIRS.values()):
                out.append(o)
        return out
    ever=set(stuck()); jam_moves=0; total=0
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m:
                    o,dr=m.group(2),m.group(3)
                    pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
                    s=stuck(); ever|=set(s); total+=1
                    if s: jam_moves+=1
    res.append((r['outcome'], r['model'], r['level'], r['proto'], r['inst'],
                sorted(ever), sorted(o for o in ever if pos[o]!=tgt[o]), jam_moves, total))
json.dump(res,open('jam.json','w'))
c=collections.Counter()
for o,m,l,p,i,ever,unsolved,jm,tot in res:
    c[(o, bool(ever))]+=1
print("6p episodes, (outcome, was any object ever completely hemmed in):")
for k,v in sorted(c.items()): print("  ",k,v)
S=[x for x in res if x[0]=='success']
print()
print("successful 6p episodes with at least one fully hemmed-in moment: %d/%d" % (sum(1 for x in S if x[5]), len(S)))
print("  median share of the episode spent with someone hemmed in: %.2f" %
      sorted(x[7]/max(x[8],1) for x in S)[len(S)//2])
print("  successes where a hemmed-in object was NOT yet on its target: %d" % sum(1 for x in S if x[6]))
