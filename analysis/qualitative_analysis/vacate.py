"""Decompose 'left its own target' moves into immediately-undone (idling) vs sustained (re-planning)."""
import json, os, re, collections
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
R=json.load(open(f'episodes.{ARM}.json'))
agg=collections.Counter()
specimens=[]
for r in R:
    if not r['depart_target']: continue
    ins=json.load(open(os.path.join(REPO,r['path'],'instance.json')))
    it=json.load(open(os.path.join(REPO,r['path'],'interactions.json')))
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    own=collections.defaultdict(list)
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m:
                    o,d=m.group(2),m.group(3)
                    own[o].append(pos[o])
                    pos[o]=(pos[o][0]+DIRS[d][0],pos[o][1]+DIRS[d][1])
                    own[o][-1]=(own[o][-1],pos[o])
    for o,steps in own.items():
        for i,(frm,to) in enumerate(steps):
            if frm!=tgt[o]: continue
            # departure: how many of this object's own moves until it is back on target (if ever)
            k=None
            for j in range(i+1,len(steps)):
                if steps[j][1]==tgt[o]: k=j-i; break
            agg[(r['outcome'], 'returned in 1 own move' if k==1 else
                 ('returned in 2-3' if k in (2,3) else ('returned later' if k else 'never returned')))]+=1
            if k==1 and len(specimens)<3: specimens.append((r['model'],r['suite'],r['level'],r['proto'],r['inst'],o,r['outcome']))
print("departure-from-own-target moves, by what happened next:")
tot=collections.Counter()
for (oc,cls),v in sorted(agg.items()): tot[oc]+=v
for (oc,cls),v in sorted(agg.items()): print(f"  {oc:8s} {cls:22s} {v:4d}  ({100*v/tot[oc]:.0f}% of that outcome's departures)")
print(" totals:",dict(tot))
