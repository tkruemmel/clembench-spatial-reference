"""Exact remaining cost at the moment the episode ended, via the repo's joint-optimum solver."""
import json, os, re, sys, time
sys.path.insert(0,"/Users/tom/uni/clembench-spatial-reference/analysis/efficiency_baseline")
import csolver
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
R=json.load(open('episodes.json'))
sel=[r for r in R if r['outcome']=='lose'] + [r for r in R if r['outcome']=='abort' and r['suite']=='6p']
out=[]
t0=time.time()
for k,r in enumerate(sel):
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    pos={o:list(p) for o,p in ins['start_positions'].items()}
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m:
                    o,dr=m.group(2),m.group(3)
                    pos[o]=[pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1]]
    gi=dict(ins); gi['start_positions']={o:list(p) for o,p in pos.items()}
    try:
        res=csolver.solve(gi, cap=8_000_000, timelimit=8.0)
    except Exception as ex:
        res={'opt':None,'lb':None,'status':str(ex)[:40]}
    out.append(dict(model=r['model'],suite=r['suite'],level=r['level'],proto=r['proto'],inst=r['inst'],
                    outcome=r['outcome'],moves=r['moves'],budget=r['max_turns'],path=r['path'],
                    remaining=res['opt'],lb=res['lb'],status=res['status'],
                    home=r['n_home_end'],nobj=r['n_obj']))
    if k%40==0: print(f"  {k}/{len(sel)} {time.time()-t0:.0f}s", flush=True)
json.dump(out,open('remaining.json','w'))
print("done", len(out), f"{time.time()-t0:.0f}s")
