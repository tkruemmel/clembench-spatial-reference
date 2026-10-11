"""Immediate reversals: X dir then X opposite-dir as that object's next own move = a no-op pair.
Tests whether oscillation is a substitute for the pass action the rules do not offer."""
import json, os, re, collections, statistics as st
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
OPP={"up":"down","down":"up","left":"right","right":"left"}
R=json.load(open(f'episodes.{ARM}.json'))
for r in R:
    it=json.load(open(os.path.join(REPO,r['path'],'interactions.json')))
    seq=[]
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m: seq.append((m.group(2),m.group(3)))
    own=collections.defaultdict(list)
    for o,d in seq: own[o].append(d)
    pairs=0; i=0
    for o,ds in own.items():
        j=0
        while j+1<len(ds):
            if OPP[ds[j]]==ds[j+1]: pairs+=1; j+=2
            else: j+=1
    r['revpairs']=pairs
json.dump(R,open(f'episodes.{ARM}.json','w'))
def show(sel,label):
    sub=[r for r in R if sel(r) and r['moves']>=10]
    if not sub: return
    fr=[2*r['revpairs']/r['moves'] for r in sub]
    print(f"  {label:34s} n={len(sub):4d}  median {100*st.median(fr):5.1f}% of moves are immediate reversals  (mean {100*sum(fr)/len(fr):4.1f}%)")
print("=== share of moves spent on immediate reversals (own-object next move) ===")
print("successes only, strong tier:")
for s in ('solo','dyad','3p','6p'): show(lambda r,s=s: r['outcome']=='success' and r['suite']==s and r['model']in('qwen38','dsv41'), f"suite {s}")
print("successes only, strong tier, by protocol (multiplayer):")
for p in ('freeform','hybrid','structured','none'): show(lambda r,p=p: r['outcome']=='success' and r['suite']!='solo' and r['proto']==p and r['model']in('qwen38','dsv41'), f"protocol {p}")
print("successes only, strong tier, by level:")
for l in ('S1','S2','S3','S4'): show(lambda r,l=l: r['outcome']=='success' and r['level']==l and r['model']in('qwen38','dsv41'), f"level {l}")
print("outcome comparison (strong tier, multiplayer):")
for o in ('success','lose','abort'): show(lambda r,o=o: r['outcome']==o and r['suite']!='solo' and r['model']in('qwen38','dsv41'), f"outcome {o}")
