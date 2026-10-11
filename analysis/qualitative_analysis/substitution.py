"""Within-player test: does a player start idling only once its OTHER objects are all home?
For every move by a player owning >1 object, classify it as a reversal or not, and record
whether that player still had unfinished work on another object at that moment."""
import json, os, re, collections, string
from scipy.stats import fisher_exact
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player (\w+) moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
OPP={"up":"down","down":"up","left":"right","right":"left"}
R=json.load(open(f'episodes.{ARM}.json'))
cnt=collections.Counter()
for r in R:
    if r['suite'] in ('solo','6p') or r['model']=='nemotron': continue
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    groups=ins.get('player_objects') or [ins['player_a_objects'],ins['player_b_objects']]
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    mine=collections.defaultdict(list)
    for i,g in enumerate(groups): mine[string.ascii_uppercase[i]]=list(g)
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    seq=[]
    for t in it['turns']:
        for e in t:
            a=e.get('action',{})
            if a.get('type')=='move':
                m=MOVE.match(a.get('content',''))
                if m: seq.append((m.group(1),m.group(2),m.group(3)))
    lastdir={}
    for who,o,dr in seq:
        others=[x for x in mine[who] if x!=o]
        busy=any(pos[x]!=tgt[x] for x in others)          # other own objects still unplaced
        rev = lastdir.get(o) is not None and OPP[lastdir[o]]==dr
        cnt[('other work pending' if busy else 'only this object left', 'reversal' if rev else 'progress')]+=1
        lastdir[o]=dr
        pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
print("dyad + 3p, strong tier: every move classified (n=%d)" % sum(cnt.values()))
for k in ('other work pending','only this object left'):
    rv,pg=cnt[(k,'reversal')],cnt[(k,'progress')]
    print(f"  while {k:22s}: {rv:5d} reversals / {rv+pg:6d} moves = {100*rv/(rv+pg):.1f}%")
a=cnt[('only this object left','reversal')]; b=cnt[('only this object left','progress')]
c=cnt[('other work pending','reversal')]; d2=cnt[('other work pending','progress')]
odds,p=fisher_exact([[a,b],[c,d2]])
print(f"  odds ratio {odds:.2f}, Fisher p={p:.3g}")
