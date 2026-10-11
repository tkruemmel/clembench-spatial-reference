"""For every abort: did the aborting player actually have a legal move available?"""
import json, os, re, collections, string
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
R=json.load(open(f'episodes.{ARM}.json'))
out=[]
for r in [x for x in R if x['outcome']=='abort']:
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json')))
    it=json.load(open(os.path.join(d,'interactions.json')))
    n=ins['grid_size']; walls={tuple(w) for w in ins['walls']}
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    groups=ins.get('player_objects') or ([ins['player_a_objects'],ins['player_b_objects']] if 'player_a_objects' in ins else [ins['objects']])
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    mine=collections.defaultdict(list)
    for o,p in owner.items(): mine[p].append(o)
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    def legal(objs):
        occ=set(pos.values())
        k=0
        for o in objs:
            r0,c0=pos[o]
            for dr,dc in DIRS.values():
                nr,nc=r0+dr,c0+dc
                if 0<=nr<n and 0<=nc<n and (nr,nc) not in walls and (nr,nc) not in occ: k+=1
        return k
    who=None
    for t in it['turns']:
        for e in t:
            a=e.get('action',{}); ty=a.get('type',''); c=a.get('content','')
            if ty=='move':
                m=MOVE.match(c)
                if m:
                    o,dr=m.group(2),m.group(3)
                    pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
            elif ty=='invalid_move':
                who=c.split(':')[0].replace('Player','').strip() or 'A'
            elif ty=='abort':
                objs=mine.get(who, sum(groups,[])) if len(groups)>1 else sum(groups,[])
                unsolved=[o for o in objs if pos[o]!=tgt[o]]
                out.append(dict(model=r['model'],suite=r['suite'],level=r['level'],proto=r['proto'],
                                inst=r['inst'],path=r['path'],moves=r['moves'],reason=c,who=who,
                                n_legal_own=legal(objs), n_legal_unsolved=legal(unsolved) if unsolved else 0,
                                nobj_own=len(objs), n_unsolved_own=len(unsolved)))
json.dump(out,open(f'aborts.{ARM}.json','w'))
print(len(out),"aborts analysed")
c=collections.Counter()
for a in out:
    c[(a['suite'], 'no legal move for own objects' if a['n_legal_own']==0 else 'had %d legal moves'%min(a['n_legal_own'],9))]+=1
for k,v in sorted(c.items()): print(k,v)
print()
print("aborts with ZERO legal moves for the aborting player's own objects:",
      sum(a['n_legal_own']==0 for a in out), "of", len(out))
print(" by suite:", collections.Counter(a['suite'] for a in out if a['n_legal_own']==0))
print(" by model:", collections.Counter(a['model'] for a in out if a['n_legal_own']==0))
print()
print("aborts where the player HAD legal moves but kept proposing invalid ones:",
      sum(a['n_legal_own']>0 for a in out))
print(" by model:", collections.Counter(a['model'] for a in out if a['n_legal_own']>0))
