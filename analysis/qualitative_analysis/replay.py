"""Compact interleaved replay: python3 replay.py <results-subpath> [from] [to] [--msg N]"""
import json, os, re, sys, string
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
p=sys.argv[1]
_m=sys.argv.index('--msg') if '--msg' in sys.argv else None
W=int(sys.argv[_m+1]) if _m else 200
_nums=[int(a) for i,a in enumerate(sys.argv[2:],start=2) if a.isdigit() and (_m is None or i!=_m+1)]
lo=_nums[0] if _nums else 0
hi=_nums[1] if len(_nums)>1 else 10**9
d=os.path.join(REPO,p)
ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
es=json.load(open(os.path.join(d,'scores.json')))['episode scores']
groups=ins.get('player_objects') or ([ins['player_a_objects'],ins['player_b_objects']] if 'player_a_objects' in ins else [ins['objects']])
owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
pos={o:tuple(q) for o,q in ins['start_positions'].items()}; tgt={o:tuple(q) for o,q in ins['target_positions'].items()}
cell=lambda c:f"R{c[0]+1}C{c[1]+1}"
print(f"# {p}")
print(f"# grid {ins['grid_size']}  level {ins['spatial_level']}  protocol {ins.get('comm_protocol','-')}  optimal {ins['optimal_moves']}  budget {ins['max_turns']}")
print(f"# outcome: success={es['Success']} lose={es['Lose']} abort={es['Aborted']}  moves={es['Move Count']}")
print("# players: " + " | ".join(f"{string.ascii_uppercase[i]}:{','.join(g)}" for i,g in enumerate(groups)))
for o in sorted(tgt): print(f"#   {o}: {cell(pos[o])} -> {cell(tgt[o])}  (manhattan {abs(pos[o][0]-tgt[o][0])+abs(pos[o][1]-tgt[o][1])})")
if ins.get('device'): print(f"# device: {ins['device']}")
print(f"# walls: {[cell(tuple(w)) for w in ins['walls']]}")
i=0
for t in it['turns']:
    for e in t:
        a=e.get('action',{}); ty=a.get('type'); c=a.get('content','')
        if ty=='move':
            m=MOVE.match(c)
            if not m: continue
            i+=1; o,dr=m.group(2),m.group(3); frm=pos[o]
            pos[o]=(frm[0]+DIRS[dr][0],frm[1]+DIRS[dr][1])
            if lo<=i<=hi:
                home=sum(pos[x]==tgt[x] for x in tgt)
                mark=" <ON TARGET" if pos[o]==tgt[o] else (" <OFF ITS TARGET" if frm==tgt[o] else "")
                print(f"{i:4d} [{owner[o]}] {o} {dr:5s} {cell(frm)}->{cell(pos[o])}  home {home}/{len(tgt)}{mark}")
        elif ty=='message' and lo<=i<=hi:
            print(f"        msg {c[:W].replace(chr(10),' | ')}")
        elif ty in ('invalid_move','invalid_format','invalid_done','player_done','lose','abort','success') and lo<=i<=hi:
            print(f"     ** {ty}: {c[:220]}")
