"""Strict: the commitment segment must assert freeness (clear/open/free/avoid/won't use).
Violation = the speaker's own object occupies that cell later. Consequence = the cell is a
teammate's target, or a teammate is later rejected for trying to move into it."""
import json, os, re, collections, string
REPO="/Users/tom/uni/clembench-spatial-reference"
MOVE=re.compile(r"^Player (\w+) moved ([A-F]) (up|down|left|right)$")
BAD=re.compile(r"^Player (\w+): \[occupied\] Cannot move ([A-FX]) (\w+): \(R(\d+),C(\d+)\) is occupied")
DIRS={"up":(-1,0),"down":(1,0),"left":(0,-1),"right":(0,1)}
CELL=re.compile(r"\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?", re.I)
PROMISE=re.compile(r"\bI\s?(?:'|’)?(?:ll|will| am going to)?\s*"
                   r"(?:keep|leave|stay out of|stay off|stay clear of|avoid|won(?:'|’)?t use|won(?:'|’)?t enter|"
                   r"won(?:'|’)?t touch|won(?:'|’)?t occupy|won(?:'|’)?t block|won(?:'|’)?t need|am not using)\b"
                   r"([^.;!?\n]{0,70})", re.I)
FREE=re.compile(r"\b(clear|free|open|empty|available|unblocked|vacant)\b|\bavoid\b|won(?:'|’)?t (use|enter|touch|occupy|block|need)|stay (out|off|clear)", re.I)
R=json.load(open('episodes.json'))
stats=collections.Counter(); broken=[]
for r in R:
    if r['suite']=='solo': continue
    d=os.path.join(REPO,r['path'])
    ins=json.load(open(os.path.join(d,'instance.json'))); it=json.load(open(os.path.join(d,'interactions.json')))
    groups=ins.get('player_objects') or [ins['player_a_objects'],ins['player_b_objects']]
    owner={o:string.ascii_uppercase[i] for i,g in enumerate(groups) for o in g}
    tgt={o:tuple(p) for o,p in ins['target_positions'].items()}
    mytgt=collections.defaultdict(set)
    for o,t in tgt.items(): mytgt[owner[o]].add(t)
    pos={o:tuple(p) for o,p in ins['start_positions'].items()}
    events=[]
    for t in it['turns']:
        for e in t:
            a=e.get('action',{}); ty=a.get('type'); c=a.get('content','')
            if ty=='move':
                m=MOVE.match(c)
                if m:
                    o,dr=m.group(2),m.group(3); pos[o]=(pos[o][0]+DIRS[dr][0],pos[o][1]+DIRS[dr][1])
                    events.append(('move',owner[o],o,pos[o]))
            elif ty=='invalid_move':
                m=BAD.match(c)
                if m: events.append(('blocked',m.group(1),m.group(2),(int(m.group(4))-1,int(m.group(5))-1)))
            elif ty=='message':
                spk=c.split(':')[0].replace('Player','').strip()
                events.append(('msg',spk,None,c.split(':',1)[1] if ':' in c else ''))
    for i,(k,who,_,payload) in enumerate(events):
        if k!='msg': continue
        for m in PROMISE.finditer(payload):
            seg=m.group(0)
            if not FREE.search(seg): continue
            if re.search(r"\b(until|unless|for now|temporar|this turn|next turn|once)\b",seg,re.I): continue
            for cell in {(int(a)-1,int(b)-1) for a,b in CELL.findall(m.group(1))}-mytgt[who]:
                later=[(j,o) for j,(k2,w2,o,p) in enumerate(events[i+1:],start=i+1) if k2=='move' and w2==who and p==cell]
                if not later: stats[('kept',r['outcome'])]+=1; continue
                j,o=later[0]
                cons=[t for t in events[j+1:] if t[0]=='blocked' and t[1]!=who and t[3]==cell]
                is_tgt=[x for x,t in tgt.items() if t==cell and owner[x]!=who]
                stats[('broken',r['outcome'])]+=1
                stats[('consequential' if (cons or is_tgt) else 'harmless', r['outcome'])]+=1
                broken.append((r['model'],r['suite'],r['level'],r['proto'],r['inst'],r['outcome'],who,o,
                               f"R{cell[0]+1}C{cell[1]+1}",seg.strip()[:140],j-i,bool(cons),is_tgt,r['path']))
json.dump(broken,open('broken_promises.json','w'))
T=collections.Counter()
for (st,oc),v in stats.items(): T[st]+=v
n=T['kept']+T['broken']
print(f"strict commitments to keep a named cell free (not the speaker's own target): {n}")
print(f"  kept {T['kept']} ({100*T['kept']/n:.0f}%)   broken {T['broken']} ({100*T['broken']/n:.0f}%)")
print(f"  of the broken ones: {T['consequential']} land on a teammate's target or a cell a teammate is then rejected from; {T['harmless']} are harmless")
print()
print("consequential violations:")
for rec in [x for x in broken if x[11] or x[12]][:12]:
    print(f"  {rec[0]:9s} {rec[1]:4s} {rec[2]} {rec[3]:10s} {rec[4]} [{rec[5]}] {rec[6]} promised {rec[8]} free, then moved {rec[7]} onto it"
          + (f"; it is {','.join(rec[12])}'s target" if rec[12] else "; a teammate was then rejected from it"))
    print(f"      \"{rec[9]}\"")
