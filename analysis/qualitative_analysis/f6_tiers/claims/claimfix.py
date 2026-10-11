import re, sys, collections
sys.path.insert(0, "/Users/tom/uni/clembench-spatial-reference/analysis/qualitative_analysis")
import qual_corpus as q
def tally(rows):
    t = collections.defaultdict(lambda: [0,0,0,0])
    for r in rows:
        if r["proto"] not in q.PROSE: continue
        k = (r["model"], r["proto"])
        t[k][0]+=r["own_ok"]; t[k][1]+=r["own_bad"]; t[k][2]+=r["oth_ok"]; t[k][3]+=r["oth_bad"]
    return t
rows,_ = q.extract("/Users/tom/uni/clembench-spatial-reference")
a = tally(rows)
q.CLAIM = re.compile(r"\b(?-i:([A-F]))\b[^.;,\n]{0,24}?\bat\s*\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?", re.I)
rows,_ = q.extract("/Users/tom/uni/clembench-spatial-reference")
b = tally(rows)
for k in sorted(a):
    for lab,t in (("orig",a[k]),("fix",b[k])):
        oo,ob,to,tb=t
        print(k,lab,"own %d/%d=%.1f%%"%(oo,oo+ob,100*oo/max(1,oo+ob)),"team %d/%d=%.1f%%"%(to,to+tb,100*to/max(1,to+tb)))
