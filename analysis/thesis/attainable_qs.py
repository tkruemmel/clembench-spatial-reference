"""Thesis Quality Score claims re-run on the per-suite attainable optimum.

Every number is printed twice, J | A: on the free-order joint optimum (the
thesis's denominator; reproduces S9 and S11 exactly) and on the attainable
optimum of the episode's own suite (analysis/efficiency_baseline/attainable_*.jsonl).
Exploratory; nothing here enters the confirmatory ladder.

    python3 analysis/thesis/attainable_qs.py
"""
import os
import sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attainable as at
from tables_comm import perm_p
A, J = at.optima(); rows = at.episodes(A, J)
def q(r, d):
    if r["aborted"]: return None
    return 100*min(1, r[d]/r["moves"]) if r["success"] else 0.0
idx = {(r["suite"], r["model"], r["level"], r["gid"], r["proto"]): r for r in rows}
M = list(at.ARMS)
def mean_qs(suite, level=None, model=None, proto=None, d="J"):
    v=[q(r,d) for r in rows if r["suite"]==suite and (level is None or r["level"]==level)
       and (model is None or r["model"]==model) and (proto is None or r["proto"]==proto)]
    v=[x for x in v if x is not None]; return st.mean(v)
def paired(a, b, level=None, model=None, d="J", protos=None):
    ds=[]
    for k, r in idx.items():
        if k[0]!=a or (level and k[2]!=level) or (model and k[1]!=model): continue
        if protos and k[4] not in protos: continue
        y = idx.get((b,)+k[1:])
        if y is None: continue
        qa, qb = q(r,d), q(y,d)
        if qa is None or qb is None: continue
        ds.append(qa-qb)
    return st.mean(ds), perm_p(ds), len(ds)

print("1. QS per suite, pooled strong tier (J | A)")
for s in at.SUITES:
    print(f"  {s:5}", "  ".join(f"{l}: {mean_qs(s,l):6.2f}|{mean_qs(s,l,d='A'):6.2f}" for l in at.LEVELS),
          f"  all {mean_qs(s):6.2f}|{mean_qs(s,d='A'):6.2f}")
print("\n2. Team-size steps, paired over (level, protocol, board); larger-minus-smaller team as 'a-b' = smaller minus larger (cost)")
for a,b in (("dyad","3p"),("3p","6p"),("dyad","6p")):
    for m in M:
        for d in "JA":
            x=paired(a,b,model=m,d=d); print(f"  {a}->{b} {m[:8]:8} {d}: cost {x[0]:+6.2f} p={x[1]:.4f} n={x[2]}")
print("\n3. dyad->6p cost per level, pooled models")
for l in at.LEVELS:
    for d in "JA":
        x=paired("dyad","6p",level=l,d=d); print(f"  {l} {d}: {x[0]:+6.2f} p={x[1]:.4f}")
print("\n4. Step costs by level (pooled)")
for a,b in (("dyad","3p"),("3p","6p")):
    for d in "JA":
        print(f"  {a}->{b} {d}:", "  ".join(f"{l} {paired(a,b,level=l,d=d)[0]:+6.2f}" for l in at.LEVELS))
print("\n5. solo vs dyad-none, unpaired means")
for d in "JA":
    print(f"  {d}: solo {mean_qs('solo',d=d):.2f} dyad-none {mean_qs('dyad',proto='none',d=d):.2f}")
print("\n6. 6p level ordering (J | A):", [(l, round(mean_qs('6p',l),2), round(mean_qs('6p',l,d='A'),2)) for l in at.LEVELS])
print("   3p level ordering:", [(l, round(mean_qs('3p',l),2), round(mean_qs('3p',l,d='A'),2)) for l in at.LEVELS])
print("\n7. Protocol QS per suite (J | A)")
for s in ("dyad","3p","6p"):
    print(f"  {s}:", "  ".join(f"{p} {mean_qs(s,proto=p):.2f}|{mean_qs(s,proto=p,d='A'):.2f}" for p in ("freeform","hybrid","structured","none")))
print("\n8. Model gap (DeepSeek - Qwen), paired over boards both played, all suites")
for d in "JA":
    ds=[]
    for k,r in idx.items():
        if k[1]!=M[1]: continue
        y=idx.get((k[0],M[0])+k[2:])
        if y and q(r,d) is not None and q(y,d) is not None: ds.append(q(r,d)-q(y,d))
    print(f"  {d}: {st.mean(ds):+.2f} p={perm_p(ds):.4f} n={len(ds)}")
print("\n9. A2-like: solo-silent vs dyad-none efficiency, paired boards solved in both, S2-S4")
for d in "JA":
    ds=[]
    for k,r in idx.items():
        if k[0]!="dyad" or k[4]!="none" or k[2]=="S1" or not r["success"]: continue
        y=idx.get(("solo",k[1],k[2],k[3],"silent"))
        if y and y["success"]: ds.append(y[d]/y["moves"]-r[d]/r["moves"])
    print(f"  {d}: {st.mean(ds):+.4f} p={perm_p(ds):.5f} n={len(ds)}")
