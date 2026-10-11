import sys, os, json, glob, re, collections, statistics as st
ROOT = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, os.path.join(ROOT, "analysis/thesis"))
import tables_comm as tc
from scipy.stats import binomtest

opt = tc.load_optima()
eps = tc.all_episodes(opt)
Q, D, N = "Qwen3.8", "DeepSeek-V4.1-Flash", "Nemotron"
for m in (Q, D, N):
    r = [e for e in eps if e["model"] == m]
    print(m, "n", len(r), "succ", sum(e["success"] for e in r), "abort", sum(e["ab"] for e in r))

# pairing Qwen vs DeepSeek
key = lambda e: (e["suite"], e["level"], e["cond"], e["game_id"])
iq = {key(e): e for e in eps if e["model"] == Q}
idd = {key(e): e for e in eps if e["model"] == D}
ks = sorted(set(iq) & set(idd))
print("paired", len(ks))
both = sum(iq[k]["success"] and idd[k]["success"] for k in ks)
neither = sum((not iq[k]["success"]) and (not idd[k]["success"]) for k in ks)
q_only = sum(iq[k]["success"] and not idd[k]["success"] for k in ks)
d_only = sum(idd[k]["success"] and not iq[k]["success"] for k in ks)
print("both", both, "neither", neither, "Q only", q_only, "D only", d_only,
      "McNemar", binomtest(q_only, q_only + d_only, .5).pvalue)
n = len(ks); po = (both + neither) / n
pq = sum(iq[k]["success"] for k in ks) / n; pd = sum(idd[k]["success"] for k in ks) / n
pe = pq * pd + (1 - pq) * (1 - pd); print("kappa", (po - pe) / (1 - pe))

# board level: (level, gid), 14 episodes each per model
bq, bd = collections.Counter(), collections.Counter()
for k in ks:
    b = (k[1], k[3]); bq[b] += iq[k]["success"]; bd[b] += idd[k]["success"]
boards = sorted(set(b for b in [(k[1], k[3]) for k in ks]))
disc_boards = set((k[1], k[3]) for k in ks if iq[k]["success"] != idd[k]["success"])
fail_boards = set((k[1], k[3]) for k in ks if not (iq[k]["success"] and idd[k]["success"]))
print("boards", len(boards), "with >=1 discordant episode", len(disc_boards),
      "with any non-success in either", len(fail_boards))
print("per-board success Q/D (non-identical boards):")
for b in boards:
    if bq[b] != bd[b]:
        print("  ", b, bq[b], bd[b])
import numpy as np
x = np.array([bq[b] for b in boards]); y = np.array([bd[b] for b in boards])
print("per-board success corr r =", np.corrcoef(x, y)[0, 1])
# boards that both models solve in all 14
print("boards perfect in both", sum(1 for b in boards if bq[b] == 14 and bd[b] == 14))

# Quality decomposition, pairs both played
pp = [(iq[k], idd[k]) for k in ks if iq[k]["q"] is not None and idd[k]["q"] is not None]
dq = [b["q"] - a["q"] for a, b in pp]  # D - Q
print("QS pairs", len(pp), "mean D-Q", st.mean(dq))
cat = collections.defaultdict(list)
for a, b in pp:
    c = ("both" if a["success"] and b["success"] else "neither" if not a["success"] and not b["success"]
         else "Donly" if b["success"] else "Qonly")
    cat[c].append(b["q"] - a["q"])
for c, v in cat.items():
    print(f"  {c:8} n={len(v):4} sum={sum(v):9.1f} contrib to mean={sum(v)/len(pp):+.3f}")

print("\n-- efficiency-only (both succeed) D-Q quality diff:")
d_both = [b["q"] - a["q"] for a, b in pp if a["success"] and b["success"]]
print("  mean", st.mean(d_both), "p", tc.perm_p(d_both))
print(f"  all {len(dq)} pairs p", tc.perm_p(dq))
# per suite: the Quality lead with its paired sign-flip p (tables_comm.perm_p,
# 100,000 resamples, seed 0), on A_k (q) and on the free-order optimum J (q_j)
for fld in ("q", "q_j"):
    for s in ("solo", "dyad", "3p", "6p", None):
        d = [idd[k][fld] - iq[k][fld] for k in ks if s in (None, k[0])
             and iq[k][fld] is not None and idd[k][fld] is not None]
        print(f"  {fld} lead D-Q {s or 'all'}: pairs {len(d)} mean {st.mean(d):+.2f} p={tc.perm_p(d):.3g}", flush=True)
# per suite decomposition
for s in ("solo", "dyad", "3p", "6p"):
    sp = [(a, b) for a, b in pp if a["suite"] == s]
    tot = st.mean([b["q"] - a["q"] for a, b in sp])
    eff = sum(b["q"] - a["q"] for a, b in sp if a["success"] and b["success"]) / len(sp)
    print(f"  {s}: pairs {len(sp)} D-Q {tot:+.2f}  of which both-succeed eff {eff:+.2f}  success-discordant {tot-eff:+.2f}")
# discordant episodes involving aborts
dis = [k for k in ks if iq[k]["success"] != idd[k]["success"]]
print("discordant", len(dis), "involving an abort", sum(1 for k in dis if iq[k]["ab"] or idd[k]["ab"]))
print("both-fail by suite", collections.Counter(k[0] for k in ks if not iq[k]["success"] and not idd[k]["success"]))
print("discordant by suite", collections.Counter(k[0] for k in dis))
