import sys, os, collections
ROOT = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, os.path.join(ROOT, "analysis/thesis"))
import tables_comm as tc
eps = tc.all_episodes(tc.load_optima())
c = collections.defaultdict(lambda: [0, 0])
for e in eps:
    k = (e["model"], e["suite"], e["level"], e["cond"]); c[k][0] += e["success"]; c[k][1] += 1
nem = {k[1:]: v for k, v in c.items() if k[0] == "Nemotron"}
strong_min = {}
for k, v in c.items():
    if k[0] != "Nemotron":
        strong_min[k[1:]] = min(strong_min.get(k[1:], 99), v[0])
print("Nemotron best cells:", sorted(((v[0], k) for k, v in nem.items()), reverse=True)[:6])
print("strong worst cells:", sorted((v, k) for k, v in strong_min.items())[:6])
print("cells where Nemotron >= strong min:", [(k, nem[k][0], strong_min[k]) for k in nem if nem[k][0] >= strong_min[k]])
