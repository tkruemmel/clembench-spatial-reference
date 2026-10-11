"""Task 1: content of teammate references, rates per 1000 messages and per episode."""
import sys
sys.dont_write_bytecode = True
import collections, random
from verbal import load

S, P, E = load()
CL = ["goalcell_mate", "nopass", "goal_mate", "route_mate", "req", "mob_mate", "stuck_mate", "mob_self", "stuck_self", "mob_none",
      "hold_self", "hold_self_place", "hold_other"]


def table(tier, suite):
    ms = [m for m in S if m["tier"] == tier and m["suite"] == suite]
    eps = [e for e in E if e["tier"] == tier and e["suite"] == suite]
    per_ep = collections.defaultdict(collections.Counter)
    for m in ms:
        for k in CL:
            per_ep[m["path"]][k] += m["tags"].get(k, False)
    row = [f"{tier:6} {suite:4} msgs={len(ms):6} eps={len(eps):4}"]
    for k in CL:
        n = sum(m["tags"].get(k, False) for m in ms)
        epw = sum(1 for e in eps if per_ep[e["path"]][k] > 0)
        row.append(f"{k}: {n} ({1000 * n / max(len(ms), 1):.1f}/1k, {n / max(len(eps), 1):.2f}/ep, {epw}/{len(eps)} eps)")
    return row


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        k, tier, nn = sys.argv[2], sys.argv[3], int(sys.argv[4])
        suite = sys.argv[5] if len(sys.argv) > 5 else None
        pool = [(m, s) for m in S if m["tier"] == tier and (suite is None or m["suite"] == suite)
                for s in m["hits"].get(k, [])]
        random.seed(7)
        print(len(pool))
        for i, (m, s) in enumerate(random.sample(pool, min(nn, len(pool)))):
            print(f"[{i}] {m['model']} {m['suite']} {m['lvl']} {m['proto']} {m['inst']} ply{m['ply']} P{m['spk']}: {s[:260]}")
    else:
        for tier in ("strong", "weak"):
            for suite in ("6p", "3p", "dyad"):
                print("\n  ".join(table(tier, suite)))
