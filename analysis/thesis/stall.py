"""§9.3.4: the stall rate, recomputed from the logged move sequences.

Remaining distance D_t is the sum, over objects off target after move t, of
the shortest-path distance to the target through the wall layout.  A window of
k consecutive moves is stalled when D at its end is no lower than at its
start; the stall rate is the share of an episode's sliding windows that are
stalled.  k = max(2n, 6) natively; the fixed-k tables hold k at 6 and 12 moves
for every team size.  Episodes shorter than k have no window and are dropped.

Exploratory and descriptive: not one of tab:15's nineteen confirmatory
contrasts.  Two strong models only.

    ARM=updated python3 analysis/thesis/stall.py          # report
    ARM=updated python3 analysis/thesis/stall.py --tex    # also thesis/numbers/E_tab_stall_level.tex
"""
import json
import os
import re
import statistics as st
import sys

from scipy.stats import kruskal, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
EPISODES = os.path.join(REPO, "analysis/qualitative_analysis", f"episodes.{ARM}.json")
STRONG = ("qwen38", "dsv41")
SUITES = ("solo", "dyad", "3p", "6p")
LEVELS = ("S1", "S2", "S3", "S4")
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def bfs(n, walls, src):
    d = {src: 0}
    q = [src]
    for c in q:
        for dr, dc in DIRS.values():
            x = (c[0] + dr, c[1] + dc)
            if 0 <= x[0] < n and 0 <= x[1] < n and x not in walls and x not in d:
                d[x] = d[c] + 1
                q.append(x)
    return d


def stall_rate(D, k):
    T = len(D) - 1
    if T < k:
        return None
    return sum(D[t + k] >= D[t] for t in range(T - k + 1)) / (T - k + 1)


def replay(e):
    """Distance series D_0..D_T plus message count and characters."""
    d = os.path.join(REPO, e["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    n = ins["grid_size"]
    n = n[0] if isinstance(n, list) else n
    walls = {tuple(w) for w in ins["walls"]}
    pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
    tgt = {o: tuple(q) for o, q in ins["target_positions"].items()}
    dist = {o: bfs(n, walls, t) for o, t in tgt.items()}
    D = [sum(dist[o][pos[o]] for o in tgt if pos[o] != tgt[o])]
    msgs = chars = 0
    for turn in it["turns"]:
        for ev in turn:
            a = ev.get("action", {})
            ty, c = a.get("type"), a.get("content", "")
            if ty == "move" and (m := MOVE.match(c)):
                o, dr = m.group(2), m.group(3)
                pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
                D.append(sum(dist[o][pos[o]] for o in tgt if pos[o] != tgt[o]))
            elif ty == "message":
                msgs += 1
                chars += len(c)
    return D, msgs, chars


# "Moves beyond the optimum" is beyond the suite's attainable optimum A_k, the
# denominator every score uses.  `optimal` in an episode is the BFS bound in the
# original arm and A_k in the updated one, so it is not read from there.
AK = {(s, d["level"], d["game_id"]): d["stuckfree"]
      for s in ("dyad", "3p", "6p")
      for d in map(json.loads, open(os.path.join(REPO, f"analysis/efficiency_baseline/attainable_{s}.jsonl")))
      if d.get("status") == "exact"}
AK.update({("solo", d["level"], d["game_id"]): d["opt"]       # solo's A is J
           for d in map(json.loads, open(os.path.join(REPO, "analysis/efficiency_baseline/joint_optimum.jsonl")))
           if d.get("status") == "exact"})


def rows(models=STRONG):
    out = []
    for e in json.load(open(EPISODES)):
        if e["model"] not in models:
            continue
        D, msgs, chars = replay(e)
        out.append(dict(
            model=e["model"], suite=e["suite"], level=e["level"], proto=e["proto"],
            outcome=e["outcome"], gid=int(e["inst"].split("_")[-1]), T=len(D) - 1,
            msgs=msgs, chars=chars,
            native=stall_rate(D, max(2 * e["nplayers"], 6)),
            k6=stall_rate(D, 6), k12=stall_rate(D, 12)))
    return out


def grid(R, key, outcome=None):
    print(f"-- mean {key}" + (f", {outcome}" if outcome else ", all outcomes"))
    print("    " + "  ".join(f"{s:>5s}" for s in SUITES))
    for lv in LEVELS:
        cells = []
        for s in SUITES:
            x = [r[key] for r in R if r["level"] == lv and r["suite"] == s
                 and (outcome is None or r["outcome"] == outcome) and r[key] is not None]
            cells.append(f"{st.mean(x):.3f}" if x else "    -")
        print(f"{lv}  " + "  ".join(cells))


def tex(R):
    """tab:stall-level, generated: mean native stall rate on successes, strong models pooled."""
    from tables_comm import table
    body = []
    for lv in LEVELS:
        cells = []
        for s in SUITES:
            x = [r["native"] for r in R if r["level"] == lv and r["suite"] == s
                 and r["outcome"] == "success" and r["native"] is not None]
            cells.append(f"{st.mean(x):.3f}")
        body.append(f"{lv} & " + " & ".join(cells))
    n = {s: sum(1 for r in R if r["suite"] == s and r["outcome"] == "success" and r["native"] is not None)
         for s in SUITES}
    return table(
        "Mean stall rate on successful episodes by level and suite, Qwen3.8 and "
        "DeepSeek pooled, native window $k = \\max(2n, 6)$. Exploratory.",
        "tab:stall-level", "lcccc", " & solo & dyad & 3p & 6p", body,
        note="Successful episodes with at least one window: "
             + ", ".join(f"{s} {n[s]}" for s in SUITES) + ".")


def main():
    R = rows()
    if "--tex" in sys.argv:
        out = os.path.join(REPO, "thesis/numbers/E_tab_stall_level.tex")
        open(out, "w").write(tex(R))
        print("wrote", out)
    E = json.load(open(EPISODES))
    w = [e["moves"] >= max(2 * e["nplayers"], 6) for e in E if e["model"] == "nemotron"]
    print(f"Nemotron episodes with a native window: {sum(w)} / {len(w)}")
    print("-- medians, native k: success vs loss (n)")
    for s in SUITES:
        w = [r["native"] for r in R if r["suite"] == s and r["outcome"] == "success" and r["native"] is not None]
        lo = [r["native"] for r in R if r["suite"] == s and r["outcome"] == "lose" and r["native"] is not None]
        print(f"{s:5s} {st.median(w):.3f} vs {st.median(lo):.3f}  ({len(w)} / {len(lo)})")
    ab = [r["native"] for r in R if r["suite"] == "6p" and r["outcome"] == "abort" and r["native"] is not None]
    print(f"6p aborts median {st.median(ab):.3f} (n = {len(ab)})")
    print("-- means, native k: success vs loss (ratio)")
    for s in SUITES:
        w = st.mean(r["native"] for r in R if r["suite"] == s and r["outcome"] == "success" and r["native"] is not None)
        lo = st.mean(r["native"] for r in R if r["suite"] == s and r["outcome"] == "lose" and r["native"] is not None)
        print(f"{s:5s} {w:.3f} vs {lo:.3f}  ({lo / w:.1f}x)")
    print("-- Spearman stall vs efficiency A_k / moves, successes, native k")
    for s in SUITES:
        g = [r for r in R if r["suite"] == s and r["outcome"] == "success" and r["native"] is not None]
        a = spearmanr([r["native"] for r in g], [AK[(s, r["level"], r["gid"])] / r["T"] for r in g])
        print(f"{s:5s} rho {a.statistic:+.2f} (n = {len(g)})")

    grid(R, "native", "success")
    print("-- Kruskal-Wallis across levels, successes, native k")
    for s in SUITES:
        g = [[r["native"] for r in R if r["suite"] == s and r["level"] == lv
              and r["outcome"] == "success" and r["native"] is not None] for lv in LEVELS]
        print(f"{s:5s} p = {kruskal(*g).pvalue:.2g}")
    grid(R, "k6")
    grid(R, "k12")

    C = [r for r in R if r["suite"] != "solo" and r["proto"] != "none"
         and r["outcome"] == "success" and r["level"] != "S1" and r["native"] is not None]
    print("-- Spearman within suite x level (successes, S2-S4, protocols with messages)")
    for s in SUITES[1:]:
        for lv in LEVELS[1:]:
            g = [r for r in C if r["suite"] == s and r["level"] == lv]
            a = spearmanr([r["native"] for r in g], [r["msgs"] / r["T"] for r in g])
            b = spearmanr([r["native"] for r in g], [r["chars"] / r["T"] for r in g])
            print(f"{s:4s} {lv} n={len(g):3d}  msgs/move {a.statistic:+.2f} (p={a.pvalue:.3f})"
                  f"  chars/move {b.statistic:+.2f} (p={b.pvalue:.3f})")
    print("-- length confound: moves beyond optimum")
    for s in SUITES[1:]:
        g = [r for r in C if r["suite"] == s]
        ex = [r["T"] - AK[(s, r["level"], r["gid"])] for r in g]
        print(f"{s:4s} msgs/move {spearmanr([r['msgs'] / r['T'] for r in g], ex).statistic:+.2f}"
              f"  stall {spearmanr([r['native'] for r in g], ex).statistic:+.2f}")

    print("-- protocol means, native k, all outcomes")
    for s in SUITES[1:]:
        ps = sorted({r["proto"] for r in R if r["suite"] == s})
        print(s, {p: round(st.mean([r["native"] for r in R if r["suite"] == s and r["proto"] == p
                                     and r["native"] is not None]), 3) for p in ps})


if __name__ == "__main__":
    assert stall_rate([4, 3, 2], 2) == 0.0          # steady progress
    assert stall_rate([4, 5, 4, 3], 2) == 0.5       # detour then recovery
    assert stall_rate([4, 3], 2) is None            # shorter than the window
    main()
