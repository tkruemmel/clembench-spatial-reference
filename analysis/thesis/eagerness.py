"""D-f: is the strong tier's cycle loss eagerness to finish?

Replaces the terminal-count comparison of §9.3.2, which compared two stopping
rules (a success ends when the board is solved, a cycle loss when the detector
fires).  The statistic here is the number of players that have declared
`done` by move k = the board's per-object BFS bound.  No episode can succeed before
its optimum and every cycle loss runs past it, so at k both groups are still
running in full and the stopping rule cannot reach the count.

Exploratory: not one of tab:15's nineteen confirmatory contrasts.
Two-sided unpaired permutation test on the difference of means, the same
helper the across-level contrasts use.

    python analysis/thesis/eagerness.py
"""
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from confirmatory import two_sample_p                        # noqa: E402
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

EPISODES = os.path.join(HERE, "..", "qualitative_analysis", f"episodes.{ARM}.json")
MODELS = (("qwen38", "Qwen3.8"), ("dsv41", "DeepSeek"))
CYCLE = "Board state repeated"


# k is the per-object BFS bound the original instances ship.  The updated
# arm stores the suite's attainable optimum in `optimal` instead, and a cycle
# loss can end before that, so k is read from the board, not the episode.
BFS = {(gi["spatial_level"], gi["game_id"]): gi["optimal_moves"]
       for ex in json.load(open(os.path.join(HERE, "..", "..", "matrixgame_covered_comm_grid",
                                             "in", "instances.json")))["experiments"]
       for gi in ex["game_instances"]}


def done_by_optimal(e):
    k = BFS[(e["level"], int(e["inst"].split("_")[-1]))]
    assert e["moves"] >= k, e["path"]           # still running at k
    return len({txt for mi, txt in e["done_decl"] if mi <= k})


def groups(E, model, suite):
    g = [e for e in E if e["model"] == model and e["suite"] == suite]
    win = [done_by_optimal(e) for e in g if e["outcome"] == "success"]
    cyc = [done_by_optimal(e) for e in g
           if e["outcome"] == "lose" and (e["end_msg"] or "").startswith(CYCLE)]
    return win, cyc


def desc(x):
    q1, q3 = np.percentile(x, [25, 75])
    return f"{st.mean(x):.2f} (SD {st.stdev(x):.2f}, IQR {q1:g}-{q3:g})"


def main():
    E = json.load(open(EPISODES))
    for suite in ("3p", "6p"):
        for m, name in MODELS:
            win, cyc = groups(E, m, suite)
            print(f"{suite} {name:8s} n={len(win)}/{len(cyc)}  "
                  f"success {desc(win)}  cycle loss {desc(cyc)}  "
                  f"diff {st.mean(win) - st.mean(cyc):+.2f}  "
                  f"p={two_sample_p(win, cyc):.4f}")


if __name__ == "__main__":
    main()
