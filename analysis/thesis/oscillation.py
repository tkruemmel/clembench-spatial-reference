"""Per-move oscillation rate by team size (tab:oscillation, eq:oscillation).

A move oscillates when it returns its object to a cell the object occupied
before one of its own last k moves (k = 4).  The window is indexed in the
object's own move sequence, not in the joint move log, so it does not go blind
as round-robin spreads an object's moves n entries apart.  The rate is pooled:
oscillating moves over all moves of the model's episodes in that suite, every
outcome included.  Only logged (legal) moves count.

Exploratory: not one of the nineteen confirmatory contrasts.

    ARM=updated python3 analysis/thesis/oscillation.py           # table
    ARM=updated python3 analysis/thesis/oscillation.py --check   # pinned values
"""
import json
import os
import re
import sys

ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EPISODES = os.path.join(REPO, "analysis/qualitative_analysis", f"episodes.{ARM}.json")
MODELS = (("qwen38", "Qwen"), ("dsv41", "DeepSeek"), ("nemotron", "Nemotron"))
SUITES = ("solo", "dyad", "3p", "6p")
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
K = 4

# Pooled rates, solo/dyad/3p/6p.  The original arm's text values (.057/.090/
# .121/.140, .056/.087/.104/.126, .202/.159/.117/.146) predate the S2 #10
# replacement; the replaced board moves them by at most .002.
CHECK = {
    "original": {"qwen38": (.056, .089, .122, .139), "dsv41": (.055, .086, .105, .125),
                 "nemotron": (.203, .157, .116, .145)},
    "updated": {"qwen38": (.056, .086, .113, .144), "dsv41": (.055, .081, .112, .127),
                "nemotron": (.203, .131, .149, .129)},
}


def osc_moves(cells, k=K):
    """Oscillating moves in one object's cell sequence c_0, c_1, ... (eq:oscillation)."""
    return sum(cells[j + 1] in cells[max(0, j - k):j] for j in range(len(cells) - 1))


def episode(path):
    d = os.path.join(REPO, path)
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(os.path.join(d, "interactions.json")))
    seq = {o: [tuple(p)] for o, p in ins["start_positions"].items()}
    for turn in it["turns"]:
        for ev in turn:
            a = ev.get("action", {})
            if a.get("type") == "move" and (m := MOVE.match(a.get("content", ""))):
                o, (dr, dc) = m.group(2), DIRS[m.group(3)]
                seq[o].append((seq[o][-1][0] + dr, seq[o][-1][1] + dc))
    return sum(osc_moves(c) for c in seq.values()), sum(len(c) - 1 for c in seq.values())


def rates():
    E = json.load(open(EPISODES))
    out = {}
    for m, _ in MODELS:
        row = []
        for s in SUITES:
            r = [episode(e["path"]) for e in E if e["model"] == m and e["suite"] == s]
            row.append(sum(h for h, _ in r) / sum(n for _, n in r))
        out[m] = row
    return out


def main():
    # eq:oscillation: i ranges over max(0, j-k) .. j-1, the cells before the
    # object's last k moves, and the move j -> j+1 lands on one of them.
    assert osc_moves([(0, 0), (0, 1), (0, 0)]) == 1          # immediate reversal
    assert osc_moves([(0, 0), (0, 1), (0, 2), (0, 3)]) == 0  # straight run
    assert osc_moves([(0, 0)] + [(0, i) for i in range(1, 6)] + [(1, 5), (1, 4), (1, 3),
                                                                (1, 2), (1, 1), (0, 1)]) == 0
    R = rates()
    print(f"oscillation rate, pooled over moves, k = {K}, ARM={ARM}")
    print(f"{'model':10s}" + "".join(f"{s:>8s}" for s in SUITES))
    for m, name in MODELS:
        print(f"{name:10s}" + "".join(f"{x:8.3f}" for x in R[m]))
    if "--check" in sys.argv:
        for m, want in CHECK[ARM].items():
            assert all(abs(x - w) < .0015 for x, w in zip(R[m], want)), (m, R[m], want)
        # The text's claim: monotone in team size in both strong models, 6p the maximum.
        for m in ("qwen38", "dsv41"):
            assert all(a < b for a, b in zip(R[m], R[m][1:])), m
        # ... and the weak model oscillates most alone.
        assert R["nemotron"][0] == max(R["nemotron"])
        print("check ok")


if __name__ == "__main__":
    main()
