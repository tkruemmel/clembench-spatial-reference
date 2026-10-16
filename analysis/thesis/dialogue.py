"""tab:dialogue on the episode, with board state held fixed.

The old tab:dialogue compared messages from successful and failed episodes.
That has two faults.  The unit was the message (tens of thousands of messages
nested in a few hundred episodes), and the comparison conditions on outcome,
which board state also drives: failures sit on harder boards and spend longer
with a teammate in the way, and that state by itself elicits requests, talk
about teammates' objects and "blocked" reports.  This script fixes both.

Design
------
scope    the two strong models, prose protocols (freeform, hybrid), S2-S4, all
         three team suites: tab:dialogue's scope (tables_comm.messages()).
unit     the episode: per episode, the share of its messages that carry a
         feature (mean words / coordinates per message for the two means).
state    each message is classed by the speaker's board state when it was
         written (the 'get message' snapshot): *blocked* if a teammate's object
         lengthens or cuts the path of one of the speaker's unplaced objects to
         its target, else *unblocked*; the same definition as
         analysis/qualitative_analysis/request_timing.py.  An episode
         contributes to a state class only if it has a message in it.
strata   model x suite x level x protocol.  Within a stratum, success minus
         failure (any non-success, aborts included) of the episode means; the
         strata are combined with weights n_s n_f / (n_s + n_f), the
         Mantel-Haenszel weights, and strata lacking either outcome drop out.
test     the outcome labels are permuted within strata, B = 10,000 at seed 0;
         two-sided p = (1 + #{|T*| >= |T|}) / (B + 1).  Exploratory, and
         uncorrected across the 27 rows x classes.

    ARM=updated python3 analysis/thesis/dialogue.py          # report
    ARM=updated python3 analysis/thesis/dialogue.py --tex    # also thesis/numbers/E_tab_dialogue.tex
"""
import collections
import glob
import json
import os
import string
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis/efficiency_baseline"))
import tables_comm as tc                                           # noqa: E402
from instance_audit import bfs_dist                                # noqa: E402
from qual_corpus import ARMS, DIRS, MOVE, PROSE, SPK, STRONG, tree  # noqa: E402

B = 10_000
CLASSES = ("all", "unblocked", "blocked")
FEATS = tc.DIALOGUE + [(label, k) for label, k, _ in tc.MEANS]


def blocked_flags():
    """{(model, suite, level, proto, gid): [blocked per message]} by board replay."""
    out = {}
    for model in STRONG:
        for suite in tc.SUITES:
            pat = os.path.join(ROOT, tree(ARMS[model], suite), "*", "*", "*", "instance_*", "instance.json")
            for f in sorted(glob.glob(pat)):
                ins = json.load(open(f))
                if ins["comm_protocol"] not in PROSE or ins["spatial_level"] == "S1":
                    continue
                it = json.load(open(os.path.join(os.path.dirname(f), "interactions.json")))
                n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
                pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
                tgt = {o: tuple(q) for o, q in ins["target_positions"].items()}
                groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
                owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
                snap, flags = dict(pos), []
                for t in it["turns"]:
                    for e in t:
                        a = e.get("action", {})
                        ty, c = a.get("type"), str(a.get("content", ""))
                        if ty == "get message":
                            snap = dict(pos)
                        elif ty == "move" and (m := MOVE.match(c)):
                            _, o, dr = m.groups()
                            pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
                        elif ty == "message" and (m := SPK.match(c)):
                            spk, b = m.group(1), False
                            for o in (o for o in tgt if owner[o] == spk and snap[o] != tgt[o]):
                                own = walls | {snap[x] for x in tgt if owner[x] == spk and x != o}
                                d0 = bfs_dist(snap[o], tgt[o], n, own)
                                d1 = bfs_dist(snap[o], tgt[o], n, own | {snap[x] for x in tgt if owner[x] != spk})
                                b = b or (d0 is not None and (d1 is None or d1 > d0))
                            flags.append(b)
                key = (model, suite, ins["spatial_level"], ins["comm_protocol"], ins["game_id"])
                assert key not in out, key
                out[key] = flags
    return out


def episodes():
    """One dict per episode: stratum, outcome, and per class the feature means."""
    msgs = collections.defaultdict(list)
    for r in tc.messages():
        msgs[(r["model"], r["suite"], r["lvl"], r["proto"], r["gid"])].append(r)
    flags = blocked_flags()
    out = []
    for k, rs in msgs.items():
        rs.sort(key=lambda r: r["midx"])
        assert len(rs) == len(flags[k]), k
        ep = dict(stratum=k[:4], su=rs[0]["su"], n={})
        for cl in CLASSES:
            sel = [r for r, b in zip(rs, flags[k]) if cl == "all" or b == (cl == "blocked")]
            ep["n"][cl] = len(sel)
            for _, f in FEATS:
                ep[(cl, f)] = float(np.mean([float(r[f]) for r in sel])) if sel else None
        out.append(ep)
    return out


def stratified(E, cl, f, seed=0):
    """MH-weighted success-minus-failure difference, weighted group means, p, n."""
    strata = collections.defaultdict(list)
    for e in E:
        if e[(cl, f)] is not None:
            strata[e["stratum"]].append((e[(cl, f)], e["su"]))
    xs, ys, ws = [], [], []
    for v in strata.values():
        x = np.array([a for a, _ in v])
        y = np.array([s for _, s in v], bool)
        if y.all() or not y.any():
            continue
        xs.append(x); ys.append(y); ws.append(y.sum() * (~y).sum() / len(y))
    w = np.array(ws) / sum(ws)
    obs = sum(wi * (x[y].mean() - x[~y].mean()) for wi, x, y in zip(w, xs, ys))
    rng = np.random.default_rng(seed)
    null = np.zeros(B)
    for wi, x, y in zip(w, xs, ys):              # labels permuted within the stratum
        Y = rng.permuted(np.tile(y, (B, 1)), axis=1)
        null += wi * (Y @ x / y.sum() - ~Y @ x / (~y).sum())
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (B + 1)
    ms = sum(wi * x[y].mean() for wi, x, y in zip(w, xs, ys))
    mf = sum(wi * x[~y].mean() for wi, x, y in zip(w, xs, ys))
    return ms, mf, obs, p, sum(int(y.sum()) for y in ys), sum(int((~y).sum()) for y in ys), len(xs)


def main():
    E = episodes()
    print(f"ARM={tc.ARM}  episodes {len(E)}: {sum(e['su'] for e in E)} successes, "
          f"{sum(not e['su'] for e in E)} failures, {len({e['stratum'] for e in E})} strata")
    for cl in CLASSES:
        print(f"messages in class {cl}: {sum(e['n'][cl] for e in E)}"
              f"  (success {sum(e['n'][cl] for e in E if e['su'])}, failure {sum(e['n'][cl] for e in E if not e['su'])})")
    res = {}
    for cl in CLASSES:
        print(f"\n-- {cl} moments: stratified (model x suite x level x protocol), episode unit")
        for label, f in FEATS:
            r = res[(cl, f)] = stratified(E, cl, f)
            sc = 1 if f in ("words", "ncoord") else 100
            print(f"  {label:<28} S {sc * r[0]:6.1f}  F {sc * r[1]:6.1f}  d {sc * r[2]:+6.1f}  p={r[3]:.4f}"
                  f"  (episodes {r[4]}/{r[5]}, strata {r[6]})")
    print("\n-- unstratified episode means, all moments (for comparison)")
    for label, f in FEATS:
        sc = 1 if f in ("words", "ncoord") else 100
        s = [e[("all", f)] for e in E if e["su"] and e[("all", f)] is not None]
        x = [e[("all", f)] for e in E if not e["su"] and e[("all", f)] is not None]
        print(f"  {label:<28} S {sc * np.mean(s):6.1f}  F {sc * np.mean(x):6.1f}  d {sc * (np.mean(s) - np.mean(x)):+6.1f}")
    if "--tex" in sys.argv:
        path = os.path.join(ROOT, "thesis/numbers/E_tab_dialogue.tex")
        open(path, "w").write(tex(E, res))
        print("wrote", path)


def tex(E, res):
    def fmt(cl, f):
        ms, mf, d, p, *_ = res[(cl, f)]
        sc, dp = (1, 1) if f in ("words", "ncoord") else (100, 1)
        pp = "$<$0.001" if p < 1e-3 else f"{p:.3f}"
        return f"{sc * ms:.{dp}f} & {sc * mf:.{dp}f} & {tc.signed(sc * d, dp)} & {pp}"
    order = sorted(tc.DIALOGUE, key=lambda t: -res[("all", t[1])][2])
    body = [f"{label} & " + " & ".join(fmt(cl, f) for cl in CLASSES) for label, f in order]
    body.append(r"\midrule")
    body += [f"{label} & " + " & ".join(fmt(cl, f) for cl in CLASSES) for label, f, _ in tc.MEANS]
    n = {cl: (res[(cl, "seq")][4], res[(cl, "seq")][5], res[(cl, "seq")][6]) for cl in CLASSES}
    head = (" & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{t}}}" for t in
                               ("All moments", "Teammate not in the way", "Teammate in the way"))
            + " \\\\\n    \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}\n    Message feature & "
            + " & ".join("S & F & $\\Delta$ & $p$" for _ in CLASSES))
    caption = (
        "Message features in successful (S) and failed (F) episodes, compared within "
        "model, suite, level and protocol and, in the right two blocks, within the "
        "speaker's board state when the message was written. Exploratory.")
    note = (
        "\\emph{Scope.} Qwen3.8 and DeepSeek, \\textsc{freeform} and \\textsc{hybrid}, "
        "S2--S4, dyad, 3p and 6p. \\emph{Unit.} The episode: each cell is the share of an "
        "episode's messages carrying the feature, in percent (words and coordinates: per-message "
        "means), averaged over episodes. \\emph{State.} A message is written with a teammate in the "
        "way if a teammate's object lengthens or cuts the path of one of the speaker's unplaced "
        "objects to its target, on the board as the speaker saw it. \\emph{Comparison.} Success "
        "minus failure within each model $\\times$ suite $\\times$ level $\\times$ protocol stratum, "
        "combined with Mantel--Haenszel weights; strata with only one outcome drop out. Episodes "
        "(S/F) entering each block: "
        + "; ".join(f"{t} {a}/{b} in {c} strata" for t, (a, b, c) in
                    zip(("all moments", "not in the way", "in the way"), n.values()))
        + ". \\emph{Test.} Outcome labels permuted within strata, 10\\,000 resamples at seed 0, "
        "two-sided, uncorrected. Failures include aborts.")
    return tc.table(caption, "tab:dialogue", "l" + "rrrr" * 3, head, body, note=note, tight=True).replace(
        r"\tabcolsep}{3pt}", r"\tabcolsep}{2pt}")      # 13 columns: 3pt still overruns the text width


if __name__ == "__main__":
    # The weighted statistic on a toy: one stratum, success 1.0 vs failure 0.0.
    toy = [{"stratum": 1, "su": True, ("all", "x"): 1.0},
           {"stratum": 1, "su": False, ("all", "x"): 0.0}]
    assert stratified(toy, "all", "x")[2] == 1.0
    main()
