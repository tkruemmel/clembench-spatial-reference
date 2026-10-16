#!/usr/bin/env python3
"""tab:hard-boards: boards split by strong-tier success within a level.

    ARM=updated python3 analysis/thesis/hard_boards.py          # rows, rho, checks
    ARM=updated python3 analysis/thesis/hard_boards.py --tex    # the float

Group.  Per board, the successes of the two strong models over its 24
multi-agent episodes (dyad, 3p, 6p x four protocols x two models).  A board is
*hard* at <= 75 % (<= 18/24) and *easy* at >= 92 % (>= 23/24); boards in
between are left out of the table but not out of the correlations.

Board features come from the instance files alone (instance_audit.py) and the
exact solvers, never from an episode:
  h0      sum of per-object BFS paths, each object ignoring the others
  J       the free-order joint optimum (joint_optimum.jsonl)
  gap     J - h0, the moves forced by objects getting out of each other's way
  ovh     six-way protocol overhead, attainable_6p / J (attainable_6p.jsonl)
  walls   wall cells
  detour  per-object BFS path / Manhattan distance; per group the median over
          the group's pooled objects, per board the median over its objects
The table reports group means of gap, ovh, walls, h0 and J.

Within each level, Spearman's rho of board success (out of 24) against gap,
ovh, detour and J over the 15 boards.  Exploratory.

ARM picks the team suites (qual_corpus.tree); solo is always the original
campaign.  Board features are read from matrixgame_covered_solo_grid, and the
script refuses to run unless the packages the reported episodes were played
from carry the same 60 boards.
"""
import collections
import json
import os
import statistics as st
import sys

from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "analysis/efficiency_baseline"))
import tables_comm as T                                          # noqa: E402
import instance_audit as IA                                      # noqa: E402

BOARDS = "matrixgame_covered_solo_grid/in/instances.json"
PLAYED = ["matrixgame_covered_comm_grid_updated_prompt/in/instances.json",
          "matrixgame_covered_comm_nplayer_grid_updated_prompt/in/instances_3p.json",
          "matrixgame_covered_comm_nplayer_grid_updated_prompt/in/instances_6p.json",
          "matrixgame_covered_comm_grid/in/instances.json",
          "matrixgame_covered_comm_nplayer_grid/in/instances_3p.json",
          "matrixgame_covered_comm_nplayer_grid/in/instances_6p.json"]
DESIGNED = ("chain", "edges", "chains", "swap", "dead_end_cells", "targets_in_dead_end")
HARD, EASY = 18, 23          # of 24: <= 75 %, >= 92 %


def features():
    boards = IA.load_boards(BOARDS)
    for p in PLAYED:
        other = IA.load_boards(p)
        bad = [k for k in boards if IA.board_key(boards[k]) != IA.board_key(other[k])]
        assert not bad, f"{p} differs from {BOARDS} on {bad}"
    jo, sf = IA.load_solver_outputs()
    a6 = {(d["level"], d["game_id"]): d for d in map(json.loads, open(
        os.path.join(ROOT, "analysis/efficiency_baseline/attainable_6p.jsonl")))}
    out, designed = {}, collections.defaultdict(set)
    for k, inst in boards.items():
        r = IA.audit_board(*k, inst, jo, sf)
        designed[k[0]].add(tuple(r[f] for f in DESIGNED))
        assert r["h0"] is not None and jo[k]["status"] == a6[k]["status"] == "exact", k
        h0 = sum(IA.bfs_dist(tuple(inst["start_positions"][o]), tuple(inst["target_positions"][o]),
                             inst["grid_size"], {tuple(w) for w in inst["walls"]})
                 for o in inst["start_positions"])
        assert h0 == r["h0"], (k, h0, r["h0"])          # joint_optimum.jsonl is this board
        out[k] = dict(h0=r["h0"], J=r["opt"], gap=r["gap"], ovh=a6[k]["stuckfree"] / r["opt"],
                      walls=r["walls"], detour=r["detour"], objs=r["detour_objs"] or [])
    # the caption's premise: the designed features of S6.2 do not vary within a level
    assert all(len(v) == 1 for v in designed.values()), dict(designed)
    print("designed features per level", DESIGNED, {l: v.pop() for l, v in sorted(designed.items())})
    return out


def outcomes():
    eps = T.all_episodes(T.load_optima())
    team, solo, weak = (collections.Counter() for _ in range(3))
    n = collections.Counter()
    for e in eps:
        k = (e["level"], e["game_id"])
        if e["model"] in T.STRONG and e["suite"] in T.SUITES:
            team[k] += e["success"]; n[k] += 1
        elif e["model"] in T.STRONG:
            solo[k] += e["success"]
        elif e["suite"] in T.SUITES:
            weak[k] += e["success"]
    assert set(n.values()) == {24}, collections.Counter(n.values())
    return team, solo, weak


def main(tex=False):
    F = features()
    team, solo, weak = outcomes()
    rows = []
    print(f"ARM={T.ARM}")
    print("level group  boards                    success  solo/4    gap   ovh  walls    h0     J  detour")
    for lv in T.ALL_LEVELS:
        gids = sorted(g for (l, g) in F if l == lv)
        for name, keep in (("hard", lambda s: s <= HARD), ("easy", lambda s: s >= EASY)):
            gs = [g for g in gids if keep(team[(lv, g)])]
            if not gs:
                print(f"{lv:5} {name:5}  --")
                continue
            f = [F[(lv, g)] for g in gs]
            m = lambda key: st.mean(x[key] for x in f)
            det = st.median(o for x in f for o in x["objs"])
            row = dict(lv=lv, name=name, gs=gs, succ=sum(team[(lv, g)] for g in gs), n=24 * len(gs),
                       solo=[solo[(lv, g)] for g in gs], gap=m("gap"), ovh=m("ovh"), walls=m("walls"),
                       h0=m("h0"), J=m("J"), det=det)
            rows.append(row)
            print(f"{lv:5} {name:5}  {','.join(map(str, gs)):24}  {row['succ']:3}/{row['n']:<3}  "
                  f"{min(row['solo'])}-{max(row['solo'])}  {row['gap']:5.1f} {row['ovh']:5.2f} "
                  f"{row['walls']:6.1f} {row['h0']:5.1f} {row['J']:5.1f}  {det:5.2f}")
        mid = [g for g in gids if HARD < team[(lv, g)] < EASY]
        print(f"{lv:5} omitted {mid}")
    print("\nWithin-level Spearman, board success (of 24) vs feature, 15 boards each (exploratory)")
    for lv in T.ALL_LEVELS:
        gids = sorted(g for (l, g) in F if l == lv)
        y = [team[(lv, g)] for g in gids]
        out = []
        for key in ("gap", "ovh", "detour", "J", "h0", "walls"):
            x = [F[(lv, g)][key] for g in gids]
            if len(set(x)) < 2 or len(set(y)) < 2:
                out.append(f"{key} --")
                continue
            r = spearmanr(y, x)
            out.append(f"{key} {r.statistic:+.2f} (p={r.pvalue:.3f})")
        print(f"  {lv}: " + "  ".join(out))
    print("\nWeak tier (Nemotron) multi-agent successes by level:",
          {lv: sum(v for (l, g), v in weak.items() if l == lv) for lv in T.ALL_LEVELS},
          " non-S1 boards:", sorted(k for k, v in weak.items() if v and k[0] != "S1"))
    hs = [s for r in rows if r["name"] == "hard" for s in r["solo"]]
    print("Solo successes on the hard boards (of 4 each):", collections.Counter(hs))
    return rows


def latex(rows):
    body = []
    for r in rows:
        lv = r["lv"] if r["name"] == "hard" or not any(
            x["lv"] == r["lv"] and x["name"] == "hard" for x in rows) else ""
        if body and lv:
            body.append(r"    \addlinespace")
        body.append(f"    {lv} & {r['name']} & {', '.join(map(str, r['gs']))} & {r['succ']}/{r['n']} & "
                    f"{r['gap']:.1f} & {r['ovh']:.2f} & {r['walls']:.1f} & {r['h0']:.1f} & "
                    f"{r['J']:.1f} & {r['det']:.2f} \\\\")
    return "\n".join([
        r"  \begin{tabular}{ll>{\raggedright\arraybackslash}p{2.4cm}rrrrrrr}", r"    \toprule",
        r"    Level & Group & Boards & Success & Gap & 6p ovh. & Walls & Paths & Optimum & Detour \\",
        r"    \midrule", *body, r"    \bottomrule", r"  \end{tabular}"])


def float_tex(rows):
    """tab:hard-boards as a float; the report goes to stderr under --tex."""
    hard = [r for r in rows if r["name"] == "hard"]
    levels = sorted({r["lv"] for r in hard})
    assert levels == ["S2", "S3", "S4"], levels      # the caption names them
    s2 = [r for r in hard if r["lv"] == "S2"]
    assert len(s2) == 1 and s2[0]["gs"] == [10], s2
    return "\n".join([
        r"\begin{table}[tbp]", r"  \centering", r"  \footnotesize",
        r"  \caption{Hard and easy boards within a level, strong tier. A board is "
        r"\emph{hard} when the two strong models solve at most 75\,\% of its 24 team "
        r"episodes (dyad, 3p and 6p, four protocols each) and \emph{easy} when they solve "
        r"92\,\% or more; boards in between are omitted. S1 has no hard board; S2 has one, "
        f"S2~\\#10 ({s2[0]['succ']} of {s2[0]['n']}), whose interaction gap is the largest at "
        r"its level. Gap, overhead, walls and paths are means over boards; detour is the "
        r"median pooled over objects. The designed features of each level "
        r"(Chapter~\ref{ch:setup}) are constant within it and are not shown.}",
        r"  \label{tab:hard-boards}", r"  \setlength{\tabcolsep}{4pt}",
        latex(rows), r"  \\[2pt]", r"  \begin{minipage}{0.9\textwidth}\footnotesize",
        r"  \emph{Gap}: the interaction gap, the free-order joint optimum $J$ minus the "
        r"per-object bound. \emph{6p ovh.}: the six-player attainable optimum over $J$."
        r" \emph{Paths}: the per-object bound, the sum of per-object shortest "
        r"paths ignoring the other objects. \emph{Optimum}: $J$. \emph{Detour}: "
        r"per-object walled path over Manhattan distance.",
        r"  \end{minipage}", r"\end{table}", ""])


if __name__ == "__main__":
    import contextlib
    tex = "--tex" in sys.argv
    with contextlib.redirect_stdout(sys.stderr if tex else sys.stdout):
        rows = main(tex)
    if tex:
        print(float_tex(rows))
