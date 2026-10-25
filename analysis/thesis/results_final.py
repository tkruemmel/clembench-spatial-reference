"""Results-chapter tables on the final arm, one model at a time.

Clembench scores and goal rate per episode (scores.json, or the log when the
episode is not scored yet), how each unsuccessful episode ended, the spatial
metrics from move_metrics.final.jsonl, and the action that cut the board.
Prints LaTeX tabular bodies for thesis/results_outline.tex.

    python3 analysis/thesis/results_final.py [model]      (default dsv41)
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from move_metrics import MOVE, DONE, pool, trees  # noqa: E402

SUITES = ["solo", "dyad", "3p", "6p"]
TEAMS = SUITES[1:]
LEVELS = ["S1", "S2", "S3", "S4"]
PROTOS = ["none", "structured", "hybrid", "freeform"]
SC = {p: r"\textsc{%s}" % p for p in PROTOS}


def end_reason(it, success):
    end = ""
    for turn in it["turns"]:
        for a in turn:
            if a["action"]["type"] in ("lose", "abort", "success"):
                end = a["action"]["content"]
    if success:
        return "success"
    for key, name in [("invalid responses", "abort:parse"), ("malformed", "abort:malformed"),
                      ("invalid moves", "abort:moves"), ("Strict", "abort:moves"),
                      ("declared done before", "lose:false_done"), ("All players declared done", "lose:all_done"),
                      ("repeated 3 times", "lose:cycle"), ("Max turns", "lose:budget")]:
        if key in end:
            return name
    raise ValueError(f"unknown end: {end!r}")


def episodes(model):
    rows = []
    for tree_model, suite, d in trees():
        if tree_model != model:
            continue
        for f in glob.glob(os.path.join(d, "*", "*", "*", "instance_*", "interactions.json")):
            e = os.path.dirname(f)
            it = json.load(open(f))
            if "Success" not in it:
                continue
            gi = json.load(open(os.path.join(e, "instance.json")))
            ok, ab = int(it["Success"]), int(it["Aborted"])
            if os.path.exists(os.path.join(e, "scores.json")):
                main = json.load(open(os.path.join(e, "scores.json")))["episode scores"]["Main Score"]
            else:  # not scored yet: eq:benchscore from the log
                main = None if ab else (min(100, round(100 * gi["optimal_moves"] / max(it["Move Count"], 1), 2)) if ok else 0)
            pos = {o: tuple(p) for o, p in gi["start_positions"].items()}
            for mv in it["Turn Moves"]:
                pos[mv["object"]] = tuple(mv["to"])
            ev = collections.Counter(a["action"]["type"] for turn in it["turns"] for a in turn)
            name = os.path.basename(os.path.dirname(e))
            rows.append(dict(
                path=os.path.relpath(e, ROOT), suite=suite, level=re.search(r"S[1-4]", name).group(0),
                proto=name.split("_")[1] if suite != "solo" else "-", success=ok, played=1 - ab, main=main,
                home=sum(pos[o] == tuple(t) for o, t in gi["target_positions"].items()),
                reason=end_reason(it, ok), parse=ev["invalid_format"], malformed=ev["malformed_message"],
                invalid=ev["invalid_move"]))
    return rows


def clem(R):
    """clemscore, % Played, Quality, success, eta, goal rate of a group of episodes.

    The clemscore scores aborted episodes 0, so clemscore = % Played * Quality / 100.
    Success is taken over all episodes, so clemscore = success * eta exactly."""
    P = [r for r in R if r["played"]]
    S = [r for r in R if r["success"]]
    q = sum(r["main"] for r in P) / len(P) if P else None
    eta = sum(r["main"] for r in S) / len(S) / 100 if S else None
    return [sum(r["main"] for r in P) / len(R), 100 * len(P) / len(R), q, 100 * len(S) / len(R), eta,
            100 * sum(r["home"] for r in R) / (6 * len(R))]


def fmt(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def clem_row(label, R):
    c, p, q, s, e, g = clem(R)
    return f"{label} & {len(R)} & {fmt(c)} & {fmt(p)} & {fmt(q)} & {fmt(s)} & {fmt(e, 3)} & {fmt(g)} \\\\"


def move_row(label, M):
    p = pool(M)
    f3 = lambda k: fmt(p[k], 3)
    return (f"{label} & {p['n']} & {f3('osc')} & {f3('stall')} & {f3('opt')} & {f3('accF')} & {f3('chF')} & "
            f"{p['nF']} & {f3('accG')} & {f3('chG')} & {f3('top')} \\\\")


def cut(model, R):
    """Kind of the action after which the board has no solution, per unsuccessful team episode."""
    mm = {r["path"]: r for r in map(json.loads, open(os.path.join(HERE, "move_metrics.final.jsonl")))
          if r["model"] == model}
    out = []
    for r in R:
        if r["success"] or r["suite"] == "solo":
            continue
        rec = mm[r["path"]]
        kinds = []
        for turn in json.load(open(os.path.join(ROOT, r["path"], "interactions.json")))["turns"]:
            for a in turn:
                c = a["action"].get("content")
                if isinstance(c, str):
                    kinds += ["move"] if MOVE.match(c) else ["done"] if DONE.match(c) else []
        assert len(kinds) == rec["M"], r["path"]
        f = rec["fatal"]
        out.append(dict(r, cut=kinds[f - 1] if f else None, after=rec["M"] - f if f else None))
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "dsv41"
    R = episodes(model)
    M = [r for r in map(json.loads, open(os.path.join(HERE, "move_metrics.final.jsonl"))) if r["model"] == model]
    sel = lambda X, **k: [r for r in X if all(r[a] == b for a, b in k.items())]
    team = lambda X: [r for r in X if r["suite"] != "solo"]

    print("% clembench per game: game & n & clemscore & %Played & Quality & Success & eta & Goal")
    for s in SUITES:
        print(clem_row(s, sel(R, suite=s)))
    rows = [clem(sel(R, suite=s)) for s in SUITES]
    pl, q, g = (statistics.mean(x[i] for x in rows) for i in (1, 2, 5))
    print(f"% macro: played {pl:.1f} quality {q:.2f} goal {g:.1f} clemscore {pl / 100 * q:.2f}")

    print("\n% clembench per game x level")
    for s in SUITES:
        for lv in LEVELS:
            print(clem_row(f"{s} & {lv}", sel(R, suite=s, level=lv)))
        print(r"\midrule")
    print("\n% clembench per game x protocol")
    for s in TEAMS:
        for p in PROTOS:
            print(clem_row(f"{s} & {SC[p]}", sel(R, suite=s, proto=p)))
        print(r"\midrule")
    print("\n% clembench per level x protocol, teams pooled")
    for lv in LEVELS:
        for p in PROTOS:
            print(clem_row(f"{lv} & {SC[p]}", sel(team(R), level=lv, proto=p)))
        print(r"\midrule")

    for s in SUITES:
        p = pool(sel(M, suite=s))
        print(f"% {s}: illegal {100 * p['illegal']:.2f}% blocked {100 * p['blocked']:.2f}% of attempts")
    print("\n% spatial per game and game x level: n osc stall opt accF chF nF accG chG top")
    for s in SUITES:
        print(move_row(f"{s} & all", sel(M, suite=s)))
        for lv in LEVELS:
            print(move_row(f" & {lv}", sel(M, suite=s, level=lv)))
        print(r"\midrule")
    print("\n% spatial per game x protocol")
    for s in TEAMS:
        for p in PROTOS:
            print(move_row(f"{s} & {SC[p]}", sel(M, suite=s, proto=p)))
        print(r"\midrule")
    print("\n% spatial per level x protocol, teams pooled")
    for lv in LEVELS:
        for p in PROTOS:
            print(move_row(f"{lv} & {SC[p]}", sel(team(M), level=lv, proto=p)))
        print(r"\midrule")

    F = [r for r in R if not r["success"]]
    ends = ["lose:cycle", "abort:moves", "abort:parse", "abort:malformed", "lose:false_done", "lose:all_done", "lose:budget"]
    print("\n% ends per game:", " ".join(ends))
    for s in SUITES:
        c = collections.Counter(r["reason"] for r in sel(F, suite=s))
        print(s, " & ".join(str(c[e]) for e in ends))
    print("% retry events per game: parse malformed invalid-move")
    for s in SUITES:
        X = sel(R, suite=s)
        print(s, sum(r["parse"] for r in X), sum(r["malformed"] for r in X), sum(r["invalid"] for r in X))
    print("\n% clemscore per level x game x protocol (solo, then dyad/3p/6p by none structured hybrid freeform)")
    for lv in LEVELS:
        cells = [fmt(clem(sel(R, suite="solo", level=lv))[0])]
        cells += [fmt(clem(sel(R, suite=s, level=lv, proto=p))[0]) for s in TEAMS for p in PROTOS]
        print(f"{lv} & " + " & ".join(cells) + r" \\")
    print("\n% unsuccessful episodes per game x level x protocol (aborts in parentheses)")
    for s in SUITES:
        for lv in LEVELS:
            cells = []
            for p in (PROTOS if s != "solo" else ["-"]):
                X = sel(F, suite=s, level=lv, proto=p)
                a = sum(r["reason"].startswith("abort") for r in X)
                cells.append(f"{len(X)}" + (f" ({a})" if a else ""))
            print(f"{s} & {lv} & " + " & ".join(cells) + r" \\")
    C = cut(model, R)
    print("\n% cut per game: unsuccessful team episodes, done cut, move cut, no cut; cycle losses after a done cut, median and range of actions to the loss")
    for s in TEAMS:
        X = sel(C, suite=s)
        k = collections.Counter(r["cut"] for r in X)
        a = [r["after"] for r in X if r["cut"] == "done" and r["reason"] == "lose:cycle"]
        print(f"{s} & {len(X)} & {k['done']} & {k['move']} & {k[None]} & {len(a)} & "
              f"{statistics.median(a) if a else '--'} & {min(a) if a else '--'}--{max(a) if a else '--'} \\\\")
    a = [r["after"] for r in C if r["cut"] == "done" and r["reason"] == "lose:cycle"]
    print(f"% all teams: median {statistics.median(a)} range {min(a)}-{max(a)} n {len(a)}")
    print("% episodes ending in a cut by move:", [(r["suite"], r["level"], r["proto"], r["reason"]) for r in C if r["cut"] == "move"])
    print("% unsuccessful without a cut:", [(r["suite"], r["level"], r["proto"], r["reason"]) for r in C if r["cut"] is None])
    fs = sel(M, outcome="lose") + sel(M, outcome="abort")
    for s in SUITES:
        X = [r for r in fs if r["suite"] == s]
        st = [r["stall"] for r in X if r["stall"] is not None]
        print(f"% failed {s}: osc {sum(r['osc'] for r in X) / max(1, sum(r['T'] for r in X)):.3f} "
              f"stall {statistics.mean(st) if st else float('nan'):.3f}")

    # Solo as the baseline, team games per protocol: the 13 conditions as columns or rows.
    conds = [("solo", "-")] + [(s, p) for s in TEAMS for p in PROTOS]
    at = lambda X, s, p, **k: sel(X, suite=s, **({} if s == "solo" else {"proto": p}), **k)
    cname = lambda s, p: "solo & baseline" if s == "solo" else f"{s} & {SC[p]}"

    print("\n% spatial per condition (solo baseline, game x protocol)")
    for s, p in conds:
        print(move_row(cname(s, p), at(M, s, p)))
    for key in ("opt", "osc"):
        print(f"\n% {key} per level x condition (solo, dyad/3p/6p by none structured hybrid freeform)")
        for lv in LEVELS:
            print(f"{lv} & " + " & ".join(fmt(pool(at(M, s, p, level=lv))[key], 3) for s, p in conds) + r" \\")
    print("\n% illegal and blocked (% of attempts) per condition")
    for key in ("illegal", "blocked"):
        print(f"{key} & " + " & ".join(fmt(100 * pool(at(M, s, p))[key], 2) for s, p in conds) + r" \\")
    print("\n% ends per condition:", " ".join(ends))
    for e in ends:
        print(f"{e} & " + " & ".join(str(sum(r["reason"] == e for r in at(F, s, p))) for s, p in conds) + r" \\")
    print("unsuccessful & " + " & ".join(str(len(at(F, s, p))) for s, p in conds) + r" \\")
    for key in ("parse", "malformed", "invalid"):
        print(f"{key} events & " + " & ".join(str(sum(r[key] for r in at(R, s, p))) for s, p in conds) + r" \\")
    print("\n% parse failures, solo baseline and per protocol (team games pooled): failures, episodes affected, of which unsuccessful, episodes")
    for name, X in [("solo baseline", sel(R, suite="solo"))] + [(SC[p], sel(team(R), proto=p)) for p in PROTOS]:
        P = [r for r in X if r["parse"]]
        print(f"{name} & {sum(r['parse'] for r in P)} & {len(P)} & {sum(not r['success'] for r in P)} & {len(X)} \\\\")
    # Counterfactual: no reprompt, so the first parse failure, malformed message or
    # invalid move aborts the episode; everything before it is unchanged.
    hit = lambda r: r["parse"] + r["malformed"] + r["invalid"] > 0
    RX = [dict(r, played=0, success=0) if hit(r) else r for r in R]
    print("\n% instant abort per condition: episodes hit; clemscore, %Played, Quality as run; same under instant abort")
    for s, p in conds:
        a, b = clem(at(R, s, p)), clem(at(RX, s, p))
        print(f"{cname(s, p)} & {sum(map(hit, at(R, s, p)))} & {fmt(a[0])} & {fmt(a[1])} & {fmt(a[2])} & "
              f"{fmt(b[0])} & {fmt(b[1])} & {fmt(b[2])} \\\\")
    for name, X in [("as run", R), ("instant abort", RX)]:
        rows = [clem(sel(X, suite=s)) for s in SUITES]
        pl, q = (statistics.mean(x[i] for x in rows) for i in (1, 2))
        print(f"% macro clemscore {name}: played {pl:.1f} quality {q:.2f} clemscore {pl / 100 * q:.2f}")

    print("\n% cut per team condition: failed, done cut, move cut, no cut, cycle after done cut, median, range")
    for s, p in conds[1:]:
        X = at(C, s, p)
        k = collections.Counter(r["cut"] for r in X)
        a = [r["after"] for r in X if r["cut"] == "done" and r["reason"] == "lose:cycle"]
        rng = f"{min(a)}--{max(a)}" if a else "--"
        print(f"{cname(s, p)} & {len(X)} & {k['done']} & {k['move']} & {k[None]} & {len(a)} & "
              f"{fmt(statistics.median(a), 1) if a else '--'} & {rng} \\\\")
    print("\n% oscillation and stall per condition: failed, osc failed, osc all, stall failed, stall successful")
    for s, p in conds:
        X = at(M, s, p)
        fx = [r for r in X if r["outcome"] != "success"]
        st = [r["stall"] for r in fx if r["stall"] is not None]
        osc_f = sum(r["osc"] for r in fx) / sum(r["T"] for r in fx) if fx else None
        print(f"{cname(s, p)} & {len(fx)} & {fmt(osc_f, 3)} & {fmt(pool(X)['osc'], 3)} & "
              f"{fmt(statistics.mean(st), 3) if st else '--'} & {fmt(pool(X)['stall'], 3)} \\\\")


if __name__ == "__main__":
    main()
