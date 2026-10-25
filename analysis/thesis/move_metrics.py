"""Spatial efficiency breakdown (sec:method-metrics-efficiency) on the final arm.

Every episode of every final-arm tree (trees()) is replayed action by action (moves and `done` declarations) on the board it was played
on, and scored on:

  illegal, blocked  rejected move attempts over all move attempts: the game's
                    Illegal / Blocked Move Rate, pooled here
  osc               eq:oscillation, k = 4 (osc_moves)
  stall             eq:stall, k = max(2n, 6) (stall_rate).  At S4 the
                    outer object, which starts on device.inner_adjacent and
                    whose target is the dead end, is left out of D_t
  regret r_m        eq:regret.  h*(s) is the exact optimum from the state
                    s = (positions, done set, player to move) under the suite's
                    rules, from hstar.c
  opt, share_p      eq:regret-rates
  acc_F, acc_G      eq:detour, each with its chance level

Checked on every episode: the replay against the logged Turn Moves, h*(s_0)
against the instance's optimal_moves (A_k), and sum r_m = T - A_k on every
success.  Once the board can no longer be solved, the action that cut it has
infinite regret and the actions after it have none.  These later actions, and
any state the solver gave up on (cap), are left out of opt, acc_F and acc_G.

Per-episode records are cached in move_metrics.final.jsonl, keyed by path and
the mtime of interactions.json, so a rerun only solves new episodes, plus the
episodes with a capped state when --cap is raised.  Build the solver once:

    cc -O3 -o analysis/thesis/hstar_c analysis/thesis/hstar.c

    python3 analysis/thesis/move_metrics.py            # report
    python3 analysis/thesis/move_metrics.py --tex      # also thesis/numbers/tab_move_metrics.final.tex
    python3 analysis/thesis/move_metrics.py --check    # hstar_c against solver.solve first

Exploratory: not one of the confirmatory contrasts.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [HERE, ROOT]
from matrixgame.utils import solver as SF                     # noqa: E402
from matrixgame.utils.history import SEATS                    # noqa: E402

BINARY = os.path.join(HERE, "hstar_c")
CACHE = os.path.join(HERE, "move_metrics.final.jsonl")
TREE = re.compile(r"^results_(\w+?)_grid_final_n15_(solo|dyad|3p|6p)$")
CHRONO = re.compile(r"^results_(\w+?)_grid_final_chrono_n15_(3p|6p)$")
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DONE = re.compile(r"^(Player \w+) declared done$")
STEP = {name: d for d, name in SF.DIRNAME.items()}


def osc_moves(cells, k=4):
    """Oscillating moves in one object's cell sequence c_0, c_1, ... (eq:oscillation)."""
    return sum(cells[j + 1] in cells[max(0, j - k):j] for j in range(len(cells) - 1))


def stall_rate(D, k):
    """Share of the k-move windows of the distance series D that make no progress (eq:stall)."""
    T = len(D) - 1
    if T < k:
        return None
    return sum(D[t + k] >= D[t] for t in range(T - k + 1)) / (T - k + 1)


def table(caption, label, spec, header, body, note=None, tight=False):
    out = [r"\begin{table}[tbp]", r"  \centering", r"  \footnotesize"]
    if tight:  # wide contrast tables: narrower column gaps, not smaller type
        out.append(r"  \setlength{\tabcolsep}{3pt}")
    out += [f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
            f"  \\begin{{tabular}}{{{spec}}}", r"    \toprule",
            "    " + header + r" \\", r"    \midrule"]
    for b in body:
        out.append(r"    \midrule" if b == r"\midrule" else "    " + b + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}"]
    if note:
        out += [r"  \\[2pt]", r"  \begin{minipage}{0.9\textwidth}\footnotesize",
                "  " + note, r"  \end{minipage}"]
    out += [r"\end{table}", ""]
    return "\n".join(out)


SUITES = ("solo", "dyad", "3p", "6p")
LEVELS = ("S1", "S2", "S3", "S4")
LABEL = {"dsv41": "DeepSeek-V4.1"}
INF = float("inf")


class Solver:
    """One hstar_c process per board and partition; answers cached per state."""

    def __init__(self, gi, cap):
        self.letters, self.groups, self.starts, self.targets, self.nbr, self.dist, free = SF.board(gi)
        self.idx = {c: k for k, c in enumerate(free)}
        lines = [f"{len(free)} {len(self.letters)} {len(self.groups)}"]
        lines += [" ".join(map(str, [len(ns)] + [v for v, _ in ns])) for ns in self.nbr]
        lines += [" ".join(map(str, self.targets))]
        lines += [" ".join(map(str, [len(g)] + list(g))) for g in self.groups]
        self.p = subprocess.Popen([BINARY, str(cap)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)
        self.p.stdin.write("\n".join(lines) + "\n")
        self.exact = {}      # state -> h*: an int, INF, or None when the cap was hit
        self.above = {}      # state -> largest bound known to be below h*
        self.capped = set()  # (state, bound) queries that hit the cap

    def close(self):
        self.p.stdin.close()
        self.p.wait()

    def run(self, queries):
        """Answer (state, bound) queries, bound -1 for exact.  Chunked so that one
        pipe round trip serves many queries and neither pipe fills up."""
        queries = list(dict.fromkeys(queries))
        for k in range(0, len(queries), 400):
            chunk = queries[k:k + 400]
            self.p.stdin.write("".join(f"{' '.join(map(str, s[0]))} {s[1]} {s[2]} {b}\n" for s, b in chunk))
            for s, b in chunk:
                out = self.p.stdout.readline().split()
                if out[0] == "h":
                    self.exact[s] = int(out[1])
                elif out[0] == "none":
                    self.exact[s] = INF
                elif out[0] == "gt":
                    self.above[s] = max(self.above.get(s, -1), b)
                elif b < 0:
                    self.exact[s] = None
                else:
                    self.capped.add((s, b))

    def hs(self, states):
        """h* along an episode.  Every state reachable from an unsolvable one is
        unsolvable too, so the states after the first infinite one are not solved."""
        out = []
        for k in range(0, len(states), 32):
            chunk = states[k:k + 32]
            self.run([(s, -1) for s in chunk if s not in self.exact])
            for s in chunk:
                out.append(self.exact[s])
                if out[-1] == INF:
                    return out + [INF] * (len(states) - len(out))
        return out

    def within(self, s, bound):
        """Whether h*(s) <= bound: True, False, None (cap), or "ask" (not known yet)."""
        if s in self.exact:
            v = self.exact[s]
            return None if v is None else v <= bound
        if self.above.get(s, -1) >= bound or sum(self.dist[i][p] for i, p in enumerate(s[0])) > bound:
            return False
        return None if (s, bound) in self.capped else "ask"


def nxt(turn, done, n):
    for k in range(1, n + 1):
        t = (turn + k) % n
        if not (done >> t) & 1:
            return t
    return -1


def legal(B, s):
    """(cost, successor, delta, label) for every legal action in state s."""
    pos, done, turn = s
    n, out = len(B.groups), []
    own = B.groups[turn]
    if all(pos[i] == B.targets[i] for i in own) and nxt(turn, done | 1 << turn, n) >= 0:
        d2 = done | 1 << turn
        out.append((0, (pos, d2, nxt(turn, d2, n)), 0, "done"))
    occ = set(pos)
    for i in own:
        for v, d in B.nbr[pos[i]]:
            if v in occ:
                continue
            p2 = pos[:i] + (v,) + pos[i + 1:]
            t2 = turn if p2 == B.targets else nxt(turn, done, n)
            out.append((1, (p2, done, t2), B.dist[i][v] - B.dist[i][pos[i]], f"{B.letters[i]} {SF.DIRNAME[d]}"))
    return out


def replay(B, it):
    """States s_0..s_M and actions (seat, cost, label, object index) from the log."""
    n = len(B.groups)
    pos, done, turn = tuple(B.starts), 0, 0
    states, acts, cells = [(pos, done, turn)], [], []
    for t in it["turns"]:
        for e in t:
            a = e.get("action", {})
            ty, c = a.get("type"), a.get("content", "")
            if ty == "move":
                m = MOVE.match(c)
                seat = SEATS.index(f"Player {m.group(1)}") if m.group(1) else 0
                i, d = B.letters.index(m.group(2)), STEP[m.group(3)]
                v = [w for w, dd in B.nbr[pos[i]] if dd == d]
                if seat != turn or i not in B.groups[turn] or not v or v[0] in pos:
                    raise ValueError(f"illegal replayed move {c!r} at action {len(acts) + 1}")
                cells.append((m.group(2), pos[i], v[0]))
                pos = pos[:i] + (v[0],) + pos[i + 1:]
                turn = turn if pos == B.targets else nxt(turn, done, n)
                acts.append((seat, 1, f"{m.group(2)} {m.group(3)}", i))
            elif ty == "player_done":
                seat = SEATS.index(DONE.match(c).group(1))
                if seat != turn or any(pos[i] != B.targets[i] for i in B.groups[seat]):
                    raise ValueError(f"illegal replayed done {c!r} at action {len(acts) + 1}")
                done |= 1 << seat
                turn = nxt(turn, done, n)
                acts.append((seat, 0, "done", None))
            else:
                continue
            states.append((pos, done, turn))
    return states, acts, cells


def episode(gi, it, B):
    """Metrics of one episode; B is the Solver of its board."""
    states, acts, cells = replay(B, it)
    logged = [(x["object"], B.idx[tuple(x["from"])], B.idx[tuple(x["to"])]) for x in it.get("Turn Moves") or []]
    if logged != cells:
        raise ValueError("replay disagrees with the logged Turn Moves")
    n, T, A = len(B.groups), len(cells), gi["optimal_moves"]
    lv = gi["spatial_level"]

    # oscillation and stall, on cells and walls-only distances
    seq = collections.defaultdict(list)
    for o, a, b in cells:
        seq[o] = seq[o] or [a]
        seq[o].append(b)
    skip = set()
    if lv == "S4":                                      # the outer object of the S4 corridor
        inner = tuple(gi["device"]["inner_adjacent"])
        skip = {B.letters.index(o) for o, p in gi["start_positions"].items() if tuple(p) == inner}
    D = [sum(B.dist[i][p] for i, p in enumerate(s[0]) if i not in skip) for k, s in enumerate(states)
         if k == 0 or acts[k - 1][1]]

    # regret
    H = B.hs(states)
    regret, fatal = [], None
    for m, (seat, c, _, _) in enumerate(acts, 1):
        a, b = H[m - 1], H[m]
        if a is None or a == INF or b is None:
            regret.append(None)
        elif b == INF:
            regret.append(None)
            fatal = m
        else:
            regret.append(c + b - a)

    # forced detours: O(s) for every state with a finite h*, the successors solved in one batch
    F, G = [0, 0, 0.0], [0, 0, 0.0]                     # actions, optimal ones, chance sum
    neither = undetermined = 0
    Ls = [legal(B, s) if h not in (None, INF) else None for s, h in zip(states, H)]
    B.run([(s2, H[m] - cost) for m, (_, _, label, _) in enumerate(acts) if Ls[m]
           for cost, s2, _, lab in Ls[m] if lab != label and B.within(s2, H[m] - cost) == "ask"])
    for m, (seat, c, label, _) in enumerate(acts, 1):
        Hs, L = H[m - 1], Ls[m - 1]
        if L is None:
            continue
        taken = (regret[m - 1] == 0) if regret[m - 1] is not None else (False if m == fatal else None)
        opt = []
        for cost, s2, delta, lab in L:
            ok = taken if lab == label else B.within(s2, Hs - cost)
            if ok is None:
                break
            if ok:
                opt.append((lab, delta))
        else:
            if any(lab == "done" for lab, _ in opt):
                neither += 1
                continue
            cls = F if all(delta == 1 for _, delta in opt) else G
            cls[0] += 1
            cls[1] += taken
            cls[2] += len(opt) / len(L)
            continue
        undetermined += 1

    by_seat = [0] * n
    for (seat, _, _, _), r in zip(acts, regret):
        by_seat[seat] += r or 0
    moves = [(m, r) for m, ((_, c, _, _), r) in enumerate(zip(acts, regret), 1) if c]
    success = bool(it.get("Success"))
    return dict(
        outcome="success" if success else ("abort" if it.get("Aborted") else "lose"),
        n=n, level=lv, T=T, M=len(acts), A=A, h0=H[0],
        attempts=it.get("Move Attempts", 0), illegal=it.get("Illegal Attempts", 0),
        blocked=it.get("Blocked Attempts", 0),
        osc=sum(osc_moves(c) for c in seq.values()),
        stall=stall_rate(D, max(2 * n, 6)),
        regret=regret, seats=[a[0] for a in acts], fatal=fatal,
        unknown=sum(h is None for h in H), by_seat=by_seat,
        opt=[sum(r == 0 for _, r in moves), sum(r is not None or m == fatal for m, r in moves)],
        F=F, G=G, neither=neither, undetermined=undetermined,
        sum_ok=(sum(regret) == T - A) if success and None not in regret else None,
    )


def trees():
    """[(model, suite, tree)] of the final arm. At 3p and 6p a model's run with
    teammates' messages in the order sent (results_<model>_grid_final_chrono_n15_<suite>)
    replaces its shuffled run when it exists."""
    out = {}
    for rx in (TREE, CHRONO):  # chrono second, so it wins
        for d in sorted(glob.glob(os.path.join(ROOT, "results_*_n15_*"))):
            m = rx.match(os.path.basename(d))
            if m:
                out[m.groups()] = d
    return [(model, suite, d) for (model, suite), d in sorted(out.items())]


def discover():
    eps = []
    for model, suite, d in trees():
        for f in sorted(glob.glob(os.path.join(d, "*", "*", "*", "instance_*", "interactions.json"))):
            eps.append((model, suite, os.path.dirname(f)))
    return eps


def board_key(gi):
    return json.dumps([gi["grid_size"], sorted(map(tuple, gi["walls"])), sorted(gi["target_positions"].items()),
                       SF.partition(gi)])


def compute(eps, cap, refresh):
    cache = {}
    if os.path.exists(CACHE) and not refresh:
        for line in open(CACHE):
            r = json.loads(line)
            cache[r["path"]] = r
    todo = collections.defaultdict(list)
    out = []
    for model, suite, d in eps:
        rel = os.path.relpath(d, ROOT)
        mtime = os.path.getmtime(os.path.join(d, "interactions.json"))
        r = cache.get(rel)
        if r and r["mtime"] == mtime and ("err" in r or not (r["unknown"] or r["undetermined"]) or r.get("cap", 0) >= cap):
            out.append(r)
            continue
        gi = json.load(open(os.path.join(d, "instance.json")))
        todo[board_key(gi)].append((model, suite, d, rel, mtime, gi))
    if todo:
        print(f"solving {sum(map(len, todo.values()))} episodes on {len(todo)} boards", file=sys.stderr)
    with open(CACHE, "a") as fh:
        for k, group in enumerate(todo.values(), 1):
            solver = Solver(group[0][5], cap)
            for model, suite, d, rel, mtime, gi in group:
                it = json.load(open(os.path.join(d, "interactions.json")))
                exp = os.path.basename(os.path.dirname(d))
                r = dict(path=rel, mtime=mtime, cap=cap, model=model, suite=suite, gid=gi["game_id"],
                         proto=gi.get("comm_protocol", "silent"), exp=exp)
                try:
                    r.update(episode(gi, it, solver))
                except (ValueError, AttributeError) as ex:
                    r["err"] = str(ex)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                out.append(r)
            solver.close()
            print(f"  board {k}/{len(todo)}", file=sys.stderr, end="\r")
    with open(CACHE, "w") as fh:                          # compact: one record per current episode
        fh.writelines(json.dumps(r) + "\n" for r in out)
    return out


# ------------------------------------------------------------------ report
def pool(R):
    """Pooled rates of a group of episode records."""
    ok = [r for r in R if "err" not in r]
    att = sum(r["attempts"] for r in ok)
    T = sum(r["T"] for r in ok)
    opt_n, opt_d = (sum(r["opt"][i] for r in ok) for i in range(2))
    F = [sum(r["F"][i] for r in ok) for i in range(3)]
    G = [sum(r["G"][i] for r in ok) for i in range(3)]
    st = [r["stall"] for r in ok if r["outcome"] == "success" and r["stall"] is not None]
    tops = [max(r["by_seat"]) / (r["T"] - r["A"]) for r in ok
            if r["n"] > 1 and r["outcome"] == "success" and r["T"] > r["A"] and r["sum_ok"]]
    return dict(
        n=len(ok), err=len(R) - len(ok),
        illegal=sum(r["illegal"] for r in ok) / att if att else None,
        blocked=sum(r["blocked"] for r in ok) / att if att else None,
        osc=sum(r["osc"] for r in ok) / T if T else None,
        stall=sum(st) / len(st) if st else None,
        opt=opt_n / opt_d if opt_d else None,
        accF=F[1] / F[0] if F[0] else None, chF=F[2] / F[0] if F[0] else None, nF=F[0],
        accG=G[1] / G[0] if G[0] else None, chG=G[2] / G[0] if G[0] else None, nG=G[0],
        top=sum(tops) / len(tops) if tops else None, ntop=len(tops),
        unknown=sum(r["unknown"] + r["undetermined"] for r in ok),
    )


def f3(x):
    return "    -" if x is None else f"{x:5.3f}"


def report(R):
    print("Spatial efficiency breakdown, final arm (pooled rates; stall = mean on successes;")
    print("top = mean largest player share of the extra moves on successes, vs 1/n)")
    hdr = f"{'model':8s} {'suite':5s} {'n':>4s} {'illeg':>5s} {'block':>5s} {'osc':>5s} {'stall':>5s} " \
          f"{'opt':>5s} {'accF':>5s} {'(ch)':>5s} {'nF':>5s} {'accG':>5s} {'(ch)':>5s} {'top':>5s} {'1/n':>5s}"
    print(hdr)
    for model in sorted({r["model"] for r in R}):
        for suite in SUITES:
            g = [r for r in R if r["model"] == model and r["suite"] == suite]
            if not g:
                continue
            p = pool(g)
            n = {"solo": 1, "dyad": 2, "3p": 3, "6p": 6}[suite]
            print(f"{model:8s} {suite:5s} {p['n']:4d} {f3(p['illegal'])} {f3(p['blocked'])} {f3(p['osc'])} "
                  f"{f3(p['stall'])} {f3(p['opt'])} {f3(p['accF'])} {f3(p['chF'])} {p['nF']:5d} "
                  f"{f3(p['accG'])} {f3(p['chG'])} {f3(p['top'])} {1 / n:5.3f}"
                  + (f"  [{p['err']} replay errors]" if p["err"] else "")
                  + (f"  [{p['unknown']} states over the cap]" if p["unknown"] else ""))
    print("\nby level, all models pooled: opt / accF (chance, nF) / accG (chance)")
    for lv in LEVELS:
        cells = []
        for suite in SUITES:
            p = pool([r for r in R if r["suite"] == suite and r.get("level") == lv])
            cells.append(f"{suite} {f3(p['opt'])} {f3(p['accF'])} ({f3(p['chF'])}, {p['nF']:4d}) "
                         f"{f3(p['accG'])} ({f3(p['chG'])})")
        print(f"{lv}  " + " | ".join(cells))


def checks(R):
    ok = [r for r in R if "err" not in r]
    bad_h0 = [r["path"] for r in ok if r["h0"] != r["A"]]
    bad_sum = [r["path"] for r in ok if r["sum_ok"] is False]
    errs = [(r["path"], r["err"]) for r in R if "err" in r]
    print(f"\nchecks: {len(ok)} episodes replayed, {len(errs)} replay errors, "
          f"h*(s_0) != A_k on {len(bad_h0)}, sum r != T - A_k on {len(bad_sum)} successes")
    for x in (bad_h0[:5] + bad_sum[:5] + errs[:5]):
        print("  ", x)
    return not (bad_h0 or bad_sum or errs)


def tex(R):
    body = []
    for suite in SUITES:
        for model in sorted({r["model"] for r in R}):
            g = [r for r in R if r["model"] == model and r["suite"] == suite]
            if not g:
                continue
            p = pool(g)
            pc = lambda x: "--" if x is None else f"{100 * x:.1f}"       # noqa: E731
            dc = lambda x: "--" if x is None else f"{x:.3f}"             # noqa: E731
            body.append(f"{suite} & {LABEL.get(model, model)} & {p['n']} & {pc(p['illegal'])} & {pc(p['blocked'])} & "
                        f"{dc(p['osc'])} & {dc(p['stall'])} & {dc(p['opt'])} & "
                        f"{dc(p['accF'])} ({dc(p['chF'])}) & {dc(p['accG'])} ({dc(p['chG'])})")
        body.append(r"\midrule")
    return table(
        "Spatial efficiency breakdown on the final arm. Illegal and blocked moves in \\% of move "
        "attempts; oscillation, optimal-move rate and detour accuracies pooled over moves; stall rate "
        "the mean over successful episodes. Chance levels in parentheses. Exploratory.",
        "tab:move-metrics", "llrrrrrrrr",
        " & model & $n$ & illegal & blocked & osc & stall & opt & $\\textit{acc}_F$ & $\\textit{acc}_G$",
        body[:-1])


# ------------------------------------------------------------------ self-check
def selfcheck(cap):
    """hstar_c against solver.solve from random reachable states."""
    rng = random.Random(0)
    files = [(suite, f"matrixgame/in/instances_{suite}.json") for suite in ("solo", "dyad", "3p", "6p")]
    n = 0
    for suite, f in files:
        boards = [gi for e in json.load(open(os.path.join(ROOT, f)))["experiments"]
                  for gi in e["game_instances"]]
        for gi in rng.sample(boards, 4):
            exact, bounded = Solver(gi, cap), Solver(gi, cap)
            s = (tuple(exact.starts), 0, 0)
            for _ in range(rng.randrange(0, 40)):
                L = legal(exact, s)
                if not L or s[0] == tuple(exact.targets):
                    break
                s = rng.choice(L)[1]
            want = SF.solve(gi, cap=2_000_000, start=s)[0]
            got = exact.hs([s])[0]
            if want != "cap" and got is not None:
                assert (INF if want is None else want) == got, (suite, gi["spatial_level"], gi["game_id"], s, want, got)
                if got not in (0, INF):     # the bounded search: h* > got - 2, h* <= got
                    bounded.run([(s, got - 2)])
                    assert bounded.within(s, got - 2) is False
                    bounded.run([(s, got)])
                    assert bounded.within(s, got) is True
                n += 1
            exact.close()
            bounded.close()
    print(f"hstar_c agrees with solver.solve on {n} random states")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=16_000_000,
                    help="expansions per solver query (16M solves every state of the final arm, ~400 MB peak)")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.access(BINARY, os.X_OK):
        sys.exit(f"build the solver first: cc -O3 -o {BINARY} {BINARY[:-2]}.c")
    if a.check:
        selfcheck(a.cap)
    R = compute(discover(), a.cap, a.refresh)
    report(R)
    good = checks(R)
    if a.tex:
        out = os.path.join(ROOT, "thesis/numbers/tab_move_metrics.final.tex")
        open(out, "w").write(tex(R))
        print("wrote", out)
    if a.check and not good:
        sys.exit(1)


if __name__ == "__main__":
    main()
