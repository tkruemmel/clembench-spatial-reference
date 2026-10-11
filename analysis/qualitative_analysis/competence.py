"""Every number of the competence reading ("What the agents can and cannot do")
and of tab:competence, on one arm.

    ARM=updated python3 competence.py [--check]       # run from this directory

Scope words on every line: "all" = the 2,520 episodes of the three models
(solo original, team suites of ARM); "strong" = Qwen3.8 + DeepSeek-V4.1 only.

Reads episodes/aborts/corpus.{ARM}.json (scan.py, forced.py, qual_corpus.py),
remaining.{ARM}.json (remaining.py), f6_tiers/claims/claims.{ARM}.json
(claims.py) and the attainable optima (analysis/thesis/tables_comm.py); runs
dominated.py, vacate.py, selftrap.py, forced2.py, promises3.py and, read-only,
f6_tiers/rejection/{rep4,retry}.py, and parses their printed lines.

Definitions fixed here (no model output is judged):
  at the optimum     a success whose move count equals the suite's attainable
                     optimum A_k (solo: the free-order joint optimum J).
  enclosure moment   after some accepted move (or at the start) a 6p object has
                     no free orthogonal neighbour (wall, edge or object).
  temporarily        an episode "contains the word" if any broadcast matches
                     qual_corpus.TEMP; universe = team episodes with >=1 message;
                     an episode "produces a temporary shift" if any object ever
                     leaves its own target (scan.py's depart_target).
  destination claim  an own-object claim whose matched CLAIM span (letter ...
                     "at" (Rr,Cc)) contains to/toward(s)/into/goal/target/
                     destination/home, e.g. "C right to my goal at (R3,C4)".
  S4 #8 minimum      A's 9 moves, the count in every solo run.
"""
import ast, collections, json, os, re, statistics as st, subprocess, sys
from scipy.stats import fisher_exact

ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO = "/Users/tom/uni/clembench-spatial-reference"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "analysis/thesis"))
import tables_comm  # noqa: E402  (load_optima only)

MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
STRONG = ("qwen38", "dsv41")
DEST = re.compile(r"\b(to|towards?|into|goal|target|destination|home)\b", re.I)
V = {}


def put(key, val, what):
    V[key] = val
    print(f"{key:<34} {str(val):<26} {what}")


def run(script, cwd=HERE):
    env = dict(os.environ, ARM=ARM)
    return subprocess.run([sys.executable, script], cwd=cwd, env=env, capture_output=True,
                          text=True, check=True).stdout


def pct(a, b, d=1):
    return f"{a}/{b} ({100 * a / b:.{d}f}%)" if b else f"{a}/0"


os.chdir(HERE)
R = json.load(open(f"episodes.{ARM}.json"))
S = [r for r in R if r["model"] in STRONG]
gid = lambda r: int(r["inst"].split("_")[1])
key = lambda r: (r["model"], r["suite"], r["level"], r["proto"], gid(r))

# ------------------------------------------------------------------ scope
boards = collections.Counter((r["level"], r["inst"]) for r in R)
prints = collections.defaultdict(set)
for r in R:
    prints[(r["level"], r["inst"])].add(r["board"])
put("scope.episodes", len(R), "all episodes")
put("scope.boards", len(boards), "distinct (level, board)")
put("scope.runs_per_board", sorted(set(boards.values())), "episodes per board")
put("scope.board_fingerprints", max(len(v) for v in prints.values()), "max distinct layouts per board id (1 = identical everywhere)")
put("scope.strong_episodes", len(S), "strong-tier episodes")

# ------------------------------------------------------------------ legality (tab row 1)
mv, il, bl = sum(r["moves"] for r in S), sum(r["illegal"] for r in S), sum(r["blocked"] for r in S)
put("legal.strong_moves", mv, "strong: accepted moves")
put("legal.strong_wall_offgrid", il, f"strong: wall/off-grid rejections = {100 * il / mv:.3f}% of moves")
put("legal.strong_occupied", bl, f"strong: occupied-cell rejections; all rejections {100 * (il + bl) / mv:.3f}% of moves")
put("legal.solo_occupied", sum(r["blocked"] for r in S if r["suite"] == "solo"), "strong: occupied-cell rejections in solo")
put("coll.strong_by_suite", {s: sum(r["blocked"] for r in S if r["suite"] == s) for s in ("solo", "dyad", "3p", "6p")}, "strong: occupied-cell attempts by suite")
for m in ("qwen38", "dsv41", "nemotron"):
    E = [r for r in R if r["model"] == m]
    put(f"clean.wall_{m}", pct(sum(r["illegal"] == 0 for r in E), len(E)), f"{m}: episodes with no wall/off-grid rejection")
    put(f"clean.any_{m}", pct(sum(r["illegal"] + r["blocked"] == 0 for r in E), len(E)), f"{m}: episodes with no rejected move of any kind")
put("clean.wall_strong", pct(sum(r["illegal"] == 0 for r in S), len(S)), "strong: episodes with no wall/off-grid rejection")

# ------------------------------------------------------------------ optimum (tab row 2)
OPT = tables_comm.load_optima()
win = [r for r in R if r["outcome"] == "success"]
at_k = [r for r in win if r["moves"] == OPT[(r["suite"], r["level"], gid(r))]]
put("opt.below_Ak", sum(r["moves"] < OPT[(r["suite"], r["level"], gid(r))] for r in win), "successes below A_k (must be 0)")
put("opt.at_Ak_all", len(at_k), "all: successes at the attainable optimum A_k")
put("opt.at_Ak_strong", sum(r["model"] in STRONG for r in at_k), "strong: successes at A_k")
put("opt.at_Ak_S2S4", sum(r["level"] != "S1" for r in at_k), "all: of those, at S2-S4 (boards with walls)")
put("opt.at_Ak_by_suite", dict(collections.Counter(r["suite"] for r in at_k)), "all: at A_k by suite")
put("opt.at_Ak_S4_by_suite", dict(collections.Counter(r["suite"] for r in at_k if r["level"] == "S4")), "all: at A_k at S4 by suite")
put("opt.at_J_all", sum(r["moves"] == r["exact"] for r in win), "all: successes at the free-order joint optimum J (old '320')")

# ------------------------------------------------------------------ S4 #8 and S4 #13 (tab row 3)
s48 = {s: [(r["model"], r["proto"], r["nmoves"].get("A", 0), r["moves"], r["outcome"])
           for r in S if r["level"] == "S4" and r["inst"] == "instance_00008" and r["suite"] == s]
       for s in ("solo", "dyad", "3p", "6p")}
put("s4_8.solo_A_moves", sorted(x[2] for x in s48["solo"]), "strong solo: A's moves on S4 #8 (the 9-move minimum)")
for s in ("dyad", "3p", "6p"):
    won = [x[2] for x in s48[s] if x[4] == "success"]
    put(f"s4_8.{s}_A_moves_won", f"{min(won)}-{max(won)} (n={len(won)})" if won else "-", f"strong {s}: A's moves in successes")
    put(f"s4_8.{s}_outcomes", dict(collections.Counter(x[4] for x in s48[s])), f"strong {s}: outcomes on S4 #8 (of {len(s48[s])})")
put("s4_8.3p_max_A", max(x[2] for x in s48["3p"]), "strong 3p: most moves A spends (excess over 9 = pacing)")
ab23 = [x for x in s48["6p"] if x[4] == "abort" and x[3] == 23]
put("s4_8.6p_abort_at_23", len(ab23), f"strong 6p: aborts at exactly move 23 {sorted((m, p) for m, p, *_ in ab23)}")
ab12 = [r for r in R if r["level"] == "S4" and r["inst"] == "instance_00013" and r["suite"] == "6p"
        and r["outcome"] == "abort" and r["moves"] == 12]
put("s4_13.6p_abort_at_12", len(ab12), f"all 6p: aborts at exactly move 12 {sorted((r['model'], r['proto']) for r in ab12)}")
put("s3_10.strong_success", {s: pct(sum(r["outcome"] == "success" for r in E), len(E), 0) for s in ("solo", "dyad", "3p", "6p")
                             for E in [[r for r in S if r["level"] == "S3" and r["inst"] == "instance_00010" and r["suite"] == s]]},
    "strong: S3 #10 successes by suite")

# ------------------------------------------------------------------ idle and reversal (tab row 4)
out = run("dominated.py")
for s, y, n in re.findall(r"(\w+)\s*:\s+(\d+) of\s+(\d+) reversals", out):
    put(f"idle.dominated_{s}", pct(int(y), int(n), 0), f"strong {s}: reversal moves strictly dominated")
med = lambda E: round(100 * st.median(2 * r["revpairs"] / r["moves"] for r in E), 1)
SW = [r for r in S if r["outcome"] == "success" and r["moves"] >= 10]
put("rev.by_suite", [med([r for r in SW if r["suite"] == s]) for s in ("solo", "dyad", "3p", "6p")], "strong successes: median reversal share %, solo/dyad/3p/6p")
put("rev.by_level", [med([r for r in SW if r["level"] == lv]) for lv in ("S1", "S2", "S3", "S4")], "strong successes: median reversal share %, S1-S4")
for lv in ("S1", "S4"):
    E = [r for r in SW if r["level"] == lv]
    put(f"rev.zero_{lv}", f"{sum(r['revpairs'] == 0 for r in E)}/{len(E)}", f"strong {lv} successes with no reversal")
MP = [r for r in S if r["suite"] != "solo" and r["moves"] >= 10]
put("rev.win_vs_loss", [med([r for r in MP if r["outcome"] == o]) for o in ("success", "lose")], "strong team episodes: median reversal share %, successes vs losses")

# ------------------------------------------------------------------ completion (tab row 5)
fd = [r for r in R if r["bad_done"]]
put("done.fatal", len(fd), f"all: losses ended by a done with an own object off target, by model {dict(collections.Counter(r['model'] for r in fd))}")
put("done.fatal_outcome", dict(collections.Counter(r["outcome"] for r in fd)), "how scores.json scores them")
put("done.fatal_at_move0", len([r for r in fd if r["moves"] == 0]), "all: of those, declared before any move")
last = 0
for r in fd:  # the declarer's own objects still off target, and how far
    who = re.match(r"Player (\w)", r["end_msg"]).group(1)
    off = [r["dist_end"][o] for o, p in r["owner"].items() if p == who and r["dist_end"][o]]
    last += off == [1]
put("done.fatal_last_cell", f"{last}/{len(fd)}", "all: false dones whose declarer had exactly one object off target, one cell out")
for t, E in (("strong", S), ("nemotron", [r for r in R if r["model"] == "nemotron"])):
    put(f"done.decls_{t}", f"{sum(len(r['done_decl']) for r in E)} valid, {sum(r['bad_done'] for r in E)} false", f"{t}: done declarations")
put("cut.strong_cycle_losses_3p6p", sum(r["outcome"] == "lose" and r["end"] == "lose" and r["end_msg"].startswith("Board state repeated")
                                        for r in S if r["suite"] in ("3p", "6p")), "strong 3p+6p: cycle losses (cut.py counts those ending on a done cut)")
for r in fd:
    if r["model"] in STRONG:
        print(f"{'':34} strong fatal done: {r['model']} {r['suite']} {r['level']} {r['proto']} #{gid(r)} "
              f"after {r['moves']}/{r['max_turns']} moves, off target {[k for k, v in r['dist_end'].items() if v]} ({r['end_msg'][:9]})")
REM = [x for x in json.load(open(f"remaining.{ARM}.json")) if x["outcome"] == "lose"]
put("rem.losses", len(REM), "all: losses")
put("rem.one_move", sum(x["remaining"] == 1 for x in REM), "all: losses one move from a win (exact solver)")
put("rem.within_two", sum(x["remaining"] is not None and x["remaining"] <= 2 for x in REM), "all: losses within two moves of a win")
put("rem.winnable", sum(x["remaining"] is not None for x in REM), "all: losses still solvable when they ended")

# ------------------------------------------------------------------ least-damaging action (tab row 6)
out = run("vacate.py")
d = {(o, c): int(n) for o, c, n in re.findall(r"^\s+(\w+)\s+(returned in 1 own move|never returned|returned in 2-3|returned later)\s+(\d+)", out, re.M)}
tot = collections.Counter()
for (o, _), n in d.items():
    tot[o] += n
put("vac.win_undone_next", pct(d[("success", "returned in 1 own move")], tot["success"], 0), "all successes: target departures undone on the object's next own move")
put("vac.loss_undone_next", d[("lose", "returned in 1 own move")], f"all losses: departures undone next move (of {tot['lose']})")
put("vac.loss_never", d.get(("lose", "never returned"), 0), "all losses: departures never undone")

# ------------------------------------------------------------------ rejection (tab row 7)
A = json.load(open(f"aborts.{ARM}.json"))
put("rej.aborts", len(A), f"all aborts, by model {dict(collections.Counter(a['model'] for a in A))}")
had = [a for a in A if a["n_legal_own"] > 0]
put("rej.aborts_with_legal_move", len(had), f"aborting player had a legal move, by model {dict(collections.Counter(a['model'] for a in had))}")
put("rej.median_legal_by_suite", {s: st.median(a["n_legal_own"] for a in had if a["suite"] == s) for s in ("solo", "dyad", "3p", "6p")},
    "median legal moves available at those aborts")
z = [r for r in R if r["outcome"] == "abort" and r["moves"] == 0]
put("rej.zero_move_aborts", len(z), f"aborts with no accepted move, by model {dict(collections.Counter(r['model'] for r in z))}")
put("rej.zero_move_budget_pct", round(100 * st.median((r["illegal"] + r["blocked"] + r["bad_format"]) / r["max_turns"] for r in z), 1),
    "median attempts as % of the move budget at those aborts")
out = run("rep4.py", cwd=os.path.join(HERE, "f6_tiers/rejection"))
for k in ("d_both_inv_gm_full", "f_final_run_same_gm_anytype"):
    s_, w_ = re.search(k + r" S (\d+) W (\d+)", out).groups()
    put(f"rej.{k[:1]}_repeat_abort", f"S {s_} W {w_}", "aborts with a byte-identical rejection back-to-back (d: anywhere, f: in the final run)")
out = run("retry.py", cwd=os.path.join(HERE, "f6_tiers/rejection"))
for m, block in re.findall(r"^(\w+) episodes with >=1 rejected move: .*?\n   legal alt available rejections followed by a retry: (\d+ \{[^}]*\})", out, re.M):
    n = int(block.split()[0]); c = ast.literal_eval(block[block.index("{"):])
    put(f"rej.retry_{m}", f"{n} rej; repeat {c.get('repeat', 0)}; revised+accepted {c.get('revised_ok', 0)}",
        f"{m}: rejections with a legal alternative followed by another attempt")

# ------------------------------------------------------------------ enclosures (tab row 8)
out = run("selftrap.py")
n_ = int(re.search(r"\(n=(\d+)\)", out).group(1))
m_ = re.search(r"(\d+)\s+a teammate sealing", out)
put("encl.sealed_by_teammate", f"{int(m_.group(1)) if m_ else 0}/{n_}", "all: zero-legal-move aborts whose last exit a teammate closed")
out = run("forced2.py")
put("encl.done_neighbour", re.search(r"declared DONE: (\d+ / \d+)", out).group(1).replace(" ", ""), "all: of those, an enclosing neighbour's owner had declared done")
done_rows = [ln for ln in out.splitlines() if re.match(r"(qwen38|dsv41) ", ln)]
put("encl.strong_done_neighbour", f"{sum('(done)' in ln for ln in done_rows)}/{len(done_rows)}", "strong: the same")


def enclosure_moments(r):
    """Per accepted move: the objects with no free neighbour, and whether each is on target."""
    d = os.path.join(REPO, r["path"])
    ins = json.load(open(os.path.join(d, "instance.json")))
    n, walls = ins["grid_size"], {tuple(w) for w in ins["walls"]}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}

    def shut():
        occ = set(pos.values())
        return [(o, pos[o] == tgt[o]) for o, (a, b) in pos.items()
                if not any(0 <= a + x < n and 0 <= b + y < n and (a + x, b + y) not in walls and (a + x, b + y) not in occ
                           for x, y in DIRS.values())]
    seen = [shut()]
    for t in json.load(open(os.path.join(d, "interactions.json")))["turns"]:
        for e in t:
            a = e.get("action", {})
            m = MOVE.match(a.get("content", "")) if a.get("type") == "move" else None
            if m:
                o, dr = m.group(2), m.group(3)
                pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
                seen.append(shut())
    return seen


for tier, E in (("all", [r for r in R if r["suite"] == "6p"]), ("strong", [r for r in S if r["suite"] == "6p"])):
    off, anyenc, okwin = collections.Counter(), collections.Counter(), 0
    for r in E:
        mom = [x for step in enclosure_moments(r) for x in step]
        if any(not on for _, on in mom):
            off[r["outcome"]] += 1
        if mom:
            anyenc[r["outcome"]] += 1
    put(f"jam.{tier}_offtarget_enclosed", dict(off), f"{tier} 6p: episodes with a moment an off-target object is enclosed, by outcome")
    put(f"jam.{tier}_any_enclosed_won", f"{anyenc['success']}/{sum(r['outcome'] == 'success' for r in E)}",
        f"{tier} 6p successes with any enclosure moment ({anyenc['success'] - off['success']} of them only on-target)")

# ------------------------------------------------------------------ the social layer
Cq = json.load(open(f"corpus.{ARM}.json"))
for m in ("qwen38", "dsv41", "nemotron"):
    cells = []
    for p in ("structured", "freeform", "hybrid"):
        e = [x for x in Cq["epi"] if x["model"] == m and x["proto"] == p]
        rq, cp = sum(x["requested"] for x in e), sum(x["complied"] for x in e)
        cells.append(f"{p[:2]} {100 * cp / rq:.1f}" if rq else f"{p[:2]} -")
    put(f"req.within6_{m}", " ".join(cells), f"{m}: requested moves performed within six turns, %")
out = run("promises3.py")
n_, k_, b_ = map(int, re.search(r"free \(not the speaker's own target\): (\d+)\n\s+kept (\d+) \(\d+%\)\s+broken (\d+)", out).groups())
c_ = int(re.search(r"of the broken ones: (\d+)", out).group(1))
put("res.reservations", f"{k_}/{n_} kept ({100 * k_ / n_:.0f}%)", "all: named-cell reservations honoured")
put("res.breaches", f"{b_} broken, {c_} consequential", "all: breaches, and those landing on a teammate's target or a cell a teammate is then rejected from")

# ------------------------------------------------------------------ caveat one: "temporarily"
EPI = {(x["model"], x["suite"], x["lvl"], x["proto"], x["gid"]): x for x in Cq["epi"]}
said = collections.defaultdict(bool)
for r in Cq["rows"]:
    said[(r["model"], r["suite"], r["lvl"], r["proto"], r["gid"])] |= r["temp"]
EK = {key(r): r for r in R}


def split(keys):
    w = [k for k in keys if said[k]]; wo = [k for k in keys if not said[k]]
    a, b = sum(EPI[k]["su"] for k in w), sum(EPI[k]["su"] for k in wo)
    p = fisher_exact([[a, len(w) - a], [b, len(wo) - b]])[1]
    return f"with {a}/{len(w)} ({100 * a / len(w):.0f}%) without {b}/{len(wo)} ({100 * b / len(wo):.0f}%) p={p:.2g}"


U = [k for k, x in EPI.items() if x["nmsg"] > 0]
put("temp.pooled", split(U), "all team episodes with messages: success with vs without 'temporar(il)y' (Fisher)")
for s in ("dyad", "3p", "6p"):
    for p in ("freeform", "hybrid"):
        put(f"temp.strong_{s}_{p}", split([k for k in EPI if k[0] in STRONG and k[1] == s and k[3] == p]), f"strong {s} {p}")
T = [k for k in U if said[k]]
put("temp.ask_produce", f"{sum(bool(EK[k]['depart_target']) for k in T)}/{len(T)}", "episodes with the word in which some object ever leaves its own target")

# ------------------------------------------------------------------ caveat two: destination-framed own claims
CL = [c for c in json.load(open(f"f6_tiers/claims/claims.{ARM}.json")) if c["proto"] in ("freeform", "hybrid") and c["own"]]
corpus_own = sum(r["own_ok"] + r["own_bad"] for r in Cq["rows"] if r["proto"] in ("freeform", "hybrid"))
put("dest.claims_match_corpus", len(CL) == corpus_own, f"claims cache own-claim count {len(CL)} equals corpus {corpus_own}")
share = {}
for m in ("nemotron", "qwen38", "dsv41"):
    for p in ("freeform", "hybrid"):
        c = [x for x in CL if x["model"] == m and x["proto"] == p]
        share[f"{m[:4]}-{p[:2]}"] = round(100 * sum(bool(DEST.search(x["span"])) for x in c) / len(c))
put("dest.share_pct", share, "own-object claims that read a destination as a position, % by model-protocol")
for s in ("dyad", "3p", "6p"):
    c = [x for x in CL if x["model"] in STRONG and x["suite"] == s]
    k = [x for x in c if not DEST.search(x["span"])]
    put(f"dest.strong_{s}", f"{sum(x['ok'] for x in c) / len(c):.3f} -> {sum(x['ok'] for x in k) / len(k):.3f}",
        f"strong {s}: own-claim accuracy, all claims -> destination claims excluded (n {len(c)} -> {len(k)})")

# ------------------------------------------------------------------ check
PIN = {
    "original": {   'scope.episodes': 2520,
        'scope.boards': 60,
        'scope.runs_per_board': [42],
        'scope.board_fingerprints': 1,
        'scope.strong_episodes': 1680,
        'legal.strong_moves': 90684,
        'legal.strong_wall_offgrid': 76,
        'legal.strong_occupied': 118,
        'legal.solo_occupied': 0,
        'coll.strong_by_suite': {'solo': 0, 'dyad': 3, '3p': 6, '6p': 109},
        'clean.wall_qwen38': '823/840 (98.0%)',
        'clean.any_qwen38': '805/840 (95.8%)',
        'clean.wall_dsv41': '794/840 (94.5%)',
        'clean.any_dsv41': '762/840 (90.7%)',
        'clean.wall_nemotron': '296/840 (35.2%)',
        'clean.any_nemotron': '14/840 (1.7%)',
        'clean.wall_strong': '1617/1680 (96.2%)',
        'opt.below_Ak': 0,
        'opt.at_Ak_all': 406,
        'opt.at_Ak_strong': 401,
        'opt.at_Ak_S2S4': 139,
        'opt.at_Ak_by_suite': {'solo': 80, 'dyad': 116, '3p': 98, '6p': 112},
        'opt.at_Ak_S4_by_suite': {'solo': 9, 'dyad': 15, '3p': 11, '6p': 26},
        'opt.at_J_all': 320,
        's4_8.solo_A_moves': [9, 9, 9, 9],
        's4_8.dyad_A_moves_won': '9-23 (n=8)',
        's4_8.dyad_outcomes': {'success': 8},
        's4_8.3p_A_moves_won': '9-41 (n=7)',
        's4_8.3p_outcomes': {'success': 7, 'lose': 1},
        's4_8.6p_A_moves_won': '11-21 (n=4)',
        's4_8.6p_outcomes': {'abort': 4, 'success': 4},
        's4_8.3p_max_A': 41,
        's4_8.6p_abort_at_23': 4,
        's4_13.6p_abort_at_12': 5,
        's3_10.strong_success': {'solo': '4/4 (100%)', 'dyad': '8/8 (100%)', '3p': '5/8 (62%)', '6p': '0/8 (0%)'},
        'idle.dominated_solo': '485/541 (90%)',
        'idle.dominated_dyad': '997/1833 (54%)',
        'idle.dominated_3p': '602/2574 (23%)',
        'idle.dominated_6p': '0/3032 (0%)',
        'rev.by_suite': [4.4, 4.9, 6.8, 8.7],
        'rev.by_level': [0.0, 7.8, 10.8, 9.0],
        'rev.zero_S1': '303/416',
        'rev.zero_S4': '6/373',
        'rev.win_vs_loss': [6.9, 30.0],
        'done.fatal': 16,
        'done.fatal_outcome': {'lose': 16},
        'done.fatal_at_move0': 2,
        'done.fatal_last_cell': '9/16',
        'done.decls_strong': '3392 valid, 2 false',
        'done.decls_nemotron': '100 valid, 14 false',
        'cut.strong_cycle_losses_3p6p': 91,
        'rem.losses': 187,
        'rem.one_move': 5,
        'rem.within_two': 7,
        'rem.winnable': 187,
        'vac.win_undone_next': '474/541 (88%)',
        'vac.loss_undone_next': 238,
        'vac.loss_never': 26,
        'rej.aborts': 781,
        'rej.aborts_with_legal_move': 742,
        'rej.median_legal_by_suite': {'solo': 13, 'dyad': 7, '3p': 4, '6p': 2},
        'rej.zero_move_aborts': 206,
        'rej.zero_move_budget_pct': 1.7,
        'rej.d_repeat_abort': 'S 11 W 570',
        'rej.f_repeat_abort': 'S 11 W 551',
        'rej.retry_qwen38': '21 rej; repeat 0; revised+accepted 21',
        'rej.retry_dsv41': '81 rej; repeat 0; revised+accepted 80',
        'rej.retry_nemotron': '2566 rej; repeat 1027; revised+accepted 920',
        'encl.sealed_by_teammate': '39/39',
        'encl.done_neighbour': '13/39',
        'encl.strong_done_neighbour': '13/32',
        'jam.all_offtarget_enclosed': {'abort': 93, 'lose': 16, 'success': 98},
        'jam.all_any_enclosed_won': '118/408',
        'jam.strong_offtarget_enclosed': {'abort': 32, 'success': 98, 'lose': 12},
        'jam.strong_any_enclosed_won': '118/402',
        'req.within6_qwen38': 'st 88.8 fr 91.4 hy 92.0',
        'req.within6_dsv41': 'st 78.1 fr 86.0 hy 96.2',
        'req.within6_nemotron': 'st - fr 50.0 hy 40.0',
        'res.reservations': '9053/10614 kept (85%)',
        'res.breaches': '1561 broken, 152 consequential',
        'temp.pooled': 'with 281/329 (85%) without 761/1154 (66%) p=1.1e-12',
        'temp.strong_dyad_freeform': 'with 50/50 (100%) without 70/70 (100%) p=1',
        'temp.strong_dyad_hybrid': 'with 41/42 (98%) without 78/78 (100%) p=0.35',
        'temp.strong_3p_freeform': 'with 51/51 (100%) without 69/69 (100%) p=1',
        'temp.strong_3p_hybrid': 'with 38/38 (100%) without 82/82 (100%) p=1',
        'temp.strong_6p_freeform': 'with 47/54 (87%) without 64/66 (97%) p=0.077',
        'temp.strong_6p_hybrid': 'with 47/51 (92%) without 69/69 (100%) p=0.03',
        'temp.ask_produce': '103/329',
        'dest.claims_match_corpus': True,
        'dest.share_pct': {   'nemo-fr': 2,
                              'nemo-hy': 2,
                              'qwen-fr': 19,
                              'qwen-hy': 16,
                              'dsv4-fr': 14,
                              'dsv4-hy': 13},
        'dest.strong_dyad': '0.799 -> 0.888',
        'dest.strong_3p': '0.841 -> 0.914',
        'dest.strong_6p': '0.489 -> 0.902'},
    "updated": {   'scope.episodes': 2520,
        'scope.boards': 60,
        'scope.runs_per_board': [42],
        'scope.board_fingerprints': 1,
        'scope.strong_episodes': 1680,
        'legal.strong_moves': 90750,
        'legal.strong_wall_offgrid': 60,
        'legal.strong_occupied': 111,
        'legal.solo_occupied': 0,
        'coll.strong_by_suite': {'solo': 0, 'dyad': 2, '3p': 3, '6p': 106},
        'clean.wall_qwen38': '824/840 (98.1%)',
        'clean.any_qwen38': '810/840 (96.4%)',
        'clean.wall_dsv41': '801/840 (95.4%)',
        'clean.any_dsv41': '771/840 (91.8%)',
        'clean.wall_nemotron': '306/840 (36.4%)',
        'clean.any_nemotron': '16/840 (1.9%)',
        'clean.wall_strong': '1625/1680 (96.7%)',
        'opt.below_Ak': 0,
        'opt.at_Ak_all': 414,
        'opt.at_Ak_strong': 409,
        'opt.at_Ak_S2S4': 130,
        'opt.at_Ak_by_suite': {'solo': 80, 'dyad': 115, '3p': 106, '6p': 113},
        'opt.at_Ak_S4_by_suite': {'solo': 9, 'dyad': 14, '3p': 12, '6p': 25},
        'opt.at_J_all': 320,
        's4_8.solo_A_moves': [9, 9, 9, 9],
        's4_8.dyad_A_moves_won': '9-13 (n=7)',
        's4_8.dyad_outcomes': {'success': 7, 'lose': 1},
        's4_8.3p_A_moves_won': '9-37 (n=8)',
        's4_8.3p_outcomes': {'success': 8},
        's4_8.6p_A_moves_won': '11-19 (n=3)',
        's4_8.6p_outcomes': {'success': 3, 'abort': 5},
        's4_8.3p_max_A': 37,
        's4_8.6p_abort_at_23': 4,
        's4_13.6p_abort_at_12': 4,
        's3_10.strong_success': {   'solo': '4/4 (100%)',
                                    'dyad': '8/8 (100%)',
                                    '3p': '5/8 (62%)',
                                    '6p': '1/8 (12%)'},
        'idle.dominated_solo': '485/541 (90%)',
        'idle.dominated_dyad': '833/1721 (48%)',
        'idle.dominated_3p': '594/2497 (24%)',
        'idle.dominated_6p': '0/3141 (0%)',
        'rev.by_suite': [4.4, 4.7, 5.9, 8.0],
        'rev.by_level': [0.0, 6.6, 10.5, 8.7],
        'rev.zero_S1': '300/416',
        'rev.zero_S4': '7/367',
        'rev.win_vs_loss': [6.2, 30.7],
        'done.fatal': 26,
        'done.fatal_outcome': {'lose': 26},
        'done.fatal_at_move0': 3,
        'done.fatal_last_cell': '9/26',
        'done.decls_strong': '3382 valid, 1 false',
        'done.decls_nemotron': '98 valid, 25 false',
        'cut.strong_cycle_losses_3p6p': 110,
        'rem.losses': 203,
        'rem.one_move': 3,
        'rem.within_two': 4,
        'rem.winnable': 203,
        'vac.win_undone_next': '474/566 (84%)',
        'vac.loss_undone_next': 160,
        'vac.loss_never': 18,
        'rej.aborts': 773,
        'rej.aborts_with_legal_move': 741,
        'rej.median_legal_by_suite': {'solo': 13, 'dyad': 6.0, '3p': 4.0, '6p': 2.0},
        'rej.zero_move_aborts': 206,
        'rej.zero_move_budget_pct': 1.4,
        'rej.d_repeat_abort': 'S 11 W 580',
        'rej.f_repeat_abort': 'S 11 W 561',
        'rej.retry_qwen38': '17 rej; repeat 0; revised+accepted 17',
        'rej.retry_dsv41': '65 rej; repeat 0; revised+accepted 65',
        'rej.retry_nemotron': '2554 rej; repeat 1064; revised+accepted 904',
        'encl.sealed_by_teammate': '32/32',
        'encl.done_neighbour': '9/32',
        'encl.strong_done_neighbour': '9/30',
        'jam.all_offtarget_enclosed': {'abort': 85, 'lose': 24, 'success': 96},
        'jam.all_any_enclosed_won': '121/399',
        'jam.strong_offtarget_enclosed': {'abort': 30, 'success': 96, 'lose': 18},
        'jam.strong_any_enclosed_won': '121/390',
        'req.within6_qwen38': 'st 82.8 fr 89.9 hy 91.6',
        'req.within6_dsv41': 'st 82.5 fr 90.5 hy 96.6',
        'req.within6_nemotron': 'st - fr 29.6 hy 45.5',
        'res.reservations': '8604/10131 kept (85%)',
        'res.breaches': '1527 broken, 196 consequential',
        'temp.pooled': 'with 259/304 (85%) without 776/1181 (66%) p=6.7e-12',
        'temp.strong_dyad_freeform': 'with 45/45 (100%) without 75/75 (100%) p=1',
        'temp.strong_dyad_hybrid': 'with 34/34 (100%) without 85/86 (99%) p=1',
        'temp.strong_3p_freeform': 'with 51/53 (96%) without 67/67 (100%) p=0.19',
        'temp.strong_3p_hybrid': 'with 36/39 (92%) without 81/81 (100%) p=0.033',
        'temp.strong_6p_freeform': 'with 46/50 (92%) without 65/70 (93%) p=1',
        'temp.strong_6p_hybrid': 'with 32/36 (89%) without 77/84 (92%) p=0.73',
        'temp.ask_produce': '109/304',
        'dest.claims_match_corpus': True,
        'dest.share_pct': {   'nemo-fr': 2,
                              'nemo-hy': 2,
                              'qwen-fr': 17,
                              'qwen-hy': 18,
                              'dsv4-fr': 14,
                              'dsv4-hy': 13},
        'dest.strong_dyad': '0.802 -> 0.895',
        'dest.strong_3p': '0.834 -> 0.911',
        'dest.strong_6p': '0.432 -> 0.856'},
}
if "--check" in sys.argv:
    bad = {k: (V.get(k), v) for k, v in PIN[ARM].items() if V.get(k) != v}
    for k, (got, want) in bad.items():
        print(f"CHECK FAIL {k}: got {got!r}, pinned {want!r}")
    assert PIN[ARM] and not bad, f"{len(bad)} of {len(PIN[ARM])} pinned values differ"
    print(f"check: all {len(PIN[ARM])} pinned values reproduce on ARM={ARM}")
