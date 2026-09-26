"""Per-episode failure/success anatomy over the three grid campaigns (2520 episodes)."""
import json, glob, os, re, sys, collections, string

REPO = "/Users/tom/uni/clembench-spatial-reference"
ARMS = {"nemotron": "results_nemotron_fixthink_grid_n15",
        "qwen38":   "results_qwen38_grid_n15",
        "dsv41":    "results_dsv41_grid_n15"}
SUITES = ("solo", "dyad", "3p", "6p")
MOVE = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def man(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def episode(f, model, suite):
    d = os.path.dirname(f)
    ins = json.load(open(os.path.join(d, "instance.json")))
    es = json.load(open(os.path.join(d, "scores.json")))["episode scores"]
    it = json.load(open(f))
    exp = os.path.basename(os.path.dirname(d))          # S3_freeform_grid / Solo_S1_Silent_Grid
    parts = exp.split("_")
    level = next(p for p in parts if re.fullmatch(r"S[1-4]", p))
    proto = parts[2].lower() if parts[0] == "Solo" else parts[1]

    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    start = dict(pos)
    groups = ins.get("player_objects") or (
        [ins["player_a_objects"], ins["player_b_objects"]] if "player_a_objects" in ins else [ins["objects"]])
    owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
    nplayers = len(groups)

    nmoves = collections.Counter()
    depart_target = collections.Counter()   # moves that leave the object's own target cell
    ever_home = set()
    first_home = {}
    tail, seq = [], []
    n_home_trace = []
    events = collections.Counter()
    end = None
    done_decl = []
    invalid_done = []
    mi = 0
    for ti, t in enumerate(it["turns"]):
        for e in t:
            a = e.get("action", {})
            ty, c = a.get("type"), a.get("content", "")
            if ty == "move":
                m = MOVE.match(c)
                if not m:
                    continue
                who, obj, dr = m.group(1) or owner[m.group(2)], m.group(2), m.group(3)
                if pos[obj] == tgt[obj]:
                    depart_target[obj] += 1
                pos[obj] = (pos[obj][0] + DIRS[dr][0], pos[obj][1] + DIRS[dr][1])
                nmoves[obj] += 1
                mi += 1
                if pos[obj] == tgt[obj] and obj not in ever_home:
                    ever_home.add(obj)
                    first_home[obj] = mi
                seq.append((mi, ti, who, obj, dr))
                n_home_trace.append(sum(pos[o] == tgt[o] for o in tgt))
            elif ty in ("invalid_move", "invalid_format", "invalid_done", "lose", "abort",
                        "success", "player_done"):
                events[ty] += 1
                if ty in ("lose", "abort"):
                    end = (ty, c)
                if ty == "success":
                    end = ("success", c)
                if ty == "player_done":
                    done_decl.append((mi, c))
                if ty == "invalid_done":
                    invalid_done.append((mi, c))

    home_end = [o for o in tgt if pos[o] == tgt[o]]
    outcome = "success" if es.get("Success") else ("abort" if es.get("Aborted") else "lose")
    return dict(
        model=model, suite=suite, level=level, proto=proto,
        inst=os.path.basename(d), path=os.path.relpath(d, REPO),
        nplayers=nplayers, outcome=outcome,
        end=end[0] if end else None, end_msg=end[1] if end else None,
        moves=mi, max_turns=ins["max_turns"], optimal=ins["optimal_moves"],
        n_obj=len(tgt), n_home_end=len(home_end), n_home_max=max(n_home_trace or [0]),
        home_end=sorted(home_end), ever_home=sorted(ever_home),
        never_moved=sorted(o for o in tgt if not nmoves[o]),
        never_moved_unsolved=sorted(o for o in tgt if not nmoves[o] and start[o] != tgt[o]),
        moved_away=sorted(o for o in tgt if man(pos[o], tgt[o]) > man(start[o], tgt[o])),
        depart_target=dict(depart_target),
        dist_end={o: man(pos[o], tgt[o]) for o in tgt},
        dist_start={o: man(start[o], tgt[o]) for o in tgt},
        nmoves=dict(nmoves), owner=owner, first_home=first_home,
        tail=[f"{w}:{o} {d_}" for _, _, w, o, d_ in seq[-8:]],
        illegal=it.get("Illegal Attempts", 0), blocked=it.get("Blocked Attempts", 0),
        bad_format=events["invalid_format"], bad_done=len(invalid_done),
        done_decl=done_decl,
        osc=es.get("Oscillation Count", 0), deadlock=es.get("Deadlock Flag", 0),
        pr_mean=es.get("Path Ratio Mean"), pr_max=es.get("Path Ratio Max"),
        main=es.get("Main Score"),
    )


rows = []
for model, pref in ARMS.items():
    for suite in SUITES:
        pat = os.path.join(REPO, f"{pref}_{suite}", "*", "*", "*", "instance_*", "interactions.json")
        for f in sorted(glob.glob(pat)):
            try:
                rows.append(episode(f, model, suite))
            except Exception as ex:
                print("ERR", f, ex, file=sys.stderr)
# board fingerprint (the same 60 boards recur across every suite and model) and the
# exact joint optimum from analysis/efficiency_baseline/joint_optimum.jsonl
import hashlib
JO = {(x["level"], x["game_id"]): x["opt"] for x in
      map(json.loads, open(os.path.join(REPO, "analysis/efficiency_baseline", "joint_optimum.jsonl")))}
for r in rows:
    ins = json.load(open(os.path.join(REPO, r["path"], "instance.json")))
    key = json.dumps([sorted(ins["start_positions"].items()),
                      sorted(ins["target_positions"].items()),
                      sorted(map(tuple, ins["walls"]))], sort_keys=True)
    r["board"] = hashlib.md5(key.encode()).hexdigest()[:8]
    r["exact"] = JO[(r["level"], int(r["inst"].split("_")[1]))]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "episodes.json")
json.dump(rows, open(out, "w"))
print(len(rows), "episodes ->", out)
print(collections.Counter((r["suite"], r["outcome"]) for r in rows))
