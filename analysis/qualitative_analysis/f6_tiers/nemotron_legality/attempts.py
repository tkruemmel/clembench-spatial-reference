"""Replay all 2520 grid episodes and emit one record per move attempt (accepted or rejected).

Run from repo root: python3 <scratch>/nemo_anatomy/attempts.py   -> writes attempts.json next to this file.
Parsing/replay logic copied from analysis/qualitative_analysis/f6_tiers/rejection/retry.py.
"""
import json, re, collections, string, os
REPO = "/Users/tom/uni/clembench-spatial-reference"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attempts.json")
R = json.load(open(f"{REPO}/analysis/qualitative_analysis/episodes.json"))
MOVEOK = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
CANNOT = re.compile(r"Cannot move ([A-Z]) (up|down|left|right)")
MOVE = re.compile(r"move:\s*\**\s*([A-Za-z])(?:@\S+)?\s+(up|down|left|right)", re.I)
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
VERD = {"move", "invalid_move", "invalid_format", "malformed_message", "player_done", "invalid_done"}
man = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])


def pm(r):
    m = MOVE.findall(r or "")
    return (m[-1][0].upper(), m[-1][1].lower()) if m else None


def episode(r):
    d = os.path.join(REPO, r["path"])
    ins = json.load(open(os.path.join(d, "instance.json"))); it = json.load(open(os.path.join(d, "interactions.json")))
    n = ins["grid_size"]; walls = {tuple(w) for w in ins["walls"]}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    tgt = {o: tuple(p) for o, p in ins["target_positions"].items()}
    groups = ins.get("player_objects") or ([ins["player_a_objects"], ins["player_b_objects"]] if "player_a_objects" in ins else [ins["objects"]])
    owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
    mine = collections.defaultdict(list)
    for o, p in owner.items(): mine[p].append(o)
    prev = {}  # object -> its position before its last accepted move (for "stale" claims)

    def status(o, dr):  # legality of o moving dr on the current board
        r0, c0 = pos[o]; nr, nc = r0 + DIRS[dr][0], c0 + DIRS[dr][1]
        if not (0 <= nr < n and 0 <= nc < n): return "off_grid"
        if (nr, nc) in walls: return "wall"
        occ = {p: x for x, p in pos.items()}
        if (nr, nc) in occ: return "occ_own" if owner[occ[(nr, nc)]] == owner[o] else "occ_other"
        return "ok"

    greedy = lambda o, dr: man((pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1]), tgt[o]) < man(pos[o], tgt[o])
    out, resp, k = [], None, 0
    for t in it["turns"]:
        for e in t:
            ty = e["action"]["type"]; c = str(e["action"].get("content", ""))
            if ty == "get message": resp = c; continue
            if ty not in VERD: continue
            who = (c.split(":")[0].replace("Player", "").strip() or "A") if ty != "move" else ((MOVEOK.match(c).group(1) or "A") if MOVEOK.match(c) else "A")
            objs = mine.get(who, []) if len(groups) > 1 else list(pos)
            if ty == "move":
                m = MOVEOK.match(c); mv = (m.group(2), m.group(3)) if m else None
            elif ty == "invalid_move":
                m = CANNOT.search(c); mv = (m.group(1), m.group(2)) if m else pm(resp)
            else:
                mv = pm(resp)
            rec = dict(model=r["model"], suite=r["suite"], level=r["level"], proto=r["proto"], ep=r["path"], i=k, who=who,
                       verdict=ty, gm=c, resp=resp, move=mv)
            k += 1
            if mv and mv[0] in pos:
                o, dr = mv
                rec["obj_owned"] = o in objs
                rec["pos"] = pos[o]; rec["tgt"] = tgt[o]; rec["prev"] = prev.get(o)
                rec["on_target"] = pos[o] == tgt[o]
                rec["status"] = status(o, dr)
                rec["greedy"] = greedy(o, dr)
                opts = [(x, y) for x in objs for y in DIRS if status(x, y) == "ok"]
                rec["n_legal"] = len(opts)
                rec["legal_greedy_any"] = any(greedy(x, y) for x, y in opts if (x, y) != mv)
                rec["legal_greedy_same"] = any(greedy(x, y) for x, y in opts if x == o and y != dr)
                rec["greedy_dirs"] = [y for y in DIRS if greedy(o, y)]
                # obstacle-blindness baselines: greedy options on the board and how many are blocked
                G = [(x, y) for x in objs if pos[x] != tgt[x] for y in DIRS if greedy(x, y)]
                rec["g_opts"] = len(G); rec["g_blocked"] = sum(status(x, y) != "ok" for x, y in G)
                rec["obj_g"] = len(rec["greedy_dirs"]); rec["obj_g_blocked"] = sum(status(o, y) != "ok" for y in rec["greedy_dirs"])
                # board snapshot for text-claim checks
                rec["occ"] = {f"{p[0]},{p[1]}": x for x, p in pos.items()}
            if ty == "invalid_move":
                tag = re.search(r"\[(\w+)\]", c); rec["tag"] = tag.group(1) if tag else "other"
            if ty == "move" and mv:
                o, dr = mv; prev[o] = pos[o]; pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
            resp = None
            out.append(rec)
    return out, sorted(walls), n


if __name__ == "__main__":
    allrec, meta = [], {}
    for r in R:
        recs, walls, n = episode(r)
        allrec += recs; meta[r["path"]] = dict(walls=walls, n=n, outcome=r["outcome"], end_msg=r["end_msg"])
    json.dump(dict(attempts=allrec, meta=meta), open(OUT, "w"))
    print(len(allrec), "attempts ->", OUT)
