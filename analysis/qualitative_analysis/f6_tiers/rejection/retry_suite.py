"""Every rejected move in all 2520 episodes: what does the next attempt do?"""
import json, re, collections, string, os, statistics
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
REPO = "/Users/tom/uni/clembench-spatial-reference"
R = json.load(open(f"{REPO}/analysis/qualitative_analysis/episodes.{ARM}.json"))
MOVEOK = re.compile(r"^Player(?: (\w+))? moved ([A-F]) (up|down|left|right)$")
MOVE = re.compile(r"move:\s*\**\s*([A-Za-z])(?:@\S+)?\s+(up|down|left|right)", re.I)
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
VERD = {"move", "invalid_move", "invalid_format", "malformed_message", "player_done", "invalid_done"}
def pm(r):
    if r is None: return None
    m = MOVE.findall(r)
    return (m[-1][0].upper(), m[-1][1].lower()) if m else None
C = collections.Counter(); eps = collections.Counter(); nleg_rep = collections.defaultdict(list)
for r in R:
    d = os.path.join(REPO, r["path"])
    ins = json.load(open(os.path.join(d, "instance.json"))); it = json.load(open(os.path.join(d, "interactions.json")))
    n = ins["grid_size"]; walls = {tuple(w) for w in ins["walls"]}
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    groups = ins.get("player_objects") or ([ins["player_a_objects"], ins["player_b_objects"]] if "player_a_objects" in ins else [ins["objects"]])
    owner = {o: string.ascii_uppercase[i] for i, g in enumerate(groups) for o in g}
    mine = collections.defaultdict(list)
    for o, p in owner.items(): mine[p].append(o)
    def legal(objs):
        occ = set(pos.values()); k = 0
        for o in objs:
            r0, c0 = pos[o]
            for dr, dc in DIRS.values():
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in walls and (nr, nc) not in occ: k += 1
        return k
    pairs, resp = [], None
    for t in it["turns"]:
        for e in t:
            ty = e["action"]["type"]; c = e["action"].get("content", "")
            if ty == "get message": resp = c
            elif ty in VERD:
                who = c.split(":")[0].replace("Player", "").strip() or "A"
                objs = mine.get(who, sum(groups, [])) if len(groups) > 1 else sum(groups, [])
                nl = legal(objs) if ty == "invalid_move" else None
                pairs.append((ty, c, resp, nl)); resp = None
                if ty == "move":
                    m = MOVEOK.match(c)
                    if m:
                        o, dr = m.group(2), m.group(3); pos[o] = (pos[o][0] + DIRS[dr][0], pos[o][1] + DIRS[dr][1])
    tier = (r["model"], re.search(r"_(solo|dyad|3p|6p)\b", r["path"]).group(1))
    eps[(tier, any(p[0] == "invalid_move" for p in pairs))] += 1
    for i in range(len(pairs) - 1):
        if pairs[i][0] != "invalid_move": continue
        nxt = pairs[i + 1]
        if nxt[0] in ("move",) and pm(nxt[2]) == pm(pairs[i][2]): k = "same_accepted?!"
        elif pm(nxt[2]) is not None and pm(nxt[2]) == pm(pairs[i][2]): k = "repeat"
        elif nxt[0] in ("move", "player_done"): k = "revised_ok"
        elif nxt[0] == "invalid_move": k = "revised_rejected"
        else: k = "format_fail"
        has_alt = pairs[i][3] > 0
        C[(tier, has_alt, k)] += 1
for (tt, a, k), v in sorted(C.items()):
    pass
agg = collections.defaultdict(collections.Counter)
for (tt, a, k), v in C.items():
    if a: agg[tt][k] += v
for tt in sorted(agg):
    tot = sum(agg[tt].values())
    print(tt, "rejections w/ legal alt:", tot, "repeat %.0f%%" % (100*agg[tt]["repeat"]/tot), "revised_ok %.0f%%" % (100*agg[tt]["revised_ok"]/tot))
