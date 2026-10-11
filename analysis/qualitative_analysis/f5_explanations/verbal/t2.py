"""Task 2: board-conditioned. Precarious (exactly one legal move) vs route-blocked spells.

Per spell (maximal run of plies in which active player T is in that state; S4
plies 0-1 excluded), does any message in the window mention it?
  window 'round' = messages written at plies [start - R + 1, end], R = active players
  'others' = speakers other than T; 'own' = T's messages written at plies [start-1, end]
"""
import sys
sys.dont_write_bytecode = True
import collections, random, re
from verbal import load, tag_sentence, SENT, MOB_ANY, MOB_VERB, REQ, cell_rx, dist_map, STEP
import json, os
_walls = {}


def walls_of(path):
    if path not in _walls:
        ins = json.load(open(os.path.join("/Users/tom/uni/clembench-spatial-reference", path, "instance.json")))
        _walls[path] = (frozenset(tuple(w) for w in ins["walls"]), ins["grid_size"])
    return _walls[path]


def exit_forward(sp):
    """Is T's sole exit at spell start a step towards its target (walls only)?"""
    ep = EP[sp["path"]]
    w, n = walls_of(sp["path"])
    s = sp["start"]
    for o in ep["mine"][sp["player"]]:
        p = ep["pos"][s][o]
        dm = dist_map(w, n, ep["tgt"][o])
        for c in sp["ply_cells"][s]:
            if abs(c[0] - p[0]) + abs(c[1] - p[1]) == 1 and dm.get(c, 1e9) < dm.get(p, 1e9):
                return True
    return False

S, P, E = load()
EP = {e["path"]: e for e in E}
NEED = re.compile(r"\b(?:please|could you|can you|would you|vacate|clear|free|need|needs|blocked|blocking|"
                  r"occupied|in my way|open|let me|so I can)\b", re.I)


def tref(ep, T, j, s, dyad):
    """Regex for references to player T at message ply j: its letters, its cells."""
    cells = {ep["pos"][min(j, len(ep["pos"]) - 1)][o] for o in ep["mine"][T]}
    pats = [r"(?-i:\b%s\b)" % o for o in ep["mine"][T]]
    pats += [r"R\s*%d\s*[,.]?\s*C\s*%d\b" % (r + 1, c + 1) for r, c in cells]
    if dyad:
        pats += [r"\byou\b", r"\byour\b"]
    return re.compile("|".join(pats), re.I)


def analyse(sp):
    ep = EP[sp["path"]]
    T, s, e = sp["player"], sp["start"], sp["end"]
    R = len(ep["mine"]) - len(ep["done"][s])
    dyad = sp["suite"] == "dyad"
    res = collections.Counter()
    ev = collections.defaultdict(list)
    for j, spk, txt in ep["msgs"]:
        if not (s - R + 1 <= j <= e) or (sp["lvl"] == "S4" and j <= 1):
            continue
        crit = sp["ply_cells"].get(j, sp["ply_cells"][s])
        crx = cell_rx(crit) if crit else None
        rx = tref(ep, T, j, s, dyad)
        own = set(ep["mine"][spk])
        mates = set(ep["tgt"]) - own
        for sent in SENT.findall(txt):
            if spk != T:
                if not rx.search(sent):
                    continue
                t = tag_sentence(sent, own, mates)
                if t["mob_mate"]:
                    res["oth_mob"] = 1; ev["oth_mob"].append((j, spk, sent))
                if t["route_mate"] or t["goal_mate"]:
                    res["oth_route"] = 1; ev["oth_route"].append((j, spk, sent))
                if crx and crx.search(sent):
                    res["oth_cell"] = 1; ev["oth_cell"].append((j, spk, sent))
            elif j >= s - 1:
                res["own_spoke"] = 1
                t = tag_sentence(sent, own, mates)
                if t["mob_self"]:
                    res["own_mob"] = 1; ev["own_mob"].append((j, spk, sent))
                if t["stuck_self"]:
                    res["own_stuck"] = 1
                if t["hold_self"]:
                    res["own_hold"] = 1
                if crx and crx.search(sent) and re.search(r"(?-i:\b(?:%s)\b)" % "|".join(mates), sent):
                    res["own_invite"] = 1; ev["own_invite"].append((j, spk, sent))
                if crx and crx.search(sent):
                    res["own_cell"] = 1
                    if NEED.search(sent):
                        res["own_cellreq"] = 1; ev["own_cellreq"].append((j, spk, sent))
                if re.search(r"\b(?:blocked|blocking|in my way|occupied|block)\b", sent, re.I):
                    res["own_blockword"] = 1
    return res, ev


def main():
    rows = collections.defaultdict(collections.Counter)
    evs = collections.defaultdict(list)
    for sp in P:
        r, ev = analyse(sp)
        key = (sp["tier"], sp["suite"], sp["kind"])
        rows[key]["n"] += 1
        rows[key]["plies"] += sp["length"]
        rows[key]["sealed"] += sp["sealed"]
        if sp["kind"] == "prec":
            fw = exit_forward(sp)
            rows[key]["fwd"] += fw
            rows[key]["fwd_oth_cell"] += fw and r["oth_cell"]
            rows[key]["back_oth_cell"] += (not fw) and r["oth_cell"]
        for k, v in r.items():
            rows[key][k] += v
        if sp["kind"] == "prec":
            tag = "S" if sp["sealed"] else "U"
            for k_ in ("own_spoke", "own_mob", "own_invite", "own_hold", "oth_mob", "oth_cell", "oth_route"):
                rows[key][tag + k_] += r[k_]
            rows[key][tag + "n"] += 1
        if sp["kind"] == "prec" and sp["sealed"]:
            rows[key]["sealed_oth_mob"] += r["oth_mob"]
            rows[key]["sealed_oth_cell"] += r["oth_cell"]
            rows[key]["sealed_own_spoke"] += r["own_spoke"]
            rows[key]["sealed_own_mob"] += r["own_mob"]
        for k, v in ev.items():
            for x in v:
                evs[(sp["tier"], sp["suite"], sp["kind"], k)].append((sp["model"], sp["lvl"], sp["proto"], sp["inst"], sp["player"]) + x)
    for key in sorted(rows):
        c = rows[key]
        n, os_ = c["n"], c["own_spoke"]
        line = (f"{key[0]:6} {key[1]:4} {key[2]:4} spells={n:5} plies={c['plies']:6} "
                f"| others: T+mob {c['oth_mob']} ({100*c['oth_mob']/n:.1f}%)  T+route/goal {c['oth_route']} ({100*c['oth_route']/n:.1f}%)  "
                f"T+critcell {c['oth_cell']} ({100*c['oth_cell']/n:.1f}%) "
                f"| own spoke {os_}: mob {c['own_mob']} ({100*c['own_mob']/max(os_,1):.1f}%)  stuck {c['own_stuck']}  "
                f"cell {c['own_cell']}  cell+req {c['own_cellreq']} ({100*c['own_cellreq']/max(os_,1):.1f}%)  blockword {c['own_blockword']} ({100*c['own_blockword']/max(os_,1):.1f}%)")
        if key[2] == "prec":
            line += (f" | exit forward {c['fwd']} (T+cell {c['fwd_oth_cell']}), exit backward {c['n']-c['fwd']} (T+cell {c['back_oth_cell']})")
            line += (f" | sealed next ply {c['sealed']}: others T+mob {c['sealed_oth_mob']}, T+cell {c['sealed_oth_cell']};"
                     f" own spoke {c['sealed_own_spoke']}, own mob {c['sealed_own_mob']}")
        if key[2] == "prec":
            for tag, lab in (("S", "sealed next"), ("U", "not sealed")):
                m = c[tag + "own_spoke"]
                line += (f"\n      {lab}: n={c[tag+'n']} others T+mob {c[tag+'oth_mob']} T+route/goal {c[tag+'oth_route']} T+exitcell {c[tag+'oth_cell']}"
                         f" | own spoke {m}: own mob {c[tag+'own_mob']} ({100*c[tag+'own_mob']/max(m,1):.1f}%), own names exit+teammate {c[tag+'own_invite']} ({100*c[tag+'own_invite']/max(m,1):.1f}%),"
                         f" own hold {c[tag+'own_hold']} ({100*c[tag+'own_hold']/max(m,1):.1f}%)")
        print(line)
    if len(sys.argv) > 1:
        tier, suite, kind, k, nn = sys.argv[1:6]
        pool = evs[(tier, suite, kind, k)]
        random.seed(3)
        print(len(pool))
        for x in random.sample(pool, min(int(nn), len(pool))):
            print("  ", x[:6], x[6], " ".join(x[7].split())[:220])


if __name__ == "__main__":
    main()
