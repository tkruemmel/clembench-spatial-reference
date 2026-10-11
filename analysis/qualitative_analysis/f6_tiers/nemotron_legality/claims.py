"""Item 4: cell claims in the response text vs the replayed board.
 (a) position claims for the moved object ("A at R5C2", "A is at (R5,C2)", "A ... from (R5,C2)")
 (b) destination claims in a sentence naming the object and the attempted direction ("A up to (R4,C2)")
 (c) model-internal direction semantics: destination minus the model's own claimed position
 (d) content claims about the attempted destination cell ("(R4,C2) is ." / "is a wall" / "wall at ...")
Usage: python3 claims.py [--examples CATEGORY]"""
import json, re, collections, os, sys
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "attempts.json")))
A, META = J["attempts"], J["meta"]
D = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
CELL = r"\(?\s*R(\d+)\s*,?\s*C(\d+)\s*\)?"
pct = lambda a, b: f"{100*a/b:5.1f}% ({a}/{b})" if b else "-"
EXAMPLES = sys.argv[sys.argv.index("--examples") + 1] if "--examples" in sys.argv else None


def pos_cat(q, p, prev, tgt):
    if q == p: return "correct"
    if prev and q == prev: return "stale(prev pos)"
    if q == tgt: return "= its target"
    if q == (p[1], p[0]): return "row/col swapped"
    if q[0] == p[0] and abs(q[1] - p[1]) == 1: return "col off by 1"
    if q[1] == p[1] and abs(q[0] - p[0]) == 1: return "row off by 1"
    return "other"


def delta_cat(dl, d):
    if dl == d: return "= dir"
    if dl == (0, 0): return "zero"
    if dl == (-d[0], -d[1]): return "opposite"
    if dl[0] * d[0] + dl[1] * d[1] == 0 and abs(dl[0]) + abs(dl[1]) == 1: return "perpendicular"
    k = dl[0] * d[0] + dl[1] * d[1]
    if k >= 2 and dl == (k * d[0], k * d[1]): return "multi-step same dir"
    return "other"


def content_truth(cell, a):
    r, c = cell[0] - 1, cell[1] - 1
    if [r, c] in META[a["ep"]]["walls"]: return "wall"
    n = META[a["ep"]]["n"]
    if not (0 <= r < n and 0 <= c < n): return "offgrid"
    return "occupied" if f"{r},{c}" in a["occ"] else "empty"


EMPTY = r"(?:is|=|:)\s*(?:an?\s+)?(?:empty|open|free|clear|available|'\.'|`\.`|\"\.\"|\.(?=[\s,)]))"
BLOCK = r"(?:is|=|:)\s*(?:an?\s+)?(?:##|'##'|`##`|wall|blocked|occupied|taken)"
MODELS = sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else ("nemotron", "qwen38", "dsv41")
for model in MODELS:
    for scope in ("solo", "comm"):
        P, Dst, Sem, Cont, Byv = (collections.Counter() for _ in range(5))
        ex = collections.defaultdict(list)
        for a in A:
            if a["model"] != model or a["verdict"] not in ("move", "invalid_move") or "pos" not in a: continue
            if (a["suite"] == "solo") != (scope == "solo"): continue
            txt = re.sub(r"(?im)^\s*move:.*$", "", a["resp"] or "")
            if len(txt) < 15: continue
            o, dr = a["move"]; d = D[dr]
            p = (a["pos"][0] + 1, a["pos"][1] + 1); tg = (a["tgt"][0] + 1, a["tgt"][1] + 1)
            pv = (a["prev"][0] + 1, a["prev"][1] + 1) if a["prev"] else None
            P["_resp"] += 1
            claims = {(int(x), int(y)) for x, y in re.findall(rf"\b{o}\b(?:'s position)?(?:\s+is)?(?:\s+currently)?(?:\s+now)?\s+(?:at|on)\s+{CELL}", txt)}
            claims |= {(int(x), int(y)) for x, y in re.findall(rf"\b{o}\b[^.\n;]{{0,30}}?\bfrom\s+{CELL}", txt)}
            cats = {pos_cat(q, p, pv, tg) for q in claims}
            if claims:  # per attempt: any stale / any other wrong claim, by verdict
                Byv[(a["verdict"], "n")] += 1; Byv[(a["verdict"], "stale")] += "stale(prev pos)" in cats
                Byv[(a["verdict"], "wrong")] += bool(cats - {"correct"})
            for q in claims:
                k = pos_cat(q, p, pv, tg); P[k] += 1
                if EXAMPLES == k and len(ex[k]) < 6: ex[k].append((a["ep"], a["i"], o, dr, p, q, txt[:300]))
            dests = set()
            for s in re.split(r"(?<=[.!?\n])\s+", txt):
                if re.search(rf"\b{o}\b", s) and re.search(rf"\b{dr}\b", s, re.I):
                    dests |= {(int(x), int(y)) for x, y in re.findall(rf"\b{dr}\b\s+(?:\w+\s+){{0,2}}?(?:to|into|onto)\s+{CELL}", s, re.I)}
            for q in dests:
                k = delta_cat((q[0] - p[0], q[1] - p[1]), d); Dst[k] += 1
                if EXAMPLES == "dst:" + k and len(ex["dst:" + k]) < 6: ex["dst:" + k].append((a["ep"], a["i"], o, dr, p, q, txt[:300]))
                for c in claims:  # model-internal semantics
                    Sem[delta_cat((q[0] - c[0], q[1] - c[1]), d)] += 1
            # (d) what does the text say about the cell it is about to move into?
            tgtcell = (p[0] + d[0], p[1] + d[1]); rc = rf"\(?\s*R{tgtcell[0]}\s*,?\s*C{tgtcell[1]}\s*\)?"
            said = "empty" if re.search(rc + r"\s*" + EMPTY, txt, re.I) else None
            if re.search(rc + r"\s*" + BLOCK, txt, re.I) or re.search(rf"(?:wall|##)\s+(?:at|in)\s+" + rc, txt, re.I):
                said = "both" if said else "blocked"
            if said:
                truth = content_truth(tgtcell, a)
                Cont[(said, truth, a["verdict"])] += 1
                if EXAMPLES == f"cont:{said}:{truth}" and len(ex[EXAMPLES]) < 6: ex[EXAMPLES].append((a["ep"], a["i"], o, dr, p, tgtcell, txt[:400]))
        n = P.pop("_resp", 0)
        print(f"== {model} {scope}: text-bearing attempts {n}")
        tot = sum(P.values()); print(f"  (a) position claims for moved object: {tot}  " + ", ".join(f"{k} {pct(v, tot)}" for k, v in P.most_common()))
        tot = sum(Dst.values()); print(f"  (b) destination claims vs true pos+dir: {tot}  " + ", ".join(f"{k} {pct(v, tot)}" for k, v in Dst.most_common()))
        tot = sum(Sem.values()); print(f"  (c) destination minus model's own claimed pos: {tot}  " + ", ".join(f"{k} {pct(v, tot)}" for k, v in Sem.most_common()))
        for v in ("move", "invalid_move"):
            print(f"  attempts with a position claim, {v}: any stale {pct(Byv[(v,'stale')], Byv[(v,'n')])}, any wrong {pct(Byv[(v,'wrong')], Byv[(v,'n')])}")
        print(f"  (d) claims about the destination cell (said, truth, verdict): {dict(Cont)}")
        for k, v in ex.items():
            print("  EXAMPLES", k)
            for e in v: print("    ", e)
