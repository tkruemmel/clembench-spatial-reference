"""Item 3b: in text-bearing Nemotron rejections, does the SAME response already state what blocks the move?
occupied: the occupant's letter (solo) or X / the letter (comm) is placed at the destination cell, or the cell is called occupied/blocked
wall:     the destination cell is called a wall / ## / blocked, or "wall at <cell>"
off_grid: no cell to name; skipped."""
import json, re, collections, os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
J = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"attempts.{ARM}.json")))
A = J["attempts"]
D = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
pct = lambda a, b: f"{100*a/b:5.1f}% ({a}/{b})" if b else "-"
for scope in ("solo", "comm"):
    C = collections.Counter()
    for a in A:
        if a["model"] != "nemotron" or a["verdict"] != "invalid_move" or a["status"] not in ("wall", "occ_own", "occ_other"): continue
        if (a["suite"] == "solo") != (scope == "solo"): continue
        txt = re.sub(r"(?im)^\s*move:.*$", "", a["resp"] or "")
        if len(txt) < 15: continue
        d = D[a["move"][1]]; r, c = a["pos"][0] + d[0], a["pos"][1] + d[1]
        cell = rf"\(?\s*R{r+1}\s*,?\s*C{c+1}\s*\)?"
        st = "occ" if a["status"].startswith("occ") else "wall"
        C[(st, "n")] += 1
        C[(st, "names cell")] += bool(re.search(cell, txt))
        if st == "occ":
            L = a["occ"][f"{r},{c}"]
            who = rf"(?:\b{L}\b|\bX\b|teammate'?s? (?:X|object))"
            hit = re.search(who + rf"(?:'s)?(?:\s+is)?(?:\s+currently)?\s*(?:at|in|on)?\s*" + cell, txt) or \
                  re.search(cell + r"\s*(?:is|=|:)?\s*(?:currently\s+)?(?:occupied|blocked|taken|has|by)\b", txt, re.I) or \
                  re.search(rf"(?:blocked|occupied) by[^.\n]{{0,25}}" + cell, txt, re.I)
        else:
            hit = re.search(cell + r"\s*(?:is|=|:)\s*(?:an?\s+)?(?:##|'##'|`##`|wall|blocked)", txt, re.I) or \
                  re.search(r"(?:wall|##)\s+(?:at|in|on)\s+" + cell, txt, re.I) or re.search(cell + r"\s*\(?(?:##|wall)\)?", txt, re.I)
        C[(st, "states blocker")] += bool(hit)
    for st in ("occ", "wall"):
        print(f"nemotron {scope} {st}: text-bearing rejections {C[(st,'n')]}; names the destination cell {pct(C[(st,'names cell')], C[(st,'n')])}; states it is blocked/occupied (and by what) {pct(C[(st,'states blocker')], C[(st,'n')])}")
