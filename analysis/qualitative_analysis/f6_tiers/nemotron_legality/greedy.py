"""Items 1-3, 6: rejection causes, greedy-toward-target, repeats + lexical acknowledgement.
Needs attempts.json (run attempts.py first)."""
import json, collections, re, os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"attempts.{ARM}.json")))["attempts"]
MODELS = ("nemotron", "qwen38", "dsv41"); SUITES = ("solo", "dyad", "3p", "6p")
pct = lambda a, b: f"{100*a/b:5.1f}% ({a}/{b})" if b else "   -   (0/0)"
CAUSE = {"wall": "wall", "off_grid": "off_grid", "occ_own": "occ_own", "occ_other": "occ_other"}

print("== 1. rejection causes (invalid_move) ==")
for m in MODELS:
    for s in SUITES:
        rej = [a for a in A if a["model"] == m and a["suite"] == s and a["verdict"] == "invalid_move"]
        att = sum(1 for a in A if a["model"] == m and a["suite"] == s and a["verdict"] in ("move", "invalid_move"))
        c = collections.Counter("not_owner" if a["tag"] == "not_owner" else a["status"] for a in rej)
        print(f"{m:8s} {s:4s} rej {len(rej):4d}/{att:5d} att  " + "  ".join(f"{k} {c[k]}" for k in ("wall", "off_grid", "occ_own", "occ_other", "not_owner")))
    if m == "nemotron":
        for cond in ("silent", "thinking"):
            rej = [a for a in A if a["model"] == m and a["suite"] == "solo" and a["proto"] == cond and a["verdict"] == "invalid_move"]
            print(f"   solo {cond}: ", collections.Counter(a["status"] for a in rej))
# occupied: is the blocked cell the mover's own target? / is the occupant already on its target?
for m in MODELS:
    occ = [a for a in A if a["model"] == m and a["verdict"] == "invalid_move" and a["status"].startswith("occ")]
    into_tgt = sum(1 for a in occ if (a["pos"][0] + {"up": -1, "down": 1}.get(a["move"][1], 0), a["pos"][1] + {"left": -1, "right": 1}.get(a["move"][1], 0)) == tuple(a["tgt"]))
    for s in SUITES: print(f"      {s}: {pct(sum(1 for a in occ if a['suite']==s and [a['pos'][0] + {'up': -1, 'down': 1}.get(a['move'][1], 0), a['pos'][1] + {'left': -1, 'right': 1}.get(a['move'][1], 0)] == a['tgt']), sum(a['suite']==s for a in occ))}")
    print(f"   {m}: occupied rejections whose blocked cell is the mover's own target: {pct(into_tgt, len(occ))}")

print("\n== 2. greedy (move reduces moved object's Manhattan distance to its own target) ==")
for m in MODELS:
    for s in SUITES + ("ALL",):
        X = [a for a in A if a["model"] == m and (s == "ALL" or a["suite"] == s) and "greedy" in a and a["verdict"] in ("move", "invalid_move")]
        acc = [a for a in X if a["verdict"] == "move"]; rej = [a for a in X if a["verdict"] == "invalid_move"]
        wall = [a for a in rej if a["status"] == "wall"]; occ = [a for a in rej if a["status"].startswith("occ")]; off = [a for a in rej if a["status"] == "off_grid"]
        g = lambda L: sum(a["greedy"] for a in L)
        print(f"{m:8s} {s:4s} greedy: accepted {pct(g(acc), len(acc))} | rejected {pct(g(rej), len(rej))} | wall {pct(g(wall), len(wall))} | occ {pct(g(occ), len(occ))} | offgrid {pct(g(off), len(off))}")
    # alternatives at rejections that had a legal alternative
    rej = [a for a in A if a["model"] == m and a["verdict"] == "invalid_move" and a["status"] != "ok" and a["n_legal"] > 0]
    gr = [a for a in rej if a["greedy"]]
    print(f"   {m} rejections w/ legal alt: {len(rej)}; greedy among them {pct(len(gr), len(rej))}; legal greedy alt exists (any obj) {pct(sum(a['legal_greedy_any'] for a in rej), len(rej))}; same obj other dir greedy+legal {pct(sum(a['legal_greedy_same'] for a in rej), len(rej))}")
    # on-target moves (moving an object that is already home)
    X = [a for a in A if a["model"] == m and "on_target" in a and a["verdict"] in ("move", "invalid_move")]
    print(f"   {m} attempts moving an object already on target: accepted {pct(sum(a['on_target'] for a in X if a['verdict']=='move'), sum(1 for a in X if a['verdict']=='move'))}, rejected {pct(sum(a['on_target'] for a in X if a['verdict']=='invalid_move'), sum(1 for a in X if a['verdict']=='invalid_move'))}")

print("\n== 3. what follows a rejection (same player, next verdict) ==")
ACK = re.compile(r"\b(walls?|##|blocked|blocks|occupied|cannot|can't|can not|invalid|rejected|illegal|not possible|impossible|failed)\b", re.I)
EXPL = re.compile(r"\b(invalid|rejected|failed|tried|previous (move|attempt)|last (move|attempt)|again|didn't work|did not work)\b", re.I)
CELL = lambda r, c: re.compile(rf"\(?R{r}\s*,?\s*C{c}\)?")
for m in MODELS:
    for s in SUITES + ("ALL",):
        X = [a for a in A if a["model"] == m and (s == "ALL" or a["suite"] == s)]
        byep = collections.defaultdict(list)
        for a in X: byep[a["ep"]].append(a)
        C = collections.Counter(); lex = collections.Counter()
        for L in byep.values():
            for a, b in zip(L, L[1:]):
                if a["verdict"] != "invalid_move" or b["who"] != a["who"] or a["n_legal"] == 0: continue
                rep = b["move"] == a["move"]
                C["n"] += 1; C["rep"] += rep; C["rep_greedy"] += rep and a["greedy"]
                C["rep_legal_greedy_alt"] += rep and a["legal_greedy_any"]
                txt = re.sub(r"move:.*", "", b["resp"] or "", flags=re.I).strip()
                if len(txt) < 15: continue  # silent / bare move line: nothing to read
                dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[a["move"][1]]
                dest = (a["pos"][0] + dr + 1, a["pos"][1] + dc + 1)
                key = "rep" if rep else "rev"
                lex[key + "_n"] += 1
                lex[key + "_ack"] += bool(ACK.search(txt)); lex[key + "_expl"] += bool(EXPL.search(txt))
                lex[key + "_cell"] += any(CELL(*dest).search(x) and ACK.search(x) for x in re.split(r"(?<=[.!?\n])\s+", txt))
        print(f"{m:8s} {s:4s} rejections(legal alt) followed by same-player attempt: {C['n']:4d}; identical repeat {pct(C['rep'], C['n'])}; repeat greedy {pct(C['rep_greedy'], C['rep'])}; repeat while a legal greedy alt existed {pct(C['rep_legal_greedy_alt'], C['rep'])}")
        if lex["rep_n"] or lex["rev_n"]:
            print(f"              text-bearing: REPEAT n={lex['rep_n']}: ack-word {pct(lex['rep_ack'], lex['rep_n'])}, explicit-rejection {pct(lex['rep_expl'], lex['rep_n'])}, sentence names blocked cell+ack {pct(lex['rep_cell'], lex['rep_n'])}"
                  f" || REVISED n={lex['rev_n']}: ack-word {pct(lex['rev_ack'], lex['rev_n'])}, explicit {pct(lex['rev_expl'], lex['rev_n'])}, cell+ack {pct(lex['rev_cell'], lex['rev_n'])}")

print("\n== 6. misc (nemotron) ==")
for m in MODELS:
    X = [a for a in A if a["model"] == m]
    fm = [a for a in X if a["verdict"] in ("invalid_format", "malformed_message", "invalid_done")]
    words = collections.Counter()
    for a in fm:
        ml = re.findall(r"move:\s*(.*)", a["resp"] or "", re.I)
        words[(a["verdict"], (ml[-1].strip().lower()[:20] if ml else "<no move line>"))] += 1
    print(m, "format/done failures", len(fm), words.most_common(8))
    xs = sum(1 for a in X if re.search(r"move:\s*\**\s*X\s+(up|down|left|right)", a["resp"] or "", re.I))
    print(f"   {m} responses whose move line moves 'X': {xs}")

WAIT = re.compile(r"\b(wait|waiting|stay|hold|pass)\b", re.I)
print("\n== 6b. 'wait/stay/hold/pass' language in text-bearing attempts ==")
for m in MODELS:
    for v in ("move", "invalid_move"):
        X = [a for a in A if a["model"] == m and a["suite"] != "solo" and a["verdict"] == v and len(re.sub(r"move:.*", "", a["resp"] or "", flags=re.I).strip()) >= 15]
        print(f"   {m} comm {v}: {pct(sum(bool(WAIT.search(a['resp'])) for a in X), len(X))}")
    ml = sum(1 for a in A if a["model"] == m and re.search(r"move:\s*(wait|stay|pass|none|nothing|hold)", a["resp"] or "", re.I))
    print(f"   {m}: move line says wait/stay/pass/none: {ml}")
