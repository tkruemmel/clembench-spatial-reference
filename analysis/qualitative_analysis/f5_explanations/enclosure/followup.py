"""Agent model vs board-pattern reflex.  python3 followup.py [--sample]
TEST 1: does seal avoidance depend on when the victim moves next / whether someone can relieve it first?
TEST 2: is the last free cell of an *identifiable* done X avoided like an active one (prose arms)?"""
import collections, io, contextlib, json, os, re, random, sys
import numpy as np
from scipy import stats
with contextlib.redirect_stdout(io.StringIO()):
    from decomp import M1, ok, clogit, predict, ps, STRONG
    from analysis import poibin_p
import plies as P
sa = P.sa
ORDER = "ABCDEF"
PROSE = ("freeform", "hybrid")
GROUPS = (("none", ("none",)), ("structured", ("structured",)), ("freeform", ("freeform",)), ("hybrid", ("hybrid",)),
          ("rigid", ("none", "structured")), ("prose", PROSE), ("all", ("none", "structured", "freeform", "hybrid")))

eps = collections.defaultdict(list)
for r in ps:
    eps[r["path"]].append(r)


def after(r):
    """Active players in turn order after the mover (mover excluded)."""
    i = ORDER.index(r["player"])
    return [ORDER[(i + j) % 6] for j in range(1, 6) if ORDER[(i + j) % 6] not in r["done"]]


# sanity: the next accepted move is by the first active player after the mover (done declarations aside)
chk = collections.Counter()
for path, rs in eps.items():
    for a, b in zip(rs, rs[1:]):
        nxt = [q for q in after(a) if q not in b["done"]]
        chk[b["player"] == (nxt[0] if nxt else a["player"])] += 1
print("turn-order check: next mover = predicted %d / %d" % (chk[True], sum(chk.values())))


def annotate(r, x):
    """d, relievable, and whether the victim was actually relieved (if this move was played)."""
    v = x["kill"][0]
    seq = after(r)
    d = seq.index(v) + 1
    nb = sa.step(r["pos"], *x["m"])
    vo = r["mine"][v]
    rel = any(sa.legal(sa.step(nb, *mq), vo, r["walls"], r["n"])
              for q in seq[:d - 1] for mq in sa.legal(nb, r["mine"][q], r["walls"], r["n"]))
    return d, rel


def relieved(r):
    """Taken seal: did the victim have a legal move when its turn came round?"""
    x = r["alts"][r["chosen"]]
    v = x["kill"][0]
    rs = eps[r["path"]]
    pos = sa.step(r["pos"], *x["m"])
    inter = set(after(r)[:after(r).index(v)])
    for s in rs[rs.index(r) + 1:]:
        if s["player"] == v:
            return True
        if s["player"] not in inter:
            break
        pos = sa.step(pos, *s["played"])
    return bool(sa.legal(pos, r["mine"][v], r["walls"], r["n"]))


for r in ps:
    for x in r["alts"]:
        if x["kill"]:
            x["d"], x["rel"] = annotate(r, x)

# ---------------- TEST 1 descriptive ----------------
print("\nTEST 1a  free pinches, strong tier: seal taken, observed vs M1-expected (M1 fitted on non-pinch plies of the same group)")
fits = {}
for lab, protos in GROUPS:
    fit = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r) and not r["pin"]]
    fits[lab] = clogit(fit, M1)[0]


def seal(r):
    return [x for x in r["alts"] if x["kill"]][0]


cells = [("d=1", lambda x: x["d"] == 1), ("d=2", lambda x: x["d"] == 2), ("d=3", lambda x: x["d"] == 3),
         ("d=4-5", lambda x: x["d"] >= 4), ("d>=2", lambda x: x["d"] >= 2),
         ("d>=2 unrelievable", lambda x: x["d"] >= 2 and not x["rel"]), ("relievable", lambda x: x["rel"]),
         ("not relievable (incl d=1)", lambda x: not x["rel"])]
for lab in ("rigid", "prose", "all"):
    protos = dict(GROUPS)[lab]
    pin = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and r["pin"] and r["pin"]["free"]]
    out = []
    for name, f in cells:
        rs = [r for r in pin if f(seal(r))]
        if not rs:
            out.append("%s --" % name); continue
        pr = [predict(fits[lab], M1, r) for r in rs]
        o = sum(bool(r["alts"][r["chosen"]]["kill"]) for r in rs)
        out.append("%s %d/%d (%.0f%%) exp %.0f%% p=%.2g" % (name, o, len(rs), 100 * o / len(rs), 100 * sum(pr) / len(rs), poibin_p(o, pr)))
    print("  %-6s " % lab + "\n         ".join(out))

print("\nTEST 1b  base rates: taken seals (strong tier, all pinches), victim had a legal move when its turn came")
t = collections.defaultdict(lambda: [0, 0])
for r in ps:
    if r["model"] in STRONG and r["pin"] and r["pin"]["taken"]:
        x = r["alts"][r["chosen"]]
        ok_ = relieved(r)
        for k in ("d=%d" % min(x["d"], 4) if x["d"] < 4 else "d=4-5", "relievable=%s" % x["rel"], "all"):
            t[k][0] += ok_; t[k][1] += 1
print("  " + " | ".join("%s: %d/%d" % (k, *t[k]) for k in sorted(t)))

# ---------------- TEST 1 logit ----------------
act = lambda x: bool(x["kill"])
don = lambda x: any(k == 1 for k, *_ in x["adjX"]) and not x["kill"]
F1 = dict(M1, **{
    "k1 (d>=2, unrelievable) [ref]": lambda r, x: act(x),
    "k1 x victim moves next (d=1)": lambda r, x: act(x) and x["d"] == 1,
    "k1 x relievable": lambda r, x: act(x) and x["rel"],
    "k1 done X (control)": lambda r, x: don(x),
})
print("\nTEST 1c  conditional logit, strong tier, M1 controls; OR [95% CI], SE clustered by board")
for lab, protos in GROUPS[4:]:
    rows = [r for r in ps if r["model"] in STRONG and r["proto"] in protos and ok(r)]
    b, V = clogit(rows, F1)
    se = np.sqrt(np.diag(V)); ks = list(F1)
    n = collections.Counter(("d1" if x["d"] == 1 else "rel" if x["rel"] else "unrel") for r in rows for x in r["alts"] if x["kill"])
    print("  %-6s k1 alternatives: d=1 %d, d>=2 unrelievable %d, relievable %d" % (lab, n["d1"], n["unrel"], n["rel"]))
    for k in ks[-4:]:
        i = ks.index(k)
        print("     %-32s OR %.2f [%.2f, %.2f] z=%.2f" % (k, np.exp(b[i]), np.exp(b[i] - 1.96 * se[i]), np.exp(b[i] + 1.96 * se[i]), b[i] / se[i]))

# ---------------- TEST 2 ----------------
MOVE = sa.MOVE
ctx = {}
for path in {r["path"] for r in ps if r["model"] in STRONG and r["proto"] in PROSE}:
    it = json.load(open(os.path.join(sa.REPO, path, "interactions.json")))
    last, ply = {}, 0
    for turn in it["turns"]:
        for e in turn:
            a = e.get("action", {})
            if a.get("type") == "send message" and e.get("to", "").startswith("Player"):
                last[e["to"]] = a.get("content", "")
            elif a.get("type") == "move" and MOVE.match(a.get("content", "")):
                mover = MOVE.match(a["content"]).group(2)
                c = last.get("Player %d" % (ORDER.index(mover) + 1), "")
                ctx[(path, ply)] = c.split("CURRENT BOARD:")[0]
                ply += 1
assert all((r["path"], r["ply"]) in ctx for r in ps if r["model"] in STRONG and r["proto"] in PROSE)

LABEL = re.compile(r"^Player [A-F](?: \(you\))? says:\s*", re.M)
SENT = re.compile(r"[^.!?\n;]+")
DONEW = re.compile(r"\b(?:done|finish\w*|complete\w*|placed|settled)\b", re.I)


def ident(r, letter, cell):
    """Context names the done object's cell in the same sentence as its letter or a done word."""
    text = LABEL.sub("", ctx[(r["path"], r["ply"])])
    rx = re.compile(sa.CELL % (cell[0] + 1, cell[1] + 1), re.I)
    hits = [s for s in SENT.findall(text) if rx.search(s) and (re.search(r"\b%s\b" % letter, s) or DONEW.search(s))]
    return hits


def done_k1(r, x):
    """Done objects whose last free cell is this move's destination (only if no active X is sealed)."""
    if x["kill"]:
        return []
    return [o for k, d, o in x["adjX"] if k == 1 and d]


rows = [r for r in ps if r["model"] in STRONG and r["proto"] in PROSE and ok(r)]
for r in rows:
    for x in r["alts"]:
        x["dk1"] = done_k1(r, x)
        x["idhits"] = [h for o in x["dk1"] for h in ident(r, o, r["pos"][o])]
        x["k1_ident"] = bool(x["idhits"])
F2 = dict(M1, **{"k1 active X": lambda r, x: act(x),
                 "k1 done X, identifiable": lambda r, x: x["k1_ident"],
                 "k1 done X, not identifiable": lambda r, x: bool(x["dk1"]) and not x["k1_ident"]})
b, V = clogit(rows, F2)
se = np.sqrt(np.diag(V)); ks = list(F2)
print("\nTEST 2  prose arms, strong tier, M1 controls; OR [95% CI], board-clustered")
cnt = lambda f: (sum(f(x) for r in rows for x in r["alts"]), sum(f(r["alts"][r["chosen"]]) for r in rows))
for k, f in (("k1 active X", act), ("k1 done X, identifiable", lambda x: x["k1_ident"]),
             ("k1 done X, not identifiable", lambda x: bool(x["dk1"]) and not x["k1_ident"])):
    i = ks.index(k); a, c = cnt(f)
    print("   %-28s alternatives %3d, chosen %3d   OR %.2f [%.2f, %.2f]" % (k, a, c, np.exp(b[i]), np.exp(b[i] - 1.96 * se[i]), np.exp(b[i] + 1.96 * se[i])))
i, j = ks.index("k1 active X"), ks.index("k1 done X, identifiable")
z = (b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
print("   active vs identifiable-done: ratio %.2f z=%.2f" % (np.exp(b[i] - b[j]), z))
# raw choice shares within plies, for transparency
for k, f in (("k1 active X", act), ("k1 done X, identifiable", lambda x: x["k1_ident"])):
    rs = [r for r in rows if any(f(x) for x in r["alts"])]
    print("   plies offering %-26s %3d, that move played %d" % (k + ":", len(rs), sum(f(r["alts"][r["chosen"]]) for r in rs)))

if "--sample" in sys.argv:
    random.seed(3)
    cases = [(r, x) for r in rows for x in r["alts"] if x["k1_ident"]]
    for r, x in random.sample(cases, min(15, len(cases))):
        print("\n%s ply %d mover %s move %s done-k1 %s at %s played=%s" % (r["path"], r["ply"], r["player"], x["m"], x["dk1"],
              ["R%dC%d" % (r["pos"][o][0] + 1, r["pos"][o][1] + 1) for o in x["dk1"]], r["alts"][r["chosen"]] is x))
        for h in x["idhits"]:
            print("    hit:", " ".join(h.split())[:300])
