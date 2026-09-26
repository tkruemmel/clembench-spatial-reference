"""Reproduce Section 5.1-5.6 of results_three_models_grid.md with no LLM in the loop.

Stage 1 (extract): read every interactions.json of the three grid campaigns'
multiplayer suites, replay the board move by move, and annotate each broadcast
message with lexical features plus a grounding check of every positional claim
against the board state at the moment of utterance.

Stage 2 (report): print every table and every percentage that Sections 5.1-5.6
of the results document assert.

Usage:  python3 qual_corpus.py [--repo PATH] [--cache FILE] [--refresh]
"""
import argparse, collections, glob, json, os, re, statistics as st, string, sys

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
MOVE = re.compile(r"^Player (\w+) moved (\w+) (up|down|left|right)$")
SPK  = re.compile(r"^Player (\w+):\s*(.*)$", re.S)

# --- feature lexicon (identical to the annotation used for the document) ------
COORD = re.compile(r"\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?", re.I)
CLAIM = re.compile(r"\b([A-F])\b[^.;,\n]{0,24}?\bat\s*\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?", re.I)
REQ   = re.compile(r"(please|could you|can you|kindly|需要)", re.I)
MOVEREQ = re.compile(r"(?:move|shift|push|vacate|step)\s+(?:the\s+)?"
                     r"(?:X\s+at\s+\(?R\s*(\d+)\s*,?\s*C\s*(\d+)\)?|([A-F])\b)"
                     r"[^.;\n]{0,20}?\b(up|down|left|right)\b", re.I)
WAIT = re.compile(r"(i(?:'| a)?m? ?(?:will )?wait|waiting|cannot move|can't move|no legal move"
                  r"|blocked|stuck|trapped|hold(?:ing)? position|stay put|pass this turn)", re.I)
ACK  = re.compile(r"(thank|thanks|acknowledg|as requested|as you asked|got it|understood|confirmed|noted)", re.I)
SEQ  = re.compile(r"\b(after|then|once|first|before|next turn|meanwhile|afterwards|subsequently)\b", re.I)
CORR = re.compile(r"(corridor|dead ?-?end|shaft|cul-de-sac|column C\d|blind alley|pocket)", re.I)
THINK = re.compile(r"(</?think>|okay,? let me|let me think|first,? i need to figure)", re.I)
CJK  = re.compile(r"[　-鿿＀-￯]")

# --- features added for 5.2, 5.3 and 5.6 ------------------------------------
# All four are exact string tests, not judgements: a pseudo-action the rules do
# not offer, and the plea for a temporary shift that the schema cannot carry.
STAY = re.compile(r"\bmove:\s*[A-F]\s+(?:stay|wait)\b", re.I)
STAYARG = re.compile(r"not a valid|invalid|must move|rules require", re.I)
TEMP = re.compile(r"\btemporar(?:y|ily)\b", re.I)
TEMPRET = re.compile(r"\btemporar(?:y|ily)\b[^.;!?\n]{0,90}?"
                     r"\b(?:return|restore|back|then you can|and back)\b"
                     r"|\b(?:return|restore|move .{0,12}back)\b"
                     r"[^.;!?\n]{0,90}?\btemporar(?:y|ily)\b", re.I)
REQNONE = re.compile(r"REQUEST\s*:\s*none\b", re.I)
QUESTION = re.compile(r"\?")

ARMS = {"nemotron": "results_nemotron_fixthink_grid_n15",
        "qwen38":   "results_qwen38_grid_n15",
        "dsv41":    "results_dsv41_grid_n15"}
SUITES = ("dyad", "3p", "6p")
STRONG = ("qwen38", "dsv41")
PROSE  = ("freeform", "hybrid")
MODEL_LABEL = {"nemotron": "Nemotron", "qwen38": "Qwen3.8", "dsv41": "DeepSeek-V4.1"}


# ---------------------------------------------------------------- extraction
def extract(repo):
    rows, epi = [], []
    for model, pref in ARMS.items():
        for suite in SUITES:
            pat = os.path.join(repo, f"{pref}_{suite}", "*", "*", "*", "instance_*", "interactions.json")
            for f in sorted(glob.glob(pat)):
                d = os.path.dirname(f)
                ins = json.load(open(os.path.join(d, "instance.json")))
                es = json.load(open(os.path.join(d, "scores.json")))["episode scores"]
                proto = ins["comm_protocol"]
                pos = {o: tuple(q) for o, q in ins["start_positions"].items()}
                tgt = {o: tuple(q) for o, q in ins["target_positions"].items()}
                groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
                owner = {o: string.ascii_uppercase[i] for i, objs in enumerate(groups) for o in objs}
                mine = collections.defaultdict(list)
                for o, p in owner.items():
                    mine[p].append(o)
                it = json.load(open(f))
                evs = [(ti, e) for ti, t in enumerate(it["turns"]) for e in t]
                nmsg = 0
                pending, complied, requested = [], 0, 0
                snapshot = dict(pos)
                for ti, e in evs:
                    a = e.get("action", {})
                    t, c = a.get("type"), str(a.get("content", ""))
                    if t == "get message":
                        snapshot = dict(pos)
                    elif t == "move":
                        m = MOVE.match(c)
                        if m:
                            _, o, dr = m.groups()
                            dy, dx = DIRS[dr]
                            pos[o] = (pos[o][0] + dy, pos[o][1] + dx)
                            for pr in list(pending):
                                if pr["obj"] == o and pr["dir"] == dr:
                                    complied += 1
                                    pending.remove(pr)
                            for pr in pending:
                                pr["age"] += 1
                            pending = [pr for pr in pending if pr["age"] <= 6]
                    elif t == "message":
                        m = SPK.match(c)
                        if not m:
                            continue
                        spk, txt = m.group(1), m.group(2)
                        nmsg += 1
                        own_ok = own_bad = oth_ok = oth_bad = 0
                        for lt, r_, c_ in CLAIM.findall(txt):
                            lt = lt.upper()
                            p = (int(r_) - 1, int(c_) - 1)
                            if lt not in snapshot:
                                continue
                            ok = snapshot[lt] == p
                            if owner.get(lt) == spk:
                                own_ok += ok; own_bad += not ok
                            else:
                                oth_ok += ok; oth_bad += not ok
                        reqs = 0
                        for r_, c_, lt, dr in MOVEREQ.findall(txt):
                            target = None
                            if lt and lt.upper() in snapshot and owner.get(lt.upper()) != spk:
                                target = lt.upper()
                            elif r_ and c_:
                                p = (int(r_) - 1, int(c_) - 1)
                                target = next((o for o, q in snapshot.items()
                                               if q == p and owner.get(o) != spk), None)
                            if target:
                                reqs += 1
                                pending.append({"obj": target, "dir": dr.lower(), "age": 0})
                        requested += reqs
                        my_t = [tgt[o] for o in mine[spk]]
                        coords = COORD.findall(txt)
                        rows.append(dict(
                            model=model, suite=suite, lvl=ins["spatial_level"], proto=proto,
                            gid=ins["game_id"], su=bool(es.get("Success")), ab=bool(es.get("Aborted")),
                            spk=spk, words=len(txt.split()), ncoord=len(coords),
                            tgt_disclosed=any((int(a_) - 1, int(b_) - 1) in my_t for a_, b_ in coords),
                            req=bool(REQ.search(txt)), movereq=reqs, wait=bool(WAIT.search(txt)),
                            ack=bool(ACK.search(txt)), seq=bool(SEQ.search(txt)), corr=bool(CORR.search(txt)),
                            think=bool(THINK.search(txt)), cjk=bool(CJK.search(txt)),
                            reqnone=bool(REQNONE.search(txt)), ques=bool(QUESTION.search(txt)),
                            stay=bool(STAY.search(txt)),
                            stayarg=bool(STAY.search(txt) and STAYARG.search(txt)),
                            temp=bool(TEMP.search(txt)), tempret=bool(TEMPRET.search(txt)),
                            midx=nmsg - 1, turn=ti,
                            text=txt if len(txt) < 1200 else txt[:1200],
                            own_ok=own_ok, own_bad=own_bad, oth_ok=oth_ok, oth_bad=oth_bad,
                            names_other=any(owner.get(l.upper()) != spk
                                            for l in re.findall(r"\b([A-F])\b", txt) if l.upper() in owner)))
                epi.append(dict(model=model, suite=suite, lvl=ins["spatial_level"], proto=proto,
                                gid=ins["game_id"], su=bool(es.get("Success")), ab=bool(es.get("Aborted")),
                                nmsg=nmsg, requested=requested, complied=complied))
    return rows, epi


# ------------------------------------------------------------------ reporting
def rate(sel, key):
    return 100.0 * sum(r[key] for r in sel) / len(sel) if sel else 0.0


def report(rows, epi):
    from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu

    def hdr(s):
        print("\n" + "=" * 104); print(s); print("=" * 104)

    # ---------------------------------------------------------------- 5.1
    hdr("5.1  corpus and method")
    print(f"  messages                     {len(rows):,}")
    print(f"  multiplayer episodes         {len(epi):,}")
    print(f"  messages under protocol none {sum(1 for r in rows if r['proto'] == 'none')}")
    print(f"  reasoning-trace markers      {sum(1 for r in rows if r['think'])}"
          f"   by model: {dict(collections.Counter(r['model'] for r in rows if r['think']))}")
    print(f"  non-Latin script             {sum(1 for r in rows if r['cjk'])}"
          f"   by model: {dict(collections.Counter(r['model'] for r in rows if r['cjk']))}")
    s2_10 = [r for r in rows if r["lvl"] == "S2" and r["gid"] == 10]
    print(f"  S2 #10 contribution          {len(s2_10)} messages "
          f"({100.0 * len(s2_10) / len(rows):.2f}% of corpus)")

    # ---------------------------------------------------------------- 5.2
    hdr("5.2  what each protocol makes sayable  (multiplayer, grid)")
    cols = ("n msgs", "words", "coords", "target", "request", "blocked", "ack", "seq", "names other")
    print(f"  {'model':<14}{'protocol':<12}" + "".join(f"{c:>12}" for c in cols))
    for m in ("nemotron", "qwen38", "dsv41"):
        for p in ("structured", "freeform", "hybrid"):
            sel = [r for r in rows if r["model"] == m and r["proto"] == p]
            if not sel:
                continue
            print(f"  {MODEL_LABEL[m]:<14}{p:<12}{len(sel):>12,}"
                  f"{st.mean([r['words'] for r in sel]):>12.0f}"
                  f"{st.mean([r['ncoord'] for r in sel]):>12.1f}"
                  + "".join(f"{rate(sel, k):>11.0f}%" for k in
                            ("tgt_disclosed", "req", "wait", "ack", "seq", "names_other")))
    print("\n  the schema's request slot, and the plea for a temporary shift")
    print(f"  {'model':<14}{'protocol':<12}{'REQUEST: none':>15}{'temporarily':>13}{'+ the return':>14}")
    for m in ("nemotron", "qwen38", "dsv41"):
        for p in ("structured", "freeform", "hybrid"):
            sel = [r for r in rows if r["model"] == m and r["proto"] == p]
            if not sel:
                continue
            print(f"  {MODEL_LABEL[m]:<14}{p:<12}{rate(sel, 'reqnone'):>14.1f}%"
                  f"{sum(r['temp'] for r in sel):>13}{sum(r['tempret'] for r in sel):>14}")
    sprose = [r for r in rows if r["model"] in STRONG and r["proto"] in PROSE]
    sstruct = [r for r in rows if r["proto"] == "structured"]
    print(f"\n    strong-tier prose : {sum(r['temp'] for r in sprose)} of {len(sprose):,} say "
          f"'temporarily', {sum(r['tempret'] for r in sprose)} also name the return")
    print(f"    structured (all)  : {sum(r['temp'] for r in sstruct)} of {len(sstruct):,} say "
          f"'temporarily', {sum(r['tempret'] for r in sstruct)} also name the return")

    # ---------------------------------------------------------------- 5.3
    hdr("5.3  grounding: positional claims checked against the board at utterance time")
    print(f"  {'model':<14}{'protocol':<12}{'own claims':>12}{'accuracy':>10}"
          f"{'other claims':>14}{'accuracy':>10}")
    for m in ("nemotron", "qwen38", "dsv41"):
        for p in PROSE:
            sel = [r for r in rows if r["model"] == m and r["proto"] == p]
            if not sel:
                continue
            a, b = sum(r["own_ok"] for r in sel), sum(r["own_bad"] for r in sel)
            c, d = sum(r["oth_ok"] for r in sel), sum(r["oth_bad"] for r in sel)
            print(f"  {MODEL_LABEL[m]:<14}{p:<12}{a + b:>12,}{100 * a / (a + b) if a + b else 0:>9.1f}%"
                  f"{c + d:>14,}{100 * c / (c + d) if c + d else 0:>9.1f}%")
    print("\n  questions in the broadcast (freeform)")
    for m in ("nemotron", "qwen38", "dsv41"):
        sel = [r for r in rows if r["model"] == m and r["proto"] == "freeform"]
        print(f"    {MODEL_LABEL[m]:<14}questions {rate(sel, 'ques'):5.1f}%")
    print("\n  the pseudo-action: `move: <X> stay` / `move: <X> wait`")
    for m in ("nemotron", "qwen38", "dsv41"):
        sel = [r for r in rows if r["model"] == m]
        print(f"    {MODEL_LABEL[m]:<14}{sum(r['stay'] for r in sel):>4} messages,"
              f" {sum(r['stayarg'] for r in sel):>4} disputing it in the same message")

    # ---------------------------------------------------------------- 5.4
    hdr("5.4  requests and compliance (requested move performed within six turns)")
    print(f"  {'model':<14}" + "".join(f"{p:>22}" for p in ("structured", "freeform", "hybrid")))
    for m in ("qwen38", "dsv41", "nemotron"):
        cells = []
        for p in ("structured", "freeform", "hybrid"):
            e = [x for x in epi if x["model"] == m and x["proto"] == p]
            rq, cp = sum(x["requested"] for x in e), sum(x["complied"] for x in e)
            cells.append(f"{cp}/{rq} ({100 * cp / rq:.1f}%)" if rq else "-")
        print(f"  {MODEL_LABEL[m]:<14}" + "".join(f"{c:>22}" for c in cells))
    print("\n  requests per episode")
    for m in ("qwen38", "dsv41", "nemotron"):
        out = []
        for p in ("structured", "freeform", "hybrid"):
            e = [x for x in epi if x["model"] == m and x["proto"] == p]
            out.append(f"{p} {st.mean([x['requested'] for x in e]):.1f}" if e else f"{p} -")
        print(f"    {MODEL_LABEL[m]:<14}" + " | ".join(out))
    print("\n  per-episode compliance rate, successes vs failures (Mann-Whitney)")
    for m in ("qwen38", "dsv41"):
        for p in ("structured", "freeform", "hybrid"):
            e = [x for x in epi if x["model"] == m and x["proto"] == p and x["requested"] > 0]
            a = [x["complied"] / x["requested"] for x in e if x["su"]]
            b = [x["complied"] / x["requested"] for x in e if not x["su"]]
            if len(a) > 2 and len(b) > 2:
                u = mannwhitneyu(a, b, alternative="two-sided")
                print(f"    {MODEL_LABEL[m]:<14}{p:<12}success {100 * st.mean(a):5.1f}% (n={len(a):3d})"
                      f"   fail {100 * st.mean(b):5.1f}% (n={len(b):3d})   p={u.pvalue:.4g}")

    # ---------------------------------------------------------------- 5.5
    hdr("5.5  successful vs failed dialogue, within prose protocols "
        "(strong models, S2-S4)")
    sel = [r for r in rows if r["model"] in STRONG and r["proto"] in PROSE and r["lvl"] != "S1"]
    su = [r for r in sel if r["su"]]
    fa = [r for r in sel if not r["su"]]
    print(f"  {len(su):,} messages from successes, {len(fa):,} from failures")
    print(f"  {'feature':<28}{'success':>10}{'failure':>10}{'fisher p':>14}{'chi2 p':>14}")
    for label, k in (("target disclosed", "tgt_disclosed"), ("acknowledgement", "ack"),
                     ("sequencing", "seq"), ("corridor/dead-end", "corr"),
                     ("request", "req"), ("names teammate's object", "names_other"),
                     ("blocked / waiting / stuck", "wait")):
        a, b = sum(r[k] for r in su), sum(r[k] for r in fa)
        tab = [[a, len(su) - a], [b, len(fa) - b]]
        pf = fisher_exact(tab)[1]
        pc = chi2_contingency(tab)[1]
        print(f"  {label:<28}{100 * a / len(su):>9.1f}%{100 * b / len(fa):>9.1f}%{pf:>14.2g}{pc:>14.2g}")
    print(f"  {'mean words':<28}{st.mean([r['words'] for r in su]):>9.0f} {st.mean([r['words'] for r in fa]):>9.0f}")
    print(f"  {'mean coordinates':<28}{st.mean([r['ncoord'] for r in su]):>9.1f} {st.mean([r['ncoord'] for r in fa]):>9.1f}")
    print("\n  naive pooled-over-protocol comparison (the misleading one)")
    pool = [r for r in rows if r["model"] in STRONG and r["proto"] != "none" and r["lvl"] != "S1"]
    ps, pf_ = [r for r in pool if r["su"]], [r for r in pool if not r["su"]]
    print(f"    sequencing  success {rate(ps, 'seq'):.1f}%   failure {rate(pf_, 'seq'):.1f}%")
    print("\n  inside the structured protocol only")
    stc = [r for r in rows if r["model"] in STRONG and r["proto"] == "structured" and r["lvl"] != "S1"]
    ss, sf = [r for r in stc if r["su"]], [r for r in stc if not r["su"]]
    a, b = sum(r["req"] for r in ss), sum(r["req"] for r in sf)
    tab = [[a, len(ss) - a], [b, len(sf) - b]]
    print(f"    request     success {100 * a / len(ss):.1f}%   failure {100 * b / len(sf):.1f}%"
          f"   fisher p={fisher_exact(tab)[1]:.2g}  chi2 p={chi2_contingency(tab)[1]:.2g}")

    # ---------------------------------------------------------------- 5.6
    hdr("5.6  two capabilities the transcripts show missing")
    print("  a) the pseudo-action the rules do not offer")
    tot = sum(r["stay"] for r in rows)
    print(f"     {tot} messages emit `move: <X> stay|wait`; "
          f"{sum(r['stayarg'] for r in rows)} dispute it in the same message")
    print(f"     by model: {dict(collections.Counter(r['model'] for r in rows if r['stay']))}")
    print("\n  b) the worked episode: DeepSeek, 6p, S3, free-form, board #14")
    ep = sorted([r for r in rows if r["model"] == "dsv41" and r["suite"] == "6p"
                 and r["lvl"] == "S3" and r["proto"] == "freeform" and r["gid"] == 14],
                key=lambda r: r["midx"])
    if not ep:
        print("     (episode not found)")
        return
    by = collections.Counter(r["spk"] for r in ep)
    B = [r for r in ep if r["spk"] == "B"]
    promise = [r for r in B if re.search(r"\bI(?:'|\u2019)?ll\s+(?:wait|hold)\b", r["text"], re.I)
               and re.search(r"R5,?C1", r["text"], re.I)]
    take = next((r for r in B if "moves right to goal R5C2" in r["text"]), None)
    done = next((r for r in B if r["text"].strip() == "done"), None)
    print(f"     {len(ep)} messages; by speaker {dict(sorted(by.items()))}")
    print(f"     B promises to wait at (R5,C1) in {len(promise)} separate messages")
    if take and done:
        print(f"     B takes (R5,C2) at message {take['midx'] + 1}, declares done at message {done['midx'] + 1}")
        after = [r for r in ep if r["midx"] > done["midx"]]
        dref = [r for r in after if r["spk"] == "D" and re.search(r"R5C2", r["text"], re.I)]
        dask = [r for r in dref if re.search(r"\b(?:please|if you can|could you)\b", r["text"], re.I)]
        eref = [r for r in after if r["spk"] == "E" and re.search(r"R5C2", r["text"], re.I)]
        print(f"     after the done: {len(after)} messages remain; D raises (R5,C2) {len(dref)} times,"
              f" asking in {len(dask)}; E raises it {len(eref)} times")
        print(f"     B speaks again: {'yes' if any(r['spk'] == 'B' for r in after) else 'no'}")


# ---------------------------------------------------------------- verification
# Every value Sections 5.1-5.6 of results_three_models_grid.md assert, checked
# against what this script computes.  tol is in the unit of the value.
DOC = [
    ("5.1", "messages in corpus",            66584, lambda R, E: len(R), 0),
    ("5.1", "multiplayer episodes",           2160, lambda R, E: len(E), 0),
    ("5.1", "messages under protocol none",      0, lambda R, E: sum(1 for r in R if r["proto"] == "none"), 0),
    ("5.1", "reasoning-trace markers",           1, lambda R, E: sum(1 for r in R if r["think"]), 0),
    ("5.1", "non-Latin script messages",         0, lambda R, E: sum(1 for r in R if r["cjk"]), 0),
    ("5.1", "S2 #10 messages",                 1137, lambda R, E: sum(1 for r in R if r["lvl"] == "S2" and r["gid"] == 10), 0),
    ("5.1", "S2 #10 share of corpus (%)",     1.71, lambda R, E: 100 * sum(1 for r in R if r["lvl"] == "S2" and r["gid"] == 10) / len(R), 0.005),
]
for m, p_, n, w, co, td, rq, wt, ak, sq, no in [
    ("nemotron", "structured",   937, 10, 2.0, 92,  2,  0,  0,  0,  0),
    ("nemotron", "freeform",    2350,101,10.0, 98, 32, 40,  1, 50, 18),
    ("nemotron", "hybrid",      1465, 65, 8.0, 98, 29, 31,  0, 32,  0),
    ("qwen38",   "structured", 10954, 11, 2.5, 87, 25,  0,  0,  2,  6),
    ("qwen38",   "freeform",    10050, 58, 9.0, 92, 60,  3, 47, 86, 86),
    ("qwen38",   "hybrid",     10334, 63, 9.3, 93, 57,  3, 23, 78, 85),
    ("dsv41",    "structured", 10653, 11, 2.4, 92, 25,  0,  0,  2, 11),
    ("dsv41",    "freeform",    9906, 44, 7.2, 88, 58,  4, 19, 79, 83),
    ("dsv41",    "hybrid",      9935, 59, 8.1, 95, 60,  5,  8, 78, 85)]:
    sub = lambda R, mm=m, pp=p_: [r for r in R if r["model"] == mm and r["proto"] == pp]
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [
        ("5.2", f"{tag} n msgs",  n,  lambda R, E, f=sub: len(f(R)), 0),
        ("5.2", f"{tag} words",   w,  lambda R, E, f=sub: st.mean([x["words"] for x in f(R)]), 0.5),
        ("5.2", f"{tag} coords",  co, lambda R, E, f=sub: st.mean([x["ncoord"] for x in f(R)]), 0.05),
        ("5.2", f"{tag} target%", td, lambda R, E, f=sub: rate(f(R), "tgt_disclosed"), 0.5),
        ("5.2", f"{tag} request%",rq, lambda R, E, f=sub: rate(f(R), "req"), 0.5),
        ("5.2", f"{tag} blocked%",wt, lambda R, E, f=sub: rate(f(R), "wait"), 0.5),
        ("5.2", f"{tag} ack%",    ak, lambda R, E, f=sub: rate(f(R), "ack"), 0.5),
        ("5.2", f"{tag} seq%",    sq, lambda R, E, f=sub: rate(f(R), "seq"), 0.5),
        ("5.2", f"{tag} names-other%", no, lambda R, E, f=sub: rate(f(R), "names_other"), 0.5),
    ]
_sp = lambda R: [r for r in R if r["model"] in STRONG and r["proto"] in PROSE]
_ss = lambda R: [r for r in R if r["proto"] == "structured"]
DOC += [
    ("5.2", "Qwen structured REQUEST: none (%)",     54, lambda R, E: rate([r for r in R if r["model"] == "qwen38" and r["proto"] == "structured"], "reqnone"), 0.5),
    # the document truncates 54.6 to "54%"
    ("5.2", "DeepSeek structured REQUEST: none (%)", 54, lambda R, E: rate([r for r in R if r["model"] == "dsv41" and r["proto"] == "structured"], "reqnone"), 1.0),
    ("5.2", "Qwen hybrid REQUEST: none (%)",         27, lambda R, E: rate([r for r in R if r["model"] == "qwen38" and r["proto"] == "hybrid"], "reqnone"), 0.5),
    ("5.2", "DeepSeek hybrid REQUEST: none (%)",     18, lambda R, E: rate([r for r in R if r["model"] == "dsv41" and r["proto"] == "hybrid"], "reqnone"), 0.5),
    ("5.2", "strong-tier prose messages",         40225, lambda R, E: len(_sp(R)), 0),
    ("5.2", "  of those saying 'temporarily'",      514, lambda R, E: sum(r["temp"] for r in _sp(R)), 0),
    ("5.2", "  of those naming the return",          52, lambda R, E: sum(r["tempret"] for r in _sp(R)), 0),
    ("5.2", "structured messages",                22544, lambda R, E: len(_ss(R)), 0),
    ("5.2", "  of those saying 'temporarily'",        8, lambda R, E: sum(r["temp"] for r in _ss(R)), 0),
    ("5.2", "  of those naming the return",           0, lambda R, E: sum(r["tempret"] for r in _ss(R)), 0),
]
def _g(R, m, p, k):
    sel = [r for r in R if r["model"] == m and r["proto"] == p]
    a, b = sum(r[k + "_ok"] for r in sel), sum(r[k + "_bad"] for r in sel)
    return a + b, 100 * a / (a + b) if a + b else 0.0
for m, p_, on, oa, tn, ta in [
    ("nemotron", "freeform", 3601, 85.6,  562, 29.9),
    ("nemotron", "hybrid",   2332, 85.2,   52,  5.8),
    ("qwen38",   "freeform", 4205, 76.8, 2137, 80.5),
    ("qwen38",   "hybrid",   3179, 74.2, 2028, 84.0),
    ("dsv41",    "freeform", 3838, 82.4, 1433, 80.7),
    ("dsv41",    "hybrid",   4157, 84.9, 2131, 83.3)]:
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [
        ("5.3", f"{tag} own claims",    on, lambda R, E, a=m, b=p_: _g(R, a, b, "own")[0], 0),
        ("5.3", f"{tag} own accuracy",  oa, lambda R, E, a=m, b=p_: _g(R, a, b, "own")[1], 0.05),
        ("5.3", f"{tag} other claims",  tn, lambda R, E, a=m, b=p_: _g(R, a, b, "oth")[0], 0),
        ("5.3", f"{tag} other accuracy",ta, lambda R, E, a=m, b=p_: _g(R, a, b, "oth")[1], 0.05),
    ]
DOC += [
    ("5.3", "Nemotron freeform questions (%)",     11.7, lambda R, E: rate([r for r in R if r["model"] == "nemotron" and r["proto"] == "freeform"], "ques"), 0.05),
    ("5.3", "Qwen freeform questions (%)",          0.3, lambda R, E: rate([r for r in R if r["model"] == "qwen38" and r["proto"] == "freeform"], "ques"), 0.05),
    ("5.3", "DeepSeek freeform questions (%)",      0.2, lambda R, E: rate([r for r in R if r["model"] == "dsv41" and r["proto"] == "freeform"], "ques"), 0.05),
    ("5.3", "`move: <X> stay|wait` messages",        19, lambda R, E: sum(r["stay"] for r in R), 0),
    ("5.3", "  of those from Nemotron",              19, lambda R, E: sum(r["stay"] for r in R if r["model"] == "nemotron"), 0),
    ("5.3", "  of those disputed in-message",        18, lambda R, E: sum(r["stayarg"] for r in R), 0),
]
def _c(E, m, p):
    e = [x for x in E if x["model"] == m and x["proto"] == p]
    return sum(x["complied"] for x in e), sum(x["requested"] for x in e)
for m, p_, cp, rq in [("qwen38", "structured", 119, 134), ("qwen38", "freeform", 423, 463),
                      ("qwen38", "hybrid", 438, 476), ("dsv41", "structured", 50, 64),
                      ("dsv41", "freeform", 49, 57), ("dsv41", "hybrid", 101, 105),
                      ("nemotron", "freeform", 13, 26), ("nemotron", "hybrid", 2, 5)]:
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [("5.4", f"{tag} complied", cp, lambda R, E, a=m, b=p_: _c(E, a, b)[0], 0),
            ("5.4", f"{tag} requested", rq, lambda R, E, a=m, b=p_: _c(E, a, b)[1], 0)]
DOC += [
    ("5.4", "Qwen freeform requests/episode",  2.6, lambda R, E: st.mean([x["requested"] for x in E if x["model"] == "qwen38" and x["proto"] == "freeform"]), 0.05),
    ("5.4", "Qwen hybrid requests/episode",    2.6, lambda R, E: st.mean([x["requested"] for x in E if x["model"] == "qwen38" and x["proto"] == "hybrid"]), 0.05),
]
def _p55(R, k, which):
    from scipy.stats import fisher_exact
    sel = [r for r in R if r["model"] in STRONG and r["proto"] in PROSE and r["lvl"] != "S1"]
    su = [r for r in sel if r["su"]]; fa = [r for r in sel if not r["su"]]
    a, b = sum(r[k] for r in su), sum(r[k] for r in fa)
    if which == "s": return 100 * a / len(su)
    if which == "f": return 100 * b / len(fa)
    return fisher_exact([[a, len(su) - a], [b, len(fa) - b]])[1]
DOC += [
    ("5.5", "messages from successes", 32997, lambda R, E: len([r for r in R if r["model"] in STRONG and r["proto"] in PROSE and r["lvl"] != "S1" and r["su"]]), 0),
    ("5.5", "messages from failures",    794, lambda R, E: len([r for r in R if r["model"] in STRONG and r["proto"] in PROSE and r["lvl"] != "S1" and not r["su"]]), 0),
]
for label, k, sv, fv, pv in [("target disclosed", "tgt_disclosed", 92.3, 83.6, 9.9e-16),
                             ("acknowledgement", "ack", 24.9, 17.4, 5.6e-07),
                             ("sequencing", "seq", 81.9, 83.4, 0.30),
                             ("corridor/dead-end", "corr", 22.7, 22.9, 0.90),
                             ("request", "req", 59.6, 72.3, 1.8e-13),
                             ("names teammate's object", "names_other", 85.4, 92.4, 1.9e-09),
                             ("blocked/waiting/stuck", "wait", 4.0, 9.3, 7.8e-11)]:
    DOC += [("5.5", f"{label} success (%)", sv, lambda R, E, a=k: _p55(R, a, "s"), 0.05),
            ("5.5", f"{label} failure (%)", fv, lambda R, E, a=k: _p55(R, a, "f"), 0.05),
            ("5.5", f"{label} fisher p",    pv, lambda R, E, a=k: _p55(R, a, "p"), abs(pv) * 0.06)]
def _m55(R, k, which):
    """The two per-message means of 5.5, and the Mann-Whitney they were missing.

    tab:dialogue prints the test, and the thesis bullet now states the size of
    the difference, so both need a check: length and coordinate density do
    separate the outcomes, by 2.3 words and 0.7 coordinates over 33 425
    messages, which is significance without a difference worth reporting as one.
    """
    from scipy.stats import mannwhitneyu
    sel = [r for r in R if r["model"] in STRONG and r["proto"] in PROSE and r["lvl"] != "S1"]
    a = [r[k] for r in sel if r["su"]]; b = [r[k] for r in sel if not r["su"]]
    if which == "s": return st.mean(a)
    if which == "f": return st.mean(b)
    return mannwhitneyu(a, b).pvalue
for label, k, sv, fv, pv in [("mean words", "words", 56.6, 58.8, 4.4e-05),
                             ("mean coordinates", "ncoord", 8.7, 9.4, 5.4e-07)]:
    DOC += [("5.5", f"{label} success", sv, lambda R, E, a=k: _m55(R, a, "s"), 0.05),
            ("5.5", f"{label} failure", fv, lambda R, E, a=k: _m55(R, a, "f"), 0.05),
            ("5.5", f"{label} Mann-Whitney p", pv, lambda R, E, a=k: _m55(R, a, "p"), abs(pv) * 0.06)]
DOC += [
    ("5.5", "pooled sequencing success (%)", 56.9, lambda R, E: rate([r for r in R if r["model"] in STRONG and r["proto"] != "none" and r["lvl"] != "S1" and r["su"]], "seq"), 0.05),
    ("5.5", "pooled sequencing failure (%)", 18.3, lambda R, E: rate([r for r in R if r["model"] in STRONG and r["proto"] != "none" and r["lvl"] != "S1" and not r["su"]], "seq"), 0.05),
]
def _struct55(R, which):
    """5.5's closing sentence: the same pattern inside the structured protocol."""
    from scipy.stats import fisher_exact
    stc = [r for r in R if r["model"] in STRONG and r["proto"] == "structured" and r["lvl"] != "S1"]
    ss, sf = [r for r in stc if r["su"]], [r for r in stc if not r["su"]]
    a, b = sum(r["req"] for r in ss), sum(r["req"] for r in sf)
    if which == "s": return 100 * a / len(ss)
    if which == "f": return 100 * b / len(sf)
    return fisher_exact([[a, len(ss) - a], [b, len(sf) - b]])[1]


def _comply55(E, which):
    """5.4's one case where compliance separates the outcomes directly."""
    from scipy.stats import mannwhitneyu
    e = [x for x in E if x["model"] == "dsv41" and x["proto"] == "structured" and x["requested"] > 0]
    a = [x["complied"] / x["requested"] for x in e if x["su"]]
    b = [x["complied"] / x["requested"] for x in e if not x["su"]]
    if which == "s": return 100 * st.mean(a)
    if which == "f": return 100 * st.mean(b)
    return mannwhitneyu(a, b, alternative="two-sided").pvalue


DOC += [
    ("5.4", "DeepSeek structured compliance, success (%)",   95, lambda R, E: _comply55(E, "s"), 0.5),
    ("5.4", "DeepSeek structured compliance, failure (%)",   37, lambda R, E: _comply55(E, "f"), 0.5),
    ("5.4", "DeepSeek structured compliance Mann-Whitney p", 0.0007, lambda R, E: _comply55(E, "p"), 5e-05),
    ("5.5", "within-structured request success (%)",        26.4, lambda R, E: _struct55(R, "s"), 0.05),
    ("5.5", "within-structured request failure (%)",        37.9, lambda R, E: _struct55(R, "f"), 0.05),
    ("5.5", "within-structured request fisher p",           4e-39, lambda R, E: _struct55(R, "p"), 4e-39 * 0.06),
]


def _ep(R):
    return sorted([r for r in R if r["model"] == "dsv41" and r["suite"] == "6p" and r["lvl"] == "S3"
                   and r["proto"] == "freeform" and r["gid"] == 14], key=lambda r: r["midx"])
def _epq(R, what):
    ep = _ep(R)
    B = [r for r in ep if r["spk"] == "B"]
    if what == "n": return len(ep)
    if what == "promise":
        return len([r for r in B if re.search(r"\bI(?:'|\u2019)?ll\s+(?:wait|hold)\b", r["text"], re.I)
                    and re.search(r"R5,?C1", r["text"], re.I)])
    take = next(r for r in B if "moves right to goal R5C2" in r["text"])
    done = next(r for r in B if r["text"].strip() == "done")
    if what == "take": return take["midx"] + 1
    if what == "done": return done["midx"] + 1
    after = [r for r in ep if r["midx"] > done["midx"]]
    if what == "after": return len(after)
    dref = [r for r in after if r["spk"] == "D" and re.search(r"R5C2", r["text"], re.I)]
    if what == "dref": return len(dref)
    if what == "dask":
        return len([r for r in dref if re.search(r"\b(?:please|if you can|could you)\b", r["text"], re.I)])
    if what == "eref":
        return len([r for r in after if r["spk"] == "E" and re.search(r"R5C2", r["text"], re.I)])
    if what == "bafter": return len([r for r in after if r["spk"] == "B"])
DOC += [
    ("5.6", "S3 #14: messages in episode",        101, lambda R, E: _epq(R, "n"), 0),
    ("5.6", "S3 #14: B promises to wait",           6, lambda R, E: _epq(R, "promise"), 0),
    ("5.6", "S3 #14: B takes (R5,C2) at message",  50, lambda R, E: _epq(R, "take"), 0),
    ("5.6", "S3 #14: B declares done at message",  56, lambda R, E: _epq(R, "done"), 0),
    ("5.6", "S3 #14: messages after the done",     45, lambda R, E: _epq(R, "after"), 0),
    ("5.6", "S3 #14: D raises (R5,C2) after",      19, lambda R, E: _epq(R, "dref"), 0),
    ("5.6", "S3 #14: D asks B to move",            18, lambda R, E: _epq(R, "dask"), 0),
    ("5.6", "S3 #14: E raises (R5,C2) after",      18, lambda R, E: _epq(R, "eref"), 0),
    ("5.6", "S3 #14: B speaks after the done",      0, lambda R, E: _epq(R, "bafter"), 0),
]


# ------------------------------------------------- Part II, Section 2.3 (RQ2)
# Table 12 answers RQ2's "presence, expressiveness or grounding" question on
# the dyad alone, because RQ2 is scoped to the dyad.  Same lexicon, same replay,
# restricted to dyad broadcasts.
_dy = lambda R, m, p: [r for r in R if r["suite"] == "dyad" and r["model"] == m and r["proto"] == p]
for m, p_, n, w, co, td, rq, ak, sq, no, rn in [
    ("qwen38",   "structured", 3539, 12, 2.5, 89, 26,  0,  1,  4, 57),
    ("qwen38",   "freeform",   3033, 59, 9.5, 96, 63, 47, 89, 79, None),
    ("qwen38",   "hybrid",     3136, 64, 9.5, 97, 59, 26, 79, 78, 24),
    ("dsv41",    "structured", 3418, 11, 2.3, 94, 20,  0,  1,  3, 66),
    ("dsv41",    "freeform",   3020, 46, 7.3, 92, 55, 14, 78, 75, None),
    ("dsv41",    "hybrid",     3082, 59, 8.1, 97, 58,  8, 76, 79, 23),
    ("nemotron", "structured",  334, 10, 2.0, 94,  3,  0,  0,  0, 95),
    ("nemotron", "freeform",    741,105,10.9, 99, 18,  0, 42,  6, None),
    ("nemotron", "hybrid",      533, 65, 8.9, 99, 23,  0, 20,  0, 79)]:
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [
        ("II.2.3", f"{tag} dyad n msgs",   n,  lambda R, E, a=m, b=p_: len(_dy(R, a, b)), 0),
        ("II.2.3", f"{tag} dyad words",    w,  lambda R, E, a=m, b=p_: st.mean([x["words"] for x in _dy(R, a, b)]), 0.5),
        ("II.2.3", f"{tag} dyad coords",   co, lambda R, E, a=m, b=p_: st.mean([x["ncoord"] for x in _dy(R, a, b)]), 0.05),
        ("II.2.3", f"{tag} dyad target%",  td, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "tgt_disclosed"), 0.5),
        ("II.2.3", f"{tag} dyad request%", rq, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "req"), 0.5),
        ("II.2.3", f"{tag} dyad ack%",     ak, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "ack"), 0.5),
        ("II.2.3", f"{tag} dyad seq%",     sq, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "seq"), 0.5),
        ("II.2.3", f"{tag} dyad names-other%", no, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "names_other"), 0.5),
    ]
    if rn is not None:
        DOC += [("II.2.3", f"{tag} dyad REQUEST: none%", rn, lambda R, E, a=m, b=p_: rate(_dy(R, a, b), "reqnone"), 0.5)]

def _dg(R, m, p, k):
    sel = _dy(R, m, p)
    a, b = sum(r[k + "_ok"] for r in sel), sum(r[k + "_bad"] for r in sel)
    return a + b, 100 * a / (a + b) if a + b else 0.0
for m, p_, tn, ta in [("qwen38", "freeform", 1317, 86.5), ("qwen38", "hybrid", 959, 89.1),
                      ("dsv41", "freeform", 722, 85.0), ("dsv41", "hybrid", 861, 87.5),
                      ("nemotron", "freeform", 46, 41.3), ("nemotron", "hybrid", 4, 25.0)]:
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [
        ("II.2.3", f"{tag} dyad other claims",   tn, lambda R, E, a=m, b=p_: _dg(R, a, b, "oth")[0], 0),
        ("II.2.3", f"{tag} dyad other accuracy", ta, lambda R, E, a=m, b=p_: _dg(R, a, b, "oth")[1], 0.05),
    ]
for m, p_, on, oa in [("qwen38", "structured", 2, 50.0), ("qwen38", "freeform", 2143, 76.5),
                      ("qwen38", "hybrid", 1283, 75.0), ("dsv41", "freeform", 1804, 83.4),
                      ("dsv41", "hybrid", 1702, 84.0), ("nemotron", "freeform", 1885, 86.5),
                      ("nemotron", "hybrid", 1137, 88.1)]:
    tag = f"{MODEL_LABEL[m]}/{p_}"
    DOC += [
        ("II.2.3", f"{tag} dyad own claims",   on, lambda R, E, a=m, b=p_: _dg(R, a, b, "own")[0], 0),
        ("II.2.3", f"{tag} dyad own accuracy", oa, lambda R, E, a=m, b=p_: _dg(R, a, b, "own")[1], 0.05),
    ]
_dyp = lambda R: [r for r in R if r["suite"] == "dyad" and r["model"] in STRONG and r["proto"] in PROSE]
_dys = lambda R: [r for r in R if r["suite"] == "dyad" and r["model"] in STRONG and r["proto"] == "structured"]
DOC += [
    ("II.2.3", "dyad messages, all models",           20836, lambda R, E: len([r for r in R if r["suite"] == "dyad"]), 0),
    ("II.2.3", "strong-tier dyad prose messages",     12271, lambda R, E: len(_dyp(R)), 0),
    ("II.2.3", "  of those saying 'temporarily'",       144, lambda R, E: sum(r["temp"] for r in _dyp(R)), 0),
    ("II.2.3", "strong-tier dyad structured messages",  6957, lambda R, E: len(_dys(R)), 0),
    ("II.2.3", "  of those saying 'temporarily'",          0, lambda R, E: sum(r["temp"] for r in _dys(R)), 0),
]


def verify(rows, epi):
    print("\n" + "=" * 104)
    print("VERIFICATION against results_three_models_grid.md")
    print("=" * 104)
    print(f"  {'sec':<9}{'claim':<46}{'document':>14}{'computed':>14}   ")
    ok = bad = 0
    for sec, label, asserted, fn, tol in DOC:
        try:
            got = fn(rows, epi)
        except Exception as e:
            print(f"  {sec:<9}{label:<46}{'':>14}{'ERROR':>14}   {e}")
            bad += 1
            continue
        hit = abs(got - asserted) <= tol
        ok, bad = ok + hit, bad + (not hit)
        fmt = (lambda v: f"{v:.2g}") if 0 < abs(asserted) < 0.01 else (
              (lambda v: f"{v:,.0f}") if float(asserted).is_integer() and abs(asserted) >= 10 else (lambda v: f"{v:.1f}"))
        print(f"  {sec:<9}{label:<46}{fmt(asserted):>14}{fmt(got):>14}   {'match' if hit else 'DIFF'}")
    print(f"\n  {ok} of {ok + bad} asserted values reproduced; {bad} differ.")
    print("  Every check is an exact string, count or replay test. No reconstructed lexicons remain.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/Users/tom/uni/clembench-spatial-reference")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check every value the document asserts")
    a = ap.parse_args()
    if a.refresh or not os.path.exists(a.cache):
        print("extracting corpus ...", file=sys.stderr)
        rows, epi = extract(a.repo)
        json.dump({"rows": rows, "epi": epi}, open(a.cache, "w"))
        print(f"cached {len(rows):,} messages -> {a.cache}", file=sys.stderr)
    else:
        d = json.load(open(a.cache))
        rows, epi = d["rows"], d["epi"]
    report(rows, epi)
    if a.verify:
        verify(rows, epi)


if __name__ == "__main__":
    main()
