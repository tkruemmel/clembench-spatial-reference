"""Does suppressed reasoning explain Nemotron's illegal moves? Reads calls.jsonl
(built by extract.py). Run from the repo root:
    python3 <this dir>/extract.py && python3 <this dir>/analyze.py [section ...]
Sections: 1 data, 2 trace, 3 length, 4 solo, 5 render, 6 request.
"""
import collections as C, glob, json, math, os, statistics as st, sys
from scipy.stats import binomtest, chi2, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
R = [json.loads(l) for l in open(os.path.join(HERE, "calls.jsonl"))]
ILL = {"wall", "off_grid", "not_owner", "unknown_object"}
REJ = ILL | {"occupied"}
MOVE_V = REJ | {"legal"}
ATT = [r for r in R if r["verdict"] in MOVE_V]          # move attempts (game counts these)
ill = lambda r: r["verdict"] in ILL
rej = lambda r: r["verdict"] in REJ


def rate(rs, f=ill):
    n = len(rs)
    k = sum(map(f, rs))
    return f"{k}/{n} ({100*k/n:.1f}%)" if n else "0/0"


def mh(rs, stratum, exposed, outcome):
    """Mantel-Haenszel risk difference (exposed - unexposed) and CMH p over strata
    containing both exposures."""
    S = C.defaultdict(lambda: [0, 0, 0, 0])            # a=exp&y, n1=exp, c=unexp&y, n0=unexp
    for r in rs:
        s = S[stratum(r)]
        if exposed(r):
            s[0] += outcome(r); s[1] += 1
        else:
            s[2] += outcome(r); s[3] += 1
    num = den = d = v = 0.0
    used = n_exp = 0
    for a, n1, c, n0 in S.values():
        if not n1 or not n0:
            continue
        N, m1 = n1 + n0, a + c
        num += (a * n0 - c * n1) / N; den += n1 * n0 / N
        d += a - n1 * m1 / N
        if N > 1:
            v += n1 * n0 * m1 * (N - m1) / (N * N * (N - 1))
        used += 1; n_exp += n1
    if not den:
        return "no informative strata"
    p = chi2.sf(d * d / v, 1) if v else float("nan")
    return f"RD {100*num/den:+.1f} pp, CMH p={p:.2g} ({used} strata, {n_exp} exposed)"


def mcnemar(pairs):
    """pairs of (x_bool, y_bool); returns discordant counts and exact p."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    p = binomtest(b, b + c).pvalue if b + c else 1.0
    return f"n={len(pairs)}, x-only {b} / y-only {c}, exact p={p:.2g}"


def retry_pairs(rs):
    """Retries after a rejected move whose own response is a move attempt."""
    return [r for r in rs if r["retry"] and r["prev_verdict"] in REJ and r["prev_move"] and r["verdict"] in MOVE_V]


rep = lambda r: r["move"] == r["prev_move"]
cell = lambda r: (r["rend"], r["suite"], r["level"], r["proto"], min(r["retry"], 1))


def s1():
    print("## 1. Data: raw logs, trace flag, prompt line")
    for rend in ("compact", "grid"):
        rs = [r for r in R if r["rend"] == rend]
        print(f"{rend}: {len(rs)} calls; closed trace (</think>) {rate(rs, lambda r: r['traced'])}; "
              f"'<think>' opener {sum(r['open_tag'] for r in rs)}; reasoningContent blocks {sum(r['reasoning_block'] for r in rs)}; "
              f"post-strip text == interactions.json {sum(r['extract_ok'] for r in rs)}/{len(rs)}; "
              f"replayed validator == GM verdict {sum(r['my_code'] == r['verdict'] for r in rs if r['my_code'])}/{sum(1 for r in rs if r['my_code'])}")
        print("  prompt line by protocol:", dict(C.Counter((r["proto"], r["prompt_line"]) for r in rs)))
        print("  trace rate by suite:", {s: rate([r for r in rs if r["suite"] == s], lambda r: r["traced"]) for s in ("solo", "dyad", "3p", "6p")})
    print("verdicts (all calls):", dict(C.Counter(r["verdict"] for r in R)))
    for rend in ("compact", "grid"):
        a = [r for r in ATT if r["rend"] == rend]
        print(f"{rend} per-attempt: illegal {rate(a)}, rejected incl. occupied {rate(a, rej)}; "
              f"episode-mean illegal {st.mean(ep_rates(a)):.3f}")


def ep_rates(a):
    E = C.defaultdict(list)
    for r in a:
        E[r["ep"]].append(ill(r))
    return [sum(v) / len(v) for v in E.values()]


def s2():
    print("\n## 2. Traced vs untraced calls (Nemotron, both renderings)")
    for rend in ("compact", "grid", None):
        a = [r for r in ATT if rend in (None, r["rend"])]
        for lab, sub in (("first attempt", [r for r in a if not r["retry"]]), ("retry", [r for r in a if r["retry"]])):
            t = [r for r in sub if r["traced"]]; u = [r for r in sub if not r["traced"]]
            print(f"{rend or 'pooled'} {lab}: traced illegal {rate(t)} rej {rate(t, rej)} | untraced illegal {rate(u)} rej {rate(u, rej)}")
    print("by suite (pooled renderings, all attempts):")
    for s in ("solo", "dyad", "3p", "6p"):
        a = [r for r in ATT if r["suite"] == s]
        print(f"  {s}: traced {rate([r for r in a if r['traced']])} | untraced {rate([r for r in a if not r['traced']])}")
    tr = lambda r: r["traced"]
    print("Stratified (exposure = traced; RD = traced - untraced):")
    print("  within episode x player x first/retry, illegal:", mh(ATT, lambda r: (r["ep"], r["player"], min(r["retry"], 1)), tr, ill))
    print("  within episode x player x first/retry, rejected:", mh(ATT, lambda r: (r["ep"], r["player"], min(r["retry"], 1)), tr, rej))
    print("  within rend x suite x level x protocol x first/retry, illegal:", mh(ATT, cell, tr, ill))
    sub = lambda r: r["traced"] and r["delib_chars"] >= 200
    a2 = [r for r in ATT if not r["traced"] or sub(r)]
    print("  substantive traces only (>=200 chars of deliberation), within episode, illegal:",
          mh(a2, lambda r: (r["ep"], r["player"], min(r["retry"], 1)), tr, ill))
    # draft vs final on the same board state (same turn)
    T = [r for r in ATT if r["traced"] and r["draft_code"]]
    print(f"Traced attempts with a parseable pre-</think> draft move: {len(T)} of {sum(r['traced'] for r in ATT)}; "
          f"draft == final {rate(T, lambda r: r['draft'] == r['move'])}")
    print(f"  draft illegal {rate(T, lambda r: r['draft_code'] in ILL)} -> final illegal {rate(T)}; McNemar (x=draft illegal, y=final illegal):",
          mcnemar([(r["draft_code"] in ILL, ill(r)) for r in T]))
    print(f"  draft rejected {rate(T, lambda r: r['draft_code'] in REJ)} -> final rejected {rate(T, rej)}; McNemar:",
          mcnemar([(r["draft_code"] in REJ, rej(r)) for r in T]))
    # repeats
    P = retry_pairs(R)
    t = [r for r in P if r["traced"]]; u = [r for r in P if not r["traced"]]
    print(f"Repeat of the just-rejected move: traced {rate(t, rep)} | untraced {rate(u, rep)}")
    for rend in ("compact", "grid"):
        print(f"  {rend}: traced {rate([r for r in t if r['rend']==rend], rep)} | untraced {rate([r for r in u if r['rend']==rend], rep)}")
    print("  stratified by cell x retry index:", mh(P, lambda r: cell(r) + (r["retry"],), lambda r: r["traced"], rep))
    TD = [r for r in t if r["draft"]]
    print(f"  traced retries with a draft: draft repeats {rate(TD, lambda r: r['draft'] == r['prev_move'])} -> final repeats {rate(TD, rep)}; McNemar:",
          mcnemar([(r["draft"] == r["prev_move"], rep(r)) for r in TD]))
    print("Selection: where do traces occur?")
    print("  trace rate by retry index:", {k: rate([r for r in R if min(r['retry'], 2) == k], lambda r: r['traced']) for k in (0, 1, 2)})
    print("  by protocol:", {p: rate([r for r in R if r['proto'] == p], lambda r: r['traced']) for p in ("none", "freeform", "hybrid", "structured", "silent", "thinking")})
    print("  by level:", {l: rate([r for r in R if r['level'] == l], lambda r: r['traced']) for l in ("S1", "S2", "S3", "S4")})
    print("  by episode call index:", {b: rate([r for r in R if lo <= r['k'] < hi], lambda r: r['traced'])
                                      for b, lo, hi in (("0", 0, 1), ("1-4", 1, 5), ("5-19", 5, 20), ("20+", 20, 10**9))})
    print("  episode outcome of traced vs untraced calls (success share):",
          rate([r for r in R if r["traced"]], lambda r: r["ep_success"]), "vs", rate([r for r in R if not r["traced"]], lambda r: r["ep_success"]))
    first = [r for r in T if not r["retry"]]
    print(f"  difficulty proxy, first attempts: traced turns' DRAFT illegal {rate(first, lambda r: r['draft_code'] in ILL)} "
          f"vs untraced first attempts illegal {rate([r for r in ATT if not r['traced'] and not r['retry']])}; "
          f"stratified (exposure = traced draft):",
          mh([r for r in ATT if not r["retry"] and (not r["traced"] or r["draft_code"])], lambda r: (r["ep"], r["player"]),
             lambda r: r["traced"], lambda r: (r["draft_code"] in ILL) if r["traced"] else ill(r)))


BINS = [(0, 20), (20, 60), (60, 100), (100, 150), (150, 250), (250, 500), (500, 1000), (1000, 3000), (3000, 10**7)]


def s3():
    print("\n## 3. Output tokens vs legality (move attempts)")
    groups = (("all", lambda r: True), ("message protocols, untraced", lambda r: r["proto"] not in ("none", "silent") and not r["traced"]),
              ("message protocols, untraced, FIRST attempts only", lambda r: r["proto"] not in ("none", "silent") and not r["traced"] and not r["retry"]),
              ("freeform only, untraced, first attempts", lambda r: r["proto"] == "freeform" and not r["traced"] and not r["retry"]),
              ("move-only (none/silent), untraced", lambda r: r["proto"] in ("none", "silent") and not r["traced"]),
              ("traced", lambda r: r["traced"]))
    for name, g in groups:
        a = [r for r in ATT if g(r) and r["out_tok"] is not None]
        print(f"{name}:")
        for lo, hi in BINS:
            b = [r for r in a if lo <= r["out_tok"] < hi]
            if b:
                print(f"  {lo}-{hi if hi < 10**7 else 'inf'} tok: illegal {rate(b)}, rejected {rate(b, rej)}, first-attempt share {sum(not r['retry'] for r in b)/len(b):.2f}")


def s4():
    print("\n## 4. Solo Silent vs Thinking")
    for rend in ("compact", "grid"):
        print(f"[{rend}]")
        rs = [r for r in R if r["rend"] == rend and r["suite"] == "solo"]
        for cond in ("silent", "thinking"):
            c = [r for r in rs if r["proto"] == cond]
            a = [r for r in c if r["verdict"] in MOVE_V]
            eps = {r["ep"]: r for r in c}
            succ = {l: sum(e["ep_success"] for e in eps.values() if e["level"] == l) for l in ("S1", "S2", "S3", "S4")}
            print(f"  {cond}: calls {len(c)}, attempts {len(a)}, illegal {rate(a)}, rejected {rate(a, rej)}, "
                  f"repeat-after-rejection {rate(retry_pairs(c), rep)}, format fails {sum(r['verdict']=='format' for r in c)}, "
                  f"traced {rate(c, lambda r: r['traced'])}, visible chars mean {st.mean(r['vis_chars'] for r in c):.0f}, "
                  f"out tokens mean {st.mean(r['out_tok'] or 0 for r in c):.0f} median {st.median(r['out_tok'] or 0 for r in c):.0f}, "
                  f"successes by level {succ}")
        # paired on identical boards: first move attempt (always the start state) and per-board illegal rate
        first = {}
        per = C.defaultdict(dict)
        for r in rs:
            if r["verdict"] not in MOVE_V:
                continue
            key = (r["pair_board"], r["start"])
            first.setdefault((key, r["proto"]), r)
            per[key].setdefault(r["proto"], []).append(ill(r))
        pairs = [(ill(first[(k, "silent")]), ill(first[(k, "thinking")])) for k in per
                 if (k, "silent") in first and (k, "thinking") in first]
        print("  first move attempt, paired boards (x=silent illegal, y=thinking illegal):", mcnemar(pairs))
        diffs = [sum(v["silent"]) / len(v["silent"]) - sum(v["thinking"]) / len(v["thinking"]) for v in per.values() if len(v) == 2]
        print(f"  per-board illegal rate silent - thinking: {len(diffs)} boards, mean {100*st.mean(diffs):+.1f} pp, "
              f"Wilcoxon p={wilcoxon([d for d in diffs if d]).pvalue:.2g}")
        print("  within cell (level x retry), RD thinking - silent, illegal:",
              mh([r for r in rs if r["verdict"] in MOVE_V], lambda r: (r["level"], min(r["retry"], 1)), lambda r: r["proto"] == "thinking", ill))


def s5():
    print("\n## 5. Compact vs grid")
    for s in ("solo", "dyad", "3p", "6p", None):
        row = []
        for rend in ("compact", "grid"):
            a = [r for r in ATT if r["rend"] == rend and s in (None, r["suite"])]
            c = [r for r in R if r["rend"] == rend and s in (None, r["suite"])]
            row.append(f"{rend}: illegal {rate(a)} rej {rate(a, rej)} repeat {rate(retry_pairs(c), rep)} traced {rate(c, lambda r: r['traced'])}")
        print(f"{s or 'all'}: " + " | ".join(row))
    first = {}
    for r in ATT:
        first.setdefault((r["board"], r["start"], r["rend"]), r)
    pairs = [(ill(first[(b, st_, "compact")]), ill(first[(b, st_, "grid")])) for (b, st_, rend) in first
             if rend == "compact" and (b, st_, "grid") in first]
    print("first move attempt on identical boards (x=compact illegal, y=grid illegal):", mcnemar(pairs))
    print("within cell (suite x level x protocol x first/retry), RD grid - compact, illegal:",
          mh(ATT, lambda r: cell(r)[1:], lambda r: r["rend"] == "grid", ill))
    print("  same, untraced calls only:", mh([r for r in ATT if not r["traced"]], lambda r: cell(r)[1:], lambda r: r["rend"] == "grid", ill))


def s6():
    print("\n## 6. Request parameters")
    for f in sorted(glob.glob("results_nemotron_fixthink*_n15_*/*/run.json")):
        if "stale" in f:
            continue
        pm = json.load(open(f))["player_models"]["0"]
        print(f"  {f}: gen_args {pm['gen_args']}, model_config {pm['model_spec'].get('model_config')}")
    ga = C.Counter()
    for f in glob.glob("results_nemotron_fixthink*_n15_*/*/*/*/instance_*/interactions.json"):
        if "stale" not in f:
            ga[json.dumps(json.load(open(f))["player_models"]["0"]["gen_args"], sort_keys=True)] += 1
    print("  interactions.json gen_args:", dict(ga))
    print("  stopReason:", dict(C.Counter(r["stop"] for r in R)), "max outputTokens", max(r["out_tok"] or 0 for r in R))
    print("  first prompt message role(s) are 'user' only; no system block; requests.json logs messages, not inferenceConfig")


def s7():
    print("\n## 7. Context: message line as partial scratchpad; strong-model per-attempt rates")
    mp = [r for r in ATT if r["suite"] != "solo"]
    for p in ("none", "structured", "hybrid", "freeform"):
        a = [r for r in mp if r["proto"] == p]
        print(f"  Nemotron multiplayer {p}: first-attempt illegal {rate([r for r in a if not r['retry']])}, all illegal {rate(a)}, "
              f"repeat {rate(retry_pairs([r for r in R if r['suite'] != 'solo' and r['proto'] == p]), rep)}")
    print("  within (rend x suite x level x first/retry), RD message-protocol - none, illegal:",
          mh(mp, lambda r: (r["rend"], r["suite"], r["level"], min(r["retry"], 1)), lambda r: r["proto"] != "none", ill))
    for pat in ("results_qwen38_grid_n15_", "results_dsv41_grid_n15_", "results_nemotron_fixthink_grid_n15_"):
        att = il = 0
        for s in ("solo", "dyad", "3p", "6p"):
            for f in glob.glob(f"{pat}{s}/*/*/*/instance_*/interactions.json"):
                x = json.load(open(f))
                att += x.get("Move Attempts", 0); il += x.get("Illegal Attempts", 0)
        print(f"  {pat}*: illegal {il}/{att} ({100*il/max(att,1):.1f}%) per move attempt")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for k in want:
        globals()["s" + k]()
