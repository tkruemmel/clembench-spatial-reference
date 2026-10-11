"""V3 follow-up: does the victim's last message plan a pass-by (step aside, let the
teammate through, then return) -- a plan that needs a turn spent in place?"""
import sys
sys.dont_write_bytecode = True
import collections, json, os, re
from scipy.stats import binomtest, fisher_exact
from verbal import load, QA, spells, cell_rx
from t3 import EP, last_msg, tags
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original

RETURN = re.compile(r"\b(?:return|re-?enter|rejoin|come back|go back|retake|step back|back (?:to|into|down|up|left|right))\b"
                    r"|\b(?:after|once|when|until)\s+(?:you|(?-i:[A-F]))\s+(?:pass|passes|clear|clears|has passed|have passed|go by|is through|are through)\b", re.I)
eps = json.load(open(os.path.join(QA, f"episodes.{ARM}.json")))
for tier, models in (("strong", ("qwen38", "dsv41")), ("weak", ("nemotron",))):
    v = collections.Counter(); b = collections.Counter(); inv = collections.Counter()
    for rec in eps:
        if rec["suite"] != "6p" or rec["proto"] not in ("freeform", "hybrid") or rec["model"] not in models:
            continue
        ep = EP[rec["path"]]
        for sp in spells(rec):
            if sp["opening"]:
                continue
            lm = last_msg(ep, sp["player"], sp["start"])
            if not lm:
                continue
            v["n"] += 1
            v["ret"] += bool(RETURN.search(lm[1]))
            t = tags(ep, sp["player"], lm[1])
            v["either"] += bool(RETURN.search(lm[1])) or t["hold_self"]
            # invitation: victim's last message names the cell the sealer then took, with the sealer's letter
            a, d = sp["cause_by"]
            pos = ep["pos"][sp["start"] - 1][a]
            dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[d]
            cell = (pos[0] + dr, pos[1] + dc)
            sent_hit = [s for s in re.split(r"[.;!?\n]", lm[1]) if cell_rx([cell]).search(s)]
            inv["n"] += 1
            inv["cell"] += bool(sent_hit)
            inv["cell+sealer"] += any(re.search(r"(?-i:\b%s\b)" % sp["cause"], s) for s in sent_hit)
        s4 = ep["lvl"] == "S4"
        for k in range(len(ep["movers"])):
            if s4 and k <= 1:
                continue
            for q in ep["mine"]:
                if q in ep["done"][k] or q == ep["movers"][k]:
                    continue
                lm = last_msg(ep, q, k)
                if lm:
                    b["n"] += 1
                    b["ret"] += bool(RETURN.search(lm[1]))
    p = binomtest(v["ret"], v["n"], b["ret"] / b["n"]).pvalue
    print(f"{tier}: victim last msg plans a return/pass-by {v['ret']}/{v['n']} ({100*v['ret']/v['n']:.0f}%)"
          f" vs baseline {b['ret']}/{b['n']} ({100*b['ret']/b['n']:.1f}%), binom p={p:.2g};"
          f" return-or-hold {v['either']}/{v['n']}")
    print(f"   victim last msg names the cell the sealer then entered: {inv['cell']}/{inv['n']},"
          f" same sentence names the sealer: {inv['cell+sealer']}/{inv['n']}")
