"""Task 3 (V3): implicit-pass language.

a) rates of self-hold / ask-others-to-hold / explicit no-pass statements
b) enclosure victims (6p, prose, spells formed in play from seal_averted.spells):
   did the victim's own last message before the seal announce holding/waiting?
   baseline: every active non-moving player's last message at every ply.
   And the sealer's message at the sealing move: does it ask someone to hold/wait?
"""
import sys
sys.dont_write_bytecode = True
import collections, json, os
from verbal import load, tag_sentence, SENT, QA, spells

S, P, E = load()
EP = {e["path"]: e for e in E}


def tags(ep, spk, txt):
    own = set(ep["mine"][spk]); mates = set(ep["tgt"]) - own
    out = collections.Counter()
    for s in SENT.findall(txt):
        for k, v in tag_sentence(s, own, mates).items():
            out[k] |= v
    return out


def rates():
    print("a) per 1000 messages (6p: speaker not all-home = an active player that must move)")
    for tier in ("strong", "weak"):
        for suite in ("6p", "3p", "dyad"):
            ms = [m for m in S if m["tier"] == tier and m["suite"] == suite]
            act = [m for m in ms if not m["home"]]
            f = lambda sel, k: sum(m["tags"].get(k, False) for m in sel)
            print(f"  {tier:6} {suite:4} msgs={len(ms):6}  hold_self {f(ms,'hold_self'):5} ({1000*f(ms,'hold_self')/len(ms):.0f}/1k)"
                  f"  in-place {f(ms,'hold_self_place'):5} ({1000*f(ms,'hold_self_place')/len(ms):.0f}/1k)"
                  f"  ask-others {f(ms,'hold_other'):5} ({1000*f(ms,'hold_other')/len(ms):.0f}/1k)"
                  f"  no-pass stated {f(ms,'nopass'):4} ({1000*f(ms,'nopass')/len(ms):.1f}/1k)"
                  f"  | not-home speakers {len(act)}: hold_self {f(act,'hold_self')} ({1000*f(act,'hold_self')/max(len(act),1):.0f}/1k)")


def last_msg(ep, q, k):
    m = [(j, txt) for j, spk, txt in ep["msgs"] if spk == q and j < k]
    return m[-1] if m else None


def victims():
    eps = json.load(open(os.path.join(QA, "episodes.json")))
    print("\nb) enclosure victims, 6p prose, spells formed in play")
    for tier, models in (("strong", ("qwen38", "dsv41")), ("weak", ("nemotron",))):
        vic = collections.Counter(); base = collections.Counter(); seal = collections.Counter()
        quotes = []
        for rec in eps:
            if rec["suite"] != "6p" or rec["proto"] not in ("freeform", "hybrid") or rec["model"] not in models:
                continue
            ep = EP[rec["path"]]
            for sp in spells(rec):
                if sp["opening"]:
                    continue
                q, s = sp["player"], sp["start"]
                lm = last_msg(ep, q, s)    # written before the sealing move (ply s-1) was played
                vic["n"] += 1
                if lm:
                    vic["spoke"] += 1
                    t = tags(ep, q, lm[1])
                    for k in ("hold_self", "hold_self_place", "hold_other", "nopass"):
                        vic[k] += t[k]
                    if t["hold_self"]:
                        quotes.append((rec["model"], rec["level"], rec["proto"], rec["inst"], q, lm[0], s, lm[1]))
                # the sealer's own message at the sealing move
                sm = [txt for j, spk, txt in ep["msgs"] if spk == sp["cause"] and j == s - 1]
                if sm:
                    seal["spoke"] += 1
                    t = tags(ep, sp["cause"], sm[0])
                    seal["hold_other"] += t["hold_other"]
                    seal["hold_self"] += t["hold_self"]
            # baseline: every active non-mover's last message at every ply
            s4 = ep["lvl"] == "S4"
            cache = {}
            for k in range(len(ep["movers"])):
                if s4 and k <= 1:
                    continue
                for q in ep["mine"]:
                    if q in ep["done"][k] or q == ep["movers"][k]:
                        continue
                    lm = last_msg(ep, q, k)
                    if not lm:
                        continue
                    base["n"] += 1
                    key = (q, lm[0])
                    if key not in cache:
                        cache[key] = tags(ep, q, lm[1])
                    for kk in ("hold_self", "hold_self_place", "hold_other", "nopass"):
                        base[kk] += cache[key][kk]
        allm = [m for m in S if m["tier"] == tier and m["suite"] == "6p"]
        ho = sum(m["tags"].get("hold_other", False) for m in allm)
        print(f"  {tier}: spells {vic['n']}, victim had spoken {vic['spoke']}")
        for k in ("hold_self", "hold_self_place", "hold_other", "nopass"):
            print(f"    victim last msg {k:16} {vic[k]:3}/{vic['spoke']} ({100*vic[k]/max(vic['spoke'],1):.1f}%)"
                  f"   baseline (active non-mover x ply) {base[k]}/{base['n']} ({100*base[k]/max(base['n'],1):.1f}%)")
        print(f"    sealer's message at the sealing move: asks others to hold {seal['hold_other']}/{seal['spoke']}"
              f" ({100*seal['hold_other']/max(seal['spoke'],1):.1f}%), self-hold {seal['hold_self']}/{seal['spoke']}"
              f"   | all 6p messages ask-others-to-hold {ho}/{len(allm)} ({100*ho/len(allm):.1f}%)")
        for x in quotes[:12]:
            print("     ", x[:7], " ".join(x[7].split())[:300])


if __name__ == "__main__":
    rates()
    victims()
