"""Output tokens per model call, the three models, by game.

    ARM=updated python3 analysis/qualitative_analysis/f6_tiers/nemotron_reasoning/tokens.py

One value per logged call in player_*.requests.json: the provider's own count
(`usage.outputTokens` on Bedrock, `usage.completion_tokens` on the OpenAI-style
endpoints), which includes hidden reasoning tokens where the provider bills
them.  Calls without a usage record are counted and skipped.  ARM picks the
team suites (qual_corpus.tree); solo is always the original campaign.
"""
import glob
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from qual_corpus import ARM, tree                                # noqa: E402

ROOT = "/Users/tom/uni/clembench-spatial-reference"
PREFIX = {"Nemotron": "results_nemotron_fixthink_grid_n15", "Qwen3.8": "results_qwen38_grid_n15",
          "DeepSeek": "results_dsv41_grid_n15"}


def out_tokens(call):
    u = (call.get("raw_response_obj") or {}).get("usage") or {}
    return u.get("outputTokens", u.get("completion_tokens"))


V = {}
for m, pre in PREFIX.items():
    for g in ("solo", "dyad", "3p", "6p"):
        v, miss = [], 0
        for f in sorted(glob.glob(os.path.join(ROOT, tree(pre, g), "**", "player_*.requests.json"), recursive=True)):
            for c in json.load(open(f))["calls"]:
                t = out_tokens(c["call"])
                if t is None:
                    miss += 1
                else:
                    v.append(t)
        V[(m, g)] = v
        print(f"{ARM} {m:9} {g:5} calls {len(v):6} (no usage {miss}) median {st.median(v):7.0f} mean {st.mean(v):7.0f}",
              flush=True)
for lab, gs in (("team suites", ("dyad", "3p", "6p")), ("all games", ("solo", "dyad", "3p", "6p"))):
    pool = {m: [t for g in gs for t in V[(m, g)]] for m in PREFIX}
    w = pool["Nemotron"]
    for m in ("Qwen3.8", "DeepSeek"):
        s = pool[m]
        print(f"{lab}: {m} / Nemotron  median {st.median(s):.0f}/{st.median(w):.0f} = {st.median(s) / st.median(w):.1f}x"
              f"  mean {st.mean(s):.0f}/{st.mean(w):.0f} = {st.mean(s) / st.mean(w):.1f}x")
