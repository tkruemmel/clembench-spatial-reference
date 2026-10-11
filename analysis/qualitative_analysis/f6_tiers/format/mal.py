import json, re, collections, os, sys, statistics as st
REPO = "/Users/tom/uni/clembench-spatial-reference"
sys.path.insert(0, REPO + "/matrixgame_covered_comm")
from master import _schema_violation, _parse_response
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
R = json.load(open(f"{REPO}/analysis/qualitative_analysis/episodes.{ARM}.json"))
why = collections.Counter(); mlen = []
runs = collections.Counter()
VERD = {"move", "invalid_move", "invalid_format", "malformed_message", "player_done", "invalid_done"}
for r in R:
    if r["model"] != "nemotron" or r["proto"] != "hybrid": continue
    it = json.load(open(os.path.join(REPO, r["path"], "interactions.json")))
    resp = None; pairs = []
    for t in it["turns"]:
        for e in t:
            ty = e["action"]["type"]; c = e["action"].get("content", "")
            if ty == "get message": resp = c
            elif ty in VERD:
                pairs.append(ty)
                if ty == "malformed_message":
                    p = _parse_response(resp, True)
                    msg = p.get("message", "") if p else ""
                    v = _schema_violation("hybrid", msg) or "NONE?"
                    mlen.append(len(msg.split()))
                    why[v.split(".")[0][:70]] += 1
                resp = None
    if r["outcome"] == "abort":
        run = []
        for p in reversed(pairs):
            if p in ("move", "player_done"): break
            run.append(p)
        runs["has_malformed" if "malformed_message" in run else "pure_invalid_move"] += 1
for k, v in why.most_common(): print(v, k)
print("malformed message mean/median words", round(st.mean(mlen),1), st.median(mlen))
print("hybrid aborts final run:", runs)
