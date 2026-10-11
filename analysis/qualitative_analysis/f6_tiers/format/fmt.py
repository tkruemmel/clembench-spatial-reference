import json, re, collections, os, statistics as st
REPO = "/Users/tom/uni/clembench-spatial-reference"
R = json.load(open(f"{REPO}/analysis/qualitative_analysis/episodes.json"))
V = collections.Counter(); W = collections.defaultdict(list); RW = collections.defaultdict(list)
ep_mal = collections.Counter(); mal_reason = collections.Counter()
for r in R:
    if r["suite"] == "solo": continue
    it = json.load(open(os.path.join(REPO, r["path"], "interactions.json")))
    key = (r["model"], r["proto"])
    has = False
    for t in it["turns"]:
        for e in t:
            ty = e["action"]["type"]; c = e["action"].get("content", "")
            V[key + (ty,)] += 1
            if ty == "message":
                W[key].append(len(c.split(":", 1)[-1].split()))
            if ty == "get message":
                RW[key].append(len(c.split()))
            if ty == "malformed_message":
                has = True
    ep_mal[key + (has, r["outcome"])] += 1
for m in ("nemotron", "qwen38", "dsv41"):
    for p in ("freeform", "hybrid", "structured", "none"):
        k = (m, p)
        print(m, p, "accepted msgs", len(W[k]), "mean words", round(st.mean(W[k]), 1) if W[k] else None,
              "| raw responses mean words", round(st.mean(RW[k]), 1),
              "| malformed", V[k + ("malformed_message",)], "inv_move", V[k + ("invalid_move",)], "inv_format", V[k + ("invalid_format",)],
              "| eps w/ malformed:", {o: ep_mal[k + (True, o)] for o in ("success", "abort", "lose")})
