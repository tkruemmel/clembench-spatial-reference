"""Item 2b: is the greedy move chosen blind to obstacles?  Fresh decisions only (first attempt of a turn).
Observed P(chosen move blocked | chosen move greedy) vs the blocked share among the greedy options on the board
(uniform over all own greedy options; uniform over the chosen object's greedy directions).
Also: off-grid / non-greedy rejections -- is the OPPOSITE direction the greedy one (direction flip)?"""
import json, collections, os
ARM = os.environ.get("ARM", "updated")  # arm of the team suites; solo is always original
A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"attempts.{ARM}.json")))["attempts"]
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
pct = lambda a, b: f"{100*a/b:5.1f}% ({a}/{b})" if b else "-"
for m in ("nemotron", "qwen38", "dsv41"):
    for s in ("solo", "dyad", "3p", "6p", "ALL"):
        X = [a for a in A if a["model"] == m and (s == "ALL" or a["suite"] == s)]
        fresh = [b for a, b in zip([None] + X, X) if b["verdict"] in ("move", "invalid_move") and "greedy" in b
                 and not (a and a["ep"] == b["ep"] and a["verdict"] == "invalid_move")]
        g = [a for a in fresh if a["greedy"] and a["g_opts"]]
        obs = sum(a["status"] != "ok" for a in g)
        exp_all = sum(a["g_blocked"] / a["g_opts"] for a in g)
        exp_obj = sum(a["obj_g_blocked"] / a["obj_g"] for a in g)
        ng = [a for a in fresh if not a["greedy"]]
        print(f"{m:8s} {s:4s} fresh {len(fresh):5d}  greedy-chosen blocked: obs {pct(obs, len(g))} | blind-uniform(all greedy opts) {100*exp_all/max(len(g),1):5.1f}% | blind(chosen obj's greedy dirs) {100*exp_obj/max(len(g),1):5.1f}%"
              f" || non-greedy chosen blocked {pct(sum(a['status']!='ok' for a in ng), len(ng))}")
    rej = [a for a in A if a["model"] == m and a["verdict"] == "invalid_move" and "greedy" in a and not a["greedy"]]
    for st in ("off_grid", "wall", "occ_own", "occ_other"):
        L = [a for a in rej if a["status"] == st]
        opp = sum(OPP[a["move"][1]] in a["greedy_dirs"] for a in L)
        ont = sum(a["on_target"] for a in L)
        print(f"   {m} non-greedy {st:9s} n={len(L):4d}: opposite dir is greedy {pct(opp, len(L))}; object already on target {pct(ont, len(L))}; axis {dict(collections.Counter('vert' if a['move'][1] in ('up','down') else 'horiz' for a in L))}")

print("\n== cleanest blindness test: fresh greedy choice for an object with 2 greedy dirs, exactly 1 blocked (blind chooser = 50%) ==")
for m in ("nemotron", "qwen38", "dsv41"):
    for s in ("solo", "dyad", "3p", "6p", "ALL"):
        X = [a for a in A if a["model"] == m and (s == "ALL" or a["suite"] == s)]
        fresh = [b for a, b in zip([None] + X, X) if b["verdict"] in ("move", "invalid_move") and "greedy" in b
                 and not (a and a["ep"] == b["ep"] and a["verdict"] == "invalid_move")]
        L = [a for a in fresh if a["greedy"] and a["obj_g"] == 2 and a["obj_g_blocked"] == 1]
        print(f"   {m:8s} {s:4s} picks the blocked greedy dir {pct(sum(a['status'] != 'ok' for a in L), len(L))}")
