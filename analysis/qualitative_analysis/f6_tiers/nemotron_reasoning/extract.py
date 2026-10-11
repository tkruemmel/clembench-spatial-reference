"""Per-model-call records for the corrected Nemotron campaigns (grid + compact).

Run from the repo root:  python3 <this dir>/extract.py
Writes calls.jsonl next to this file. One record per Nemotron call, aligned
with interactions.json (n-th 'Player k -> GM' event == n-th call in
player_k.requests.json), with the raw (pre-strip) text from the Bedrock log,
the game's verdict on the resulting move, and a replayed board so that the
pre-</think> draft move can be judged on the same state.
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from qual_corpus import ARM, tree                                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"calls.{ARM}.jsonl")
ROOTS = [tree("results_nemotron_fixthink_grid_n15", s) for s in ("solo", "dyad", "3p", "6p")]
if ARM == "original":  # the compact campaigns have no updated-prompt run
    ROOTS += [f"results_nemotron_fixthink_n15_{s}" for s in ("solo", "dyad", "3p", "6p")]

# Copied from matrixgame_covered_comm/master.py and matrixgame_covered_solo/master.py
MOVE_ONLY = re.compile(r"\A\s*move:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b", re.I)
COMM_MOVE = re.compile(r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b", re.I | re.S)
SOLO_MSG_MOVE = re.compile(r"\A\s*message:\s*(?P<message>.+?)\s*\nmove:\s*(?P<object>[A-Z])\s+(?P<direction>up|down|left|right)\b", re.I)
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OUTCOME_TYPES = {"move", "invalid_move", "invalid_format", "malformed_message", "player_done", "invalid_done"}
THINK = "</think>"


def pattern_for(proto):
    return {"none": MOVE_ONLY, "silent": MOVE_ONLY, "thinking": SOLO_MSG_MOVE}.get(proto, COMM_MOVE)


def parse_move(pat, text):
    m = pat.match(text)
    return (f"{m['object'].upper()} {m['direction'].lower()}", m.end()) if m else (None, 0)


def validate(pos, walls, n, allowed, move):
    """Mirror of utils/board.py Board.validate_move; returns code or 'legal'."""
    obj, d = move.split()
    if obj not in pos:
        return "unknown_object"
    if obj not in allowed:
        return "not_owner"
    r, c = pos[obj][0] + DIRS[d][0], pos[obj][1] + DIRS[d][1]
    if not (0 <= r < n and 0 <= c < n):
        return "off_grid"
    if (r, c) in walls:
        return "wall"
    if any(p == (r, c) for o, p in pos.items() if o != obj):
        return "occupied"
    return "legal"


def raw_text(call):
    blocks = call["raw_response_obj"]["output"]["message"]["content"]
    return "\n".join(str(b["text"]).strip() for b in blocks if "text" in b).strip(), \
        any(isinstance(b.get("reasoningContent"), dict) for b in blocks)


def episode(epdir, rend, suite):
    inter = json.load(open(f"{epdir}/interactions.json"))
    inst = json.load(open(f"{epdir}/instance.json"))
    proto = inst.get("comm_protocol") or ("thinking" if inst.get("thinking") else "silent")
    pat = pattern_for(proto)
    n, walls = inst["grid_size"], {tuple(w) for w in inst["walls"]}
    pos = {o: tuple(p) for o, p in inst["start_positions"].items()}
    players = [p for p in inter["players"] if p != "GM"]
    if "player_objects" in inst:
        own = {f"Player {i+1}": set(o) for i, o in enumerate(inst["player_objects"])}
    elif "player_a_objects" in inst:
        own = {"Player 1": set(inst["player_a_objects"]), "Player 2": set(inst["player_b_objects"])}
    else:
        own = {p: set(inst["objects"]) for p in players}
    calls = {}
    for p in players:
        f = f"{epdir}/player_{p.split()[-1]}.requests.json"
        calls[p] = json.load(open(f))["calls"] if os.path.exists(f) else []
    exp = inter["meta"]["experiment_name"]
    base = dict(rend=rend, suite=suite, exp=exp, level=inst["spatial_level"], proto=proto,
                ep=epdir, board=f"{re.sub(r'_(grid|Grid|Compact)$', '', exp)}#{inst['game_id']}",
                pair_board=f"{inst['spatial_level']}#{inst['game_id']}",
                start=json.dumps(sorted(inst["start_positions"].items())),
                ep_success=int(inter.get("Success", 0)), ep_aborted=int(inter.get("Aborted", 0)))
    events = [e for t in inter["turns"] for e in t]
    recs, cursor, streak, last = [], {p: 0 for p in players}, {p: 0 for p in players}, {}
    for i, e in enumerate(events):
        who = e["from"]
        if who == "GM" or e["to"] != "GM":
            continue
        call = calls[who][cursor[who]]["call"]
        cursor[who] += 1
        raw, has_rc = raw_text(call)
        vis = e["action"]["content"]
        # outcome = first GM self-log of an outcome type before the next player response
        outcome, code = "none", None
        for f in events[i + 1:]:
            if f["from"] != "GM":
                break
            t = f["action"]["type"]
            if f["to"] == "GM" and t in OUTCOME_TYPES:
                outcome = t
                if t == "invalid_move":
                    m = re.search(r"\[(\w+)\]", f["action"]["content"])
                    code = m.group(1) if m else "masked"
                break
        verdict = {"move": "legal", "invalid_move": code, "invalid_format": "format",
                   "malformed_message": "schema", "player_done": "done",
                   "invalid_done": "bad_done", "none": "none"}[outcome]
        move, _ = parse_move(pat, vis) if verdict not in ("format", "masked") else (None, 0)
        traced = THINK in raw
        draft = draft_code = None
        delib = 0
        if traced:
            pre = raw.rsplit(THINK, 1)[0]
            draft, end = parse_move(pat, pre)
            delib = len(pre) - end
            if draft:
                draft_code = validate(pos, walls, n, own[who], draft)
        my_code = validate(pos, walls, n, own[who], move) if move and verdict not in ("schema", "done", "bad_done") else None
        usage = call["raw_response_obj"].get("usage", {})
        prompt0 = call["manipulated_prompt_obj"][0]["content"]
        line = ("preamble, thinking, or scratchpad" if "Do not output any preamble, thinking, or scratchpad" in prompt0
                else "preamble or thinking" if "Do not output any preamble or thinking" in prompt0 else "absent")
        prev = last.get(who)
        recs.append(dict(base, player=who, k=len(recs), retry=streak[who],
                         prev_move=prev["move"] if prev and streak[who] else None,
                         prev_verdict=prev["verdict"] if prev and streak[who] else None,
                         verdict=verdict, move=move, my_code=my_code, traced=traced,
                         n_close=raw.count(THINK), open_tag="<think>" in raw, reasoning_block=has_rc,
                         draft=draft, draft_code=draft_code, delib_chars=delib,
                         out_tok=usage.get("outputTokens"), in_tok=usage.get("inputTokens"),
                         raw_chars=len(raw), vis_chars=len(vis), stop=call["raw_response_obj"].get("stopReason"),
                         extract_ok=(raw.rsplit(THINK, 1)[1].strip() if traced else raw) == vis.strip(),
                         prompt_line=line))
        if verdict in ("legal", "done"):
            streak[who] = 0
        else:
            streak[who] += 1
        last[who] = recs[-1]
        if verdict == "legal":
            obj, d = move.split()
            pos[obj] = (pos[obj][0] + DIRS[d][0], pos[obj][1] + DIRS[d][1])
    return recs


def main():
    n = 0
    with open(OUT, "w") as out:
        for root in ROOTS:
            rend = "grid" if "_grid_" in root else "compact"
            suite = root.rsplit("_", 1)[1]
            for f in sorted(glob.glob(f"{root}/*/*/*/instance_*/interactions.json")):
                for r in episode(os.path.dirname(f), rend, suite):
                    out.write(json.dumps(r) + "\n")
                    n += 1
    print(f"{n} call records -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
