"""Message corpus: parser, owner map, replay, visibility, message and hazard records, self-checks.

Spec: thesis/message_analysis_design.md §2-§3. Promoted from the X1 study
(thesis/review/qual_design/X1/x1lib.py); one state machine serves both arms.

    python3 -m analysis.messages.corpus dev      # build, self-check, print sizes
    python3 -m analysis.messages.corpus calib

Scopes: `dev` is the updated-prompt arm. `calib` is final-arm S1 plus the
exposed S2 episodes (§2, exposure rule). `test` is the rest of the final arm.
"""
import ast
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ROUNDS = 4  # matrixgame_covered_shared/history.py
FINAL_SEATS = tuple(f"Player {c}" for c in ("Green", "Orange", "Blue", "Yellow", "Red", "Purple"))
DEV_SEATS = tuple(f"Player {c}" for c in "ABCDEF")
MODELS = ("qwen38", "dsv41", "nemotron")
TIER = {"qwen38": "strong", "dsv41": "strong", "nemotron": "weak"}
SUITES = {2: "dyad", 3: "3p", 6: "6p"}
PROTOS = ("none", "structured", "hybrid", "freeform")
MOVE = re.compile(r"^(Player \w+) moved ([A-F]) (up|down|left|right)$")
SPK = re.compile(r"^(Player \w+): (.*)$", re.S)
DONE = re.compile(r"^(Player \w+) declared done$")
REJECT = {"invalid_format", "invalid_move", "malformed_message", "invalid_done"}
TREES = {
    ("dev", "qwen38"): "results_qwen38_grid_updated_prompt_n15",
    ("dev", "dsv41"): "results_dsv41_grid_updated_prompt_n15",
    ("dev", "nemotron"): "results_nemotron_grid_updated_prompt_n15",
    ("final", "qwen38"): "results_qwen38_grid_final_n15",
    ("final", "dsv41"): "results_dsv41_grid_final_n15",
    ("final", "nemotron"): "results_nemotron_grid_final_n15",
}
# At 3p and 6p a final-arm run with teammates' messages in the order sent
# (`_final_chrono`) replaces the shuffled run when it exists (tree()).
FINAL_GAME = re.compile(r"_final(_chrono)?$")
# Final-arm episodes outside S1 that were read during development (spec §2):
# (model, suite, protocol, level, board). The structured ones are X1's
# "six DeepSeek S2 structured episodes", reconstructed from completion times
# with the 3p tie at 08:58 resolved by taking both; the hybrid one is the
# context printed on 2026-10-17 while checking the MOVE BOARD format.
EXPOSED = ({("dsv41", "3p", "structured", "S2", g) for g in (1, 2, 3, 4, 5)}
           | {("dsv41", "6p", "structured", "S2", g) for g in (1, 2)}
           | {("dsv41", "3p", "hybrid", "S2", 3)})
# All of them are episodes of DeepSeek's shuffled 3p/6p run, so they do not apply to its chrono run.


def _master_regexes():
    """The game's own protocol regexes (G-slot), read from the master's source."""
    path = os.path.join(REPO, "matrixgame_covered_comm", "master.py")
    names = {"COMM_MOVE_PATTERN", "COMM_DONE_PATTERN", "STRUCTURED_MSG_PATTERN", "HYBRID_MSG_PATTERN"}
    ns = {"re": re}
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    assert names <= set(ns), names - set(ns)
    return ns


_M = _master_regexes()
SCHEMA = {"structured": _M["STRUCTURED_MSG_PATTERN"], "hybrid": _M["HYBRID_MSG_PATTERN"]}
PLAN = re.compile(r"move\s+(?P<obj>\S+)\s+from\s*\((?P<fr>R\d+,C\d+)\)\s+to\s*\((?P<to>R\d+,C\d+)\)", re.I)


def cell(p):
    """(r, c) zero-based -> canonical '(Rr,Cc)'."""
    return f"(R{p[0] + 1},C{p[1] + 1})"


def uncell(s):
    """'R3,C4' / '(R3,C4)' -> (2, 3)."""
    r, c = re.findall(r"\d+", s)
    return int(r) - 1, int(c) - 1


def norm(s):
    return " ".join((s or "").split())


# ---------------------------------------------------------------- episodes

def tree(arm, model, suite):
    base = TREES[(arm, model)]
    chrono = base.replace("_grid_final_", "_grid_final_chrono_")
    if arm == "final" and suite in ("3p", "6p") and os.path.isdir(os.path.join(REPO, f"{chrono}_{suite}")):
        return f"{chrono}_{suite}"
    return f"{base}_{suite}"


def episode_paths(arm, model, suite):
    pat = os.path.join(REPO, tree(arm, model, suite), "*", "*", "*", "instance_*", "interactions.json")
    return sorted(p for p in glob.glob(pat)
                  if not re.search(r"_stale_|_killed|/_campaign/", p))


def owner_map(ins, it):
    """Object -> seat, seats in turn order, from the logged `Player <S> Done` keys.

    The masters log one `<seat> Done` key per seat in seat order, and
    instance.json lists the object groups in the same order, so seat i owns
    groups[i] whatever the arm names its seats (X1 §2.4)."""
    groups = ins.get("player_objects") or [ins["player_a_objects"], ins["player_b_objects"]]
    seats = [k[:-len(" Done")] for k in it if k.startswith("Player ") and k.endswith(" Done")]
    assert len(seats) == len(groups), f"{len(seats)} Done keys for {len(groups)} object groups"
    assert len(set(seats)) == len(seats), "duplicate seat names"
    expected = FINAL_SEATS if FINAL_GAME.search(it["meta"]["game_name"]) else DEV_SEATS
    assert tuple(seats) == expected[:len(seats)], f"seat names {seats} not the arm's"
    owner = {o: seats[i] for i, g in enumerate(groups) for o in g}
    assert sorted(owner) == sorted(ins["start_positions"]), "partition does not cover the objects"
    return seats, owner


def slots(text, proto):
    """G-slot: the master's schema regex, then the PLAN fields. None if no schema."""
    if proto not in SCHEMA or not text:
        return None
    m = SCHEMA[proto].match(text)
    if not m:
        return None
    p = PLAN.match(m.group("plan"))
    req = norm(m.group("request"))
    return dict(prose=(m.groupdict().get("prose") or "").strip() or None, plan=norm(m.group("plan")),
                plan_obj=p.group("obj").upper(), plan_from=uncell(p.group("fr")), plan_to=uncell(p.group("to")),
                request=req, request_none=req.strip(" .\"'").lower() == "none")


def parse(path, arm, model):
    """One episode -> dict(meta, seats, owner, objs, turns, calls, ...)."""
    d = os.path.dirname(path)
    ins = json.load(open(os.path.join(d, "instance.json")))
    it = json.load(open(path))
    seats, owner = owner_map(ins, it)
    pos = {o: tuple(p) for o, p in ins["start_positions"].items()}
    calls, turns, cur = [], [], None
    for rnd, t in enumerate(it["turns"]):
        for e in t:
            ty, c = e["action"]["type"], e["action"].get("content")
            if ty == "send message":
                cur = dict(player=e["to"], ctx=c, round=rnd, response=None, verdict=None, events=[])
                calls.append(cur)
            elif ty == "get message":
                assert cur is not None and e["from"] == cur["player"]
                cur["response"] = c
            else:
                cur["events"].append((ty, c))
                if ty in REJECT:
                    cur["verdict"] = ty
                elif ty == "move":
                    seat, o, dr = MOVE.match(c).groups()
                    src = pos[o]
                    dst = (src[0] + DIRS[dr][0], src[1] + DIRS[dr][1])
                    turns.append(dict(ply=len(turns), seat=seat, player=cur["player"], obj=o, dir=dr,
                                      src=src, dst=dst, done=False, msg=None, board=dict(pos), call=len(calls) - 1))
                    pos[o] = dst
                    cur["verdict"] = "accepted"
                elif ty == "player_done":
                    seat = DONE.match(c).group(1)
                    turns.append(dict(ply=len(turns), seat=seat, player=cur["player"], obj=None, dir=None,
                                      src=None, dst=None, done=True, msg=None, board=dict(pos), call=len(calls) - 1))
                    cur["verdict"] = "accepted"
                elif ty == "message":
                    seat, txt = SPK.match(c).groups()
                    assert turns and turns[-1]["seat"] == seat and turns[-1]["call"] == len(calls) - 1
                    turns[-1]["msg"] = txt
    n = len(seats)
    final = bool(FINAL_GAME.search(it["meta"]["game_name"]))
    assert final == (arm == "final"), path
    proto, level, gid = ins["comm_protocol"], ins["spatial_level"], ins["game_id"]
    meta = dict(id=f"{arm}|{model}|{SUITES[n]}|{level}|{proto}|{gid:02d}", arm=arm, model=model, tier=TIER[model],
                suite=SUITES[n], n=n, level=level, board=gid, proto=proto, game=it["meta"]["game_name"],
                completed=it["meta"].get("completed") is True,
                success=int(it.get("Success", 0)), aborted=int(it.get("Aborted", 0)), lose=int(it.get("Lose", 0)),
                path=os.path.relpath(d, REPO))
    ep = dict(meta=meta, seats=seats, owner=owner, objs={s: sorted(o for o in owner if owner[o] == s) for s in seats},
              grid=ins["grid_size"], walls=frozenset(tuple(w) for w in ins.get("walls", [])),
              start={o: tuple(p) for o, p in ins["start_positions"].items()},
              targets={o: tuple(p) for o, p in ins["target_positions"].items()},
              device=ins.get("device") or {}, turns=turns, calls=calls, final_pos=pos, ins=ins, it=it)
    _annotate(ep)
    return ep


def _annotate(ep):
    """Per-turn fields: attempts, retry, own-turn index, board after, slots,
    duplicate flag, visible plies with lags; per-seat done ply."""
    T, proto = ep["turns"], ep["meta"]["proto"]
    prev, own_n, seen = -1, collections.Counter(), collections.defaultdict(set)
    for i, t in enumerate(T):
        t["attempts"] = list(range(prev + 1, t["call"] + 1))
        t["retry"] = len(t["attempts"]) > 1
        prev = t["call"]
        t["si"] = ep["seats"].index(t["seat"])
        t["own_turn"] = own_n[t["seat"]]
        own_n[t["seat"]] += 1
        t["after"] = T[i + 1]["board"] if i + 1 < len(T) else ep["final_pos"]
        t["is_msg"] = bool(t["msg"]) and not t["done"] and proto != "none"
        t["slots"] = slots(t["msg"], proto) if t["is_msg"] else None
        k = norm(t["msg"])
        t["dup"] = t["is_msg"] and k in seen[t["seat"]]
        if t["is_msg"]:
            seen[t["seat"]].add(k)
        t["vis"] = visible(ep, i, t["retry"])
    ep["done_ply"] = {t["seat"]: t["ply"] for t in T if t["done"]}


# ---------------------------------------------------------------- visibility

def visible(ep, idx, retry):
    """Plies of earlier messages shown in the context of the call that produced turn idx.

    dev  (matrixgame_covered_comm/master.py:332-349, nplayer master.py:672-693):
         every seat's latest message, own included; retry contexts carry none.
    final (matrixgame_covered_shared/history.py): every message since the
         viewer's ROUNDS-th last own turn, first call and retries alike; a done
         turn whose message is just "done" shows as `declared done` only.
    """
    T = ep["turns"]
    if ep["meta"]["proto"] == "none":
        return []
    if ep["meta"]["arm"] == "dev":
        if retry:
            return []
        latest = {}
        for t in T[:idx]:
            latest[t["seat"]] = t["ply"] if t["msg"] else None
        return sorted(p for p in latest.values() if p is not None)
    start = window_start(ep, idx)
    return [t["ply"] for t in T[start:idx] if t["msg"] and not (t["done"] and t["msg"].strip().lower() == "done")]


def window_start(ep, idx):
    """First ply of the final arm's 4-round window for the viewer of turn idx."""
    viewer = ep["turns"][idx]["seat"]
    own = [t["ply"] for t in ep["turns"][:idx] if t["seat"] == viewer]
    return own[-ROUNDS] if len(own) >= ROUNDS else 0


def move_board(ep, idx):
    """Final arm: the MOVE BOARD shown at turn idx, as (ply, own, src, dst) in play order.
    Teammates' moves are anonymous (`X moved ...`); own moves name the object."""
    if ep["meta"]["arm"] != "final":
        return []
    viewer = ep["turns"][idx]["seat"]
    return [(t["ply"], t["seat"] == viewer, t["src"], t["dst"])
            for t in ep["turns"][window_start(ep, idx):idx] if not t["done"]]


DONE_SIGNAL = re.compile(r"^(Player \w+)(?: has finished\.| says: done$| declared done$)|^(Your teammate) has finished\.", re.M)


def done_visible(ep, idx):
    """Seats whose finish is signalled in the logged context of turn idx (§5.6):
    dev `says: done` entries and `has finished` notices, final `declared done`.
    Precomputed by `load`, which then drops the contexts."""
    tr = ep["turns"][idx]
    if "done_vis" in tr:
        return tr["done_vis"]
    out = set()
    for m in DONE_SIGNAL.finditer(ep["calls"][tr["call"]]["ctx"]):
        if m.group(2):  # dyad notice: the partner
            out |= set(ep["seats"]) - {tr["seat"]}
        else:
            out.add(m.group(1))
    return out & set(ep["done_ply"])


FINAL_LINE = re.compile(r'^(Player \w+)(?: \(you\))? (?:said "(.*)"|declared done)$')


def logged_entries(ep, tr):
    """Message entries in the logged context of turn tr's accepting call: [(seat, text|None)].
    text None is a final-arm `declared done` line."""
    ctx = ep["calls"][tr["call"]]["ctx"]
    if ep["meta"]["arm"] == "final":
        if "MESSAGE BOARD" not in ctx:
            return []
        block = ctx.split("MESSAGE BOARD", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
        out = []
        for line in block:
            m = FINAL_LINE.match(line)
            assert m, line[:80]
            out.append((m.group(1), m.group(2)))
        return out
    head = ctx.split("CURRENT BOARD:", 1)[0]
    return [(m.group(1), None) for m in re.finditer(r"^(Player \w+)(?: \(you\))? says: ", head, re.M)]


# ---------------------------------------------------------------- scopes

def scope_of(arm, model, suite, proto, level, board, chrono=False):
    if arm == "dev":
        return "dev"
    exposed = not chrono and (model, suite, proto, level, board) in EXPOSED
    return "calib" if level == "S1" or exposed else "test"


PATH_KEY = re.compile(r"/(S\d)_(none|structured|hybrid|freeform)_.*?/instance_(\d+)/")


_LOCK = None


def _machine_lock():
    """One corpus-holding process at a time on this machine, held until exit.
    The laptop also runs the campaign; parallel loads exhausted its 16 GB."""
    global _LOCK
    if _LOCK is None:
        import fcntl
        os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
        _LOCK = open(os.path.join(HERE, "out", ".load.lock"), "w")
        try:
            fcntl.flock(_LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("corpus.load: waiting for another analysis process to finish", file=sys.stderr)
            fcntl.flock(_LOCK, fcntl.LOCK_EX)


def slim(ep):
    """Keep what the analysis reads: precompute the done signal per turn, then
    drop the raw interactions, the logged contexts and the preamble."""
    for i, t in enumerate(ep["turns"]):
        t["done_vis"] = done_visible(ep, i)
    ep.pop("it", None)
    ep["ins"] = {k: v for k, v in ep["ins"].items() if k != "player_prompt"}
    ep["calls"] = [dict(player=c["player"], verdict=c["verdict"]) for c in ep["calls"]]
    return ep


def load(scope, models=MODELS, suites=("dyad", "3p", "6p"), protos=PROTOS, check=True):
    """Included episodes of a scope (spec §2): completed, not solo, not stale.
    `none` episodes are included; callers use them as baselines only.
    Episodes come back slimmed (no raw logs, no contexts); `parse` keeps them."""
    assert scope in ("dev", "calib", "test")
    _machine_lock()
    arm = "dev" if scope == "dev" else "final"
    out = []
    for m in models:
        for s in suites:
            for p in episode_paths(arm, m, s):
                # scope from the path, so no test-arm file is opened outside `test`
                lv, pr, g = PATH_KEY.search(p).groups()
                if pr not in protos or scope_of(arm, m, s, pr, lv, int(g), "_final_chrono_" in p) != scope:
                    continue
                ep = parse(p, arm, m)
                assert (ep["meta"]["level"], ep["meta"]["board"]) == (lv, int(g)), p
                if not ep["meta"]["completed"]:
                    continue
                if check:
                    selfcheck(ep)
                out.append(slim(ep))
    return out


def messages(eps):
    """The message units (spec §2): accepted non-done turns under a messaging protocol."""
    return [(ep, t) for ep in eps for t in ep["turns"] if t["is_msg"]]


def dist_map(ep, target, blocked=frozenset()):
    """BFS distance to `target` over non-wall cells, `blocked` cells as walls
    (d_W with no blocked cells; d_t with the other objects' cells)."""
    cache = ep.setdefault("_dist", {})
    if not blocked and target in cache:
        return cache[target]
    n, walls = ep["grid"], ep["walls"]
    seen, queue = {target: 0}, collections.deque([target])
    while queue:
        c = queue.popleft()
        for dr, dc in DIRS.values():
            nb = (c[0] + dr, c[1] + dc)
            if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls and nb not in blocked and nb not in seen:
                seen[nb] = seen[c] + 1
                queue.append(nb)
    if not blocked:
        cache[target] = seen
    return seen


def goalward(ep, obj, src, d):
    """Does a step of obj from src in direction d lower its wall-only distance to its target?"""
    dm = dist_map(ep, ep["targets"][obj])
    dst = (src[0] + DIRS[d][0], src[1] + DIRS[d][1])
    return dst in dm and src in dm and dm[dst] < dm[src]


def exposures(ep):
    """(message ply, listener seat) -> ply of the listener's first decision
    whose context shows that message (spec §5.5). The speaker is no listener."""
    out = {}
    for t in ep["turns"]:
        for q in t["vis"]:
            if ep["turns"][q]["seat"] != t["seat"]:
                out.setdefault((q, t["seat"]), t["ply"])
    return out


# ---------------------------------------------------------------- hazards (§6)

def _components(n, walls, blocked):
    comp, cid = {}, 0
    for r in range(n):
        for c in range(n):
            if (r, c) in walls or (r, c) in blocked or (r, c) in comp:
                continue
            stack, cid = [(r, c)], cid + 1
            comp[(r, c)] = cid
            while stack:
                cr, cc = stack.pop()
                for dr, dc in DIRS.values():
                    nb = (cr + dr, cc + dc)
                    if 0 <= nb[0] < n and 0 <= nb[1] < n and nb not in walls and nb not in blocked and nb not in comp:
                        comp[nb] = cid
                        stack.append(nb)
    return comp


def legal(ep, board, objs):
    occ = set(board.values())
    out = []
    for o in objs:
        r0, c0 = board[o]
        for d, (dr, dc) in DIRS.items():
            nb = (r0 + dr, c0 + dc)
            if 0 <= nb[0] < ep["grid"] and 0 <= nb[1] < ep["grid"] and nb not in ep["walls"] and nb not in occ:
                out.append((o, d))
    return out


def step(board, o, d):
    b = dict(board)
    b[o] = (b[o][0] + DIRS[d][0], b[o][1] + DIRS[d][1])
    return b


def first_fatal(ep):
    """Ply of the episode's first fatal decision (a fork declared into or a
    pinch taken), or None. Outcome comparisons use messages before it (§6)."""
    return min((h["ply"] for h in hazards(ep) if h["case"]), default=None)


def hazards(ep):
    if "_haz" not in ep:
        ep["_haz"] = _hazards(ep)
    return ep["_haz"]


def _hazards(ep):
    """Every decision of a player at which a fork or a free pinch is live.

    Ported from analysis/qualitative_analysis/averted.py (fork) and
    seal_averted.py (pinch), on the seat-index owner map:
    - fork: every object of the decider is home, and freezing them (as `done`
      would) severs a live teammate object's last route; finished objects are
      walls, active ones are not. Case: the decider declares done.
    - pinch: at a move decision, some legal move leaves an active teammate
      that can move now with no legal move. Case: the decider takes one. The
      pinch is `free` when a harmless legal move also exists (seal_averted.py's
      risk set); a forced pinch is still a fatal decision for `first_fatal`.
    Returns [dict(ply, seat, kind, case, victims, critical, free)]: `critical`
    are the decider's door cells (fork) or the sealing destinations (pinch)."""
    T, n, walls, tg = ep["turns"], ep["grid"], ep["walls"], ep["targets"]
    out = []
    for t in T:
        b, p, done = t["board"], t["seat"], {s for s, q in ep["done_ply"].items() if q < t["ply"]}
        if all(b[o] == tg[o] for o in ep["objs"][p]):
            frozen = {b[o] for o in b if ep["owner"][o] in done}
            add = {b[o] for o in ep["objs"][p]}
            before, after = _components(n, walls, frozen), _components(n, walls, frozen | add)
            vic = [o for o in tg if ep["owner"][o] != p and ep["owner"][o] not in done and b[o] != tg[o]
                   and before.get(b[o]) and before.get(b[o]) == before.get(tg[o])
                   and after.get(b[o]) != after.get(tg[o])]
            if vic:
                doors = [b[o] for o in ep["objs"][p]
                         if any((lab := _components(n, walls, (frozen | add) - {b[o]})).get(b[v])
                                and lab.get(b[v]) == lab.get(tg[v]) for v in vic)]
                out.append(dict(ply=t["ply"], seat=p, kind="fork", case=t["done"], victims=vic,
                                critical=doors or sorted(add), free=True))
        if t["done"]:
            continue
        active = [s for s in ep["seats"] if s not in done and s != p]
        mobile_now = {s for s in active if legal(ep, b, ep["objs"][s])}
        seals, moves = {}, legal(ep, b, ep["objs"][p])
        for m in moves:
            nb = step(b, *m)
            killed = sorted(s for s in mobile_now if not legal(ep, nb, ep["objs"][s]))
            if killed:
                seals[m] = killed
        if seals:
            out.append(dict(ply=t["ply"], seat=p, kind="pinch", case=(t["obj"], t["dir"]) in seals,
                            victims=sorted({s for v in seals.values() for s in v}),
                            critical=sorted({step(b, *m)[m[0]] for m in seals}), free=len(seals) < len(moves)))
    return out


# ---------------------------------------------------------------- self-checks (§3)

def selfcheck(ep):
    """X1 A1-A13 and V1; raises AssertionError naming the episode and check."""
    it, ins, T, C = ep["it"], ep["ins"], ep["turns"], ep["calls"]
    seats, owner, proto, eid = ep["seats"], ep["owner"], ep["meta"]["proto"], ep["meta"]["id"]

    def need(ok, what):
        if not ok:
            raise AssertionError(f"{eid}: {what}")

    need(it["meta"].get("completed") is True, "A1 meta.completed")
    need(all(owner[t["obj"]] == t["seat"] for t in T if not t["done"]), "A2 mover owns object")
    need(all(t["seat"] == seats[int(t["player"].split()[1]) - 1] for t in T), "A3 Player k is seat k-1")
    tm = [(m["player"], m["object"], tuple(m["from"]), tuple(m["to"])) for m in it["Turn Moves"]]
    need(tm == [(t["seat"], t["obj"], t["src"], t["dst"]) for t in T if not t["done"]], "A4 Turn Moves == replay")
    pos = dict(ep["start"])
    for t in T:
        if t["done"]:
            continue
        r, c = t["dst"]
        need(0 <= r < ep["grid"] and 0 <= c < ep["grid"] and t["dst"] not in ep["walls"]
             and t["dst"] not in set(pos.values()), f"A5 legal replay at ply {t['ply']}")
        pos[t["obj"]] = t["dst"]
    need(bool(it["Success"]) == (pos == ep["targets"]), "A6 success iff all home")
    unresolved = [i for i, x in enumerate(C) if x["verdict"] is None]
    need(all(i == len(C) - 1 for i in unresolved), "A7 every call resolved")
    need(it["Request Count"] == len(C), "A8 Request Count == calls")
    need(it["Parsed Request Count"] == len(T), "A8 Parsed == accepted")
    need(it["Move Count"] == sum(not t["done"] for t in T), "A8 Move Count == moves")
    for s in seats:
        need(bool(it[f"{s} Done"]) == (s in ep["done_ply"]), f"A9 Done key {s}")
    if proto == "none":
        need(not any(t["msg"] for t in T), "A10 no messages under none")
    else:
        need(all(t["msg"] for t in T if not t["done"]), "A10 every non-done turn has a message")
        need(sum(t["is_msg"] for t in T) == sum(not t["done"] for t in T), "message count == non-done turns")
    if proto in SCHEMA:
        need(all(t["slots"] for t in T if t["is_msg"]), "A11 schema holds for accepted messages")
    need(tuple(seats) == (FINAL_SEATS if ep["meta"]["arm"] == "final" else DEV_SEATS)[:len(seats)], "A12 seat names")
    need(not re.search(r"_stale|killed", ep["meta"]["path"]), "A13 path not stale")
    # V1: the simulated visible set equals the logged context. Each visible
    # message's text is in the context, and the context holds as many message
    # entries as the model shows (texts that are substrings of one another and
    # identical twins make a line-by-line match ambiguous; X1 §2.5).
    for t in T:
        if proto == "none":
            continue
        ctx = norm(C[t["call"]]["ctx"])
        for q in t["vis"]:
            need(norm(T[q]["msg"]) in ctx, f"V1 visible text missing at ply {t['ply']}")
        ent = logged_entries(ep, t)
        n_msgs = sum(1 for s, x in ent if not (ep["meta"]["arm"] == "final" and x is None))
        need(n_msgs == len(t["vis"]), f"V1 entry count {n_msgs} != {len(t['vis'])} at ply {t['ply']}")


def own_claim_share(eps):
    """Share of PLAN slots whose object belongs to the speaker; > 0 under a correct owner map."""
    k = n = 0
    for ep in eps:
        for t in ep["turns"]:
            if t["slots"]:
                n += 1
                k += ep["owner"].get(t["slots"]["plan_obj"]) == t["seat"]
    return k / n if n else None


def main(argv):
    scope = argv[1] if len(argv) > 1 else "dev"
    eps = load(scope)
    share = own_claim_share(eps)
    assert share is None or share > 0, "own-claim share is 0: owner map broken"
    c = collections.Counter((e["meta"]["model"], e["meta"]["suite"], e["meta"]["proto"]) for e in eps)
    nm = collections.Counter((e["meta"]["model"]) for e, t in messages(eps))
    print(f"{scope}: {len(eps)} episodes, {len(messages(eps))} messages {dict(nm)}, own-claim share {share:.3f}")
    for k in sorted(c):
        print(" ", "|".join(k), c[k])


if __name__ == "__main__":
    main(sys.argv)
