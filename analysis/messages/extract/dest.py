"""E5 destinations: the writer's own announced destinations in freeform prose,
extracted by the E5 annotator and scored by the replay like a PLAN (extension §4).

    python3 -m analysis.messages.extract.dest dev --dry-run
    python3 -m analysis.messages.extract.dest dev --limit 60 --max-usd 1    # pilot
    python3 -m analysis.messages.extract.dest dev --max-usd 5

Freeform only: hybrid and structured carry the same fact in the PLAN slot (E2),
and G-move pairs an own letter with a direction, never with a cell, so "I'll
take D to (R2,C3)" is otherwise unscored. Eligible: freeform messages naming an
own letter (G-letter) and a cell (G-cell). The prompt is `prompt_dest.txt`; the
cache, batching and the acceptance rules are E5's (extract.run, records), plus
one rule here: the letter is the writer's own (reason `not_own`). The call uses
OpenAI's structured outputs (a strict JSON schema) and medium reasoning effort:
long route lists broke the free JSON of a first run in 28 % of batches. The
schema's items are stored as E5's item arrays, unchanged in content.

Each accepted record (letter L, cell c) of message t is one of
- already: L stands on c at B_t- (no destination; left out);
- now: the move sent with the message puts L on c, a description of this
  turn's step (as G-move's present frame), counted but not scored;
- later: a commitment. Per message and object the furthest stated later cell
  (the span ending last, the end of a stated route) is scored, as E2 scores a
  PLAN-to: same-turn say-do (the move sent with the message moves L and lowers
  its d_W to c: progress, same_no_progress, other_object; none on a done
  turn), next-turn progress (the writer's next own turn moves L and lowers its
  d_W to c; None when no own move follows), and reached (L stands on c after
  one of the writer's next H own turns, H = corpus.ROUNDS; None, censored, when
  fewer follow and L never got there).
Rates use messages before the episode's first fatal decision and drop exact
repeats within a seat (spec §6), as E2 does. Every number is exploratory unless
the author check in coding/dest passes (precision >= .90, as spec §9.6's
model part).
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys

from analysis.messages import corpus, grammar, stats
from analysis.messages.extract import client, records
from analysis.messages.extract import run as x
from analysis.messages.measures import requests_plans as rp

PROMPT = "dest"
SEED = 0
H = corpus.ROUNDS
MODEL_API = "gpt-5.6-luna"
EFFORT = "medium"
METRICS = ("commit_stated", "described_now", "saydo_progress", "next_progress", "reached")
VALIDATION = os.path.join(corpus.HERE, "coding", "dest", "metrics.json")


SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {
    "type": "array", "items": {"type": "object", "additionalProperties": False,
                               "required": ["message", "letter", "cell", "span"],
                               "properties": {"message": {"type": "integer"},
                                              "letter": {"type": "string", "enum": list("ABCDEF")},
                                              "cell": {"type": "string", "pattern": r"^\(R[1-9],C[1-9]\)$"},
                                              "span": {"type": "string"}}}}}}
# the cache is keyed by (prompt sha, model): the model label carries the effort and the schema, so changing
# either never reuses old replies; the price lookup matches the API id inside it
MODEL = f"{MODEL_API}-{EFFORT}-schema-{hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()[:8]}"


def to_array(text):
    """The schema's object as E5's reply, [[message, letter, cell, span], ...]; other text unchanged (it fails parsing)."""
    try:
        items = json.loads(text)["items"]
        return json.dumps([[i["message"], i["letter"], i["cell"], i["span"]] for i in items], ensure_ascii=False)
    except (ValueError, KeyError, TypeError):
        return text


def backend(effort=EFFORT, max_tokens=32768):
    """OpenAI chat completions with the strict schema; returns E5's (text, usage)."""
    import openai
    conf = json.load(open(os.path.join(client.REPO, "key.json")))["openai"]
    api = openai.OpenAI(api_key=conf["api_key"], organization=conf.get("organisation") or None, timeout=600,
                        max_retries=5)
    fmt = {"type": "json_schema", "json_schema": {"name": "destinations", "strict": True, "schema": SCHEMA}}

    def complete(prompt, model_id):
        system, user = prompt
        r = api.chat.completions.create(model=MODEL_API, reasoning_effort=effort, max_completion_tokens=max_tokens,
                                        response_format=fmt, messages=[{"role": "developer", "content": system},
                                                                       {"role": "user", "content": user}])
        u, det = r.usage, getattr(r.usage, "completion_tokens_details", None)
        return to_array(r.choices[0].message.content or ""), dict(
            inputTokens=u.prompt_tokens, outputTokens=u.completion_tokens,
            reasoningTokens=getattr(det, "reasoning_tokens", None), stop=r.choices[0].finish_reason)
    return complete


def eligible(eps, A):
    """{input key: [(ep, t), ...]} in seeded order: freeform messages naming an own letter and a cell."""
    groups = {}
    for ep, t in corpus.messages(eps):
        if ep["meta"]["proto"] != "freeform":
            continue
        an = A[(ep["meta"]["id"], t["ply"])]
        own = set(ep["objs"][t["seat"]])
        if an["cells"] and any(v in own for _, _, v, _ in an["letters"]):
            groups.setdefault(client.key(ep, t), []).append((ep, t))
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    return {k: groups[k] for k in keys}


def load(eps, A, model=MODEL, units=None):
    """({(episode id, ply): [records]} own letters only, reject Counter, annotated message keys, units)."""
    units = units if units is not None else eligible(eps, A)
    recs, rejects, _, annotated, _ = x.load_records(eps, PROMPT, model, A, units)
    idx = {(ep["meta"]["id"], t["ply"]): (ep, t) for ms in units.values() for ep, t in ms}
    out = {}
    for k, rs in recs.items():
        ep, t = idx[k]
        own = set(ep["objs"][t["seat"]])
        out[k] = [r for r in rs if r["letter"] in own]
        rejects["not_own"] += len(rs) - len(out[k])
    return out, rejects, annotated, units


def kind(ep, t, rec):
    L, c = rec["letter"], rec["cell"]
    return "already" if t["board"][L] == c else "now" if not t["done"] and t["after"][L] == c else "later"


def score(ep, t, rec, nxt):
    """saydo, next_progress, reached for a later destination; nxt = the writer's later turns in order."""
    L, c = rec["letter"], rec["cell"]
    closer = lambda u: u["obj"] == L and rp._dw(ep, u["after"][L], c) < rp._dw(ep, u["board"][L], c)
    saydo = None if t["done"] else "other_object" if t["obj"] != L else "progress" if closer(t) else "same_no_progress"
    nx = next((u for u in nxt[:1] if not u["done"]), None)
    reached = any(not u["done"] and u["after"][L] == c for u in nxt[:H]) or (None if len(nxt) < H else False)
    return dict(saydo=saydo, next_progress=None if nx is None else closer(nx), reached=reached)


def commitments(ep, t, rs):
    """Per object the later record whose span ends last in the message (ties: the longer span)."""
    end = lambda r: max(i + len(r["span"]) for i in range(len(t["msg"])) if t["msg"].startswith(r["span"], i))
    best = {}
    for r in rs:
        if kind(ep, t, r) == "later" and (r["letter"] not in best or
                                          (end(r), len(r["span"])) > (end(best[r["letter"]]), len(best[r["letter"]]["span"]))):
            best[r["letter"]] = r
    return list(best.values())


def validation():
    if not os.path.exists(VALIDATION):
        return "pending"
    return "validated" if json.load(open(VALIDATION))["passed"] else "exploratory (author check failed)"


def run(scope, eps, A, model=MODEL):
    """{table: rows} from the cache; no API call. Empty estimates while the extraction is incomplete."""
    units = eligible(eps, A)
    recs, rejects, annotated, _ = load(eps, A, model, units)
    n_msgs = sum(len(ms) for ms in units.values())
    meta = [dict(prompt=PROMPT, prompt_sha256=client.prompt_sha(PROMPT), model=model, eligible_messages=n_msgs,
                 annotated=len(annotated), validation=validation(), **{f"reject_{k}": v for k, v in rejects.items()})]
    if not units or len(annotated) < n_msgs:
        return dict(dest_meta=meta)
    rows, ep_rows = [], []
    for ep in eps:
        m = ep["meta"]
        if m["proto"] != "freeform":
            continue
        T = ep["turns"]
        ff = corpus.first_fatal(ep)
        by_seat = collections.defaultdict(list)
        for t in T:
            by_seat[t["seat"]].append(t)
        stated, now, saydo, nextp, reached = [], [], [], [], []
        for t in T[:len(T) if ff is None else ff]:
            if not t["is_msg"] or t["dup"]:
                continue
            rs = recs.get((m["id"], t["ply"]), [])
            cs = commitments(ep, t, rs)
            stated.append(bool(cs))
            now.append(any(kind(ep, t, r) == "now" for r in rs))
            nxt = by_seat[t["seat"]][t["own_turn"] + 1:]
            for r in cs:
                s = score(ep, t, r, nxt)
                rows.append(dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], board=m["board"],
                                 episode=m["id"], ply=t["ply"], seat=t["seat"], letter=r["letter"],
                                 cell=corpus.cell(r["cell"]), span=r["span"], records_in_msg=len(rs), **s))
                if s["saydo"]:
                    saydo.append(s["saydo"] == "progress")
                for v, out in ((s["next_progress"], nextp), (s["reached"], reached)):
                    if v is not None:
                        out.append(v)
        mean = lambda v: sum(v) / len(v) if v else None
        ep_rows.append(dict(model=m["model"], tier=m["tier"], suite=m["suite"], level=m["level"], board=m["board"],
                            episode=m["id"], commit_stated=mean(stated), commit_stated_n=len(stated),
                            described_now=mean(now), described_now_n=len(now), saydo_progress=mean(saydo),
                            saydo_progress_n=len(saydo), next_progress=mean(nextp), next_progress_n=len(nextp),
                            reached=mean(reached), reached_n=len(reached)))
    cells = []
    for (mo, su), rs in sorted(_group(ep_rows).items()):
        for lv in ["all"] + sorted({r["level"] for r in rs}):
            sub = [r for r in rs if lv == "all" or r["level"] == lv]
            for met in METRICS:
                vals = [(r[met], r["level"]) for r in sub if r[met] is not None]
                if vals:
                    est = stats.cell_estimate([v for v, _ in vals], [l for _, l in vals])
                    cells.append(dict(model=mo, tier=corpus.TIER[mo], suite=su, level=lv, metric=met,
                                      units=sum(r[f"{met}_n"] for r in sub), **est))
    return dict(dest_meta=meta, dest_records=rows, dest_episodes=ep_rows, dest_cells=cells)


def _group(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["model"], r["suite"])].append(r)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scope", choices=("dev", "calib", "test"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    eps = corpus.load(a.scope)
    A = grammar.analyse_all(eps)
    units = eligible(eps, A)
    if a.dry_run:
        return x.dry_run([PROMPT], units, a.limit, a.model)
    state = x.extract(PROMPT, a.model, backend(), units, a.limit, a.workers, a.max_usd)
    if state["unanswered"] and a.limit is None:
        raise SystemExit(f"incomplete extraction, {state['unanswered']} batches unanswered; rerun to continue")


if __name__ == "__main__":
    main(sys.argv[1:])
