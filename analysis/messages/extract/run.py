"""E5 run: selection, dry-run cost estimate, extraction and the report tables.

    python3 -m analysis.messages.extract.run dev --dry-run [--prompts all]
    python3 -m analysis.messages.extract.run dev --backend openai --model gpt-5.6-luna --limit 60    # pilot
    python3 -m analysis.messages.extract.run dev --backend openai --model gpt-5.6-luna               # primary
    python3 -m analysis.messages.extract.run dev --backend openai --model gpt-5.6-luna --prompts alt1,alt2 --limit 5000
    python3 -m analysis.messages.extract.run dev --report --model <id>

Only messages that name a teammate's letter (G-letter) and a cell (G-cell) are
annotated: an accepted teammate `at` record needs both literally in its span,
so no other message can yield one. Identical inputs (the writer's letters and
the text) are annotated once. Inputs are put in a seeded order and cut into
batches of client.BATCH; `--limit N` takes the first N inputs of that order,
so a pilot's batches are reused by the full run when N is a multiple of BATCH.
Every call goes through the cache, so reruns never pay twice. Nothing runs
without an explicit backend and model, and a run stops submitting batches
once its observed spend reaches --max-usd.
"""
import argparse
import collections
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from analysis.messages import corpus, grammar
from analysis.messages.extract import client, records

SEED = 0
# ---- dry-run assumptions, replaced by observed usage once a pilot is cached.
# USD per million tokens: Anthropic list prices of September 2026 (assumed equal
# on Bedrock on-demand) and OpenAI's GPT-5.6 prices after the 2026-07-30 cut.
# OpenAI bills reasoning tokens as output.
CHARS_PER_TOKEN = 3.0      # cell-dense text: "(R3,C4)" is about one token per 1-2 characters
OUT_PER_MSG = 20           # reply tokens per message: about one ~20-token item per message
OUT_PER_BATCH = 5          # the array's brackets
PRICES = {"gpt-5.6-luna": (0.20, 1.20), "gpt-5.6-terra": (2.00, 12.00), "gpt-5.6-sol": (5.00, 30.00),
          "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}


def eligible(eps, A):
    """{input key: [(ep, t), ...]} in seeded order: the messages that name a
    teammate's letter and a cell, grouped by identical input."""
    groups = {}
    for ep, t in corpus.messages(eps):
        an = A[(ep["meta"]["id"], t["ply"])]
        if an["cells"] and any(records.teammate(ep, t, v) for _, _, v, _ in an["letters"]):
            groups.setdefault(client.key(ep, t), []).append((ep, t))
    keys = sorted(groups)
    random.Random(SEED).shuffle(keys)
    return {k: groups[k] for k in keys}


def batches(units, limit=None):
    ks = list(units)[:limit] if limit else list(units)
    return [ks[i:i + client.BATCH] for i in range(0, len(ks), client.BATCH)]


def tok(s):
    return len(s) / CHARS_PER_TOKEN


def price_of(model):
    return next((v for k, v in PRICES.items() if model and k in model), None)


def dry_run(names, units, limit=None, model=None):
    """Token and cost estimates for the batches not yet cached; no call is made."""
    todo_all = batches(units, limit)
    print(f"{len(units)} distinct inputs ({sum(map(len, units.values()))} messages); "
          f"{len(todo_all)} batches of up to {client.BATCH}")
    total = collections.Counter()
    for name in names:
        cached = client.load_cache(name, model) if model else {}
        todo = [b for b in todo_all if not all(k in cached for k in b)]
        if not todo:
            print(f"{name}: all batches cached")
            continue
        tin = sum(tok("\n".join(client.render(name, [units[k][0] for k in b]))) for b in todo)
        tout = sum(OUT_PER_BATCH + OUT_PER_MSG * len(b) for b in todo)
        print(f"{name}: {len(todo)} batches to call, input {tin / 1e6:.2f}M, output {tout / 1e6:.2f}M tokens "
              "(estimated, before reasoning tokens)")
        for m, (pin, pout) in PRICES.items():
            cost = tin * pin / 1e6 + tout * pout / 1e6
            total[m] += cost
            print(f"  {m:20s} ${cost:8.2f}")
        used = [r["usage"] for r, _ in {id(r): (r, i) for r, i in cached.values()}.values() if r.get("usage")]
        if used and price_of(model):
            pin, pout = price_of(model)
            mi = sum(u.get("inputTokens", 0) for u in used) / len(used)
            mo = sum(u.get("outputTokens", 0) for u in used) / len(used)
            print(f"  observed on {len(used)} cached batches of {model}: {mi:,.0f} input and {mo:,.0f} output tokens "
                  f"per batch -> about ${len(todo) * (mi * pin + mo * pout) / 1e6:,.2f} for the rest")
    if len(names) > 1:
        print("all wordings:", ", ".join(f"{m} ${total[m]:,.2f}" for m in PRICES))


def cost(usage, model):
    p = price_of(model)
    return (usage.get("inputTokens", 0) * p[0] + usage.get("outputTokens", 0) * p[1]) / 1e6 if p and usage else 0.0


def _answered(done, b):
    """Is every input of batch b cached with a reply that parses? An unparseable
    reply is called again; the newer row replaces it in load_cache."""
    return all(k in done and records.parse(done[k][0]["text"]) is not None for k in b)


def extract(name, model, complete, units, limit=None, workers=4, max_usd=25.0):
    """Call the backend for every batch not yet answered for (prompt sha, model),
    stopping once the observed spend of this run reaches max_usd (needs a priced model)."""
    if price_of(model) is None:
        raise SystemExit(f"no price for {model} in run.PRICES; add it before spending")
    done = client.load_cache(name, model)
    todo = [b for b in batches(units, limit) if not _answered(done, b)]
    print(f"{name} / {model}: {len(batches(units, limit))} batches, {len(todo)} to call, cap ${max_usd:.2f}", flush=True)
    state = collections.Counter()
    spent = [0.0]

    def one(b):
        if (state["error"] >= 10 and not state["ok"]) or spent[0] >= max_usd:  # broken setup or cap reached
            state["skipped"] += 1
            return
        try:
            row = client.store(name, model, b, complete(client.render(name, [units[k][0] for k in b]), model))
            spent[0] += cost(row["usage"], model)
            state["ok"] += 1
        except Exception as e:  # not cached, so a rerun retries it
            state["error"] += 1
            if state["error"] <= 5:
                print(f"  error: {e!r}"[:400], flush=True)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} ({state['error']} errors, ${spent[0]:.2f})", flush=True)
    done = client.load_cache(name, model)
    state["unanswered"] = sum(not _answered(done, b) for b in batches(units, limit))
    print(f"  done: {state['ok']} batches stored, {state['error']} errors, {state['skipped']} skipped, "
          f"${spent[0]:.2f} spent, {state['unanswered']} batches unanswered (rerun to continue)")
    return state


# ---------------------------------------------------------------- report

def load_records(eps, name, model, A=None, units=None):
    """({(episode id, ply): [records]} for every message of an annotated input,
    reject Counter, first rejected example per reason, annotated message keys,
    number of eligible inputs)."""
    units = units if units is not None else eligible(eps, A if A is not None else grammar.analyse_all(eps))
    arm = eps[0]["meta"]["arm"] if eps else "dev"
    rows, seen = client.load_cache(name, model), {}
    for k, (row, i) in rows.items():
        seen.setdefault(id(row), row)
    by_key, rejects, examples = {}, collections.Counter(), {}
    for row in seen.values():
        if not all(k in units for k in row["keys"]):
            continue
        ok, bad = records.accept(row["text"], [units[k][0][1]["msg"] for k in row["keys"]], arm)
        if records.parse(row["text"]) is not None:  # an unparseable batch leaves its inputs unannotated
            for i, k in enumerate(row["keys"], 1):
                by_key[k] = ok.get(i, [])
        for why, item in bad:
            rejects[why] += 1
            examples.setdefault(why, item)
    recs, annotated = {}, set()
    for k, rs in by_key.items():
        for ep, t in units[k]:
            mk = (ep["meta"]["id"], t["ply"])
            recs[mk] = rs
            annotated.add(mk)
    return recs, rejects, examples, annotated, len(units)


def report(eps, A, name, model):
    units = eligible(eps, A)
    recs, rejects, examples, annotated, n_units = load_records(eps, name, model, A, units)
    n_items = sum(map(len, recs.values()))
    base = dict(model=model, prompt=name, inputs_eligible=n_units, messages_annotated=len(annotated))
    out = dict(e5_rejects=[dict(base, reason=r, n=rejects[r], example=str(examples.get(r))[:200]) for r in records.REASONS],
               e5_agreement=[dict(base, **r) for r in records.agreement(eps, A, recs, annotated)])
    mo = records.model_only_at(eps, A, recs)
    by = collections.defaultdict(collections.Counter)
    idx = {(ep["meta"]["id"], t["ply"]): (ep, t) for ep, t in corpus.messages(eps)}
    for eid, ply, L, c in mo:
        ep, t = idx[(eid, ply)]
        by[ep["meta"]["tier"]]["n"] += 1
        by[ep["meta"]["tier"]]["true"] += records.truth(ep, t, dict(letter=L, cell=c))
    out["e5_model_only"] = [dict(base, tier=k, n=v["n"], true=v["true"],
                                 note="accuracy exploratory until the §9.6 check") for k, v in sorted(by.items())]
    out["e5_accepted"] = [dict(base, accepted_records=n_items)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scope", choices=("dev", "calib", "test"))
    ap.add_argument("--prompts", default="primary", help="primary, all, or a comma list of primary,alt1,alt2")
    ap.add_argument("--model", help="annotator model id (Bedrock model or inference-profile id, or endpoint model name)")
    ap.add_argument("--backend", choices=("openai", "bedrock", "compatible"))
    ap.add_argument("--reasoning-effort", default="low", help="OpenAI reasoning models: minimal, low, medium, high")
    ap.add_argument("--max-usd", type=float, default=25.0, help="stop submitting once this run has spent this much")
    ap.add_argument("--region", default="eu-central-1")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key")
    ap.add_argument("--temperature", default="0", help="a number, or 'none' to omit it")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="the first N inputs of the seeded order (pilot, subsets)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    names = client.PROMPTS if a.prompts == "all" else tuple(a.prompts.split(","))
    eps = corpus.load(a.scope)
    A = grammar.analyse_all(eps)
    units = eligible(eps, A)
    if a.dry_run:
        return dry_run(names, units, a.limit, a.model)
    if a.report:
        import pandas as pd
        for name in names:
            for k, rows in report(eps, A, name, a.model).items():
                print(f"\n== {k}\n{pd.DataFrame(rows).to_string(index=False) if rows else '(none)'}")
        return
    if not (a.backend and a.model):
        ap.error("an extraction run needs --backend and --model (see --dry-run for costs)")
    temp = None if a.temperature == "none" else float(a.temperature)
    complete = (client.openai_api(a.reasoning_effort) if a.backend == "openai"
                else client.bedrock(a.region, temp, a.max_tokens) if a.backend == "bedrock"
                else client.openai_compatible(a.base_url, a.api_key, temp, a.max_tokens))
    left = {name: extract(name, a.model, complete, units, a.limit, a.workers, a.max_usd)["unanswered"] for name in names}
    if any(left.values()):  # non-zero exit, so finish.py never runs the once-only test arm on a partial extraction
        raise SystemExit(f"incomplete extraction, unanswered batches {left}; rerun to continue")


if __name__ == "__main__":
    main(sys.argv[1:])
