"""E3/E4 backends, the reply cache and the spending cap.

Campaign settings (interactions.json `player_models`, model_registry.json):
- dsv41: deepseek-flash on api.deepseek.com, thinking enabled, reasoning_effort
  high, T 0.6, max_tokens 16384 (doubled up to twice when a reply is cut
  mid-think, as deepseek_api.py does), 1500 s hard cap per attempt;
- nemotron: nvidia.nemotron-super-3-120b, Bedrock Converse in eu-central-1,
  T 0.6, no maxTokens (`uncapped_output`), the logged message structure
  (dev: preamble and turn context as two user messages, each stripped as
  amazon_bedrock_api.py did); thinking arrives inline, ending in `</think>`;
- qwen38: the Potsdam endpoint, streamed, T 0.6, no max_tokens, thinking on.
  Never called while its campaign holds the GPU (the CLI refuses it).

The cache is JSONL per model, one row per reply, keyed by (edited-prompt sha,
model, sample index); errors are not cached, so a rerun retries them.
"""
import collections
import datetime
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
CACHE = os.path.join(HERE, "cache")
HARD_CAP_SEC = 1500
WORKERS = {"dsv41": 16, "nemotron": 8, "qwen38": 8}
# USD per million tokens (input, output). DeepSeek off-peak list price, doubled
# 01-04 and 06-10 UTC Mon-Fri; every input token is priced as a cache miss.
# Nemotron: Bedrock on-demand, Europe (Frankfurt), aws.amazon.com/bedrock/pricing
# read 2026-10-17. Qwen on Potsdam is free.
PRICES = {"dsv41": (0.15, 0.60), "nemotron": (0.18, 0.78), "qwen38": (0.0, 0.0)}
_lock = threading.Lock()


def peak(t):
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def usd(model, usage, t=None):
    pin, pout = PRICES[model]
    f = 2 if model == "dsv41" and peak(t or datetime.datetime.now(datetime.timezone.utc)) else 1
    return f * (usage.get("in", 0) * pin + usage.get("out", 0) * pout) / 1e6


# ---------------------------------------------------------------- cache

def cache_path(model):
    return os.path.join(CACHE, f"replies_{model}.jsonl")


def load_cache(model):
    """{(sha, sample index): row}; a later row replaces an earlier one."""
    out, p = {}, cache_path(model)
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line)
            out[(r["sha"], r["i"])] = r
    return out


def store(model, sha, i, reply):
    text, usage, finish = reply
    row = dict(sha=sha, model=model, i=i, text=text, usage=usage, finish=finish, usd=usd(model, usage),
               ts=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    with _lock:
        os.makedirs(CACHE, exist_ok=True)
        with open(cache_path(model), "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


# ---------------------------------------------------------------- backends

def _hard_cap(fn):
    """Bound one attempt's wall clock from outside (a black-holed DeepSeek
    socket keeps resetting httpx's read timeout); the thread leaks on timeout."""
    ex = ThreadPoolExecutor(1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=HARD_CAP_SEC)
    except FutureTimeout:
        raise TimeoutError(f"request exceeded the {HARD_CAP_SEC} s hard cap")
    finally:
        ex.shutdown(wait=False)


TRANSIENT = re.compile(r"timeout|timed out|429|50[0234]|rate|busy|overload|connection|throttl|unavailable|"
                       r"internal ?server|modelnotready|eventstream", re.I)


def _retry(fn, tries=6, delay=10):
    """Retry transient failures with backoff; a bad request, a refused key or an
    empty account raises at once."""
    for a in range(tries):
        try:
            return fn()
        except Exception as e:
            if a == tries - 1 or not TRANSIENT.search(f"{type(e).__name__} {e}"):
                raise
            time.sleep(min(120, delay * 2 ** a))


def deepseek():
    import openai
    key = json.load(open(os.path.join(REPO, "key.json")))["deepseek"]["api_key"]
    client = openai.OpenAI(base_url="https://api.deepseek.com", api_key=key, timeout=1800.0, max_retries=0)

    def complete(msgs):
        mt, tot = 16384, collections.Counter()
        for _ in range(3):
            r = _retry(lambda: _hard_cap(lambda: client.chat.completions.create(
                model="deepseek-flash", messages=msgs, temperature=0.6, max_tokens=mt,
                extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"})))
            ch, u = r.choices[0], r.usage
            det = getattr(u, "completion_tokens_details", None)
            tot.update({"in": u.prompt_tokens, "out": u.completion_tokens, "hit": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
                        "reasoning": getattr(det, "reasoning_tokens", 0) or 0, "attempts": 1})
            text = ch.message.content or ""
            if text.strip() or ch.finish_reason != "length":
                break
            mt *= 2  # cut mid-think: deepseek_api.py doubles max_tokens, twice at most; every attempt is billed
        return text, dict(tot), ch.finish_reason
    return complete


def bedrock(model_id="nvidia.nemotron-super-3-120b", region="eu-central-1"):
    import boto3
    from botocore.config import Config
    client = boto3.client("bedrock-runtime", region_name=region,
                          config=Config(read_timeout=HARD_CAP_SEC, retries={"max_attempts": 8, "mode": "adaptive"}))

    def complete(msgs):
        conv = [{"role": m["role"], "content": [{"text": m["content"].strip()}]} for m in msgs if m["content"].strip()]
        r = _retry(lambda: client.converse(modelId=model_id, messages=conv, inferenceConfig={"temperature": 0.6}))
        text = "\n".join(b["text"].strip() for b in r["output"]["message"]["content"] if "text" in b)
        u = r.get("usage", {})
        return text, {"in": u.get("inputTokens", 0), "out": u.get("outputTokens", 0)}, r.get("stopReason")
    return complete


def potsdam():
    """Streamed, bounded by 900 s of silence and never by total duration
    (project memory: the proxy kills non-streamed requests at 300 s)."""
    import httpx
    import openai
    conf = json.load(open(os.path.join(REPO, "key.json")))["openai_compatible"]
    client = openai.OpenAI(base_url=conf["base_url"], api_key=conf["api_key"], max_retries=0,
                           http_client=httpx.Client(verify=False, timeout=httpx.Timeout(900.0, connect=30.0)))

    def complete(msgs):
        def once():
            text, usage, finish = [], None, None
            for ev in client.chat.completions.create(
                    model="Qwen/Qwen3.8-27B-FP8", messages=msgs, temperature=0.6, stream=True,
                    stream_options={"include_usage": True},
                    extra_body={"chat_template_kwargs": {"enable_thinking": True}}):
                if ev.usage:
                    usage = ev.usage
                for ch in ev.choices:
                    text.append(ch.delta.content or "")
                    finish = ch.finish_reason or finish
            return "".join(text), {"in": getattr(usage, "prompt_tokens", 0), "out": getattr(usage, "completion_tokens", 0)}, finish
        return _retry(once)
    return complete


BACKENDS = {"dsv41": deepseek, "nemotron": bedrock, "qwen38": potsdam}


# ---------------------------------------------------------------- execution

def execute(model, jobs, complete, max_usd, workers=None):
    """Call every (sha, i, msgs) job not cached, stopping once this run's
    spend reaches max_usd. Returns dict(ok, error, skipped, usd)."""
    done = load_cache(model)
    todo = [j for j in dict(((s, i), (s, i, m)) for s, i, m in jobs).values() if (j[0], j[1]) not in done]
    st = dict(ok=0, error=0, skipped=0, usd=0.0)
    print(f"{model}: {len(jobs)} jobs, {len(todo)} to call, cap ${max_usd:.2f}", flush=True)

    def one(job):
        s, i, msgs = job
        if st["usd"] >= max_usd or (st["error"] >= 10 and not st["ok"]):
            st["skipped"] += 1
            return
        try:
            row = store(model, s, i, complete(msgs))
            with _lock:
                st["ok"] += 1
                st["usd"] += row["usd"]
        except Exception as e:
            with _lock:
                st["error"] += 1
            if st["error"] <= 5:
                print(f"  error: {e!r}"[:400], flush=True)

    with ThreadPoolExecutor(workers or WORKERS[model]) as ex:
        for n, _ in enumerate(ex.map(one, todo), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}  {st['error']} errors  ${st['usd']:.2f}", flush=True)
    print(f"  done: {st['ok']} stored, {st['error']} errors, {st['skipped']} skipped, ${st['usd']:.2f}", flush=True)
    return st
