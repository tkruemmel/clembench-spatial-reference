"""E5 client: prompt files, batch rendering, backends and the on-disk cache.

One call annotates a batch of up to BATCH messages. A prompt file holds the
fixed instructions above the `=== INPUT ===` line and the batch template below
it. Each message is shown with the letters of the writer's own objects (what
the writer saw on its own board) and nothing else: no message board, outcome,
model, tier or owner map.

The unit of annotation is the input itself, `key(ep, t)` = the writer's
letters plus the message text, so a message repeated word for word is
annotated once. The cache is JSONL per (prompt sha, model), one row per batch
with its keys, so a rerun never pays twice.
"""
import hashlib
import json
import os
import re
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PROMPTS = ("primary", "alt1", "alt2")
CACHE = os.path.join(HERE, "cache")
SPLIT = "=== INPUT ===\n"
BATCH = 20
_lock = threading.Lock()


def prompt_path(name):
    return os.path.join(HERE, f"prompt_{name}.txt")


def prompt_sha(name):
    return hashlib.sha256(open(prompt_path(name), "rb").read()).hexdigest()


def own_letters(ep, t):
    return ", ".join(ep["objs"][t["seat"]])


def key(ep, t):
    """The annotator's whole input for one message, hashed."""
    return hashlib.sha1(f"{own_letters(ep, t)}\n{t['msg']}".encode()).hexdigest()[:16]


def render(name, batch):
    """(system, user) for a batch [(ep, t)]; messages are numbered from 1."""
    system, user = open(prompt_path(name)).read().split(SPLIT)
    blocks = [f"### Message {i} (the writer's own objects are {own_letters(ep, t)})\n{t['msg']}"
              for i, (ep, t) in enumerate(batch, 1)]
    return system.rstrip(), user.replace("<<MESSAGES>>", "\n\n".join(blocks))


# ---------------------------------------------------------------- cache

def cache_path(name, model):
    return os.path.join(CACHE, f"{name}_{prompt_sha(name)[:12]}_{re.sub(r'[^A-Za-z0-9.-]+', '_', model)}.jsonl")


def load_cache(name, model):
    """{message key: (batch row, 1-based position)} for this prompt's current sha and model."""
    path, sha, out = cache_path(name, model), prompt_sha(name), {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["prompt_sha"] == sha and r["model"] == model:
                for i, k in enumerate(r["keys"], 1):
                    out[k] = (r, i)
    return out


def store(name, model, keys, reply):
    text, usage = reply if isinstance(reply, tuple) else (reply, None)
    row = dict(prompt_sha=prompt_sha(name), model=model, keys=keys, text=text, usage=usage)
    with _lock:
        os.makedirs(CACHE, exist_ok=True)
        with open(cache_path(name, model), "a") as f:
            f.write(json.dumps(row) + "\n")
    return row


# ---------------------------------------------------------------- backends

def bedrock(region="eu-central-1", temperature=0.0, max_tokens=2048):
    """Anthropic Claude on Amazon Bedrock, bedrock-runtime Converse. temperature
    None omits it (the newest Claude models reject sampling parameters)."""
    import boto3
    from botocore.config import Config
    client = boto3.client("bedrock-runtime", region_name=region,
                          config=Config(read_timeout=300, retries={"max_attempts": 8, "mode": "adaptive"}))

    def complete(prompt, model_id):
        system, user = prompt
        cfg = {"maxTokens": max_tokens}
        if temperature is not None:
            cfg["temperature"] = temperature
        r = client.converse(modelId=model_id, system=[{"text": system}],
                            messages=[{"role": "user", "content": [{"text": user}]}], inferenceConfig=cfg)
        text = "".join(b["text"] for b in r["output"]["message"]["content"] if "text" in b)
        return text, dict(r.get("usage", {}), stop=r.get("stopReason"))
    return complete


def openai_api(reasoning_effort="low", max_tokens=16384):
    """OpenAI's API with the key from key.json `openai`. GPT-5-series models accept
    only the default temperature, so none is sent; reasoning tokens bill as output."""
    import openai
    conf = json.load(open(os.path.join(REPO, "key.json")))["openai"]
    client = openai.OpenAI(api_key=conf["api_key"], organization=conf.get("organisation") or None,
                           timeout=600, max_retries=5)

    def complete(prompt, model_id):
        system, user = prompt
        r = client.chat.completions.create(model=model_id, reasoning_effort=reasoning_effort,
                                           max_completion_tokens=max_tokens,
                                           messages=[{"role": "developer", "content": system},
                                                     {"role": "user", "content": user}])
        u = r.usage
        det = getattr(u, "completion_tokens_details", None)
        return r.choices[0].message.content or "", dict(
            inputTokens=u.prompt_tokens, outputTokens=u.completion_tokens,
            reasoningTokens=getattr(det, "reasoning_tokens", None), stop=r.choices[0].finish_reason)
    return complete


def openai_compatible(base_url=None, api_key=None, temperature=0.0, max_tokens=2048):
    """An OpenAI-compatible chat endpoint; defaults from key.json `generic_openai_compatible`."""
    import openai
    keys = os.path.join(REPO, "key.json")
    conf = json.load(open(keys)).get("generic_openai_compatible", {}) if os.path.exists(keys) else {}
    client = openai.OpenAI(base_url=base_url or conf.get("base_url"), api_key=api_key or conf.get("api_key") or "none",
                           timeout=600, max_retries=5)

    def complete(prompt, model_id):
        system, user = prompt
        kw = dict(model=model_id, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if temperature is not None:
            kw["temperature"] = temperature
        r = client.chat.completions.create(**kw)
        u = r.usage
        return r.choices[0].message.content or "", (dict(inputTokens=u.prompt_tokens, outputTokens=u.completion_tokens,
                                                          stop=r.choices[0].finish_reason) if u else None)
    return complete
