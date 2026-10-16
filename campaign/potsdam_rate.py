#!/usr/bin/env python3
"""Endpoint throughput per 15-minute bucket for the Potsdam Qwen campaign.

From the machine-wide clembench.log (httpx lines): 200 OK calls, 502s, read
timeouts, instance exceptions. From finished episodes' requests.json under
--prefix: completion tokens and tokens per call. Completion tokens/h is the
endpoint's real capacity (calls/h varies 4x with the protocol mix: `none` calls
are ~250-600 tokens, `hybrid` ~2000+). Buckets younger than ~1-2 h undercount
tokens because in-flight episodes have not written requests.json yet.

Cascade signature: timeouts in several consecutive buckets. Saturation: more
streams, same or fewer tokens/h (seen at 24 streams on 2026-09-05).

    python3 campaign/potsdam_rate.py [hours] [--prefix results_qwen38_grid_n15]
"""
import argparse
import glob
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("hours", nargs="?", type=float, default=3.0)
ap.add_argument("--prefix", default="results_qwen38_grid_n15")
args = ap.parse_args()
REPO = Path(__file__).resolve().parent.parent
since = datetime.now() - timedelta(hours=args.hours)

stamp = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}):\d{2},\d+ - (.*)$")
log = defaultdict(lambda: [0, 0, 0, 0])
with open(REPO / "clembench.log", errors="ignore") as fh:
    for line in fh:
        m = stamp.match(line)
        if not m:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M")
        if t < since:
            continue
        b = t.replace(minute=t.minute // 15 * 15)
        rest = m.group(2)
        if "jarvis" in rest and '"HTTP/1.1 200 OK"' in rest:
            log[b][0] += 1
        elif "jarvis" in rest and '"HTTP/1.1 502' in rest:
            log[b][1] += 1
        elif "Request timed out" in rest or "ReadTimeout" in rest or "APITimeoutError" in rest:
            log[b][2] += 1
        elif "Exception for instance" in rest:
            log[b][3] += 1

# Per-file cache of (bucket -> calls, completion tokens): a finished episode's
# requests.json never changes, and parsing hundreds of multi-MB files every check
# took minutes once the campaign had a few hundred episodes.
import os
cache_path = REPO / "campaign" / f".potsdam_rate_cache_{args.prefix}.json"
try:
    cache = json.load(open(cache_path))
except Exception:
    cache = {}
tok = defaultdict(lambda: [0, 0])
dirty = False
for rf in glob.glob(f"{REPO}/{args.prefix}_*/*/*/*/instance_*/*.requests.json"):
    if "_stale" in rf or "_oldprompt" in rf:   # parked trees share the prefix
        continue
    try:
        stat = os.stat(rf)
    except OSError:
        continue
    key = f"{stat.st_size}:{int(stat.st_mtime)}"
    entry = cache.get(rf)
    if not entry or entry["key"] != key:
        per = defaultdict(lambda: [0, 0])
        try:
            b = json.load(open(rf))
            for rec in (b.get("calls", []) if isinstance(b, dict) else b):
                c = rec.get("call", rec)
                try:
                    t = datetime.fromisoformat(c["timestamp"])
                except Exception:
                    continue
                u = (c.get("raw_response_obj") or {}).get("usage") or {}
                k = t.replace(minute=t.minute // 15 * 15, second=0, microsecond=0).isoformat()
                per[k][0] += 1
                per[k][1] += u.get("completion_tokens", 0) or 0
        except Exception:
            continue
        entry = cache[rf] = {"key": key, "buckets": dict(per)}
        dirty = True
    for k, (n, comp) in entry["buckets"].items():
        t = datetime.fromisoformat(k)
        if t >= since:
            tok[t][0] += n
            tok[t][1] += comp
if dirty:
    tmp = str(cache_path) + ".tmp"
    json.dump(cache, open(tmp, "w"))
    os.replace(tmp, cache_path)

print(f"{'bucket':<13}{'200OK':>6}{'/h':>6}{'502':>4}{'tmo':>4}{'exc':>4} | {'fin.calls':>9}{'compl_tok/h':>12}{'tok/s':>6}{'tok/call':>9}")
for b in sorted(set(log) | set(tok)):
    ok, bad, to, ex = log.get(b, [0, 0, 0, 0])
    n, comp = tok.get(b, [0, 0])
    print(f"{b:%m-%d %H:%M}  {ok:6d}{ok*4:6d}{bad:4d}{to:4d}{ex:4d} | {n:9d}{comp*4:12d}{comp*4/3600:6.0f}{comp//max(n,1):9d}")
