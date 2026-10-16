#!/usr/bin/env python3
"""One-screen health check for the grid campaign (results_qwen38_grid_n15_*).

Prints, per suite: episodes done / expected, driver and shard liveness, live
sockets to the endpoint, each shard's tqdm progress, per-call latency and
episode duration of episodes finished since launch (and in the last window),
and transport errors from clembench.log since launch. Ends with heuristics for
the two decisions the monitor has to make: is the endpoint cascading into 300 s
timeouts (scale down), and has a suite finished (re-allocate its streams).

    python3 campaign/grid_status.py [--window-hours 3]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "Qwen3.8-27B-FP8"
SUITES = [
    ("solo", "matrixgame_covered_solo_grid", 120),
    ("dyad", "matrixgame_covered_comm_grid", 240),
    ("3p", "matrixgame_covered_comm_3p_grid", 240),
    ("6p", "matrixgame_covered_comm_6p_grid", 240),
]
ENDPOINT_IP = "141.89.97.67"
TQDM = re.compile(r"Playing game instances:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]")
LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def launch_time(chain_log):
    starts = [l for l in chain_log.read_text(errors="ignore").splitlines() if "SUITE_START" in l]
    if not starts:
        return None
    return datetime.strptime(starts[-1][1:20], "%Y-%m-%d %H:%M:%S")


def episodes(results, game):
    rows = []
    for f in glob.glob(f"{results}/{MODEL}/{game}/*/instance_*/interactions.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ts = [e["timestamp"] for t in d["turns"] for e in t if "timestamp" in e]
        if not ts:
            continue
        t0, t1 = datetime.fromisoformat(min(ts)), datetime.fromisoformat(max(ts))
        rc = d.get("Request Count") or 1
        level = f.split("/")[-3].split("_")[0]
        rows.append(dict(level=level, t0=t0, t1=t1, dur=(t1 - t0).total_seconds(), rc=rc,
                         aborted=bool(d.get("Aborted")), success=bool(d.get("Success"))))
    return rows


def summarize(rows):
    if not rows:
        return "none"
    dur = [r["dur"] for r in rows]
    spc = sum(dur) / sum(r["rc"] for r in rows)
    lv = {}
    for r in rows:
        lv.setdefault(r["level"], []).append(r)
    per = " ".join(f"{k}:{len(v)}" for k, v in sorted(lv.items()))
    ab = sum(r["aborted"] for r in rows)
    su = sum(r["success"] for r in rows)
    return (f"n={len(rows)} ({per}) median {st.median(dur)/60:.0f} min/ep, "
            f"{spc:.0f} s/call, success {su}, aborted {ab}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-hours", type=float, default=3.0)
    args = ap.parse_args()
    now = datetime.now()
    print(f"grid campaign status @ {now:%F %T}")

    sockets = sh(f"lsof -nP -iTCP@{ENDPOINT_IP}:443 -sTCP:ESTABLISHED 2>/dev/null | tail -n +2")
    sock_pids = [l.split()[1] for l in sockets.splitlines() if l.strip()]
    finished, cascade_flags, total_streams, since_global = [], [], 0, None

    for suite, game, expected in SUITES:
        results = REPO / f"results_qwen38_grid_n15_{suite}"
        camp = results / "_campaign"
        chain = camp / "chain.log"
        since = launch_time(chain) if chain.exists() else None
        if since and (since_global is None or since < since_global):
            since_global = since
        done = len(glob.glob(f"{results}/{MODEL}/{game}/*/instance_*/interactions.json"))
        drv_pid = (camp / "driver.lock" / "pid").read_text().strip() if (camp / "driver.lock" / "pid").exists() else ""
        drv = f"driver {drv_pid} alive" if drv_pid and alive(drv_pid) else "driver NOT running"
        # Shards are the driver's direct children. Do not grep command lines:
        # other campaigns (e.g. the DeepSeek grid run) play the same games.
        shard_pids = []
        if drv_pid and alive(drv_pid):
            for line in sh("ps -axo pid,ppid,command -ww").splitlines():
                parts = line.split(None, 2)
                if len(parts) == 3 and parts[1] == drv_pid and "clem run " in parts[2]:
                    shard_pids.append(parts[0])
        n_sock = sum(1 for p in sock_pids if p in shard_pids)
        total_streams += len(shard_pids)
        # Episode dirs without interactions.json: one per shard is normal (in flight);
        # any surplus is an episode that died mid-flight (timeout/exception) and will be replayed.
        partial = sum(1 for d in glob.glob(f"{results}/{MODEL}/{game}/*/instance_*")
                      if not os.path.exists(os.path.join(d, "interactions.json")))
        dead = max(0, partial - len(shard_pids))
        print(f"\n== {suite:4s} {game}: {done}/{expected} done | {drv} | {len(shard_pids)} shards, {n_sock} live sockets"
              f" | {partial} in flight, {dead} died"
              + (f" | started {since:%m-%d %H:%M} ({(now - since).total_seconds()/3600:.1f} h ago)" if since else ""))
        if chain.exists():
            tail = [l for l in chain.read_text(errors="ignore").splitlines() if l.strip()][-2:]
            for l in tail:
                print("   chain:", l[:160])
        if done >= expected:
            finished.append(suite)

        sits = []
        for log in sorted(camp.glob("instances_resume_*_grid*.log")):
            text = log.read_text(errors="ignore").replace("\r", "\n")
            m = None
            for m_ in TQDM.finditer(text):
                m = m_
            if m:
                k, n, meta = m.groups()
                sit = re.search(r"([\d.]+)s/it", meta)
                sits.append(float(sit.group(1)) if sit else None)
                print(f"   {log.stem[-3:]}: {k}/{n} [{meta}]")
        vals = [s for s in sits if s]
        if len(vals) >= 3 and max(vals) - min(vals) < 10 and 280 <= st.mean(vals) <= 310:
            cascade_flags.append(suite)

        rows = episodes(results, game)
        rows = [r for r in rows if since is None or r["t1"] >= since]
        recent = [r for r in rows if r["t1"] >= now - timedelta(hours=args.window_hours)]
        print(f"   since launch : {summarize(rows)}")
        print(f"   last {args.window_hours:g} h     : {summarize(recent)}")

    # transport errors since launch, from the shared clembench.log at repo root
    # clembench.log at the repo root is shared by every clem process on this machine
    # (the DeepSeek grid campaign writes to it too). httpx lines carry the host, so
    # responses and 502s are scoped to Potsdam; timeout/exception lines carry no host
    # and are reported as machine-wide counts.
    log = REPO / "clembench.log"
    ok = e502 = timeouts = exc = 0
    if log.exists() and since_global:
        for line in log.read_text(errors="ignore").splitlines()[-300000:]:
            m = LOG_TS.match(line)
            if m and datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") < since_global:
                continue
            if "jarvis.ling.uni-potsdam.de" in line:
                if '"HTTP/1.1 200' in line:
                    ok += 1
                elif '"HTTP/1.1 502' in line:
                    e502 += 1
            if m and ("Request timed out" in line or "ReadTimeout" in line or "APITimeoutError" in line):
                timeouts += 1
            if "Exception for instance" in line:
                exc += 1
    print(f"\n== transport since launch: Potsdam {ok} x 200 OK, {e502} x 502 | machine-wide (shared log): "
          f"{timeouts} timeouts, {exc} instance exceptions | {total_streams} shards, {len(sock_pids)} live sockets to {ENDPOINT_IP}")

    print("\n== assessment")
    if cascade_flags:
        print(f"   CASCADE SUSPECTED in {cascade_flags}: identical ~294 s/it across shards = the 300 s read "
              f"timeout firing, not real work. Scale down (fewer shards) now; in-flight work is worthless.")
    elif timeouts:
        print(f"   {timeouts} timeouts since launch: watch the trend; a rising hourly count means the "
              f"endpoint is saturating (reasoning traces > ~9k tokens at current speed).")
    else:
        print("   no timeouts, no cascade signature.")
    if finished:
        print(f"   finished suites: {finished}. Their streams are idle; re-allocating means restarting "
              f"the other drivers (discarding their in-flight episodes) - worth it only if days remain.")
    print(f"   {total_streams} streams running; healthy reference from the compact campaign: 13-16, cascade at 24.")


if __name__ == "__main__":
    main()
