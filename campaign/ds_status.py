#!/usr/bin/env python3
"""One-screen health check for a worker-based campaign (default: the
DeepSeek-V4-Flash grid run, results_dsflash_grid_n15_*).

Per suite: episodes done / expected, live workers and what each is playing (with
age), stale claims and failed attempts, latency (s/call) and duration of the
episodes finished since launch and within the last --window-hours, worker
failures. Then the account balance, an empirical spend rate (each run appends a
balance snapshot to <prefix>_solo/_campaign/balance.log, so $/episode needs no
price table), an ETA, and the assessment the monitor acts on: is the API
latency inflating with concurrency (hold / reduce workers), is there headroom
(add workers), is the balance about to run out, did workers give up.

    python3 campaign/ds_status.py [--window-hours 2] [--tokens]

Scaling is done with `SOLO=.. DYAD=.. P3=.. P6=.. bash campaign/dsflash_grid.sh`
(workers stop only between episodes, so nothing in flight is lost).
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import subprocess
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# ARM=_final selects the final arm's games (suite and game names carry the suffix).
ARM = os.environ.get("ARM", "")
# SUITES="3p 6p" restricts it to an arm's suites (ARM=_final_chrono has no solo/dyad);
# hetzner_dsflash.sh and dsv41_grid.sh read the same variable.
_ONLY = os.environ.get("SUITES", "").split()
SUITES = [s for s in [
    ("solo", "solo_grid" + ARM, "matrixgame_covered_solo_grid" + ARM, 60 if ARM == "_final" else 120),  # final: silent only
    ("dyad", "dyad_grid" + ARM, "matrixgame_covered_comm_grid" + ARM, 240),
    ("3p", "3p_grid" + ARM, "matrixgame_covered_comm_3p_grid" + ARM, 240),
    ("6p", "6p_grid" + ARM, "matrixgame_covered_comm_6p_grid" + ARM, 240),
] if not _ONLY or s[0] in _ONLY]
# Compact DeepSeek-V4-Flash campaign (Hetzner, 2026-08): ~45-60 s/call, flat up to ~10 streams.
BASELINE_S_PER_CALL = 55
STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] w(\d+): (.*)$")


def alive(pid):
    """True if pid is a running campaign worker shell (after a reboot the old pids
    are reused by unrelated processes, so a bare kill -0 is not enough)."""
    try:
        os.kill(int(pid), 0)
    except Exception:
        return False
    out = subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))], capture_output=True, text=True).stdout
    return "campaign/worker.sh" in out


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def worker_events(campaign):
    """[(time, worker, message)] over all worker logs, sorted."""
    rows = []
    for log in glob.glob(f"{campaign}/worker_w*.log"):
        for line in Path(log).read_text(errors="ignore").splitlines():
            m = STAMP.match(line)
            if m:
                rows.append((parse_ts(m.group(1)), int(m.group(2)), m.group(3)))
    rows.sort()
    return rows


def request_count(episode_dir, d):
    rc = d.get("Request Count")
    if rc:
        return rc
    n = 0
    for rf in glob.glob(f"{episode_dir}/*.requests.json"):
        try:
            b = json.load(open(rf))
            n += len(b.get("calls", [])) if isinstance(b, dict) else len(b)
        except Exception:
            pass
    return n or 1


def episodes(results, model, game):
    rows = []
    for f in glob.glob(f"{results}/{model}/{game}/*/instance_*/interactions.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ts = [e["timestamp"] for t in d["turns"] for e in t if "timestamp" in e]
        if not ts:
            continue
        t0, t1 = datetime.fromisoformat(min(ts)), datetime.fromisoformat(max(ts))
        exp = f.split("/")[-3]
        level = next((s for s in ("S1", "S2", "S3", "S4") if s in exp), "?")
        rows.append(dict(level=level, exp=exp, t0=t0, t1=t1, dur=(t1 - t0).total_seconds(),
                         rc=request_count(os.path.dirname(f), d),
                         aborted=bool(d.get("Aborted")), success=bool(d.get("Success")),
                         dir=os.path.dirname(f)))
    return rows


def summarize(rows):
    if not rows:
        return "none", None
    dur = [r["dur"] for r in rows]
    calls = sum(r["rc"] for r in rows)
    spc = sum(dur) / calls if calls else float("nan")
    lv = {}
    for r in rows:
        lv.setdefault(r["level"], []).append(r)
    per = " ".join(f"{k}:{len(v)}" for k, v in sorted(lv.items()))
    return (f"n={len(rows)} ({per}) median {st.median(dur)/60:.0f} min/ep, {spc:.0f} s/call, "
            f"success {sum(r['success'] for r in rows)}, aborted {sum(r['aborted'] for r in rows)}"), spc


def tokens(rows):
    tot = dict(calls=0, hit=0, miss=0, completion=0)
    for r in rows:
        for rf in glob.glob(f"{r['dir']}/*.requests.json"):
            try:
                b = json.load(open(rf))
            except Exception:
                continue
            for rec in (b.get("calls", []) if isinstance(b, dict) else b):
                c = rec.get("call", rec) if isinstance(rec, dict) else {}
                u = (c.get("raw_response_obj") or {}).get("usage") or {}
                if not u:
                    continue
                tot["calls"] += 1
                tot["hit"] += u.get("prompt_cache_hit_tokens") or 0
                tot["miss"] += u.get("prompt_cache_miss_tokens") or (u.get("prompt_tokens", 0) - (u.get("prompt_cache_hit_tokens") or 0))
                tot["completion"] += u.get("completion_tokens", 0)
    return tot


def balance():
    try:
        key = json.load(open(REPO / "key.json"))["deepseek"]["api_key"]
        req = urllib.request.Request("https://api.deepseek.com/user/balance",
                                     headers={"Authorization": f"Bearer {key}"})
        info = json.loads(urllib.request.urlopen(req, timeout=20).read())
        return float(info["balance_infos"][0]["total_balance"]), info["balance_infos"][0]["currency"]
    except Exception as exc:
        return None, str(exc)[:80]


def probe(prefix, model, window_hours, suites=SUITES):
    """Machine-readable snapshot of the same state main() prints.

    Written for campaign/supervise_dsv41.py, which runs this over ssh on the
    campaign host once per tick: one round trip instead of a dozen, and the
    supervisor never has to parse the human report. Missing trees are normal
    (a fresh campaign has none yet) and come back as zeros, not errors.
    """
    now = datetime.now()
    window = now - timedelta(hours=window_hours)
    out = {"now": now.isoformat(timespec="seconds"), "model": model, "prefix": prefix, "suites": {}}
    total_done = total_expected = total_workers = 0
    remaining_hours = 0.0
    finished_window = 0

    for short, suite, game, expected in suites:
        results = REPO / f"{prefix}_{short}"
        campaign = results / "_campaign"
        rows = episodes(results, model, game)
        recent = [r for r in rows if r["t1"] >= window]
        events = worker_events(campaign)

        live = []
        for pidfile in glob.glob(f"{campaign}/worker_w*.pid"):
            wid = int(re.search(r"worker_w(\d+)\.pid", pidfile).group(1))
            if alive(Path(pidfile).read_text().strip()):
                live.append(wid)
        live.sort()

        # A stop file outlives the worker it retired, so only those whose worker
        # is still up count as a drain in progress (see supervise.py).
        stops = []
        for path in glob.glob(f"{campaign}/stop_w*"):
            m = re.search(r"stop_w(\d+)$", path)
            if not m:
                continue
            wid = int(m.group(1))
            pidfile = Path(f"{campaign}/worker_w{wid}.pid")
            try:
                running = pidfile.exists() and alive(pidfile.read_text().strip())
            except OSError:
                running = False
            if running:
                stops.append(wid)
        stops.sort()

        claims, dead_claims = [], 0
        for c in sorted(glob.glob(f"{campaign}/claims/*")):
            name = os.path.basename(c)
            if not os.path.isdir(c) or ".stale." in name:
                continue
            pid = (Path(c) / "pid").read_text().strip() if (Path(c) / "pid").exists() else ""
            since = (Path(c) / "since").read_text().strip() if (Path(c) / "since").exists() else ""
            wid = (Path(c) / "worker").read_text().strip() if (Path(c) / "worker").exists() else "?"
            owner_alive = bool(pid) and alive(pid)
            dead_claims += not owner_alive
            claims.append({"worker": int(wid) if wid.isdigit() else -1, "episode": name,
                           "age_min": round((now - parse_ts(since)).total_seconds() / 60, 1) if since else None,
                           "owner_alive": owner_alive})

        fails = glob.glob(f"{campaign}/claims/*.fails")
        blocked = []
        for f in fails:
            try:
                attempts = len(Path(f).read_text().splitlines())
                age_h = (now - datetime.fromtimestamp(os.path.getmtime(f))).total_seconds() / 3600
            except OSError:
                continue
            blocked.append({"episode": os.path.basename(f)[:-len(".fails")],
                            "attempts": attempts, "age_h": round(age_h, 1)})

        _, spc_all = summarize(rows)
        _, spc_win = summarize(recent)
        med_min = st.median([r["dur"] for r in rows]) / 60 if rows else None
        if med_min:
            remaining_hours += max(0, expected - len(rows)) * med_min / 60
        finished_window += len(recent)
        total_done += len(rows)
        total_expected += expected
        total_workers += len(live)

        out["suites"][short] = {
            "suite": suite, "game": game, "done": len(rows), "expected": expected,
            "live": live, "stops": stops, "claims": claims, "dead_claims": dead_claims,
            "blocked": blocked,
            "median_min": round(med_min, 1) if med_min else None,
            "s_per_call": round(spc_all) if spc_all else None,
            "window": {"n": len(recent),
                       "s_per_call": round(spc_win) if spc_win else None},
            "success": sum(r["success"] for r in rows),
            "aborted": sum(r["aborted"] for r in rows),
            "gave_up": sum("giving up" in m for _, _, m in events),
            "recent_failures": [f"{t:%H:%M} w{w}: {m[:100]}"
                                for t, w, m in events if m.startswith("FAILED") and t >= window][-3:],
        }

    bal, cur = balance()
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = None
    out["total"] = {"done": total_done, "expected": total_expected, "workers": total_workers}
    out["throughput_ep_h"] = round(finished_window / window_hours, 2) if window_hours else None
    out["remaining_hours"] = round(remaining_hours)
    out["balance"] = {"usd": bal, "currency": cur if bal is not None else None,
                      "error": None if bal is not None else cur}
    out["load"] = [load1, load5, load15]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="results_dsflash_grid_n15")
    ap.add_argument("--model", default="DeepSeek-V4-Flash")
    ap.add_argument("--window-hours", type=float, default=2.0)
    ap.add_argument("--tokens", action="store_true", help="also sum token usage of finished episodes")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable snapshot for campaign/supervise_dsv41.py (one ssh round trip)")
    args = ap.parse_args()
    if args.json:
        print(json.dumps(probe(args.prefix, args.model, args.window_hours)))
        return
    now = datetime.now()
    window = now - timedelta(hours=args.window_hours)
    print(f"{args.model} worker campaign status @ {now:%F %T}  (window {args.window_hours:g} h)")

    total_done = total_expected = total_workers = 0
    remaining_hours = 0.0
    spc_window, spc_all, notes = [], [], []
    all_rows_window = []
    launch = None
    for short, suite, game, expected in SUITES:
        results = REPO / f"{args.prefix}_{short}"
        campaign = results / "_campaign"
        rows = episodes(results, args.model, game)
        done = len(rows)
        total_done += done
        total_expected += expected

        events = worker_events(campaign)
        starts = [t for t, _, m in events if m.startswith("WORKER_START")]
        if starts:
            launch = min(launch, starts[0]) if launch else starts[0]
        live = []
        for pidfile in glob.glob(f"{campaign}/worker_w*.pid"):
            wid = int(re.search(r"worker_w(\d+)\.pid", pidfile).group(1))
            if alive(Path(pidfile).read_text().strip()):
                live.append(wid)
        live.sort()
        total_workers += len(live)
        stops = sorted(int(m.group(1)) for p in glob.glob(f"{campaign}/stop_w*")
                       for m in [re.search(r"stop_w(\d+)$", p)] if m)

        claims, stale = [], 0
        for c in sorted(glob.glob(f"{campaign}/claims/*")):
            name = os.path.basename(c)
            if not os.path.isdir(c) or ".stale." in name:
                stale += ".stale." in name
                continue
            pid = (Path(c) / "pid").read_text().strip() if (Path(c) / "pid").exists() else ""
            since = (Path(c) / "since").read_text().strip() if (Path(c) / "since").exists() else ""
            wid = (Path(c) / "worker").read_text().strip() if (Path(c) / "worker").exists() else "?"
            age = (now - parse_ts(since)).total_seconds() / 60 if since else float("nan")
            claims.append((int(wid) if wid.isdigit() else -1, name, age, alive(pid)))
        fails = glob.glob(f"{campaign}/claims/*.fails")
        fail_lines = sum(len(Path(f).read_text().splitlines()) for f in fails)

        print(f"\n== {suite}: {done}/{expected} done, {len(live)} live worker(s) {live}"
              + (f", stop requested for {stops}" if stops else "")
              + (f", {stale} stale claim(s)" if stale else "")
              + (f", {fail_lines} failed attempt(s) on {len(fails)} episode(s)" if fails else ""))
        for wid, name, age, ok in sorted(claims):
            print(f"   w{wid:<3} {name:<40} {age:5.0f} min" + ("" if ok else "  [owner dead]"))
        recent = [r for r in rows if r["t1"] >= window]
        s_all, spc_a = summarize(rows)
        s_win, spc_w = summarize(recent)
        print(f"   all finished : {s_all}")
        print(f"   last window  : {s_win}")
        if spc_a:
            spc_all.append(spc_a)
        if spc_w:
            spc_window.append(spc_w)
        all_rows_window += recent
        failed = [(t, w, m) for t, w, m in events if m.startswith("FAILED") and t >= window]
        gave_up = [(t, w, m) for t, w, m in events if "giving up" in m]
        for t, w, m in failed[-5:]:
            print(f"   {t:%H:%M} w{w}: {m[:110]}")
        if gave_up:
            print(f"   {len(gave_up)} worker(s) gave up, last: {gave_up[-1][0]:%F %H:%M} w{gave_up[-1][1]}")
            notes.append(f"{suite}: {len(gave_up)} worker(s) gave up (see FAILED lines) - fix cause, then re-run dsflash_grid.sh")
        if rows:
            med_min = st.median([r["dur"] for r in rows]) / 60
            remaining_hours += (expected - done) * med_min / 60
        if done >= expected and live:
            notes.append(f"{suite}: complete but {len(live)} worker(s) still alive - they exit on their own")
        if done >= expected:
            notes.append(f"{suite}: COMPLETE - reassign its worker budget to the other suites")
        if args.tokens and rows:
            tk = tokens(rows)
            print(f"   tokens (all finished): {tk['calls']} calls, cache-hit {tk['hit']/1e6:.1f}M, "
                  f"cache-miss {tk['miss']/1e6:.1f}M, completion {tk['completion']/1e6:.1f}M")

    print(f"\n== total: {total_done}/{total_expected} episodes, {total_workers} live workers"
          + (f", launched {launch:%F %H:%M} ({(now - launch).total_seconds()/3600:.1f} h ago)" if launch else ""))
    finished_window = len(all_rows_window)
    rate = finished_window / args.window_hours
    if rate:
        print(f"   throughput last window: {finished_window} episodes / {args.window_hours:g} h = {rate:.1f} ep/h"
              f" -> naive ETA {(total_expected - total_done) / rate:.1f} h")
    if remaining_hours and total_workers:
        print(f"   remaining ~{remaining_hours:.0f} episode-hours at observed per-suite medians"
              f" -> ~{remaining_hours / total_workers:.1f} h at {total_workers} workers")

    if not args.no_balance:
        bal, cur = balance()
        blog = REPO / f"{args.prefix}_solo" / "_campaign" / "balance.log"
        if bal is not None:
            blog.parent.mkdir(parents=True, exist_ok=True)
            with open(blog, "a") as fh:
                fh.write(f"{now:%F %T} {bal:.2f} {total_done}\n")
            hist = [l.split() for l in blog.read_text().splitlines() if l.strip()]
            print(f"   balance: {bal:.2f} {cur}")
            # Spend rate from the two most distant snapshots without a top-up between them.
            spent = eps = 0.0
            for a, b in zip(hist, hist[1:]):
                d = float(a[2]) - float(b[2])
                if d <= 0:  # balance rose: top-up; skip this interval
                    continue
                spent += d
                eps += int(b[3]) - int(a[3])
            if eps > 0:
                per_ep = spent / eps
                need = (total_expected - total_done) * per_ep
                print(f"   spend: {spent:.2f} $ over {eps:.0f} finished episodes = {per_ep:.3f} $/ep"
                      f" -> remaining {total_expected - total_done} episodes ~{need:.0f} $"
                      + (f"  ** TOP UP ~{need - bal:.0f} $ **" if need > bal else "  (covered)"))
                if need > bal:
                    notes.append(f"balance {bal:.2f} $ covers ~{bal / per_ep:.0f} of {total_expected - total_done} remaining episodes - top up ~{need - bal:.0f} $")
            if bal < 10:
                notes.append("BALANCE < 10 $: workers will start failing with 402 (InsufficientBalanceError) very soon")
        else:
            print(f"   balance: unavailable ({cur})")

    print("\n== assessment")
    if spc_window:
        w = st.median(spc_window)
        ratio = w / BASELINE_S_PER_CALL
        if ratio > 1.5:
            notes.append(f"latency inflated: {w:.0f} s/call in window vs ~{BASELINE_S_PER_CALL} compact baseline"
                         f" ({ratio:.1f}x) - do not add workers; consider scaling down")
        elif ratio > 1.2:
            notes.append(f"latency mildly up: {w:.0f} s/call ({ratio:.1f}x baseline) - hold the current worker count")
        else:
            notes.append(f"latency flat: {w:.0f} s/call ({ratio:.1f}x baseline) - headroom to add workers if balance allows")
    else:
        notes.append("no episode finished in the window yet - latency not measurable; check claims' ages instead")
    for n in notes:
        print(f" - {n}")


if __name__ == "__main__":
    main()
