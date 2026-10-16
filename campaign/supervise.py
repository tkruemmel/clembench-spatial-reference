#!/usr/bin/env python3
"""Unattended supervisor for the Qwen3.8 grid campaign.

Replaces the hourly human/LLM check with a loop that does the same four jobs and
nothing else:

  1. keeps the worker count up      - restarts workers that died (a reboot kills
                                      them all; scale_workers.sh is idempotent)
  2. rebalances the per-suite split - runs campaign/rebalance.py and applies its
                                      proposal under the anti-churn rule below
  3. unblocks skipped episodes      - resets stale claims/*.fails files, which
                                      otherwise make claim_next.py skip an
                                      episode permanently after 5 attempts
  4. finishes the campaign          - scores and summarises each suite once all
                                      840 episodes exist, then exits

Everything it observes goes to campaign/supervisor.log, one line per tick plus a
line per action, so `tail -f` is the whole status view.

    bash campaign/supervise.sh            # start detached under caffeinate
    tail -f campaign/supervisor.log
    python3 campaign/supervise.py --once --dry-run    # inspect one tick

Anti-churn rule (learned from watching the proposals by hand): once the suites
are balanced the proposal oscillates by one worker as the long pole flips
between them, so a +-1 proposal is only applied when the same one appears twice
in a row, while a shift of 2+ workers is a structural signal and is applied at
once. Applying is lossless either way: surplus workers exit after their current
episode, so nothing in flight is ever discarded.
"""
import argparse
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# ARM=_final supervises the final arm: its own trees, log, state and lock.
ARM = os.environ.get("ARM", "")
LOG = REPO / "campaign" / f"supervisor{ARM}.log"
STATE = REPO / "campaign" / f".supervisor{ARM}_state.json"
LOCK = REPO / "campaign" / f".supervisor{ARM}.lock"
PREFIX = f"results_qwen38_grid{ARM}_n15"
MODEL = "Qwen3.8-27B-FP8"
METRICS = "https://jarvis.ling.uni-potsdam.de/clp-chat-2/api/metrics"

# short, game, results suffix, expected episodes, env var used by qwen38_grid.sh
SUITES = [
    ("solo", "matrixgame_covered_solo_grid" + ARM, "solo", 60 if ARM == "_final" else 120, "SOLO"),  # final: silent only
    ("dyad", "matrixgame_covered_comm_grid" + ARM, "dyad", 240, "DYAD"),
    ("3p", "matrixgame_covered_comm_3p_grid" + ARM, "3p", 240, "P3"),
    ("6p", "matrixgame_covered_comm_6p_grid" + ARM, "6p", 240, "P6"),
]
# Parked pre-fix runs of the original arm; no same-game reference for other arms.
REF_ROOTS = [] if ARM else [
    "3p=results_qwen38_grid_n15_3p_stale_20260906_nplayerfix",
    "6p=results_qwen38_grid_n15_6p_stale_20260906_nplayerfix",
]
TOTAL_STREAMS = int(os.environ.get("STREAMS", 28))
# A .fails file blocks an episode after 5 attempts. Failures cluster in outages,
# so give a blocked episode another chance once its newest attempt is this old.
FAILS_RESET_AFTER_H = 3


def log(msg):
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def load_state():
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {}


def save_state(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def results_dir(suffix):
    return REPO / f"{PREFIX}_{suffix}"


def done_counts():
    return {short: len(list(results_dir(sfx).glob("*/*/*/instance_*/interactions.json")))
            for short, _, sfx, _, _ in SUITES}


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def live_workers(suffix):
    n = 0
    for pidfile in (results_dir(suffix) / "_campaign").glob("worker_w*.pid"):
        try:
            if alive(pidfile.read_text().strip()):
                n += 1
        except Exception:
            pass
    return n


def pending_stops(suffix):
    """Stop files for workers that are still draining.

    A stop file outlives the worker it retired (worker.sh removes its pidfile on
    exit but not the stop file), and a stale one must not be mistaken for a drain
    in progress: that would suppress the restart branch below forever. So prune
    the ones whose worker is gone and count only the live ones.
    """
    campaign = results_dir(suffix) / "_campaign"
    live = 0
    for stop in campaign.glob("stop_w*"):
        wid = stop.name[len("stop_w"):]
        pidfile = campaign / f"worker_w{wid}.pid"
        try:
            running = pidfile.exists() and alive(pidfile.read_text().strip())
        except Exception:
            running = False
        if running:
            live += 1
        else:
            stop.unlink(missing_ok=True)
    return live


def vpn_up():
    try:
        out = subprocess.run(["ifconfig"], capture_output=True, text=True, timeout=30).stdout
        return "inet 141.89." in out
    except Exception:
        return False


def api_key():
    try:
        return json.loads((REPO / "key.json").read_text())["openai_compatible"]["api_key"]
    except Exception:
        return None


def metrics():
    """(generation_tokens_total, running, waiting, kv_frac) or None."""
    key = api_key()
    if not key:
        return None
    bundle = REPO / ".certs" / "clem-ca-bundle.pem"
    for ctx in (ssl.create_default_context(cafile=str(bundle)) if bundle.exists() else None,
                ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(METRICS, headers={"Authorization": f"Bearer {key}"})
            body = urllib.request.urlopen(req, timeout=30, context=ctx).read().decode()
        except Exception:
            continue
        got = {}
        for line in body.splitlines():
            for name in ("generation_tokens_total", "num_requests_running",
                         "num_requests_waiting", "kv_cache_usage_perc"):
                if line.startswith(f"vllm:{name}") and name not in got:
                    try:
                        got[name] = float(line.rsplit(None, 1)[1])
                    except ValueError:
                        pass
        if len(got) == 4:
            return (got["generation_tokens_total"], got["num_requests_running"],
                    got["num_requests_waiting"], got["kv_cache_usage_perc"])
    return None


def reset_stale_fails():
    """Free episodes that claim_next.py would skip, once the cause looks past."""
    freed = []
    cutoff = datetime.now() - timedelta(hours=FAILS_RESET_AFTER_H)
    for _, _, sfx, _, _ in SUITES:
        for f in (results_dir(sfx) / "_campaign" / "claims").glob("*.fails"):
            try:
                attempts = len(f.read_text().splitlines())
                newest = datetime.fromtimestamp(f.stat().st_mtime)
            except OSError:
                continue
            if attempts >= 5 and newest < cutoff:
                f.unlink(missing_ok=True)
                freed.append(f.name.replace(".fails", ""))
    return freed


def clear_stale_claims():
    """Drop claim dirs whose owner died, so the episode can be claimed again.

    claim_next.py already takes these over, but only while some worker is still
    looking; clearing them keeps the claims dir honest for the status tools.
    """
    cleared = 0
    for _, _, sfx, _, _ in SUITES:
        for c in (results_dir(sfx) / "_campaign" / "claims").glob("*"):
            if not c.is_dir() or ".stale." in c.name:
                continue
            pid = (c / "pid").read_text().strip() if (c / "pid").exists() else ""
            if pid and not alive(pid):
                shutil.rmtree(c, ignore_errors=True)
                cleared += 1
    return cleared


def proposal():
    """Run rebalance.py and return {'SOLO': n, ...} or None."""
    cmd = [sys.executable, str(REPO / "campaign" / "rebalance.py"), "--total", str(TOTAL_STREAMS), "--prefix", PREFIX]
    for r in REF_ROOTS:
        cmd += ["--ref-root", r]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=REPO).stdout
    except Exception as exc:
        log(f"rebalance.py failed: {exc}")
        return None
    m = re.search(r"SOLO=(\d+) DYAD=(\d+) P3=(\d+) P6=(\d+)", out)
    if not m:
        return None
    return dict(zip(("SOLO", "DYAD", "P3", "P6"), (int(g) for g in m.groups())))


def apply_targets(targets, why):
    env = {**os.environ, **{k: str(v) for k, v in targets.items()}}
    try:
        out = subprocess.run(["bash", str(REPO / "campaign" / "qwen38_grid.sh")],
                             capture_output=True, text=True, timeout=600, cwd=REPO, env=env)
    except Exception as exc:
        log(f"apply {targets} failed: {exc}")
        return False
    log(f"APPLIED {targets} ({why})")
    for line in out.stdout.splitlines():
        if "started worker" in line or "will stop" in line:
            log("  " + line.strip())
    return True


def finish():
    """Score and summarise every suite, then report."""
    log("ALL EPISODES PRESENT - scoring")
    for short, game, sfx, expected, _ in SUITES:
        res = results_dir(sfx)
        subprocess.run(["clem", "score", "-g", game, "-r", str(res)],
                       capture_output=True, text=True, cwd=REPO, timeout=7200)
        played = len(list(res.glob("*/*/*/instance_*/interactions.json")))
        scored = len(list(res.glob("*/*/*/instance_*/scores.json")))
        log(f"  {short}: scored {scored} of {played}")
        summary = subprocess.run(
            [sys.executable, str(REPO / "campaign" / "summarize.py"), "--results", str(res),
             "--model", MODEL, "--game", game, "--label", short, "--expected", str(expected)],
            capture_output=True, text=True, cwd=REPO, timeout=3600).stdout
        (res / "_campaign" / f"summary_{short}.txt").write_text(summary)
        for line in summary.splitlines():
            log("  " + line)
    log("CAMPAIGN COMPLETE")


def tick(state, dry_run):
    done = done_counts()
    total = sum(done.values())
    expected_total = sum(e for _, _, _, e, _ in SUITES)
    live = {short: live_workers(sfx) for short, _, sfx, _, _ in SUITES}
    stops = {short: pending_stops(sfx) for short, _, sfx, _, _ in SUITES}
    targets = state.get("targets") or {env: live[short] for short, _, _, _, env in SUITES}

    m = metrics()
    if m:
        gen, running, waiting, kv = m
        prev = state.get("gen_tokens")
        prev_t = state.get("gen_time")
        rate = ""
        if prev and prev_t and gen > prev:
            rate = f", {(gen - prev) / (time.time() - prev_t):.0f} tok/s"
        state["gen_tokens"], state["gen_time"] = gen, time.time()
        server = f"running {running:.0f}, waiting {waiting:.0f}, KV {kv * 100:.0f}%{rate}"
    else:
        server = "metrics unavailable"

    log(f"{total}/{expected_total} done "
        f"({' '.join(f'{k} {v}' for k, v in done.items())}) | "
        f"workers {' '.join(f'{k} {live[k]}' for k in live)}"
        f"{' | stops pending ' + str(sum(stops.values())) if sum(stops.values()) else ''} | "
        f"{server}{'' if vpn_up() else ' | VPN DOWN'}")

    if total >= expected_total:
        if not dry_run:
            finish()
        return True

    if not vpn_up():
        log("VPN address missing - skipping actions until it returns")
        return False

    freed = reset_stale_fails()
    if freed:
        log(f"reset {len(freed)} blocked episode(s): {', '.join(freed[:6])}")
    stale = clear_stale_claims()
    if stale:
        log(f"cleared {stale} claim(s) whose worker had died")

    # 1. restart anything that died (a reboot kills every worker at once)
    missing = {env: targets.get(env, 0) for short, _, _, _, env in SUITES
               if live[short] < targets.get(env, 0) and stops[short] == 0}
    if missing:
        log(f"workers below target: {', '.join(f'{k}<{v}' for k, v in missing.items())}")
        if not dry_run:
            apply_targets(targets, "restarting dead workers")
        return False

    # 2. rebalance, under the anti-churn rule
    prop = proposal()
    if not prop:
        return False
    if prop == targets:
        state["last_proposal"] = prop
        return False
    shift = max(abs(prop[k] - targets.get(k, 0)) for k in prop)
    repeated = state.get("last_proposal") == prop
    state["last_proposal"] = prop
    if shift >= 2 or repeated:
        why = f"shift of {shift}" if shift >= 2 else "proposed twice in a row"
        if dry_run:
            log(f"WOULD APPLY {prop} ({why}); current {targets}")
        elif apply_targets(prop, why):
            state["targets"] = prop
    else:
        log(f"holding {targets}; proposal {prop} is a single-worker change seen once")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="seconds between ticks")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report actions without taking them")
    args = ap.parse_args()

    if not args.dry_run and not args.once:
        try:
            LOCK.mkdir()
        except FileExistsError:
            owner = (LOCK / "pid").read_text().strip() if (LOCK / "pid").exists() else ""
            if owner and alive(owner):
                print(f"supervisor already running (pid {owner})", file=sys.stderr)
                return 3
            shutil.rmtree(LOCK, ignore_errors=True)
            LOCK.mkdir()
        (LOCK / "pid").write_text(f"{os.getpid()}\n")
        log(f"SUPERVISOR START pid {os.getpid()}, interval {args.interval}s, target {TOTAL_STREAMS} streams")

    try:
        while True:
            state = load_state()
            try:
                complete = tick(state, args.dry_run)
            except Exception as exc:
                log(f"tick failed: {type(exc).__name__}: {exc}")
                complete = False
            if not args.dry_run:
                save_state(state)
            if complete or args.once:
                return 0
            time.sleep(args.interval)
    finally:
        if not args.dry_run and not args.once:
            shutil.rmtree(LOCK, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
