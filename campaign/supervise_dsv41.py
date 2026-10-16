#!/usr/bin/env python3
"""Laptop-side supervisor for the DeepSeek-V4.1-Flash grid campaign on Hetzner.

Same four jobs as campaign/supervise.py (the Qwen3.8 supervisor), except the
workers run on the cloud runner and this loop drives them over ssh:

  1. keeps the worker count up      - restarts workers that died; a server reboot
                                      kills all of them and scale_workers.sh is
                                      idempotent, so re-applying the targets is
                                      the whole recovery
  2. redistributes the streams      - runs campaign/rebalance.py ON THE SERVER
                                      (the results trees live there) and applies
                                      its proposal under the anti-churn rule
  3. unblocks skipped episodes      - resets stale claims/*.fails files, which
                                      otherwise make claim_next.py skip an
                                      episode permanently after 5 attempts, and
                                      drops claim dirs whose worker is gone
  4. finishes the campaign          - scores each suite on the server once all
                                      840 episodes exist, fetches the trees and
                                      summarises them locally, then exits

Everything it observes goes to campaign/supervisor_dsv41.log, one line per tick
plus a line per action, so `tail -f` is the whole status view on the laptop.

    bash campaign/supervise_dsv41.sh                      # start detached
    tail -f campaign/supervisor_dsv41.log
    python3 campaign/supervise_dsv41.py --once --dry-run  # inspect one tick

Design notes worth keeping in mind when editing:

* One ssh round trip per job per tick. The probe is campaign/ds_status.py --json,
  so the supervisor never parses the human report and the server does the file
  walking (the trees are tens of thousands of files - rsyncing them every tick
  to decide anything would be absurd).
* Every remote call is allowed to fail. A dropped laptop network, a closed lid
  or a rebooting server produces a logged line and a skipped tick, never a
  crash and never a half-applied rebalance: the campaign keeps running on the
  server whether or not this process is alive.
* The balance floor is enforced by the workers themselves (worker.sh MIN_BALANCE
  waits instead of claiming), not here. This loop only reports the balance, the
  measured $/episode and whether the remaining episodes are affordable, so a
  top-up resumes the campaign with nothing re-run.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# ARM=_final supervises the final arm: its own trees, log, state and lock.
# MODEL/TAG select another registry entry on the same API, e.g.
# MODEL=DeepSeek-V4.1-Flash-NoThink TAG=dsv41nt: again its own trees, log, state and lock.
ARM = os.environ.get("ARM", "")
MODEL = os.environ.get("MODEL", "DeepSeek-V4.1-Flash")
TAG = os.environ.get("TAG", "dsv41")
LOG = REPO / "campaign" / f"supervisor_{TAG}{ARM}.log"
STATE = REPO / "campaign" / f".supervisor_{TAG}{ARM}_state.json"
LOCK = REPO / "campaign" / f".supervisor_{TAG}{ARM}.lock"

PREFIX = f"results_{TAG}_grid{ARM}_n15"
CAMPAIGN_SCRIPT = "campaign/dsv41_grid.sh"
MIN_BALANCE = 12          # USD; passed to the workers, which park below it. Lower it
                          # (--min-balance) to spend the tail of an account on the last
                          # few episodes: the floor guards against dying mid-episode, so
                          # once a handful are left it costs more than it protects.
TOTAL_STREAMS = 24        # what the box ran comfortably during the V4-Flash campaign
WINDOW_H = 2.0            # throughput/latency window reported each tick
FAILS_RESET_AFTER_H = 3   # a blocked episode gets another chance once this old
FETCH_EVERY = 6           # ticks between rsync mirrors of the results trees

# short, env var used by dsv41_grid.sh, game, expected episodes
_ONLY = os.environ.get("SUITES", "").split()   # e.g. "3p 6p", as in ds_status.py
SUITES = [s for s in [
    ("solo", "SOLO", "matrixgame_covered_solo_grid" + ARM, 60 if ARM == "_final" else 120),  # final: silent only
    ("dyad", "DYAD", "matrixgame_covered_comm_grid" + ARM, 240),
    ("3p", "P3", "matrixgame_covered_comm_3p_grid" + ARM, 240),
    ("6p", "P6", "matrixgame_covered_comm_6p_grid" + ARM, 240),
] if not _ONLY or s[0] in _ONLY]
# Reference for rebalance.py's per-level minutes: the FINISHED compact DeepSeek
# campaign, same API and same prompts, whose solo arm is the Markovian rerun.
REF_ARGS = ["--ref-prefix", "results_dsflash_n15", "--ref-model", "DeepSeek-V4-Flash",
            "--ref-dir", "solo=results_dsflash_n15_solo_markov"]


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


def cloud_config():
    cfg = {}
    for line in (REPO / "cloud" / "config.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k] = v.strip().strip('"').strip("'")
    return cfg


CFG = cloud_config()
SSH = ["ssh", "-i", CFG["SSH_KEY"], "-o", "StrictHostKeyChecking=no",
       "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
       "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
       f"{CFG['REMOTE_USER']}@{CFG['SERVER_IP']}"]
REMOTE_DIR = CFG["REMOTE_DIR"]


def remote(cmd, timeout=300):
    """Run cmd in the repo on the campaign host. Returns (ok, stdout, detail)."""
    try:
        p = subprocess.run(SSH + [f"cd {shlex.quote(REMOTE_DIR)} && export ARM={ARM} SUITES={shlex.quote(' '.join(_ONLY))} "
                                  f"MODEL={shlex.quote(MODEL)} TAG={TAG} RESULTS_PREFIX={PREFIX} && {cmd}"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "", f"ssh timed out after {timeout}s"
    except Exception as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    if p.returncode != 0:
        return False, p.stdout, (p.stderr or p.stdout).strip().splitlines()[-1:][0] if (p.stderr or p.stdout).strip() else f"rc={p.returncode}"
    return True, p.stdout, ""


def probe():
    """Snapshot of the whole campaign in one round trip, or None."""
    ok, out, why = remote(
        f"venv/bin/python3 campaign/ds_status.py --json --prefix {PREFIX} "
        f"--model {shlex.quote(MODEL)} --window-hours {WINDOW_H}", timeout=600)
    if not ok:
        log(f"probe failed: {why}")
        return None
    for line in reversed(out.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    log("probe returned no JSON")
    return None


def proposal(total):
    """rebalance.py's split, run where the trees are. {'SOLO': n, ...} or None."""
    cmd = (f"venv/bin/python3 campaign/rebalance.py --prefix {PREFIX} "
           f"--model {shlex.quote(MODEL)} --total {total} " + " ".join(shlex.quote(a) for a in REF_ARGS))
    ok, out, why = remote(cmd, timeout=900)
    if not ok:
        log(f"rebalance failed: {why}")
        return None
    m = re.search(r"SOLO=(\d+) DYAD=(\d+) P3=(\d+) P6=(\d+)", out)
    if not m:
        log("rebalance produced no proposal line")
        return None
    prop = dict(zip(("SOLO", "DYAD", "P3", "P6"), (int(g) for g in m.groups())))
    return {env: prop[env] for _, env, _, _ in SUITES}


def apply_targets(targets, why, floor=MIN_BALANCE):
    env = " ".join(f"{k}={v}" for k, v in targets.items())
    ok, out, detail = remote(
        f"source venv/bin/activate && {env} MIN_BALANCE={floor} bash {CAMPAIGN_SCRIPT}",
        timeout=900)
    if not ok:
        log(f"apply {targets} failed: {detail}")
        return False
    log(f"APPLIED {targets} ({why})")
    for line in out.splitlines():
        if "started worker" in line or "will stop" in line:
            log("  " + line.strip())
    return True


def maintain():
    """Reset blocked episodes and drop claims whose worker died, on the server.

    Both are the same rules supervise.py applies locally: a .fails file blocks an
    episode after 5 attempts, and failures cluster in outages, so a blocked
    episode is freed once its newest attempt is FAILS_RESET_AFTER_H old.
    """
    trees = " ".join(f"{PREFIX}_{short}" for short, _, _, _ in SUITES)
    script = f'''
freed=0; cleared=0; now=$(date +%s)
for t in {trees}; do
  for f in "$t"/_campaign/claims/*.fails; do
    [ -e "$f" ] || continue
    n=$(wc -l < "$f"); age=$(( (now - $(stat -c %Y "$f")) / 3600 ))
    if [ "$n" -ge 5 ] && [ "$age" -ge {FAILS_RESET_AFTER_H} ]; then rm -f "$f"; freed=$((freed+1)); fi
  done
  for c in "$t"/_campaign/claims/*/; do
    [ -d "$c" ] || continue
    pid=$(cat "$c/pid" 2>/dev/null || echo 0)
    ps -o command= -p "${{pid:-0}}" 2>/dev/null | grep -q "campaign/worker.sh" || {{ rm -rf "$c"; cleared=$((cleared+1)); }}
  done
done
echo "MAINT freed=$freed cleared=$cleared"
'''
    ok, out, why = remote(script, timeout=300)
    if not ok:
        log(f"maintenance failed: {why}")
        return
    m = re.search(r"MAINT freed=(\d+) cleared=(\d+)", out)
    if m and (int(m.group(1)) or int(m.group(2))):
        log(f"unblocked {m.group(1)} episode(s), cleared {m.group(2)} dead claim(s)")


def fetch():
    """Mirror the server's results trees onto the laptop (hetzner_dsflash.sh fetch)."""
    try:
        p = subprocess.run(["bash", str(REPO / "campaign" / "hetzner_dsflash.sh"), "fetch"],
                           capture_output=True, text=True, timeout=3600, cwd=REPO,
                           env={**os.environ, "RESULTS_PREFIX": PREFIX})
    except Exception as exc:
        log(f"fetch failed: {type(exc).__name__}: {exc}")
        return
    if p.returncode != 0:
        log(f"fetch failed: {(p.stderr or p.stdout).strip().splitlines()[-1:]}")
        return
    log("mirrored results trees to the laptop: " + "; ".join(
        line.strip() for line in p.stdout.splitlines() if "episodes" in line))


def economics(state, probe_data):
    """($/episode, $/hour, affordable?) from the balance snapshots seen so far."""
    bal = (probe_data.get("balance") or {}).get("usd")
    done = probe_data["total"]["done"]
    hist = state.setdefault("balance_history", [])
    hist.append([time.time(), bal, done])
    del hist[:-60]
    first = next((h for h in hist if h[1] is not None), None)
    if bal is None or not first or first is hist[-1]:
        return None, None, None
    d_bal, d_done, d_t = first[1] - bal, done - first[2], (hist[-1][0] - first[0]) / 3600
    per_ep = d_bal / d_done if d_done > 0 and d_bal > 0 else None
    per_h = d_bal / d_t if d_t > 0.25 and d_bal > 0 else None
    left = probe_data["total"]["expected"] - done
    return per_ep, per_h, (bal >= per_ep * left if per_ep else None)


def status_line(p, state):
    t = p["total"]
    suites = p["suites"]
    done = " ".join(f"{k} {suites[k]['done']}" for k in suites)
    workers = " ".join(f"{k} {len(suites[k]['live'])}" for k in suites)
    stops = sum(len(s["stops"]) for s in suites.values())
    bal = (p.get("balance") or {}).get("usd")
    per_ep, per_h, affordable = economics(state, p)

    money = f"bal {bal:.2f} $" if bal is not None else "bal unavailable"
    if per_ep:
        left_cost = per_ep * (t["expected"] - t["done"])
        money += f" ({per_ep:.2f} $/ep"
        money += f", {per_h:.1f} $/h" if per_h else ""
        # %.0f turned the last two episodes' $0.36 into "~0 $ to go", which reads
        # like the campaign is finished rather than one top-up short.
        money += f", ~{left_cost:.2f} $ to go)" if left_cost < 10 else f", ~{left_cost:.0f} $ to go)"
        if affordable is False:
            money += " SHORT"
    eta = ""
    if t["workers"] and p.get("remaining_hours"):
        eta = f" | ETA ~{p['remaining_hours'] / t['workers']:.0f} h"
    load = p.get("load") or [None]
    return (f"{t['done']}/{t['expected']} done ({done}) | "
            f"workers {t['workers']} ({workers})"
            f"{f' | {stops} draining' if stops else ''} | "
            f"{p.get('throughput_ep_h', 0):.1f} ep/h{eta} | {money}"
            f"{f' | load {load[0]:.1f}' if load[0] is not None else ''}")


def parked(p, floor):
    """True when the workers are alive but refusing to claim for lack of credit.

    worker.sh checks the balance before every claim and waits instead of playing
    while it is under the floor. That is the designed behaviour, but without this
    check the tick line reports a parked campaign exactly like a hung one - 24
    live workers, 0.0 ep/h - which is how 11 hours passed unnoticed on
    2026-09-12 with two episodes left and 25x their cost sitting in the account.
    """
    bal = (p.get("balance") or {}).get("usd")
    return bal is not None and bal < floor and p["total"]["workers"] > 0


def anomalies(p, floor=MIN_BALANCE):
    out = []
    bal = (p.get("balance") or {}).get("usd")
    if parked(p, floor):
        out.append(f"PARKED: balance {bal:.2f} $ is below the {floor} $ floor - all "
                   f"{p['total']['workers']} worker(s) are waiting, not playing. Top up the "
                   f"account (they resume within 300 s) or restart them with a lower floor.")
    for short, s in p["suites"].items():
        if s["gave_up"]:
            out.append(f"{short}: {s['gave_up']} worker(s) gave up")
        if s["blocked"]:
            out.append(f"{short}: {len(s['blocked'])} episode(s) blocked by .fails")
        if s["dead_claims"]:
            out.append(f"{short}: {s['dead_claims']} claim(s) with a dead owner")
        for line in s["recent_failures"]:
            out.append(f"{short}: {line}")
        stuck = [c for c in s["claims"] if (c["age_min"] or 0) > 90]
        if stuck:
            out.append(f"{short}: {len(stuck)} episode(s) running over 90 min "
                       f"({', '.join(c['episode'] for c in stuck[:3])})")
    return out


def finish():
    """Score on the server, mirror, summarise locally."""
    log("ALL EPISODES PRESENT - scoring on the server")
    for short, _, game, expected in SUITES:
        ok, _, why = remote(f"source venv/bin/activate && clem score -g {game} -r {PREFIX}_{short}",
                            timeout=7200)
        log(f"  {short}: scored" if ok else f"  {short}: scoring failed: {why}")
    fetch()
    for short, _, game, expected in SUITES:
        res = REPO / f"{PREFIX}_{short}"
        try:
            summary = subprocess.run(
                [sys.executable, str(REPO / "campaign" / "summarize.py"), "--results", str(res),
                 "--model", MODEL, "--game", game, "--label", short, "--expected", str(expected)],
                capture_output=True, text=True, cwd=REPO, timeout=3600).stdout
        except Exception as exc:
            log(f"  {short}: summarize failed: {type(exc).__name__}: {exc}")
            continue
        (res / "_campaign").mkdir(parents=True, exist_ok=True)
        (res / "_campaign" / f"summary_{short}.txt").write_text(summary)
        for line in summary.splitlines():
            log("  " + line)
    log("CAMPAIGN COMPLETE")


def tick(state, args):
    p = probe()
    if p is None:
        return False   # server unreachable: say so, change nothing, try again next tick

    log(status_line(p, state))
    for note in anomalies(p, args.min_balance):
        log("  ! " + note)

    if p["total"]["done"] >= p["total"]["expected"]:
        if not args.dry_run:
            finish()
        return True

    state["ticks"] = state.get("ticks", 0) + 1
    if state["ticks"] % args.fetch_every == 0 and not args.dry_run:
        fetch()

    if not args.dry_run:
        maintain()

    live = {env: len(p["suites"][short]["live"]) for short, env, _, _ in SUITES}
    draining = {env: bool(p["suites"][short]["stops"]) for short, env, _, _ in SUITES}
    targets = state.get("targets")
    if not targets:
        # First tick after a launch: adopt what is actually running as the target,
        # so a campaign started by hand is supervised without being re-scaled.
        targets = live
        state["targets"] = targets
        log(f"adopting running split as target: {targets}")

    # 1. restart anything that died (a server reboot kills every worker at once)
    missing = {env: n for env, n in targets.items() if live[env] < n and not draining[env]}
    if missing:
        log(f"workers below target: {', '.join(f'{k} {live[k]}<{v}' for k, v in missing.items())}")
        if not args.dry_run:
            apply_targets(targets, "restarting dead workers", args.min_balance)
        else:
            log(f"WOULD APPLY {targets} (restarting dead workers)")
        return False

    # 2. redistribute, under the anti-churn rule: once the suites are balanced the
    # proposal oscillates by one worker as the long pole flips between them, so a
    # +-1 proposal is applied only when it shows up twice running, while a shift
    # of 2+ is structural and applied at once. Either way nothing in flight is
    # lost: surplus workers exit after their current episode.
    prop = proposal(args.total)
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
        if args.dry_run:
            log(f"WOULD APPLY {prop} ({why}); current {targets}")
        elif apply_targets(prop, why, args.min_balance):
            state["targets"] = prop
    else:
        log(f"holding {targets}; proposal {prop} is a single-worker change seen once")
    return False


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="seconds between ticks")
    ap.add_argument("--total", type=int, default=TOTAL_STREAMS, help="streams to spread over the suites")
    ap.add_argument("--fetch-every", type=int, default=FETCH_EVERY, help="ticks between result mirrors")
    ap.add_argument("--min-balance", type=float, default=MIN_BALANCE,
                    help="USD floor handed to the workers; below it they wait instead of claiming")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report actions without taking them")
    args = ap.parse_args()

    detached = not args.dry_run and not args.once
    if detached:
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
        log(f"SUPERVISOR START pid {os.getpid()}, {CFG['SERVER_IP']}:{REMOTE_DIR}, "
            f"interval {args.interval}s, {args.total} streams, floor {args.min_balance} $, model {MODEL}")

    try:
        while True:
            state = load_state()
            try:
                complete = tick(state, args)
            except Exception as exc:
                log(f"tick failed: {type(exc).__name__}: {exc}")
                complete = False
            if not args.dry_run:
                save_state(state)
            if complete or args.once:
                return 0
            time.sleep(args.interval)
    finally:
        if detached:
            shutil.rmtree(LOCK, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
