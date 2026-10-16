#!/usr/bin/env python3
"""Claim the next outstanding episode of a suite for one worker process.

Worker model (see campaign/worker.sh): instead of dealing every remaining episode
into K fixed shards up front (run_chain.sh), each worker claims ONE episode at a
time, plays it with `clem run -i <single-instance file>` and comes back for the
next. Workers can therefore be added or stopped at any moment without discarding
in-flight work, which is what makes the shard count adjustable mid-campaign.

Outstanding means no
  <results>/<model>/<game>/<experiment>/instance_<game_id:05d>/interactions.json
(clemcore writes it from on_game_end only: a crashed episode leaves nothing and
is replayed, an aborted one keeps its result). Order: S1 -> S4, then file order,
so complete, analysable levels accumulate as the campaign runs.

Claims are directories under <results>/_campaign/claims/ (mkdir is atomic, so two
workers never get the same episode). A claim whose recorded pid is dead is stale
(worker killed mid-episode) and is taken over. An episode that failed
--fail-limit times without producing interactions.json is skipped; its .fails
file lists the attempts. Delete the .fails files to make those eligible again
(e.g. after topping up the DeepSeek account).

The canonical instances file is never modified; the single-instance file is
written next to it via temp + os.replace.

Prints one line:  NONE  |  CLAIM=<claim> FILE=<stem> EXP=<experiment> ID=<game_id> REMAINING=<n>
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def alive(pid):
    """True if pid is a running campaign worker shell (not merely any process:
    after a reboot the old pids are reused by unrelated processes)."""
    try:
        os.kill(int(pid), 0)
    except (ValueError, ProcessLookupError, PermissionError, TypeError):
        return False
    out = subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))], capture_output=True, text=True).stdout
    return "campaign/worker.sh" in out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--game-dir", required=True)
    ap.add_argument("--src", required=True, help="canonical instances basename, no .json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True, help="results run dir (model identifier)")
    ap.add_argument("--tag", required=True, help="run tag, part of the instance file name")
    ap.add_argument("--worker", required=True, type=int)
    ap.add_argument("--pid", required=True, type=int,
                    help="pid of the worker shell that will play the episode (liveness check)")
    ap.add_argument("--fail-limit", type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.game_dir) / "in"
    data = json.loads((in_dir / f"{args.src}.json").read_text())
    game_root = Path(args.results) / args.model / args.game
    claims = Path(args.results) / "_campaign" / "claims"
    claims.mkdir(parents=True, exist_ok=True)

    outstanding = []
    for exp_index, experiment in enumerate(data["experiments"]):
        level = next((s for s in ("S1", "S2", "S3", "S4") if s in experiment["name"]), "S9")
        for ins_index, instance in enumerate(experiment["game_instances"]):
            episode = game_root / experiment["name"] / f"instance_{instance['game_id']:05d}"
            if (episode / "interactions.json").exists():
                continue
            outstanding.append(((level, exp_index, ins_index), experiment["name"], instance))
    outstanding.sort(key=lambda row: row[0])

    meta = {e["name"]: {k: v for k, v in e.items() if k != "game_instances"}
            for e in data["experiments"]}

    for _, exp_name, instance in outstanding:
        name = f"{exp_name}__{instance['game_id']:05d}"
        claim = claims / name
        fails = claims / f"{name}.fails"
        if fails.exists() and len(fails.read_text().splitlines()) >= args.fail_limit:
            continue
        try:
            claim.mkdir()
        except FileExistsError:
            owner = (claim / "pid").read_text().strip() if (claim / "pid").exists() else ""
            if owner and alive(owner):
                continue
            # stale claim: the worker died mid-episode. Rename it away (only one
            # contender wins the rename) and try to take the slot.
            try:
                claim.rename(claims / f"{name}.stale.{int(time.time())}.{os.getpid()}")
            except OSError:
                continue
            try:
                claim.mkdir()
            except FileExistsError:
                continue
        (claim / "pid").write_text(f"{args.pid}\n")
        (claim / "worker").write_text(f"{args.worker}\n")
        (claim / "since").write_text(f"{time.strftime('%F %T')}\n")

        stem = f"instances_{args.tag}_w{args.worker}"
        payload = {**data, "experiments": [{**meta[exp_name], "game_instances": [instance]}]}
        tmp = in_dir / f".{stem}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, in_dir / f"{stem}.json")
        print(f"CLAIM={name} FILE={stem} EXP={exp_name} ID={instance['game_id']} "
              f"REMAINING={len(outstanding)}")
        return
    print(f"NONE REMAINING={len(outstanding)}")


main()
