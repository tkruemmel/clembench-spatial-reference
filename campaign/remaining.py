#!/usr/bin/env python3
"""Write filtered instances files holding only the episodes a suite still owes.

Resume signal: an episode counts as finished iff
  <results>/<model>/<game>/<experiment>/instance_<game_id:05d>/interactions.json
exists. clemcore writes that file from on_game_end only, so an episode killed
mid-flight (network drop, laptop sleep, SIGKILL) leaves nothing behind and is
replayed, while an episode the game legitimately aborted keeps its result.

Outstanding episodes are ordered by spatial level (S1 -> S2 -> S3 -> S4) before
being dealt round-robin into K disjoint files, so K `clem run` processes can work
the same suite concurrently without ever touching the same episode directory.
Level order matters because only S1 episodes exit early on success; S2+ run their
full 4x-optimal turn budget. Playing levels in order means a suite yields
complete, analysable levels as it goes instead of partial coverage everywhere,
and an early stop loses whole levels rather than every level at once.

The canonical instances.json is never modified; filtered copies get their own
run-specific names and are passed to `clem run -i`.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--game-dir", required=True)
    ap.add_argument("--src", required=True, help="canonical instances basename, no .json")
    ap.add_argument("--out", required=True, help="filtered instances basename prefix, no .json")
    ap.add_argument("--results", required=True)
    ap.add_argument("--model", required=True, help="results run dir (model identifier)")
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    in_dir = Path(args.game_dir) / "in"
    data = json.loads((in_dir / f"{args.src}.json").read_text())
    game_root = Path(args.results) / args.model / args.game

    # Collect what is still owed, ordered by spatial level then by file order.
    outstanding, total = [], 0
    for exp_index, experiment in enumerate(data["experiments"]):
        level = next((s for s in ("S1", "S2", "S3", "S4") if s in experiment["name"]), "S9")
        for ins_index, instance in enumerate(experiment["game_instances"]):
            total += 1
            episode = game_root / experiment["name"] / f"instance_{instance['game_id']:05d}"
            if (episode / "interactions.json").exists():
                continue
            outstanding.append(((level, exp_index, ins_index), experiment["name"], instance))
    outstanding.sort(key=lambda row: row[0])

    # shard -> experiment name -> [instances]
    shards = [{} for _ in range(args.nshards)]
    for order, (_, exp_name, instance) in enumerate(outstanding):
        shards[order % args.nshards].setdefault(exp_name, []).append(instance)
    order = len(outstanding)

    meta = {e["name"]: {k: v for k, v in e.items() if k != "game_instances"}
            for e in data["experiments"]}
    written = []
    for i, shard in enumerate(shards):
        path = in_dir / (f"{args.out}_s{i}.json" if args.nshards > 1 else f"{args.out}.json")
        if shard:
            experiments = [{**meta[name], "game_instances": instances}
                           for name, instances in shard.items()]
            path.write_text(json.dumps({**data, "experiments": experiments}, indent=2))
            written.append(path.stem)
        elif path.exists():
            path.unlink()

    print(f"REMAINING={order} DONE={total - order} TOTAL={total} "
          f"SHARD_FILES={','.join(written)}")


main()
