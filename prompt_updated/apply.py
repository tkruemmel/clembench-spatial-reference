"""Swap player_prompt (and optionally the turn budget) in a copy of an instances file.

The master reads the prompt from the instance (master.py:299), never from
resources, so rewriting player_prompt is enough for a new prompt arm — the
boards stay byte-identical to the run they were copied from, which no
regeneration can guarantee.

With --budget, optimal_moves becomes the A* attainable optimum A_k and max_turns
is recomputed as max(4 * A_k, 20), replacing the generator's per-object BFS sum.
That sum understates what a k-player team can reach under round-robin and a
permanent `done` — by up to 34 moves at 6p/S4 — so it both starved the turn
budget and inflated the scorer's efficiency denominator (master.py:704).

Usage:
  python3 prompt_updated/apply.py \
      --src  matrixgame_covered_comm_grid/in/instances.json \
      --prompts prompt_updated/dyad \
      --budget analysis/efficiency_baseline/attainable_dyad.jsonl \
      --out  matrixgame_covered_comm_grid_<tag>/in/instances.json
"""

import argparse
import json
import os
import re

PLACEHOLDER = re.compile(r"\$[A-Z_]+\$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="existing instances.json to copy")
    ap.add_argument("--prompts", required=True, help="directory of player_prompt_<protocol>.template")
    ap.add_argument("--out", required=True, help="instances.json to write")
    ap.add_argument("--budget", help="attainable_<suite>.jsonl; recompute max_turns from A_k")
    args = ap.parse_args()

    with open(args.src) as fh:
        data = json.load(fh)

    attainable = {}
    if args.budget:
        for line in open(args.budget):
            row = json.loads(line)
            # A non-exact row is a solver timeout, not an optimum; using it would
            # set a budget from a bound nobody verified.
            assert row["status"] == "exact", f"{args.budget}: {row['level']}/{row['game_id']} is {row['status']}"
            attainable[(row["level"], row["game_id"])] = row["stuckfree"]

    templates = {}
    changed = 0
    rebudgeted = 0
    for experiment in data["experiments"]:
        for instance in experiment["game_instances"]:
            protocol = instance["comm_protocol"]
            if protocol not in templates:
                path = os.path.join(args.prompts, f"player_prompt_{protocol}.template")
                with open(path) as fh:
                    templates[protocol] = fh.read()
            old, new = instance["player_prompt"], templates[protocol]
            # A dropped or renamed $VAR$ would silently ship a literal to the
            # model; the master substitutes blind.
            assert set(PLACEHOLDER.findall(old)) == set(PLACEHOLDER.findall(new)), (
                f"{protocol}: placeholders differ from the original prompt: "
                f"{set(PLACEHOLDER.findall(old)) ^ set(PLACEHOLDER.findall(new))}"
            )
            if old != new:
                changed += 1
            instance["player_prompt"] = new

            if attainable:
                key = (instance["spatial_level"], instance["game_id"])
                assert key in attainable, f"{args.budget} has no row for {key}"
                optimum = attainable[key]
                budget = max(4 * optimum, 20)
                # A_k is an attainable optimum, so it can only exceed the BFS sum;
                # a shrinking value means the two were computed on different boards.
                assert optimum >= instance["optimal_moves"], (
                    f"{key}: A_k {optimum} < stored optimal_moves "
                    f"{instance['optimal_moves']} — the jsonl is from a different board"
                )
                if optimum != instance["optimal_moves"]:
                    rebudgeted += 1
                instance["optimal_moves"] = optimum
                instance["max_turns"] = budget

    # Everything except the prompt (and the budget, when asked) must survive untouched.
    with open(args.src) as fh:
        original = json.load(fh)
    mutable = {"player_prompt": "",
               **({"max_turns": 0, "optimal_moves": 0} if attainable else {})}
    strip = lambda d: [  # noqa: E731
        {**i, **mutable} for e in d["experiments"] for i in e["game_instances"]
    ]
    assert strip(original) == strip(data), "instances changed beyond the prompt and optimum"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        # No trailing newline: matches how the generator writes these files, so
        # an unedited template round-trips byte-identical.
        json.dump(data, fh, indent=2, ensure_ascii=False)

    total = sum(len(e["game_instances"]) for e in data["experiments"])
    budget_note = f", {rebudgeted}/{total} a larger optimum and budget" if attainable else ""
    print(f"{args.out}: {changed}/{total} instances got a new prompt{budget_note} "
          f"({len(data['experiments'])} experiments, protocols: {', '.join(sorted(templates))})")


if __name__ == "__main__":
    main()
