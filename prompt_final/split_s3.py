"""Put the S3 entrance holder and the corridor object on different dyad players.

At S3, B's target is the corridor entrance and D's target is the dead end. The
played dyad split ([D,B,E] / [C,A,F]) gives both to Player A, who can declare
`done` only once D is home, so in the dyad a `done` can never seal D out of the
corridor; at 3p and 6p B and D always belong to different players and it can
(thesis/design_review_2026-10-12.md, K2). Swapping A and B gives
[D,A,E] / [C,B,F], so the trap exists in every team size.

With two players, the chain A<-B<-C<-D cannot alternate owners and still split
B from D, so one link must stay with one player. This split keeps that to the
single link B<-C; A<-B, C<-D and E<-F still cross players.

The partition changes the dyad's attainable optimum, so the S3 budgets come from
stuckfree.py run on the swapped file. Run after apply.py (see README); a second
run without --budget is a no-op.

  python3 prompt_final/split_s3.py matrixgame_covered_comm_grid_final/in/instances.json
  python3 analysis/efficiency_baseline/stuckfree.py --suite dyad --c --all \
      --instances matrixgame_covered_comm_grid_final/in/instances.json \
      --save analysis/efficiency_baseline/attainable_dyad_final.jsonl
  python3 prompt_final/split_s3.py matrixgame_covered_comm_grid_final/in/instances.json \
      --budget analysis/efficiency_baseline/attainable_dyad_final.jsonl
"""

import argparse
import copy
import json

PARTITION = ("player_a_objects", "player_b_objects")


def owner(inst, obj):
    return 0 if obj in inst["player_a_objects"] else 1


def check_s3(inst):
    """The S3 roles this split relies on, read from the board, not assumed."""
    s, t, dev = inst["start_positions"], inst["target_positions"], inst["device"]
    assert t["B"] == dev["entrance"] and t["D"] == dev["dead_end"], inst["game_id"]
    for head, tail in ("AB", "BC", "CD", "EF"):  # tail starts on head's target
        assert s[tail] == t[head], (inst["game_id"], head, tail)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", help="final dyad instances.json, rewritten in place")
    ap.add_argument("--budget", help="attainable_dyad_final.jsonl from stuckfree.py on this file")
    a = ap.parse_args()

    data = json.load(open(a.instances))
    original = copy.deepcopy(data)
    optima = {}
    if a.budget:
        for line in open(a.budget):
            row = json.loads(line)
            assert row["status"] == "exact", (row["level"], row["game_id"], row["status"])
            optima[(row["level"], row["game_id"])] = row["stuckfree"]

    swapped = rebudgeted = 0
    for exp in data["experiments"]:
        for inst in exp["game_instances"]:
            key = (inst["spatial_level"], inst["game_id"])
            if inst["spatial_level"] != "S3":
                # Same board and partition as the source: the optimum must agree.
                assert not optima or optima[key] == inst["optimal_moves"], key
                continue
            check_s3(inst)
            if owner(inst, "B") == owner(inst, "D"):
                pa, pb = inst["player_a_objects"], inst["player_b_objects"]
                assert "B" in pa and "A" in pb, key
                pa[pa.index("B")], pb[pb.index("A")] = "A", "B"
                swapped += 1
            crossing = [owner(inst, h) != owner(inst, t) for h, t in ("AB", "BC", "CD", "EF")]
            assert owner(inst, "B") != owner(inst, "D") and crossing == [True, False, True, True], key
            if optima:
                inst["optimal_moves"] = optima[key]
                inst["max_turns"] = max(4 * optima[key], 20)
                rebudgeted += 1

    # Only the S3 partition and, with --budget, the S3 budget may change.
    strip = lambda d: [  # noqa: E731
        {**i, **({k: 0 for k in (*PARTITION, "optimal_moves", "max_turns")} if i["spatial_level"] == "S3" else {})}
        for e in d["experiments"] for i in e["game_instances"]
    ]
    assert strip(original) == strip(data), "instances changed beyond the S3 partition and budget"

    with open(a.instances, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)  # no trailing newline, as apply.py
    print(f"{a.instances}: {swapped} S3 instances re-split, {rebudgeted} rebudgeted")


if __name__ == "__main__":
    main()
