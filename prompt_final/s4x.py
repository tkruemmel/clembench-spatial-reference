"""Play S4 on the S4x boards in every final file.

S4x keeps S4's corridor and swap but makes the swap pair the head link of the
deep chain A<-B<-C<-D, with the pair E<-F in open space
(matrixgame_covered_crossed_shared/boards.py). The level keeps the name S4.

Each final S4 instance takes the board of the crossed package's S4x instance
with the same protocol and game_id: positions, walls, device, dyad split and
turn budget. Prompt, names and level label stay. The team budgets are already
A_k on these boards (attainable_{dyad,3p,6p}_crossed.jsonl, applied by the
crossed generators); solo, whose crossed budget is the BFS sum, takes
max(4 * J, 20) from joint_optimum_crossed.jsonl, as attainable_solo.jsonl does
for the other levels.

Run after apply.py (see README); a second run is a no-op.

  python3 prompt_final/s4x.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FILES = (  # final file, crossed file holding its S4x boards
    ("matrixgame_covered_comm_grid_final/in/instances.json",
     "matrixgame_covered_comm_crossed_grid/in/instances.json"),
    ("matrixgame_covered_comm_nplayer_grid_final/in/instances_3p.json",
     "matrixgame_covered_comm_nplayer_crossed_grid/in/instances_3p.json"),
    ("matrixgame_covered_comm_nplayer_grid_final/in/instances_6p.json",
     "matrixgame_covered_comm_nplayer_crossed_grid/in/instances_6p.json"),
    ("matrixgame_covered_solo_grid_final/in/instances.json",
     "matrixgame_covered_solo_crossed_grid/in/instances.json"),
)
BOARD = ("start_positions", "target_positions", "walls", "device",
         "player_a_objects", "player_b_objects", "optimal_moves", "max_turns")
JOINT = REPO / "analysis" / "efficiency_baseline" / "joint_optimum_crossed.jsonl"


def _instances(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, {(e["name"], i["game_id"]): i for e in data["experiments"] for i in e["game_instances"]}


def main():
    joint = {}
    for row in map(json.loads, JOINT.read_text(encoding="utf-8").splitlines()):
        if row["level"] == "S4x":
            assert row["status"] == "exact", row
            joint[row["game_id"]] = row["opt"]

    for final, crossed in FILES:
        data, arm = _instances(REPO / final)
        _, boards = _instances(REPO / crossed)
        changed = 0
        for (name, gid), inst in arm.items():
            if inst["spatial_level"] != "S4":
                continue
            board = boards[(name.replace("S4_", "S4x_"), gid)]
            # Same protocol, masking and objects; only the board may differ.
            same = set(inst) - set(BOARD) - {"player_prompt", "spatial_level"}
            assert all(board[k] == inst[k] for k in same), (final, name, gid)
            new = {k: board[k] if k in BOARD and k in inst else v for k, v in inst.items()}
            if "comm_protocol" not in inst:
                assert joint[gid] >= board["optimal_moves"], (name, gid)
                new["optimal_moves"] = joint[gid]
                new["max_turns"] = max(4 * joint[gid], 20)
            changed += new != inst
            inst.clear()
            inst.update(new)
        # No trailing newline, as apply.py writes these files.
        (REPO / final).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{final}: {changed} S4 instances moved onto S4x boards")


if __name__ == "__main__":
    main()
