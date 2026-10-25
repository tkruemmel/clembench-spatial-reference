"""Build the *_final_chrono 3p/6p instances from the final ones.

The chrono variant lists teammates' messages in the order sent instead of
shuffled within a round. Its instances are the final arm's with the one
sentence announcing the shuffle removed, which leaves the 3p/6p prompts worded
like the dyad's. Rerun after every rebuild of the final nplayer instances.
"""
import json
from pathlib import Path

SENTENCE = " Within a round, your teammates' messages are listed in random order."
REPO = Path(__file__).resolve().parent.parent
for n in (3, 6):
    src = REPO / f"matrixgame_covered_comm_nplayer_grid_final/in/instances_{n}p.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    for exp in data["experiments"]:
        for inst in exp["game_instances"]:
            prompt = inst["player_prompt"]
            assert prompt.count(SENTENCE) == (inst["comm_protocol"] != "none"), (exp["name"], inst["game_id"])
            inst["player_prompt"] = prompt.replace(SENTENCE, "")
    out = REPO / f"matrixgame_covered_comm_nplayer_grid_final_chrono/in/instances_{n}p.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO)}")
