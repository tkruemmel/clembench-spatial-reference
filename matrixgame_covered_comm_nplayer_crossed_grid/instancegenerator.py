"""Instance generator for matrixgame_covered_comm_nplayer_crossed_grid.

Runs the released n-player generator's on_generate unchanged, with boards
from matrixgame_covered_crossed_shared.boards (S3x/S4x) and this package's
config.json (16 variants). Prompt and budget follow the updated-prompt arm
(matrixgame_covered_comm_nplayer_grid_updated_prompt): the templates are copies
of prompt_updated/nplayer, and prompt_updated/apply.py sets optimal_moves to the
attainable optimum A_k (analysis/efficiency_baseline/attainable_{3p,6p}_crossed.jsonl)
and max_turns to max(4 * A_k, 20). Experiment names: S3x_none_3p_grid, ...
The combined instances.json is split into instances_3p.json / instances_6p.json
(which clemgame.json loads) and then removed, as in the updated-prompt arm.

Usage:
  python3 matrixgame_covered_comm_nplayer_crossed_grid/instancegenerator.py --n 15
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

import matrixgame_covered_comm_nplayer.instancegenerator as _compact
from matrixgame_covered_crossed_shared.boards import crossed_boards


class CrossedGridNPlayerInstanceGenerator(_compact.MatrixGameNPlayerInstanceGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)

    def on_generate(self, seed: int, **kwargs):
        with crossed_boards(_compact):
            super().on_generate(seed, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate crossed-device 3p/6p grid instances.")
    parser.add_argument("--n", type=int, default=15, help="Instances per condition (default: 15)")
    args = parser.parse_args()

    CrossedGridNPlayerInstanceGenerator().generate(filename="instances.json", seed=42, num_instances=args.n)
    in_dir = os.path.join(_HERE, "in")
    _compact.split_by_player_count(in_dir)
    os.remove(os.path.join(in_dir, "instances.json"))
    for n in (3, 6):
        out = os.path.join(in_dir, f"instances_{n}p.json")
        subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "prompt_updated", "apply.py"),
                        "--src", out, "--out", out,
                        "--prompts", os.path.join(_REPO_ROOT, "prompt_updated", "nplayer"),
                        "--budget", os.path.join(_REPO_ROOT, "analysis", "efficiency_baseline",
                                                 f"attainable_{n}p_crossed.jsonl")], check=True)
