"""Instance generator for matrixgame_covered_comm_crossed_grid.

Runs the released dyad generator's on_generate unchanged, with boards from
matrixgame_covered_crossed_shared.boards (S3x/S4x) and this package's
config.json. Prompt and budget follow the updated-prompt arm
(matrixgame_covered_comm_grid_updated_prompt): the templates are copies of
prompt_updated/dyad, and prompt_updated/apply.py sets optimal_moves to the
attainable optimum A_k (analysis/efficiency_baseline/attainable_dyad_crossed.jsonl)
and max_turns to max(4 * A_k, 20). Experiment names: S3x_none_grid, ...

Usage:
  python3 matrixgame_covered_comm_crossed_grid/instancegenerator.py --n 15
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

import matrixgame_covered_comm.instancegenerator as _compact
from matrixgame_covered_crossed_shared.boards import crossed_boards

PROTOCOLS = ("none", "structured", "freeform", "hybrid")
LEVELS = ("s3x", "s4x")


class CrossedGridInstanceGenerator(_compact.MatrixGameInstanceGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)

    def on_generate(self, seed: int, **kwargs):
        with crossed_boards(_compact):
            super().on_generate(seed, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate crossed-device dyad grid instances.")
    parser.add_argument("--n", type=int, default=15, help="Instances per condition (default: 15)")
    args = parser.parse_args()

    gen = CrossedGridInstanceGenerator()
    for level in LEVELS:
        for protocol in PROTOCOLS:
            variant = f"{level}_{protocol}_grid"
            gen.generate(filename="instances.json", seed=42, variant=variant, num_instances=args.n)
            print(f"Generated: {variant} (n={args.n})")
    out = os.path.join(_HERE, "in", "instances.json")
    subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "prompt_updated", "apply.py"),
                    "--src", out, "--out", out,
                    "--prompts", os.path.join(_REPO_ROOT, "prompt_updated", "dyad"),
                    "--budget", os.path.join(_REPO_ROOT, "analysis", "efficiency_baseline",
                                             "attainable_dyad_crossed.jsonl")], check=True)
