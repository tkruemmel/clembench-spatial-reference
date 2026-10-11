"""Instance generator for matrixgame_covered_solo_crossed_grid.

Runs the released solo generator's on_generate unchanged, with boards from
matrixgame_covered_crossed_shared.boards (S3x/S4x) and this package's
config.json (four *_grid variants) and grid templates.

Usage:
  python3 matrixgame_covered_solo_crossed_grid/instancegenerator.py --num-instances 15
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

import matrixgame_covered_solo.instancegenerator as _compact
from matrixgame_covered_crossed_shared.boards import crossed_boards


class CrossedGridSoloInstanceGenerator(_compact.SoloMatrixGameInstanceGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)

    def on_generate(self, seed: int, **kwargs):
        with crossed_boards(_compact):
            super().on_generate(seed, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate crossed-device solo grid instances.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=15)
    args = parser.parse_args()

    CrossedGridSoloInstanceGenerator().generate(
        filename="instances.json", seed=args.seed, num_instances=args.num_instances
    )
    print("Wrote in/instances.json")
