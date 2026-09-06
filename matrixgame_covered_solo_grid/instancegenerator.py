"""Instance generator for matrixgame_covered_solo_grid.

Same boards as matrixgame_covered_solo (shared boardgen, seed 42, same
instance count) with compact_board=false and the grid prompt templates from
this package. Reuses the compact generator's on_generate; only the resource
root (this directory, whose config.json holds the eight *_grid variants)
differs.

Usage:
  python3 matrixgame_covered_solo_grid/instancegenerator.py --num-instances 15
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_solo.instancegenerator import (
    SoloMatrixGameInstanceGenerator as _CompactGenerator,
)


class GridSoloMatrixGameInstanceGenerator(_CompactGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate grid-rendered solo instances.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-instances", type=int, default=15)
    parser.add_argument("--out", default="instances.json")
    args = parser.parse_args()

    gen = GridSoloMatrixGameInstanceGenerator()
    gen.generate(filename=args.out, seed=args.seed, num_instances=args.num_instances)
    print(f"Wrote instances to {args.out}")
