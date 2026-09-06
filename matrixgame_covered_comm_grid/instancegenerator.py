"""Instance generator for matrixgame_covered_comm_grid.

Same boards as matrixgame_covered_comm (shared boardgen, seed 42, same
instance count) with compact_board=false and the grid prompt templates from
this package. Reuses the compact generator's on_generate; only the resource
root (this directory, whose config.json holds the sixteen *_grid variants)
and the variant keys differ. Experiment names carry the `_grid` suffix
(S1_none_grid, ...) where the compact package has `_compact`.

Usage:
  python3 matrixgame_covered_comm_grid/instancegenerator.py --n 15
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_comm.instancegenerator import (
    MatrixGameInstanceGenerator as _CompactGenerator,
)

PROTOCOLS = ("none", "structured", "freeform", "hybrid")
# Same order as the compact generator's main: Exp2 (S2) first, then Exp3.
LEVELS = (("exp2", "s2"), ("exp3", "s1"), ("exp3", "s3"), ("exp3", "s4"))


class GridMatrixGameInstanceGenerator(_CompactGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate grid-rendered dyad instances.")
    parser.add_argument("--n", type=int, default=15, help="Instances per condition (default: 15)")
    args = parser.parse_args()

    gen = GridMatrixGameInstanceGenerator()
    for experiment, level in LEVELS:
        for protocol in PROTOCOLS:
            variant = f"{experiment}_{level}_{protocol}_grid"
            gen.generate(filename="instances.json", seed=42, variant=variant, num_instances=args.n)
            print(f"Generated: {variant} (n={args.n})")
