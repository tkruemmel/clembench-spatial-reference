"""Instance generator for matrixgame_covered_comm_nplayer_grid.

Same boards as matrixgame_covered_comm_nplayer (shared boardgen, seed 42, same
instance count) with compact_board=false and the grid prompt templates from
this package. Reuses the compact generator's on_generate over this package's
config.json (the same 32 variants with compact_board=false); experiment names
carry the `_grid` suffix (S1_none_3p_grid, ...). Like the compact package, the
combined instances.json is split into instances_3p.json / instances_6p.json,
which clemgame.json loads.

Usage:
  python3 matrixgame_covered_comm_nplayer_grid/instancegenerator.py --n 15
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clemcore.clemgame import GameInstanceGenerator

from matrixgame_covered_comm_nplayer.instancegenerator import (
    MatrixGameNPlayerInstanceGenerator as _CompactGenerator,
    split_by_player_count,
)


class GridMatrixGameNPlayerInstanceGenerator(_CompactGenerator):
    def __init__(self):
        # Bypass the compact generator's __init__, which pins its own directory.
        GameInstanceGenerator.__init__(self, _HERE)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate grid-rendered 3p/6p instances.")
    parser.add_argument("--n", type=int, default=15, help="Instances per condition (default: 15)")
    args = parser.parse_args()

    gen = GridMatrixGameNPlayerInstanceGenerator()
    gen.generate(filename="instances.json", seed=42, num_instances=args.n)
    split_by_player_count(os.path.join(_HERE, "in"))
    print("Wrote in/instances.json, in/instances_3p.json, in/instances_6p.json")
