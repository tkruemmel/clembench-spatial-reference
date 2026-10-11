"""matrixgame_covered_comm_crossed_grid: crossed-device levels S3x/S4x, grid rendering.

Same game master, scorer, rules and rendering as matrixgame_covered_comm_grid,
with the prompts and A_k turn budget of matrixgame_covered_comm_grid_updated_prompt; only the
boards differ (matrixgame_covered_crossed_shared.boards). Sharing the master by
import keeps the comparison with S3/S4 controlled.

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so only the benchmark class is imported.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMPACT_PKG = os.path.join(_REPO_ROOT, "matrixgame_covered_comm")

# The compact master imports its helpers as `utils.*`, resolved from its own
# directory; clemcore only puts *this* directory on sys.path while loading.
for _path in (_REPO_ROOT, _COMPACT_PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from matrixgame_covered_comm.master import MatrixGameBenchmark  # noqa: E402,F401
