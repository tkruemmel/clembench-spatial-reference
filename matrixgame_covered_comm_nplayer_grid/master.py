"""matrixgame_covered_comm_nplayer_grid: grid-rendered twin of
matrixgame_covered_comm_nplayer (games matrixgame_covered_comm_3p_grid and
matrixgame_covered_comm_6p_grid).

Same game master, scorer and rules as the compact package. The only differences
are the instances (compact_board=false, so the board is sent as the ASCII grid)
and the prompt templates that describe that rendering. Sharing the master by
import keeps the two conditions controlled: a fix in the compact master applies
here too, so the comparison stays rendering-only.

clemcore loads this file by path and requires exactly one GameBenchmark subclass
in the module namespace, so only the benchmark class is imported.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_COMPACT_PKG = os.path.join(_REPO_ROOT, "matrixgame_covered_comm_nplayer")

# The compact master imports its helpers as `utils.*`, resolved from its own
# directory; clemcore only puts *this* directory on sys.path while loading.
for _path in (_REPO_ROOT, _COMPACT_PKG):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from matrixgame_covered_comm_nplayer.master import NPlayerMatrixGameBenchmark  # noqa: E402,F401
