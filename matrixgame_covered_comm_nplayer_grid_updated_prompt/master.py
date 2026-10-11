"""matrixgame_covered_comm_nplayer_grid_updated_prompt: second prompt arm of
matrixgame_covered_comm_nplayer_grid (games matrixgame_covered_comm_3p_grid_updated_prompt
and matrixgame_covered_comm_6p_grid_updated_prompt).

Same game master, scorer, rules and boards as the grid twin — the instances are
a copy of its in/instances_{3p,6p}.json with only player_prompt swapped (see
prompt_updated/). Sharing the master by import keeps the arms controlled: the
only difference between them is the prompt text.

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
