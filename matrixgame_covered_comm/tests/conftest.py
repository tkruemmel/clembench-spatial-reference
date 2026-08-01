"""pytest configuration for matrixgame_covered_comm tests.

Adds the game package root to sys.path so that 'from utils.board import ...'
and 'from master import ...' resolve to this package without requiring
PYTHONPATH on the command line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
