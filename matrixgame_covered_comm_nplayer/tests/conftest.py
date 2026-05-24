"""pytest configuration for matrixgame_covered_comm_nplayer tests.

Adds the game package root to sys.path so that 'from utils.board import ...'
resolves to matrixgame_covered_comm_nplayer/utils/board.py without requiring
PYTHONPATH on the command line.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
