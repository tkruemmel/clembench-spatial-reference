"""pytest configuration for matrixgame_covered_solo tests.

Adds the game package root to sys.path so that 'from utils.board import ...'
resolves to matrixgame_covered_solo/utils/board.py without requiring
PYTHONPATH=matrixgame_covered_solo on the command line.
"""

import sys
from pathlib import Path

# matrixgame_covered_solo/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))
