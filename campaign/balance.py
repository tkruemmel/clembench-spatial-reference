#!/usr/bin/env python3
"""Print the DeepSeek account balance in USD (nothing on failure). Used by worker.sh's MIN_BALANCE floor."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_status import balance  # noqa: E402
bal, _ = balance()
if bal is not None:
    print(f"{bal:.2f}")
