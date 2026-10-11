"""Cross-check tab:clembench's team columns against clemcore's own aggregation.

In the updated-prompt arm the shipped `Main Score` divides by the suite's
attainable optimum, the score tab:clembench recomputes, so
clemeval.save_clem_table on the shipped scores must reproduce % Played and the
Quality Score of dyad, 3p and 6p.  Solo ships the BFS-bound score and is not
compared.  (`clem eval` itself refuses these trees: the games log `Played`.)

    python3 analysis/thesis/clem_crosscheck.py
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import tempfile

import pandas as pd
from clemcore.clemeval import save_clem_table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tables_comm as T                                      # noqa: E402

assert T.ARM == "updated", "the shipped score equals A_k only in the updated arm"
rows = []
for model, prefix in T.ARMS.items():
    for game in T.SUITES:
        for f in sorted(glob.glob(os.path.join(T.ROOT, T.tree(prefix, game), "**", "scores.json"),
                                  recursive=True)):
            s = json.load(open(f))["episode scores"]
            rows += [(game, model, "Main Score", s["Main Score"]),
                     (game, model, "Played", 1 - s["Aborted"])]
df = pd.DataFrame(rows, columns=["game", "model", "metric", "value"])
with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
    clem = save_clem_table(df, d)

out = io.StringIO()
with contextlib.redirect_stdout(out):
    T.clembench_table(T.load_optima(), check=True)
ours = {(m, g): (float(p), float(q)) for m, g, p, q in re.findall(
    r"^(\S+)\s+(dyad|3p|6p)\s+played\s+([\d.]+)\s+quality\s+([\d.]+)", out.getvalue(), re.M)}

bad = 0
for (model, game), (p, q) in sorted(ours.items()):
    cp, cq = clem.loc[model, f"{game}, % Played"], clem.loc[model, f"{game}, Quality Score"]
    ok = abs(cp - p) <= 0.01 and abs(cq - q) <= 0.01
    bad += not ok
    print(f"{model:<20} {game:<4}  played {p:6.2f} / {cp:6.2f}  quality {q:6.2f} / {cq:6.2f}  "
          f"{'ok' if ok else 'DIFF'}")
assert len(ours) == 3 * len(T.ARMS), len(ours)
print(f"{len(ours) - bad} of {len(ours)} team cells match clemcore to 0.01")
sys.exit(1 if bad else 0)
