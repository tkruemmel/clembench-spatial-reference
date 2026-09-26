"""Compute the exact joint optimum for every board in an instance file.

Writes one JSON object per board to a .jsonl file as results arrive, so a long
run can be inspected or resumed.  Boards already solved exactly are skipped,
which makes the script restartable: raise ``--cap`` and re-run with
``--redo-cutoffs`` to finish off boards whose search was cut short.

    python3 analysis/efficiency_baseline/run_all_boards.py

By default it uses the C solver, which needs building once:

    cc -O3 -o analysis/efficiency_baseline/joint_optimum_c analysis/efficiency_baseline/joint_optimum.c

``--engine python`` forces the pure-Python reference implementation instead.
It gives identical answers and is roughly 100x slower.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csolver  # noqa: E402
from joint_optimum import load_instances, solve as py_solve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INSTANCES = "matrixgame_covered_comm_grid/in/instances.json"
DEFAULT_OUT = os.path.join(HERE, "joint_optimum.jsonl")

_CFG = {}


def _init(cfg):
    global _CFG
    _CFG = cfg


def _work(item):
    (level, gid), gi = item
    t = time.time()
    engine = _CFG["engine"]
    solver = csolver.solve if engine == "c" else py_solve
    r = solver(gi, cap=_CFG["cap"], timelimit=_CFG["timelimit"])
    r.update(
        level=level,
        game_id=gid,
        stored=gi["optimal_moves"],
        gap=(None if r["opt"] is None else r["opt"] - r["h0"]),
        engine=engine,
        secs=round(time.time() - t, 1),
    )
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", default=DEFAULT_INSTANCES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--engine", choices=("c", "python"), default=None,
                    help="default: c when the binary is built, else python")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cap", type=int, default=None,
                    help="max expanded states; default 60M for c, 3M for python")
    ap.add_argument("--timelimit", type=float, default=900.0)
    ap.add_argument("--redo-cutoffs", action="store_true",
                    help="also re-run boards whose previous search was cut off")
    args = ap.parse_args()

    engine = args.engine or ("c" if csolver.available() else "python")
    if engine == "c" and not csolver.available():
        sys.exit(f"C solver not built; run: cc -O3 -o {csolver.BINARY} {csolver.BINARY}.c"
                 .replace("_c.c", ".c"))
    cap = args.cap if args.cap is not None else (60_000_000 if engine == "c" else 3_000_000)

    done = {}
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done[(r["level"], r["game_id"])] = r
    skip = {k for k, r in done.items()
            if r["status"] == "exact" or not args.redo_cutoffs}

    instances = load_instances(args.instances)
    todo = [kv for kv in instances.items() if kv[0] not in skip]
    todo.sort(key=lambda kv: kv[1]["optimal_moves"])
    print(f"engine={engine} cap={cap}: {len(instances)} boards, "
          f"{len(skip)} already done, {len(todo)} to solve", flush=True)
    if not todo:
        return

    kept = [r for k, r in done.items() if k in skip]
    with open(args.out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    cfg = dict(engine=engine, cap=cap, timelimit=args.timelimit)
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        for r in pool.imap_unordered(_work, todo):
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"{r['level']} #{r['game_id']:>2} stored={r['stored']:>3} "
                  f"joint={r['opt']} lb={r['lb']} gap={r['gap']} "
                  f"{r['status']} {r['secs']}s", flush=True)


if __name__ == "__main__":
    main()
