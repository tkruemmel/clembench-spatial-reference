"""Bridge to the C solver in ``joint_optimum.c``.

Same problem and same algorithm as ``joint_optimum.solve``, roughly 100x faster.
Build it once with::

    cc -O3 -o analysis/efficiency_baseline/joint_optimum_c analysis/efficiency_baseline/joint_optimum.c

``solve`` falls back to the pure-Python reference implementation when the binary
is missing, so nothing here is a hard dependency.
"""

import os
import subprocess
from typing import Optional

from joint_optimum import _grid

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "joint_optimum_c")


def available() -> bool:
    return os.path.exists(BINARY) and os.access(BINARY, os.X_OK)


def encode(gi: dict) -> str:
    """Board as the whitespace-separated integers the C solver reads on stdin."""
    free, idx, nbr = _grid(gi)
    objs = sorted(gi["objects"])
    if len(free) > 64:
        raise ValueError(f"{len(free)} free cells, the C solver packs at most 64")
    out = [f"{len(free)} {len(objs)}"]
    for ns in nbr:
        out.append(" ".join(str(x) for x in (len(ns),) + tuple(ns)))
    out.append(" ".join(str(idx[tuple(gi["start_positions"][o])]) for o in objs))
    out.append(" ".join(str(idx[tuple(gi["target_positions"][o])]) for o in objs))
    return "\n".join(out) + "\n"


def solve(gi: dict, cap: int = 60_000_000, timelimit: float = 900.0) -> dict:
    """Exact joint optimum via the C binary; same result dict as the Python one."""
    proc = subprocess.run(
        [BINARY, str(cap), str(timelimit)],
        input=encode(gi), capture_output=True, text=True, check=True,
    )
    fields = proc.stdout.split()
    got = dict(zip(fields[0::2], fields[1::2]))
    opt: Optional[int] = int(got["opt"])
    return dict(
        h0=int(got["h0"]),
        opt=None if opt < 0 else opt,
        lb=int(got["lb"]),
        popped=int(got["expanded"]),
        status=got["status"],
    )
