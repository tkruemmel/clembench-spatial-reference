"""Episode-level metric helpers computed from the enriched move log.

Move-log entries carry: object, from ([r, c]), to ([r, c]), message.
All functions are pure so they can be re-run offline on stored logs.
"""

from typing import Dict, List, Optional, Tuple


def oscillation_stats(move_log: List[Dict], k: int = 4) -> Tuple[int, int]:
    """(count, flag): moves where an object returns to a cell it left
    within the last k moves (global move index) — the local signature of
    pathological looping."""
    left: Dict[str, List[Tuple[int, Tuple[int, int]]]] = {}
    count = 0
    for i, m in enumerate(move_log):
        obj = m["object"]
        frm = tuple(m["from"])
        to = tuple(m["to"])
        hist = left.setdefault(obj, [])
        if any(cell == to and i - idx <= k for idx, cell in hist):
            count += 1
        hist.append((i, frm))
    return count, int(count > 0)


def deadlock_flag(
    move_log: List[Dict],
    start_positions: Dict[str, List[int]],
    target_positions: Dict[str, List[int]],
    k: int = 6,
) -> int:
    """1 if there is any window of k consecutive moves with no net progress:
    the summed Manhattan distance of all objects to their targets does not
    decrease over the window."""
    pos = {o: tuple(p) for o, p in start_positions.items()}
    targets = {o: tuple(t) for o, t in target_positions.items()}

    def total_dist() -> int:
        return sum(
            abs(pos[o][0] - t[0]) + abs(pos[o][1] - t[1])
            for o, t in targets.items()
        )

    dists = [total_dist()]
    for m in move_log:
        pos[m["object"]] = tuple(m["to"])
        dists.append(total_dist())
    return int(any(dists[i] >= dists[i - k] for i in range(k, len(dists))))


def path_ratios(
    move_log: List[Dict],
    start_positions: Dict[str, List[int]],
    target_positions: Dict[str, List[int]],
) -> Dict[str, float]:
    """Per object: cells actually traversed / Manhattan(start, target).
    Objects with Manhattan distance 0 are skipped (cannot occur in
    generated instances, which enforce start != target)."""
    moves_per: Dict[str, int] = {}
    for m in move_log:
        moves_per[m["object"]] = moves_per.get(m["object"], 0) + 1
    ratios: Dict[str, float] = {}
    for o, s in start_positions.items():
        t = target_positions[o]
        man = abs(s[0] - t[0]) + abs(s[1] - t[1])
        if man > 0:
            ratios[o] = moves_per.get(o, 0) / man
    return ratios


def message_lengths(move_log: List[Dict]) -> List[int]:
    """Word count of each non-empty message, in move order."""
    return [
        len(m["message"].split())
        for m in move_log
        if m.get("message")
    ]
