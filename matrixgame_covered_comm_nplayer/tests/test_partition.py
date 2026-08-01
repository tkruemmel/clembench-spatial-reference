"""Round-robin ownership must keep chain neighbours and swap partners on
different agents at 3p (thesis S2-S4 cross-agent claims)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from instancegenerator import round_robin_partition


def test_3p_partition():
    objs = ["A", "B", "C", "D", "E", "F"]
    assert round_robin_partition(objs, 3) == [["A", "D"], ["B", "E"], ["C", "F"]]


def test_6p_partition():
    objs = ["A", "B", "C", "D", "E", "F"]
    assert round_robin_partition(objs, 6) == [["A"], ["B"], ["C"], ["D"], ["E"], ["F"]]


def _owner(partition, obj):
    for i, group in enumerate(partition):
        if obj in group:
            return i
    raise AssertionError(obj)


def test_3p_cross_agent_properties():
    p = round_robin_partition(["A", "B", "C", "D", "E", "F"], 3)
    # S4 swap partners OUTER=A / INNER=B on different agents
    assert _owner(p, "A") != _owner(p, "B")
    # S2 chain A<-B<-C alternates owners
    assert _owner(p, "A") != _owner(p, "B") != _owner(p, "C")
    # S3/S4 length-3 chains: no two consecutive members share an owner
    for chain in (["A", "B", "C", "D"], ["C", "D", "E", "F"]):
        owners = [_owner(p, o) for o in chain]
        assert all(x != y for x, y in zip(owners, owners[1:]))
    # S3 corridor pair C (entrance) / D (passer) on different agents
    assert _owner(p, "C") != _owner(p, "D")
