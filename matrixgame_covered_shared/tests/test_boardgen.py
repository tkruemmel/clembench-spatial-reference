"""Property tests for the shared S1-S4 board generators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from matrixgame_covered_shared.boardgen import generate_board, instance_rng

OBJS = ["A", "B", "C", "D", "E", "F"]


def test_deterministic_per_seed_level_idx():
    b1 = generate_board("S3", 8, OBJS, instance_rng(42, "S3", 0))
    b2 = generate_board("S3", 8, OBJS, instance_rng(42, "S3", 0))
    assert b1 == b2
    b3 = generate_board("S3", 8, OBJS, instance_rng(42, "S3", 1))
    assert b3 != b1


def test_s1_free_placement_no_walls_no_chains():
    for idx in range(5):
        b = generate_board("S1", 8, OBJS, instance_rng(42, "S1", idx))
        assert b["walls"] == []
        cells = [tuple(v) for v in list(b["start_positions"].values())
                 + list(b["target_positions"].values())]
        # 12 pairwise-distinct cells => no accidental dependency chains
        assert len(set(cells)) == 12
        assert b["optimal_moves"] > 0


def test_s2_chain_and_walls():
    for idx in range(5):
        b = generate_board("S2", 8, OBJS, instance_rng(42, "S2", idx))
        s, t = b["start_positions"], b["target_positions"]
        assert s["B"] == t["A"], "B.start must equal A.target"
        assert s["C"] == t["B"], "C.start must equal B.target"
        assert len(b["walls"]) > 0
        assert b["dyad_split"] == (["A", "C", "D"], ["B", "E", "F"])


def test_s3_corridor_length4_and_threading():
    for idx in range(5):
        b = generate_board("S3", 8, OBJS, instance_rng(42, "S3", idx))
        s, t, dev = b["start_positions"], b["target_positions"], b["device"]
        assert len(dev["corridor"]) == 4, "corridor must be exactly length 4"
        assert s["C"] == dev["entrance"], "C starts at the corridor entrance"
        assert t["D"] == dev["dead_end"], "D targets the dead-end cell"
        assert t["B"] == dev["entrance"], "chain: C.start = B.target = entrance"
        assert s["D"] == t["C"], "chain: D.start = C.target"
        assert s["B"] == t["A"], "chain: B.start = A.target"
        assert s["F"] == t["E"], "secondary dependency: F.start = E.target"
        assert dev["entrance"] not in dev["corridor"]
        assert b["dyad_split"] == (["D", "B", "E"], ["C", "A", "F"])


def test_s4_corridor_length4_and_swap():
    for idx in range(5):
        b = generate_board("S4", 8, OBJS, instance_rng(42, "S4", idx))
        s, t, dev = b["start_positions"], b["target_positions"], b["device"]
        assert len(dev["corridor"]) == 4
        assert s["B"] == dev["dead_end"], "INNER (B) starts at the dead end"
        assert t["A"] == dev["dead_end"], "OUTER (A) targets the dead end"
        assert s["A"] == dev["inner_adjacent"], "OUTER starts adjacent to the dead end"
        assert s["D"] == t["C"] and s["E"] == t["D"] and s["F"] == t["E"], \
            "length-3 chain C<-D<-E<-F"
        assert b["dyad_split"] == (["A", "D", "F"], ["B", "C", "E"])


def test_no_start_equals_target():
    for level in ("S1", "S2", "S3", "S4"):
        b = generate_board(level, 8, OBJS, instance_rng(42, level, 0))
        for o in OBJS:
            assert b["start_positions"][o] != b["target_positions"][o]
