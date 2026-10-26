"""Tests for the spatial-level figure logic.

The chain and swap derivations decide what each figure's caption claims, so
they are pinned both against hand-built boards and against every instance.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_levels import (  # noqa: E402
    EXPECTED_CHAIN_SIGNATURE,
    LEVELS,
    chain_signature,
    check_chains,
    dependency_chains,
    load_instances,
    pick_representative,
    swap_pair,
)


# ------------------------------------------------------------------ unit level


def test_no_overlap_yields_no_chains():
    starts = {"A": [0, 0], "B": [4, 4]}
    targets = {"A": [0, 1], "B": [4, 5]}
    assert dependency_chains(starts, targets) == []


def test_single_link_orders_blocked_before_blocker():
    # B starts on A's target, so B must vacate before A can land: A <- B.
    starts = {"A": [0, 0], "B": [0, 1]}
    targets = {"A": [0, 1], "B": [3, 3]}
    assert dependency_chains(starts, targets) == [["A", "B"]]


def test_length_two_chain_resolves_in_order():
    # C sits on B's target, B sits on A's target: A <- B <- C.
    starts = {"A": [0, 0], "B": [0, 1], "C": [0, 2]}
    targets = {"A": [0, 1], "B": [0, 2], "C": [5, 5]}
    assert dependency_chains(starts, targets) == [["A", "B", "C"]]


def test_multiple_chains_are_returned_longest_first():
    starts = {"A": [0, 0], "B": [0, 1], "C": [0, 2], "E": [4, 0], "F": [4, 1]}
    targets = {
        "A": [0, 1],
        "B": [0, 2],
        "C": [7, 7],
        "E": [4, 1],
        "F": [6, 6],
    }
    assert dependency_chains(starts, targets) == [["A", "B", "C"], ["E", "F"]]


def test_object_on_its_own_target_is_not_a_chain():
    starts = {"A": [2, 2]}
    targets = {"A": [2, 2]}
    assert dependency_chains(starts, targets) == []


def test_swap_pair_reads_the_device_block():
    instance = {
        "device": {"dead_end": [7, 3], "inner_adjacent": [6, 3], "entrance": [4, 3]},
        "start_positions": {"A": [6, 3], "B": [7, 3], "C": [0, 0]},
    }
    assert swap_pair(instance) == ("A", "B")


def test_swap_pair_is_none_without_a_swap_device():
    assert swap_pair({"device": {"dead_end": [7, 3]}, "start_positions": {}}) is None
    assert swap_pair({"start_positions": {}}) is None


def test_pick_representative_prefers_the_median_board():
    instances = [
        {"game_id": "far-low", "optimal_moves": 20, "walls": [[0, 0]] * 4},
        {"game_id": "median", "optimal_moves": 40, "walls": [[0, 0]] * 12},
        {"game_id": "far-high", "optimal_moves": 60, "walls": [[0, 0]] * 20},
    ]
    assert pick_representative(instances)["game_id"] == "median"


def test_pick_representative_is_deterministic_under_ties():
    tied = [
        {"game_id": "b", "optimal_moves": 30, "walls": []},
        {"game_id": "a", "optimal_moves": 30, "walls": []},
    ]
    assert pick_representative(tied)["game_id"] == "a"


# ------------------------------------------------------- against real instances


# ------------------------------------------------------- against the instances


@pytest.fixture(scope="module")
def released():
    return load_instances()


def test_every_level_has_instances(released):
    for level in LEVELS:
        assert released[level], f"no instances found for {level}"


def test_boards_are_deduplicated_across_protocols(released):
    # Four protocols share one board per index, so the 60 experiment instances
    # per level collapse to the 15 distinct puzzles the campaign runs.
    for level in LEVELS:
        assert len(released[level]) == 15


def test_documented_chain_signature_holds_for_every_instance(released):
    for level in LEVELS:
        expected = tuple(sorted(EXPECTED_CHAIN_SIGNATURE[level], reverse=True))
        for inst in released[level]:
            chains = dependency_chains(
                inst["start_positions"], inst["target_positions"]
            )
            assert chain_signature(chains) == expected, (
                f"{level} {inst.get('game_id')}: {chain_signature(chains)}"
            )


def test_s4_swap_heads_the_long_chain(released):
    for inst in released["S4"]:
        swap = swap_pair(inst)
        assert swap is not None
        chains = dependency_chains(inst["start_positions"], inst["target_positions"])
        assert chains[0][:2] == list(swap)


def test_s4_outer_target_is_the_dead_end(released):
    for inst in released["S4"]:
        outer, _inner = swap_pair(inst)
        assert tuple(inst["target_positions"][outer]) == tuple(
            inst["device"]["dead_end"]
        )


def test_s4_retreat_prefix_moves_outer_away_from_its_target(released):
    """The claim the S4 panel makes: OUTER's correct prefix increases its distance."""
    for inst in released["S4"]:
        outer, _inner = swap_pair(inst)
        start = tuple(inst["start_positions"][outer])
        target = tuple(inst["target_positions"][outer])
        entrance = tuple(inst["device"]["entrance"])
        # Retreating to the entrance is a prerequisite, and every step of it
        # increases the distance to OUTER's own target (the dead end).
        assert abs(entrance[0] - target[0]) + abs(entrance[1] - target[1]) > abs(
            start[0] - target[0]
        ) + abs(start[1] - target[1])


def test_s1_has_no_walls_and_s2_upward_have_them(released):
    assert all(inst["walls"] == [] for inst in released["S1"])
    for level in ("S2", "S3", "S4"):
        assert all(inst["walls"] for inst in released[level])


def test_corridor_is_recorded_at_s3_and_s4_only(released):
    for level in ("S1", "S2"):
        assert all(not inst.get("device") for inst in released[level])
    for level in ("S3", "S4"):
        for inst in released[level]:
            assert len(inst["device"]["corridor"]) == 4


def test_check_chains_rejects_a_board_whose_chains_drifted():
    broken = {
        "spatial_level": "S2",
        "game_id": "broken",
        "start_positions": {"A": [0, 0], "B": [4, 4]},
        "target_positions": {"A": [0, 1], "B": [4, 5]},
    }
    with pytest.raises(ValueError, match="chain signature"):
        check_chains(broken)
