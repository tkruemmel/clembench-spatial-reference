"""Tests for the spatial-level figure logic.

The chain and swap derivations decide what each figure's caption claims, so
they are pinned both against hand-built boards and against every released
instance.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatial_levels import (  # noqa: E402
    EXPECTED_CHAIN_SIGNATURE,
    LEVELS,
    cellname,
    corridor_extent,
    mean_detour_ratio,
    retreat_cell,
    chain_signature,
    dependency_chains,
    describe,
    free_path,
    load_instances,
    pick_representative,
    swap_pair,
    worst_detour,
)


# ------------------------------------------------------------------ unit level


def test_cellname_is_one_indexed():
    assert cellname((0, 0)) == "R1C1"
    assert cellname((7, 3)) == "R8C4"


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


def test_free_path_routes_around_a_wall():
    walls = {(0, 1), (1, 1), (2, 1)}
    path = free_path((0, 0), (0, 2), walls, 8)
    assert path[0] == (0, 0) and path[-1] == (0, 2)
    assert not set(path) & walls
    assert len(path) - 1 == 8  # around the wall's open end: 3 down, 2 across, 3 up


def test_free_path_is_empty_when_target_is_walled_off():
    walls = {(0, 1), (1, 0), (1, 1)}
    assert free_path((0, 0), (7, 7), walls, 8) == []


def test_swap_pair_reads_the_device_block():
    instance = {
        "device": {"dead_end": [7, 3], "inner_adjacent": [6, 3], "entrance": [4, 3]},
        "start_positions": {"A": [6, 3], "B": [7, 3], "C": [0, 0]},
    }
    assert swap_pair(instance) == ("A", "B")


def test_swap_pair_is_none_without_a_swap_device():
    assert swap_pair({"device": {"dead_end": [7, 3]}, "start_positions": {}}) is None
    assert swap_pair({"start_positions": {}}) is None


def test_worst_detour_is_none_on_an_open_board():
    instance = {
        "walls": [],
        "grid_size": 8,
        "objects": ["A"],
        "start_positions": {"A": [0, 0]},
        "target_positions": {"A": [0, 3]},
    }
    assert worst_detour(instance) is None


def test_worst_detour_reports_the_forced_object():
    instance = {
        "walls": [[0, 1], [1, 1], [2, 1]],
        "grid_size": 8,
        "objects": ["A", "B"],
        "start_positions": {"A": [0, 0], "B": [5, 0]},
        "target_positions": {"A": [0, 2], "B": [5, 3]},
    }
    obj, path, base = worst_detour(instance)
    assert obj == "A"
    assert base == 2 and len(path) - 1 == 8


def test_retreat_cell_avoids_walls_and_the_corridor():
    # Corridor runs down column 4 from row 4 to row 7; the entrance at (3,4)
    # is outside it. (2,4) is walled, so OUTER must park sideways.
    instance = {
        "grid_size": 8,
        "walls": [[2, 4]],
        "objects": ["A", "B"],
        "device": {
            "entrance": [3, 4],
            "dead_end": [7, 4],
            "inner_adjacent": [6, 4],
            "corridor": [[4, 4], [5, 4], [6, 4], [7, 4]],
        },
        "start_positions": {"A": [6, 4], "B": [7, 4]},
        "target_positions": {"A": [7, 4], "B": [0, 0]},
    }
    cell = retreat_cell(instance)
    assert cell in {(3, 3), (3, 5)}
    assert cell != (2, 4)  # walled
    assert cell not in {(4, 4), (5, 4), (6, 4), (7, 4)}  # in the corridor


def test_retreat_cell_prefers_the_side_away_from_inners_target():
    instance = {
        "grid_size": 8,
        "walls": [[2, 4]],
        "objects": ["A", "B"],
        "device": {
            "entrance": [3, 4],
            "dead_end": [7, 4],
            "inner_adjacent": [6, 4],
            "corridor": [[4, 4], [5, 4], [6, 4], [7, 4]],
        },
        "start_positions": {"A": [6, 4], "B": [7, 4]},
        "target_positions": {"A": [7, 4], "B": [3, 7]},  # inner exits rightward
    }
    assert retreat_cell(instance) == (3, 3)  # park left, out of inner's way


def test_corridor_extent_excludes_the_entrance():
    device = {
        "entrance": [3, 4],
        "dead_end": [7, 4],
        "corridor": [[4, 4], [5, 4], [6, 4], [7, 4]],
    }
    assert corridor_extent(device) == ((4, 4), (7, 4))
    assert corridor_extent({}) is None


def test_mean_detour_ratio_is_none_without_walls():
    assert mean_detour_ratio({"walls": [], "objects": [], "grid_size": 8}) is None


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


def test_s4_swap_is_one_of_the_derived_chains(released):
    for inst in released["S4"]:
        swap = swap_pair(inst)
        assert swap is not None
        chains = dependency_chains(inst["start_positions"], inst["target_positions"])
        assert list(swap) in chains


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


def test_describe_succeeds_for_every_released_instance(released):
    for level in LEVELS:
        for inst in released[level]:
            described = describe(inst)
            assert described["level"] == level


def test_describe_rejects_a_board_whose_chains_drifted():
    broken = {
        "spatial_level": "S2",
        "game_id": "broken",
        "grid_size": 8,
        "objects": ["A", "B"],
        "walls": [],
        "start_positions": {"A": [0, 0], "B": [4, 4]},
        "target_positions": {"A": [0, 1], "B": [4, 5]},
    }
    with pytest.raises(ValueError, match="chain signature"):
        describe(broken)


def test_representative_boards_are_solvable_object_by_object(released):
    """Every object on a chosen board can reach its target ignoring other objects."""
    for level in LEVELS:
        inst = pick_representative(released[level])
        walls = {tuple(w) for w in inst["walls"]}
        for obj in inst["objects"]:
            path = free_path(
                tuple(inst["start_positions"][obj]),
                tuple(inst["target_positions"][obj]),
                walls,
                inst["grid_size"],
            )
            assert path, f"{level} {obj} cannot reach its target"


def test_every_s4_board_has_a_free_retreat_cell(released):
    """The S4 panel draws OUTER parking here, so it must exist and be free."""
    for inst in released["S4"]:
        cell = retreat_cell(inst)
        assert cell is not None, inst.get("game_id")
        walls = {tuple(w) for w in inst["walls"]}
        corridor = {tuple(c) for c in inst["device"]["corridor"]}
        assert cell not in walls and cell not in corridor
        assert 0 <= cell[0] < inst["grid_size"] and 0 <= cell[1] < inst["grid_size"]


def test_retreat_cell_is_adjacent_to_the_entrance(released):
    for inst in released["S4"]:
        cell = retreat_cell(inst)
        entrance = tuple(inst["device"]["entrance"])
        assert abs(cell[0] - entrance[0]) + abs(cell[1] - entrance[1]) == 1


def test_achieved_detour_ratio_falls_short_of_the_target_at_s3_and_s4(released):
    """Pins a real property of the released instances.

    boardgen carves toward 1.5 at S2 and 2.0 at S3/S4, but at S3/S4 every
    corridor cell plus every start and target is protected and dead ends are
    refused, so _carve_for_ratio exhausts its attempts. Wall pressure therefore
    *decreases* up the ladder: the S2-to-S4 escalation is carried by the
    corridor device and chain depth, not by wall-induced detours. Any figure or
    prose claiming ~2.0 detours at S3/S4 contradicts these files.
    """
    import statistics

    medians = {}
    for lvl in ("S2", "S3", "S4"):
        ratios = []
        for inst in released[lvl]:
            walls = {tuple(w) for w in inst["walls"]}
            for obj in inst["objects"]:
                start = tuple(inst["start_positions"][obj])
                goal = tuple(inst["target_positions"][obj])
                base = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                path = free_path(start, goal, walls, inst["grid_size"])
                if base and path:
                    ratios.append((len(path) - 1) / base)
        medians[lvl] = statistics.median(ratios)

    assert medians["S2"] == pytest.approx(1.5, abs=0.15), medians
    assert medians["S3"] < medians["S2"], medians
    assert medians["S4"] < medians["S3"], medians


def test_corridor_extent_never_includes_the_entrance_on_real_boards(released):
    for level in ("S3", "S4"):
        for inst in released[level]:
            mouth, dead_end = corridor_extent(inst["device"])
            entrance = tuple(inst["device"]["entrance"])
            corridor = {tuple(c) for c in inst["device"]["corridor"]}
            assert entrance not in corridor
            assert mouth in corridor and dead_end in corridor
