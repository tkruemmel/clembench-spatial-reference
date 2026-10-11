"""Unit tests for the episode-level metric helpers (thesis metric families)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import (
    oscillation_stats, deadlock_flag, deadlock_window, path_ratios,
    message_lengths,
)


def test_oscillation_detects_return_within_k():
    log = [
        {"object": "A", "from": [0, 0], "to": [0, 1]},
        {"object": "A", "from": [0, 1], "to": [0, 0]},
    ]
    count, flag = oscillation_stats(log, k=4)
    assert count == 1 and flag == 1


def test_oscillation_ignores_return_beyond_k_of_the_objects_own_moves():
    # A leaves (0, 0), makes five further moves of its own, then comes back:
    # six of A's own moves apart, outside k = 4.
    log = [{"object": "A", "from": [0, c], "to": [0, c + 1]} for c in range(6)]
    log += [{"object": "A", "from": [0, 6], "to": [0, 0]}]
    count, flag = oscillation_stats(log, k=4)
    assert count == 0 and flag == 0


def test_oscillation_window_is_independent_of_team_size():
    # The same A -> B -> A reversal, with zero, two and five other players'
    # moves interleaved. The window counts A's own moves, so all three fire.
    for n_between in (0, 2, 5):
        log = [{"object": "A", "from": [0, 0], "to": [0, 1]}]
        log += [
            {"object": f"O{i}", "from": [5, i], "to": [6, i]}
            for i in range(n_between)
        ]
        log += [{"object": "A", "from": [0, 1], "to": [0, 0]}]
        count, flag = oscillation_stats(log, k=4)
        assert (count, flag) == (1, 1), n_between


def test_deadlock_window_is_two_rounds_with_a_floor_of_six():
    assert deadlock_window(1) == 6   # solo
    assert deadlock_window(2) == 6   # dyad
    assert deadlock_window(3) == 6   # 3p
    assert deadlock_window(6) == 12  # 6p


def test_deadlock_flags_six_moves_without_net_progress():
    starts = {"A": [0, 0], "B": [7, 7]}
    targets = {"A": [0, 5], "B": [7, 0]}
    log = []
    for _ in range(3):
        log.append({"object": "A", "from": [0, 0], "to": [0, 1]})
        log.append({"object": "A", "from": [0, 1], "to": [0, 0]})
    assert deadlock_flag(log, starts, targets, num_players=2) == 1


def test_deadlock_not_flagged_on_steady_progress():
    starts = {"A": [0, 0]}
    targets = {"A": [0, 7]}
    log = [{"object": "A", "from": [0, c], "to": [0, c + 1]} for c in range(7)]
    assert deadlock_flag(log, starts, targets, num_players=2) == 0


def test_deadlock_window_scales_with_team_size():
    # Six moves of stalling: a deadlock at three players (window 6), not yet
    # one at six players (window 12).
    starts = {"A": [0, 0], "B": [7, 7]}
    targets = {"A": [0, 5], "B": [7, 0]}
    log = []
    for _ in range(3):
        log.append({"object": "A", "from": [0, 0], "to": [0, 1]})
        log.append({"object": "A", "from": [0, 1], "to": [0, 0]})
    assert deadlock_flag(log, starts, targets, num_players=3) == 1
    assert deadlock_flag(log, starts, targets, num_players=6) == 0


def test_path_ratios_detour_counts_double():
    starts = {"A": [0, 0]}
    targets = {"A": [0, 2]}
    log = [
        {"object": "A", "from": [0, 0], "to": [1, 0]},
        {"object": "A", "from": [1, 0], "to": [1, 1]},
        {"object": "A", "from": [1, 1], "to": [1, 2]},
        {"object": "A", "from": [1, 2], "to": [0, 2]},
    ]
    assert path_ratios(log, starts, targets) == {"A": 2.0}


def test_message_lengths_skips_missing():
    log = [
        {"object": "A", "from": [0, 0], "to": [0, 1], "message": "clear R3C3 please"},
        {"object": "A", "from": [0, 1], "to": [0, 2], "message": None},
    ]
    assert message_lengths(log) == [3]


def test_scorer_logs_new_metric_family():
    from master import (
        MatrixGameScorer, MOVE_COUNT, MOVE_ATTEMPTS, ILLEGAL_ATTEMPTS,
        BLOCKED_ATTEMPTS, TURN_MOVES,
    )
    from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS

    gi = {
        "optimal_moves": 4,
        "start_positions": {"A": [0, 0], "B": [7, 7]},
        "target_positions": {"A": [0, 2], "B": [7, 5]},
    }
    scorer = MatrixGameScorer("matrixgame_covered_comm", {}, gi)
    logged = {}
    scorer.log_episode_score = lambda k, v: logged.__setitem__(k, v)

    interactions = {
        METRIC_ABORTED: 0, METRIC_SUCCESS: 1, MOVE_COUNT: 8,
        MOVE_ATTEMPTS: 10, ILLEGAL_ATTEMPTS: 1, BLOCKED_ATTEMPTS: 1,
        TURN_MOVES: [
            {"object": "A", "from": [0, 0], "to": [0, 1], "message": "hi there"},
            {"object": "A", "from": [0, 1], "to": [0, 2], "message": None},
        ],
    }
    scorer.compute_scores(interactions)
    assert logged["Move Efficiency"] == 0.5
    assert logged["Illegal Move Rate"] == 0.1
    assert logged["Blocked Move Rate"] == 0.1
    assert logged["Oscillation Flag"] == 0
    assert logged["Deadlock Flag"] == 0
    assert logged["Path Ratio Mean"] == 0.5  # A: 2 moves / man 2 = 1.0; B: 0 / 2 = 0.0
    assert logged["Message Count"] == 1
    assert logged["Message Length Mean"] == 2.0
    assert logged["Played"] == 1
