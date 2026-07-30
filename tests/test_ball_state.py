"""
tests/test_ball_state.py
Pure unit tests for the ball state machine — no torch, no models, deterministic.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ball_state import (
    classify_floor_level,
    classify_contact_vs_bounce,
    FLOOR_LEVEL,
    IN_FLIGHT,
)


# ── classify_floor_level ─────────────────────────────────────────────────────

def test_floor_level_marks_every_reversal_no_exceptions():
    n = 10
    reversals = [2, 5, 8]
    states = classify_floor_level(reversals, n)

    assert len(states) == n
    for f in reversals:
        assert states[f] == FLOOR_LEVEL
    for f in range(n):
        if f not in reversals:
            assert states[f] == IN_FLIGHT


def test_floor_level_out_of_range_reversal_ignored():
    states = classify_floor_level([-1, 2, 99], n_frames=5)
    assert len(states) == 5
    assert states[2] == FLOOR_LEVEL
    assert sum(1 for s in states if s == FLOOR_LEVEL) == 1


def test_floor_level_empty_reversals_all_in_flight():
    states = classify_floor_level([], n_frames=5)
    assert states == [IN_FLIGHT] * 5


# ── classify_contact_vs_bounce (best-effort, secondary) ──────────────────────

def test_contact_vs_bounce_uses_proximity():
    n = 10
    ball = [{1: [90, 90, 110, 110]} for _ in range(n)]
    players = [{} for _ in range(n)]
    players[3] = {1: [80, 40, 120, 105]}   # foot (100,105), ~5px from ball → contact
    players[6] = {1: [400, 400, 440, 500]} # foot (420,500), far from ball → bounce

    contacts, bounces = classify_contact_vs_bounce([3, 6], ball, players, shot_player_distance_px=300)

    assert contacts == [3]
    assert bounces == [6]


def test_contact_and_bounce_are_disjoint_and_subset_of_reversals():
    n = 20
    ball = [{1: [10, 10, 30, 30]} for _ in range(n)]
    players = [{1: [0, 0, 40, 30]} for _ in range(n)]  # foot (20,30) near ball everywhere
    reversals = [2, 5, 11, 17]

    contacts, bounces = classify_contact_vs_bounce(reversals, ball, players)

    assert set(contacts).isdisjoint(bounces)
    assert set(contacts) | set(bounces) <= set(reversals)
    assert sorted(contacts) == reversals   # everyone's nearby here
    assert bounces == []


def test_missing_ball_frame_is_skipped():
    n = 5
    ball = [{} for _ in range(n)]
    players = [{1: [0, 0, 10, 10]} for _ in range(n)]
    contacts, bounces = classify_contact_vs_bounce([1, 3], ball, players)
    assert contacts == [] and bounces == []
