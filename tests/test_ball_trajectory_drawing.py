"""
tests/test_ball_trajectory_drawing.py
Unit tests for MiniCourt.draw_ball_trajectory (was a no-op before Phase 1, Step 4).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mini_visual_court import MiniCourt


def _make_mini_court() -> MiniCourt:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    return MiniCourt(frame)


def test_no_op_when_no_positions():
    mini = _make_mini_court()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    before = [f.copy() for f in frames]

    out = mini.draw_ball_trajectory(frames, positions={})

    for a, b in zip(out, before):
        assert np.array_equal(a, b)


def test_draws_something_when_trail_has_movement():
    mini = _make_mini_court()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    before = [f.copy() for f in frames]

    positions = {
        0: {1: (100, 100)},
        1: {1: (110, 105)},
        2: {1: (120, 110)},
        3: {1: (130, 115)},
        4: {1: (140, 120)},
    }
    out = mini.draw_ball_trajectory(frames, positions)

    # by the last frame, a 5-point trail exists — the frame should differ from blank
    assert not np.array_equal(out[4], before[4])


def test_single_point_trail_does_not_crash_or_draw():
    mini = _make_mini_court()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    before = [f.copy() for f in frames]

    positions = {0: {1: (100, 100)}}   # only one point ever — no line possible
    out = mini.draw_ball_trajectory(frames, positions)

    for a, b in zip(out, before):
        assert np.array_equal(a, b)


def test_returns_same_number_of_frames():
    mini = _make_mini_court()
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(7)]
    out = mini.draw_ball_trajectory(frames, positions={0: {1: (50, 50)}, 6: {1: (60, 60)}})
    assert len(out) == 7
