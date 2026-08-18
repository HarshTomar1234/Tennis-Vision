"""
tests/test_mini_court_ball_projection.py
Unit tests for MiniCourt.convert_ball_to_mini_court_coordinates - the state-aware
ball projection that only trusts the homography at floor-level frames and
interpolates in between.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mini_visual_court import MiniCourt
from utils.ball_state import FLOOR_LEVEL, IN_FLIGHT


def _make_mini_court() -> MiniCourt:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    return MiniCourt(frame)


def test_floor_level_frames_are_projected_directly():
    mini = _make_mini_court()
    # Use the mini-court's own keypoints as "video" keypoints → near-identity homography,
    # so a ball centered exactly on the source keypoints projects to ~ the same point.
    kp = mini.drawing_key_points
    n = 5

    ball_boxes = [{} for _ in range(n)]
    # Frame 0 and 4: ball sits exactly at keypoint 0 (a known court corner).
    x0, y0 = kp[0], kp[1]
    ball_boxes[0] = {1: [x0 - 5, y0 - 5, x0 + 5, y0 + 5]}
    ball_boxes[4] = {1: [x0 - 5, y0 - 5, x0 + 5, y0 + 5]}

    states = [IN_FLIGHT] * n
    states[0] = FLOOR_LEVEL
    states[4] = FLOOR_LEVEL

    out = mini.convert_ball_to_mini_court_coordinates(ball_boxes, kp, states)

    got_x, got_y = out[0][1]
    assert abs(got_x - x0) < 5
    assert abs(got_y - y0) < 5


def test_in_flight_frames_interpolate_between_anchors():
    mini = _make_mini_court()
    kp = mini.drawing_key_points
    n = 5

    x0, y0 = kp[0], kp[1]
    x1, y1 = kp[2], kp[3]   # a different, known keypoint

    ball_boxes = [{} for _ in range(n)]
    ball_boxes[0] = {1: [x0 - 5, y0 - 5, x0 + 5, y0 + 5]}
    ball_boxes[4] = {1: [x1 - 5, y1 - 5, x1 + 5, y1 + 5]}
    # frames 1,2,3 have no usable ball box - irrelevant, they're in_flight anyway

    states = [FLOOR_LEVEL, IN_FLIGHT, IN_FLIGHT, IN_FLIGHT, FLOOR_LEVEL]

    out = mini.convert_ball_to_mini_court_coordinates(ball_boxes, kp, states)

    ax, ay = out[0][1]
    bx, by = out[4][1]
    mx, my = out[2][1]   # frame 2 is the midpoint (t=0.5)

    assert abs(mx - (ax + bx) / 2) < 2
    assert abs(my - (ay + by) / 2) < 2


def test_holds_at_nearest_anchor_outside_range():
    mini = _make_mini_court()
    kp = mini.drawing_key_points
    n = 5

    x0, y0 = kp[0], kp[1]
    ball_boxes = [{} for _ in range(n)]
    ball_boxes[2] = {1: [x0 - 5, y0 - 5, x0 + 5, y0 + 5]}   # only one anchor, mid-clip

    states = [IN_FLIGHT, IN_FLIGHT, FLOOR_LEVEL, IN_FLIGHT, IN_FLIGHT]
    out = mini.convert_ball_to_mini_court_coordinates(ball_boxes, kp, states)

    # frames before and after the single anchor hold at that anchor's position
    assert out[0][1] == out[2][1]
    assert out[4][1] == out[2][1]


def test_no_anchors_returns_empty_frames():
    mini = _make_mini_court()
    kp = mini.drawing_key_points
    n = 3
    ball_boxes = [{} for _ in range(n)]
    states = [IN_FLIGHT] * n

    out = mini.convert_ball_to_mini_court_coordinates(ball_boxes, kp, states)
    assert all(out[f] == {} for f in range(n))
