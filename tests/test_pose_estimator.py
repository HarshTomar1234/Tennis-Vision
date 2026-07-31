"""
tests/test_pose_estimator.py
Unit tests for the pure-math part of PoseEstimator — bbox padding and frame-boundary
clamping. detect_in_bbox itself needs a real model + real image data (already validated
empirically against real footage in docs/journal/0013 — the 0.45 padding value below is
measured, not a round-number guess).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_estimator import PoseEstimator


def _estimator():
    # bypass model loading — these tests only exercise _pad_bbox, pure math
    est = PoseEstimator.__new__(PoseEstimator)
    est.bbox_padding = 0.45
    return est


def test_padding_expands_bbox_symmetrically():
    est = _estimator()
    x1, y1, x2, y2 = est._pad_bbox([100.0, 100.0, 200.0, 300.0], frame_w=1280, frame_h=720)
    # width=100 -> pad_x=45, height=200 -> pad_y=90
    assert x1 == 55 and x2 == 245
    assert y1 == 10 and y2 == 390


def test_padding_clamps_to_frame_left_top():
    est = _estimator()
    x1, y1, x2, y2 = est._pad_bbox([10.0, 10.0, 60.0, 60.0], frame_w=1280, frame_h=720)
    assert x1 == 0   # would be negative unclamped
    assert y1 == 0


def test_padding_clamps_to_frame_right_bottom():
    est = _estimator()
    x1, y1, x2, y2 = est._pad_bbox([1200.0, 650.0, 1270.0, 715.0], frame_w=1280, frame_h=720)
    assert x2 == 1280
    assert y2 == 720


def test_zero_padding_returns_original_bbox_as_ints():
    est = _estimator()
    est.bbox_padding = 0.0
    x1, y1, x2, y2 = est._pad_bbox([50.0, 60.0, 150.0, 260.0], frame_w=1280, frame_h=720)
    assert (x1, y1, x2, y2) == (50, 60, 150, 260)


def test_default_padding_is_the_measured_value():
    """0.45 was chosen empirically (recovered a real failed contact frame, didn't
    regress any previously-successful one at 0.15; 0.60 over-shot and did regress one)
    — not a default that should silently drift back to a round number."""
    import inspect
    sig = inspect.signature(PoseEstimator.__init__)
    assert sig.parameters["bbox_padding"].default == 0.45
