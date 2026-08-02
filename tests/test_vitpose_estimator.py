"""
tests/test_vitpose_estimator.py
Unit tests for the parts of ViTPoseEstimator testable without loading the real model
(network + GPU/CPU inference). detect_in_bbox's actual output was validated empirically
against real footage in docs/journal/0021 (wrist estimates on frames 166/167, where
MediaPipe finds nothing at all).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_estimator import UPPER_BODY_LANDMARKS
from utils.vitpose_estimator import ViTPoseEstimator, _LABEL_MAP


def _estimator():
    est = ViTPoseEstimator.__new__(ViTPoseEstimator)
    est._model = None
    est._processor = None
    est.min_score = 0.3
    return est


def test_unavailable_when_model_not_loaded():
    est = _estimator()
    assert est.available is False


def test_detect_in_bbox_returns_none_without_model():
    est = _estimator()
    assert est.detect_in_bbox(frame=object(), bbox=[0, 0, 10, 10]) is None


def test_detect_in_bbox_returns_none_on_missing_frame_or_bbox():
    est = _estimator()
    assert est.detect_in_bbox(frame=None, bbox=[0, 0, 10, 10]) is None
    assert est.detect_in_bbox(frame=object(), bbox=None) is None


def test_label_map_covers_every_landmark_pose_shot_classifier_needs():
    """classify_forehand_backhand only ever looks up the same anatomical names
    PoseEstimator produces -- if this fallback's label map drifts from that set, it
    would silently fail to supply shoulders/wrists even when ViTPose found them."""
    assert set(_LABEL_MAP.values()) == set(UPPER_BODY_LANDMARKS)
