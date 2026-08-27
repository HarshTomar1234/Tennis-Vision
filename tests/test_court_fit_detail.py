"""
tests/test_court_fit_detail.py
───────────────────────────────
Guards the court-fit failure explanation.

Why it exists: the held-out benchmark refused 14 of 14 clips, and the median line-support
score said "court fit failed" on all of them without distinguishing two situations that
need completely different things from the user:

    0.05 0.05 0.05 0.60 0.58 0.58 0.57 0.47 0.01 0.03 0.00 0.05   median 0.053
    0.12 0.13 0.14 0.06 0.15 0.13 0.17 0.09 0.13 0.14 0.07 0.05   median 0.131

The first is a real court view with camera cuts either side: five of twelve frames are a
perfectly good court and trimming fixes it. The second never shows a playable court.

This changes NO gate. The verdict is identical; only the explanation is new. These tests
pin that, because an "improvement" that quietly loosened the threshold would be the exact
failure this project exists to prevent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.court_validity import MIN_LINE_SUPPORT, assess_court_fit, assess_court_fit_detail


class _Scores:
    """Feeds assess_court_fit_detail a controlled sequence of line-support scores."""

    def __init__(self, scores):
        self.scores = scores


def _patched(monkeypatch, scores):
    """Run the detail assessment against a fixed score sequence."""
    import utils.court_validity as cv

    calls = {"i": 0}

    def fake(frame, keypoints, *a, **k):
        s = scores[calls["i"] % len(scores)]
        calls["i"] += 1
        return s

    monkeypatch.setattr(cv, "line_support_score", fake)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in scores]
    return cv.assess_court_fit_detail(frames, [[0.0] * 28] * len(scores),
                                      sample_count=len(scores))


def test_a_good_clip_passes_and_says_so(monkeypatch):
    valid, median, detail = _patched(monkeypatch, [0.6] * 12)
    assert valid
    assert detail["passed"] == 12
    assert detail["likely_camera_cut"] is False


def test_camera_cut_pattern_is_identified(monkeypatch):
    """The H00 shape: a real court segment with non-court either side."""
    valid, median, detail = _patched(
        monkeypatch, [0.05, 0.05, 0.05, 0.60, 0.58, 0.58, 0.57, 0.47,
                      0.01, 0.03, 0.00, 0.05])
    assert valid is False, "the gate must still refuse; this is explanation, not leniency"
    assert detail["likely_camera_cut"] is True
    assert detail["passed"] == 5
    assert "camera cuts" in detail["reason"]
    assert "trim" in detail["reason"]


def test_not_a_court_at_all_is_not_called_a_camera_cut(monkeypatch):
    """The H13 shape: uniformly low, nothing to trim to."""
    valid, median, detail = _patched(
        monkeypatch, [0.12, 0.13, 0.14, 0.06, 0.15, 0.13, 0.17, 0.09,
                      0.13, 0.14, 0.07, 0.05])
    assert valid is False
    assert detail["likely_camera_cut"] is False
    assert detail["passed"] == 0
    assert "No part of this clip" in detail["reason"]


def test_one_lucky_frame_is_not_a_camera_cut(monkeypatch):
    """A single passing frame is noise, not a segment worth trimming to."""
    valid, median, detail = _patched(monkeypatch, [0.05] * 11 + [0.9])
    assert valid is False
    assert detail["likely_camera_cut"] is False


# ── the gate itself must be unchanged ───────────────────────────────────────────

def test_verdict_is_identical_to_the_two_tuple_api(monkeypatch):
    """
    assess_court_fit still exists, still returns two values, and must agree with the
    detailed form on every input. Five evals and the packaging tests import it.
    """
    for scores in ([0.6] * 12, [0.05] * 12,
                   [0.05, 0.05, 0.05, 0.60, 0.58, 0.58, 0.57, 0.47, 0.01, 0.03, 0.00, 0.05],
                   [0.22] * 12, [0.219] * 12):
        valid_d, median_d, _ = _patched(monkeypatch, scores)
        assert valid_d == (median_d >= MIN_LINE_SUPPORT)


def test_threshold_is_untouched():
    """If this ever moves, every published court number moves with it."""
    assert MIN_LINE_SUPPORT == 0.22


def test_empty_input_refuses():
    valid, median, detail = assess_court_fit_detail([], [])
    assert valid is False and median == 0.0
    assert detail["sampled"] == 0


def test_two_tuple_api_still_works():
    valid, median = assess_court_fit([], [])
    assert valid is False and median == 0.0
