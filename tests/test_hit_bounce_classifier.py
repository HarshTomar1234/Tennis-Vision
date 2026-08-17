"""
tests/test_hit_bounce_classifier.py
Tests for the trajectory-based hit/bounce classifier: feature extraction (pure math,
no trained weights needed) and classification (both against a synthetic weights file,
to isolate the logistic-regression math from the real trained coefficients, and against
the real shipped weights, as a sanity check on unambiguous examples).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ball_state import CONTACT, BOUNCE
from utils.hit_bounce_classifier import compute_event_features, classify_hit_or_bounce


# ── compute_event_features ───────────────────────────────────────────────────

def test_computes_velocity_change_around_event():
    # ball moving steadily in +x before the event, reverses to -x after (hit-like)
    positions = [
        (0.0, 100.0), (10.0, 100.0), (20.0, 100.0), (30.0, 100.0),
        (40.0, 100.0),                                              # event at index 4
        (30.0, 100.0), (20.0, 100.0), (10.0, 100.0), (0.0, 100.0),
    ]
    feats = compute_event_features(positions, event_frame=4, window=4)
    assert feats is not None
    assert feats["height_y"] == 100.0
    assert feats["vx_change_mag"] > 15   # velocity flipped from +10 to -10 px/frame
    assert feats["vy_change_mag"] == 0.0


def test_none_when_event_frame_has_no_detection():
    positions = [(0.0, 0.0), None, (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    assert compute_event_features(positions, event_frame=1, window=2) is None


def test_none_when_out_of_range():
    positions = [(0.0, 0.0)] * 5
    assert compute_event_features(positions, event_frame=-1, window=2) is None
    assert compute_event_features(positions, event_frame=99, window=2) is None


def test_none_when_not_enough_context_before():
    # only 1 valid point before the event, window needs >= 2
    positions = [None, (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    assert compute_event_features(positions, event_frame=1, window=4) is None


def test_none_when_not_enough_context_after():
    positions = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), None]
    assert compute_event_features(positions, event_frame=3, window=4) is None


def test_handles_gaps_in_context_window():
    # a missing frame inside the window should just be skipped, not break extraction
    positions = [
        (0.0, 50.0), None, (10.0, 50.0), (15.0, 50.0),
        (20.0, 50.0),
        (25.0, 50.0), None, (30.0, 50.0), (35.0, 50.0),
    ]
    feats = compute_event_features(positions, event_frame=4, window=4)
    assert feats is not None
    assert feats["height_y"] == 50.0


# ── classify_hit_or_bounce - logistic math, isolated with a synthetic model ──

def _write_synthetic_weights(tmpdir) -> str:
    """A hand-built model: only vx_change_mag matters, threshold ~ around 10."""
    path = os.path.join(tmpdir, "weights.json")
    weights = {
        "feature_names": ["height_y", "vy_change_mag", "vx_change_mag"],
        "mu":    [300.0, 20.0, 10.0],
        "sigma": [100.0, 10.0, 10.0],
        "w":     [0.0, 0.0, 5.0],     # only vx_change_mag drives the decision
        "b":     0.0,
    }
    with open(path, "w") as f:
        json.dump(weights, f)
    return path


def test_classifies_obvious_hit_with_synthetic_model():
    with tempfile.TemporaryDirectory() as d:
        path = _write_synthetic_weights(d)
        # vx_change_mag well above mu -> positive z -> CONTACT
        result = classify_hit_or_bounce(
            {"height_y": 300.0, "vy_change_mag": 20.0, "vx_change_mag": 30.0},
            weights_path=path,
        )
    assert result is not None
    label, prob = result
    assert label == CONTACT
    assert prob > 0.5


def test_classifies_obvious_bounce_with_synthetic_model():
    with tempfile.TemporaryDirectory() as d:
        path = _write_synthetic_weights(d)
        # vx_change_mag well below mu -> negative z -> BOUNCE
        result = classify_hit_or_bounce(
            {"height_y": 300.0, "vy_change_mag": 20.0, "vx_change_mag": 0.5},
            weights_path=path,
        )
    assert result is not None
    label, prob = result
    assert label == BOUNCE
    assert prob > 0.5


def test_returns_none_for_none_features():
    assert classify_hit_or_bounce(None) is None


def test_returns_none_when_weights_file_missing():
    assert classify_hit_or_bounce(
        {"height_y": 1, "vy_change_mag": 1, "vx_change_mag": 1},
        weights_path="does/not/exist.json",
    ) is None


# ── classify_hit_or_bounce - sanity check against the real shipped model ─────

REAL_WEIGHTS = "models/hit_bounce_classifier.json"


def test_real_model_classifies_unambiguous_hit():
    """Large horizontal redirection, moderate height - matches the measured hit profile
    (mean |vx change| ~19px vs bounce's ~3px)."""
    if not os.path.exists(REAL_WEIGHTS):
        return   # not trained in this environment - skip rather than fail the suite
    result = classify_hit_or_bounce(
        # vx_sign_flip=1: a 40px/frame horizontal redirect is a racket reversing the
        # ball, which is the profile 71.8% of real hits show.
        {"height_y": 250.0, "vy_change_mag": 20.0, "vx_change_mag": 40.0,
         "vx_sign_flip": 1.0}
    )
    assert result is not None
    assert result[0] == CONTACT


def test_real_model_classifies_unambiguous_bounce():
    """Near-zero horizontal velocity change - matches the measured bounce profile
    (mean |vx change| ~3px, std ~4px, i.e. almost always small)."""
    if not os.path.exists(REAL_WEIGHTS):
        return
    result = classify_hit_or_bounce(
        # vx_sign_flip=0: the floor preserves horizontal direction, which is why only
        # 2.1% of real bounces flip it.
        {"height_y": 330.0, "vy_change_mag": 18.0, "vx_change_mag": 0.5,
         "vx_sign_flip": 0.0}
    )
    assert result is not None
    assert result[0] == BOUNCE


def test_computed_features_cover_everything_the_weights_expect():
    """
    compute_event_features() must produce every feature the trained model consumes.

    These two drifted apart when the model gained a feature the inference path did not
    compute, and the symptom was a bare KeyError from deep inside classification. This
    fails loudly at the contract instead, and catches the more dangerous direction too:
    a retrain that adds a feature nobody wired into the pipeline.
    """
    import json

    if not os.path.exists(REAL_WEIGHTS):
        return

    # A straightforward descending-then-ascending trajectory: enough clean context on
    # both sides for every velocity feature to be computable.
    positions = [(float(x), float(100 + abs(x - 10) * 5)) for x in range(20)]
    features = compute_event_features(positions, event_frame=10)
    assert features is not None

    expected = json.loads(Path(REAL_WEIGHTS).read_text())["feature_names"]
    missing = [name for name in expected if name not in features]
    assert not missing, f"compute_event_features is missing trained features: {missing}"


def test_missing_feature_raises_a_diagnosable_error():
    """A features/weights mismatch is a bug, and must not be silently defaulted to 0."""
    if not os.path.exists(REAL_WEIGHTS):
        return
    with pytest.raises(ValueError, match="out of sync"):
        classify_hit_or_bounce({"height_y": 250.0})
