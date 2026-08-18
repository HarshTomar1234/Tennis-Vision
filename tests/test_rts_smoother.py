"""
tests/test_rts_smoother.py
──────────────────────────
Tests for the fixed-interval (Rauch-Tung-Striebel) trajectory smoother.

The smoother is not enabled in the pipeline (see the measurement table in
utils/kalman_smoother.py), but it is exercised by
eval/ball_localization_accuracy.py --smooth and is the foundation for the gated
version described there, so its maths needs to be correct and its edge cases safe.

The property that matters, and the one a forward-only filter cannot have, is that the
estimate at frame k is informed by frames AFTER k. That is what the gap tests check.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.kalman_smoother import rts_smooth


def test_empty_input():
    assert rts_smooth([]) == []


def test_single_measurement_passes_through_with_zero_velocity():
    """One frame gives a smoother nothing to work with; it must not invent motion."""
    out = rts_smooth([(5.0, 7.0)])
    assert len(out) == 1
    assert out[0][0] == 5.0 and out[0][1] == 7.0
    assert out[0][2] == 0.0 and out[0][3] == 0.0


def test_all_missing_measurements_does_not_crash():
    """A clip where the ball is never detected must return a defined result, not raise."""
    out = rts_smooth([None] * 6)
    assert len(out) == 6
    assert all(math.isfinite(v) for frame in out for v in frame)


def test_constant_velocity_line_is_recovered():
    """With noiseless measurements on a straight line, the smoother should reproduce it."""
    truth = [(float(i), 2.0 * i) for i in range(15)]
    out = rts_smooth(truth)
    for i, (x, y) in enumerate(truth):
        assert abs(out[i][0] - x) < 0.5, f"frame {i} x drifted"
        assert abs(out[i][1] - y) < 0.5, f"frame {i} y drifted"


def test_velocity_estimate_matches_the_trajectory():
    """The state carries velocity, and it should match the real per-frame step."""
    out = rts_smooth([(3.0 * i, -1.0 * i) for i in range(15)])
    mid = out[7]
    assert abs(mid[2] - 3.0) < 0.3, f"vx was {mid[2]}"
    assert abs(mid[3] + 1.0) < 0.3, f"vy was {mid[3]}"


def test_gap_is_bridged_using_frames_after_the_gap():
    """
    The whole point of the backward pass.

    A forward-only filter entering a gap can only coast at its last velocity. Here the
    trajectory turns during the gap, so an estimate that used only the past would
    overshoot along the original heading. Being close to the truth in the middle of the
    gap is only possible using frames from after it.
    """
    truth = [(float(i), 0.0) for i in range(10)] + [(9.0, float(i)) for i in range(1, 11)]
    measurements = [None if 8 <= i <= 12 else truth[i] for i in range(len(truth))]

    out = rts_smooth(measurements)
    gap_mid = out[10]
    # Coasting on the pre-gap heading would put x near 10-11 and y at 0.
    assert gap_mid[1] > 0.5, f"backward pass did not inform the gap (y={gap_mid[1]})"


def test_gap_output_covers_every_frame():
    """Coverage is the measured benefit: a position on every frame, gaps included."""
    measurements = [(float(i), float(i)) if i % 3 == 0 else None for i in range(12)]
    out = rts_smooth(measurements)
    assert len(out) == 12
    assert all(math.isfinite(v) for frame in out for v in frame)


def test_leading_gap_does_not_drag_the_estimate_from_the_origin():
    """
    Initialisation starts at the first real measurement, not at (0, 0).

    Without that, a clip whose ball appears late spends its first frames being pulled in
    from the origin, which is a large error on exactly the frames a viewer sees first.
    """
    out = rts_smooth([None, None, None] + [(500.0 + i, 300.0) for i in range(10)])
    assert out[0][0] > 400.0, f"leading frames were dragged toward the origin: {out[0]}"


def test_higher_process_noise_follows_measurements_more_closely():
    """
    The tuning knob must behave monotonically, since the eval sweeps it.

    A step in the data is followed more faithfully when the motion model is trusted less.
    """
    step = [(0.0, 0.0)] * 8 + [(50.0, 0.0)] * 8
    loose = rts_smooth(step, process_noise=1.0, measurement_noise=1.0)
    tight = rts_smooth(step, process_noise=1e-4, measurement_noise=1.0)
    assert abs(loose[-1][0] - 50.0) < abs(tight[-1][0] - 50.0)
