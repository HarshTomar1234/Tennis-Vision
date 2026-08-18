"""
tests/test_swing_candidates.py
──────────────────────────────
Tests for the body-based contact candidate generator.

This generator exists to reach the 15.4% of real contacts that no ball-trajectory
generator proposes, so the properties that matter are: it fires on a swing, it stays
silent on ordinary movement, it produces one candidate per swing rather than one per
fast frame, and it degrades quietly when pose is missing rather than raising.

The normalisation is worth a test of its own. The same real swing at the far end of the
court covers far fewer pixels, and if the score were raw pixel speed one threshold could
not serve both players.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.swing_candidates import (
    MIN_SWING_SPEED,
    detect_swing_candidates,
    swing_score,
)

JOINTS = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
          "LEFT_WRIST", "RIGHT_WRIST")


def _pose(x_offsets: dict, base=(100.0, 100.0)):
    """One frame of landmarks, each joint displaced from base by its offset."""
    return {j: (base[0] + x_offsets.get(j, 0.0), base[1]) for j in JOINTS}


def _still_track(n, height=200.0):
    """A player standing still: every joint fixed."""
    return [_pose({}) for _ in range(n)], [height] * n


def _swing_track(n, peak_frame, amplitude, height=200.0):
    """
    A player whose right arm accelerates to `peak_frame` and decelerates after.

    Displacement is triangular in time, so speed is roughly constant either side of the
    peak and the turning point sits at peak_frame, which is what the detector looks for.
    """
    track = []
    for f in range(n):
        distance = amplitude * max(0.0, 1.0 - abs(f - peak_frame) / 4.0)
        track.append(_pose({
            "RIGHT_WRIST": distance,
            "RIGHT_ELBOW": distance * 0.6,
            "RIGHT_SHOULDER": distance * 0.3,
        }))
    return track, [height] * n


def test_still_player_produces_no_candidates():
    track, heights = _still_track(40)
    assert detect_swing_candidates(track, heights) == []


def test_swing_is_detected_near_its_peak():
    track, heights = _swing_track(40, peak_frame=20, amplitude=60.0)
    found = detect_swing_candidates(track, heights)
    assert found, "no candidate proposed for a clear swing"
    assert min(abs(f - 20) for f in found) <= 3, f"peak missed, got {found}"


def test_one_candidate_per_swing_not_one_per_fast_frame():
    """A swing holds high speed for several frames; only its peak should be proposed."""
    track, heights = _swing_track(40, peak_frame=20, amplitude=60.0)
    assert len(detect_swing_candidates(track, heights)) == 1


def test_two_separated_swings_give_two_candidates():
    track_a, heights = _swing_track(60, peak_frame=15, amplitude=60.0)
    track_b, _ = _swing_track(60, peak_frame=45, amplitude=60.0)
    # Splice: first swing in the first half, second in the second half.
    track = track_a[:30] + track_b[30:]
    found = detect_swing_candidates(track, heights)
    assert len(found) == 2, f"expected two swings, got {found}"


def test_score_is_normalised_by_player_height():
    """
    The same swing, half the pixel size, at half the player height, must score the same.

    This is what lets one threshold serve the near and far player. Without it the far
    player's strokes fall below any threshold tuned on the near one.
    """
    near_track, near_h = _swing_track(40, peak_frame=20, amplitude=60.0, height=200.0)
    far_track, far_h = _swing_track(40, peak_frame=20, amplitude=30.0, height=100.0)

    near = swing_score(near_track, near_h, 20)
    far = swing_score(far_track, far_h, 20)
    assert near is not None and far is not None
    assert abs(near - far) < 1e-6, f"normalisation failed: near={near}, far={far}"


def test_far_player_swing_is_still_detected():
    """The behavioural consequence of the normalisation, stated separately."""
    track, heights = _swing_track(40, peak_frame=20, amplitude=30.0, height=100.0)
    assert detect_swing_candidates(track, heights), "far player's swing was missed"


def test_missing_pose_frames_do_not_raise():
    track, heights = _swing_track(40, peak_frame=20, amplitude=60.0)
    track[18] = None
    track[19] = None
    assert isinstance(detect_swing_candidates(track, heights), list)


def test_all_pose_missing_returns_empty():
    assert detect_swing_candidates([None] * 30, [200.0] * 30) == []


def test_zero_height_is_rejected_rather_than_dividing_by_zero():
    track, _ = _swing_track(30, peak_frame=15, amplitude=60.0)
    assert swing_score(track, [0.0] * 30, 15) is None
    assert detect_swing_candidates(track, [0.0] * 30) == []


def test_empty_input():
    assert detect_swing_candidates([], []) == []


def test_slow_movement_stays_below_the_threshold():
    """Running between shots must not be proposed as a stroke."""
    track, heights = _swing_track(40, peak_frame=20, amplitude=2.0)
    score = swing_score(track, heights, 20)
    assert score is not None
    assert score < MIN_SWING_SPEED, f"walking pace scored {score}"
