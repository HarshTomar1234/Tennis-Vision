"""
tests/test_serve_detector.py
────────────────────────────
Tests for evidence-based serve detection, and for the minimum separation between serves.

The separation rule exists because of a measured failure. Running
eval/serve_false_positive_check.py over the eval suite, one clip reported two serves 18
frames apart, which is 0.6 seconds. A point does not start twice inside a second: even a
fault followed by a second serve leaves time to retrieve a ball and reset. Both candidates
came from the ball sitting above the player across a short window, not from two deliveries.

The rule is expressed in seconds rather than frames, because 18 frames is 0.6s at 30fps
and 0.3s at 60fps, and only one of those readings is about tennis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.serve_detector import MIN_SERVE_SEPARATION_S, detect_serve_frames, is_serve

# Mini-court geometry: far baseline at the top, near baseline at the bottom.
FAR_Y, NEAR_Y = 100.0, 500.0


def _scene(n_frames, contacts, ball_high=True, at_baseline=True):
    """
    Build ball/player/mini-court inputs where every contact frame is a serve or not.

    A serve needs two facts: the ball above the player's head, and the hitter at or
    behind a baseline. Each can be switched off independently to test them separately.
    """
    ball, players, mini = [], [], {}
    for f in range(n_frames):
        player_box = [400.0, 300.0, 460.0, 460.0]     # top edge y=300, height 160
        ball_y = 250.0 if ball_high else 380.0        # above or below the box top
        ball.append({1: [425.0, ball_y - 5, 435.0, ball_y + 5]})
        players.append({7: player_box})
        mini[f] = {7: (300.0, NEAR_Y if at_baseline else 300.0)}
    return ball, players, mini


def test_ball_above_head_at_baseline_is_a_serve():
    ball, players, mini = _scene(60, [10])
    assert is_serve(10, ball, players, mini, FAR_Y, NEAR_Y) is True


def test_ball_below_head_is_not_a_serve():
    ball, players, mini = _scene(60, [10], ball_high=False)
    verdict, reason = is_serve(10, ball, players, mini, FAR_Y, NEAR_Y, explain=True)
    assert verdict is False
    assert "above" in reason


def test_mid_court_contact_above_head_is_a_smash_not_a_serve():
    """The clause that separates a serve from a smash: where the player stands."""
    ball, players, mini = _scene(60, [10], at_baseline=False)
    verdict, reason = is_serve(10, ball, players, mini, FAR_Y, NEAR_Y, explain=True)
    assert verdict is False
    assert "baseline" in reason


def test_two_serves_within_the_minimum_separation_collapse_to_one():
    """The measured failure: two serves 18 frames apart at 30fps."""
    ball, players, mini = _scene(200, [33, 51])
    found = detect_serve_frames([33, 51], ball, players, mini, FAR_Y, NEAR_Y, fps=30.0)
    assert len(found) == 1, f"expected one serve, got {found}"


def test_serves_far_apart_are_both_kept():
    """A fault and a second serve, or two points, are separated by seconds."""
    ball, players, mini = _scene(400, [30, 300])
    found = detect_serve_frames([30, 300], ball, players, mini, FAR_Y, NEAR_Y, fps=30.0)
    assert found == [30, 300]


def test_separation_is_a_duration_not_a_frame_count():
    """
    The same frame gap must be judged differently at different frame rates.

    120 frames is 4s at 30fps (two serves, plausible) and 2s at 60fps (one serve).
    """
    ball, players, mini = _scene(400, [30, 150])
    at_30 = detect_serve_frames([30, 150], ball, players, mini, FAR_Y, NEAR_Y, fps=30.0)
    at_60 = detect_serve_frames([30, 150], ball, players, mini, FAR_Y, NEAR_Y, fps=60.0)
    assert len(at_30) == 2, "4s apart should be two serves"
    assert len(at_60) == 1, "2s apart should be one serve"


def test_cluster_keeps_the_frame_with_the_ball_highest():
    """
    A serve's contact is the peak of the toss, so the highest ball in a cluster is it.

    Frame 40 has the ball well above the player; 45 has it barely clear. Both pass the
    serve test, and the delivery is 40.
    """
    ball, players, mini = _scene(200, [40, 45])
    ball[45] = {1: [425.0, 292.0, 435.0, 298.0]}   # only just above the box top (y=300)
    ball[40] = {1: [425.0, 195.0, 435.0, 205.0]}   # clearly above
    found = detect_serve_frames([40, 45], ball, players, mini, FAR_Y, NEAR_Y, fps=30.0)
    assert found == [40], f"kept the wrong frame: {found}"


def test_no_contacts_gives_no_serves():
    ball, players, mini = _scene(60, [])
    assert detect_serve_frames([], ball, players, mini, FAR_Y, NEAR_Y) == []


def test_single_serve_passes_through_unchanged():
    ball, players, mini = _scene(60, [20])
    assert detect_serve_frames([20], ball, players, mini, FAR_Y, NEAR_Y) == [20]


def test_missing_mini_court_position_refuses_rather_than_guesses():
    """Without a trusted court there is no baseline test, so no serve may be claimed."""
    ball, players, _ = _scene(60, [10])
    verdict, reason = is_serve(10, ball, players, {}, FAR_Y, NEAR_Y, explain=True)
    assert verdict is False
    assert "mini-court" in reason


def test_separation_constant_is_physically_sensible():
    """A guard on the constant itself: seconds, and long enough to mean a new point."""
    assert 1.0 <= MIN_SERVE_SEPARATION_S <= 10.0
