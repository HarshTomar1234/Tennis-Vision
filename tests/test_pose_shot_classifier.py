"""
tests/test_pose_shot_classifier.py

Geometry tests for pose-based forehand/backhand. Landmarks are (x_px, y_px, z_px);
the body axis is built in the horizontal (x, z) plane.

The three that matter most are the left-handed, facing-away, and side-on cases: those are
exactly where simpler rules get the answer backwards or become numerically unusable.

Coordinate convention (image space, y grows downward; z is depth, negative = toward camera):
  - Facing the CAMERA: the player's anatomical LEFT shoulder is at a LARGER image x.
  - Facing AWAY: the reverse.
  - SIDE-ON: shoulders share almost the same image x and separate in z instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pose_shot_classifier import classify_forehand_backhand, FOREHAND, BACKHAND


# ── Facing the camera ────────────────────────────────────────────────────────

def test_right_handed_forehand_facing_camera():
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
        "RIGHT_WRIST":    (60.0, 120.0, 0.0),    # extended on the player's own right
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(50.0, 120.0))
    assert result is not None
    assert result[0] == FOREHAND


def test_right_handed_backhand_facing_camera():
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
        "RIGHT_WRIST":    (240.0, 120.0, 0.0),   # crossed to the opposite side
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(250.0, 120.0))
    assert result is not None
    assert result[0] == BACKHAND


def test_left_handed_forehand_is_the_mirror_not_a_backhand():
    """A left-hander's forehand is on the opposite image side from a right-hander's.
    A handedness-hardcoded rule calls this a backhand; this one must not."""
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
        "LEFT_WRIST":     (240.0, 120.0, 0.0),   # extended on the player's own left
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(250.0, 120.0))
    assert result is not None
    assert result[0] == FOREHAND


# ── Facing away - the case image-space rules get wrong ───────────────────────

def test_right_handed_forehand_facing_away_is_still_a_forehand():
    """Same shot as facing the camera, but the player has turned around, so the hitting
    wrist is now on the OPPOSITE side of the image. The label must not flip."""
    landmarks = {
        "LEFT_SHOULDER":  (100.0, 100.0, 0.0),   # facing away -> anatomical left on image left
        "RIGHT_SHOULDER": (200.0, 100.0, 0.0),
        "RIGHT_WRIST":    (240.0, 120.0, 0.0),   # own right side, now image-right
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(250.0, 120.0))
    assert result is not None
    assert result[0] == FOREHAND


def test_right_handed_backhand_facing_away():
    landmarks = {
        "LEFT_SHOULDER":  (100.0, 100.0, 0.0),
        "RIGHT_SHOULDER": (200.0, 100.0, 0.0),
        "RIGHT_WRIST":    (60.0, 120.0, 0.0),    # crossed over the midline
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(50.0, 120.0))
    assert result is not None
    assert result[0] == BACKHAND


# ── Side-on - the stance players actually hit from ───────────────────────────

def test_side_on_forehand_uses_depth_not_image_x():
    """A side-on player's shoulders are ~3px apart in image x - unusable in 2-D - but
    well separated in depth. The horizontal (x, z) axis must still resolve this."""
    landmarks = {
        "RIGHT_SHOULDER": (150.0, 100.0, -30.0),   # nearer the camera
        "LEFT_SHOULDER":  (153.0, 100.0,  30.0),   # further away
        "RIGHT_WRIST":    (160.0, 120.0, -50.0),   # extended on the player's own right
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(165.0, 120.0))
    assert result is not None
    assert result[0] == FOREHAND


def test_side_on_backhand_uses_depth_not_image_x():
    landmarks = {
        "RIGHT_SHOULDER": (150.0, 100.0, -30.0),
        "LEFT_SHOULDER":  (153.0, 100.0,  30.0),
        "RIGHT_WRIST":    (160.0, 120.0,  55.0),   # crossed past the midline in depth
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(165.0, 120.0))
    assert result is not None
    assert result[0] == BACKHAND


# ── Contact validation ───────────────────────────────────────────────────────

def test_rejects_frame_when_wrist_is_far_from_the_ball():
    """A wrist ~300px from the ball is not hitting it. On the reference clip this is what
    separated a wrongly-flagged ready-stance frame (289px) from real contacts (31-71px)."""
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
        "RIGHT_WRIST":    (60.0, 120.0, 0.0),
    }
    assert classify_forehand_backhand(
        landmarks, ball_pos=(400.0, 300.0), max_contact_distance=120.0
    ) is None
    # the same pose with the ball actually nearby is still classified
    assert classify_forehand_backhand(
        landmarks, ball_pos=(50.0, 120.0), max_contact_distance=120.0
    ) is not None


# ── Honest refusals ──────────────────────────────────────────────────────────

def test_wrists_together_and_equidistant_is_a_two_handed_backhand():
    """
    This case used to return None, and that was the bug.

    The wrists are 20 apart on a 100-wide shoulder axis, which is a fifth of a shoulder
    width, roughly 8cm on a real body. Two hands on a racket grip sit about that far
    apart, so this is a two-handed stroke rather than an unreadable one. Discarding it
    threw away the commonest stroke in tennis and left the pipeline reporting 7
    forehands to 1 backhand on the reference clip.

    The genuinely unreadable case, two arms spread wide that happen to be equidistant,
    is covered by test_wrists_far_apart_but_equidistant_is_still_unreadable.
    """
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
        "LEFT_WRIST":     (160.0, 200.0, 0.0),
        "RIGHT_WRIST":    (140.0, 200.0, 0.0),
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(150.0, 210.0))
    assert result is not None
    assert result[0] == BACKHAND
    assert result[1] > 0.0, "hands this close should carry real confidence"


def test_returns_none_without_both_shoulders():
    landmarks = {"LEFT_SHOULDER": (200.0, 100.0, 0.0), "RIGHT_WRIST": (60.0, 120.0, 0.0)}
    assert classify_forehand_backhand(landmarks, ball_pos=(50.0, 120.0)) is None


def test_returns_none_without_any_wrist():
    landmarks = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
    }
    assert classify_forehand_backhand(landmarks, ball_pos=(50.0, 120.0)) is None


def test_returns_none_on_empty_input():
    assert classify_forehand_backhand(None, (50.0, 120.0)) is None
    assert classify_forehand_backhand({}, (50.0, 120.0)) is None
    assert classify_forehand_backhand({"LEFT_SHOULDER": (1.0, 1.0, 0.0)}, None) is None


def test_returns_none_on_fully_collapsed_shoulders():
    """Shoulders collapsed in BOTH x and z - genuinely no body axis to project onto."""
    landmarks = {
        "RIGHT_SHOULDER": (150.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (150.0, 100.0, 0.0),
        "RIGHT_WRIST":    (60.0, 120.0, 0.0),
    }
    assert classify_forehand_backhand(landmarks, ball_pos=(50.0, 120.0)) is None


# ── Confidence behaviour ─────────────────────────────────────────────────────

def test_confidence_is_higher_for_a_fully_extended_arm():
    shoulders = {
        "RIGHT_SHOULDER": (100.0, 100.0, 0.0),
        "LEFT_SHOULDER":  (200.0, 100.0, 0.0),
    }
    far  = classify_forehand_backhand({**shoulders, "RIGHT_WRIST": (20.0, 120.0, 0.0)},
                                       ball_pos=(10.0, 120.0))
    near = classify_forehand_backhand({**shoulders, "RIGHT_WRIST": (140.0, 120.0, 0.0)},
                                       ball_pos=(130.0, 120.0))

    assert far is not None and near is not None
    assert far[0] == near[0] == FOREHAND
    assert far[1] > near[1]
    assert 0.0 <= near[1] <= 1.0 and 0.0 <= far[1] <= 1.0


# ── two-handed strokes ────────────────────────────────────────────────────────
# Both wrists near the ball used to return None. That discarded the commonest stroke in
# tennis: a two-handed backhand puts both hands on the grip, so both wrists are near the
# ball by construction. On the reference clip the pipeline reported 7 forehands to 1
# backhand, which is not a plausible rally.

def _two_handed_scene(grip_x, gap=6.0):
    """
    Landmarks for a two-handed stroke with the grip at `grip_x`.

    Shoulders span x=-40..40 at z=0, so the body axis is 80 wide and both wrists sit
    `gap` apart, well inside TWO_HANDED_MAX_GAP.
    """
    return {
        "LEFT_SHOULDER":  (40.0, 100.0, 0.0),
        "RIGHT_SHOULDER": (-40.0, 100.0, 0.0),
        "LEFT_WRIST":     (grip_x + gap / 2, 120.0, 0.0),
        "RIGHT_WRIST":    (grip_x - gap / 2, 120.0, 0.0),
    }


def test_hands_together_on_the_grip_is_called_a_backhand():
    """Two-handed forehands barely exist; hands together is evidence of a backhand."""
    landmarks = _two_handed_scene(grip_x=60.0)
    result = classify_forehand_backhand(landmarks, ball_pos=(60.0, 120.0))
    assert result is not None, "two-handed stroke was discarded as ambiguous"
    assert result[0] == BACKHAND


def test_hands_together_on_the_other_side_is_still_a_backhand():
    """The call must not depend on which side of the body the grip is on."""
    result = classify_forehand_backhand(_two_handed_scene(grip_x=-60.0),
                                        ball_pos=(-60.0, 120.0))
    assert result is not None
    assert result[0] == BACKHAND


def test_wrists_far_apart_but_equidistant_is_still_unreadable():
    """
    The case the ambiguity guard was actually written for.

    Two arms spread wide, coincidentally the same distance from the ball, carry no
    information about which one is hitting. That must still return None.
    """
    landmarks = {
        "LEFT_SHOULDER":  (40.0, 100.0, 0.0),
        "RIGHT_SHOULDER": (-40.0, 100.0, 0.0),
        "LEFT_WRIST":     (90.0, 120.0, 0.0),
        "RIGHT_WRIST":    (-90.0, 120.0, 0.0),
    }
    assert classify_forehand_backhand(landmarks, ball_pos=(0.0, 120.0)) is None


def test_one_handed_stroke_is_unaffected_by_the_two_handed_path():
    """A clear single-arm stroke must still be judged by which side it came off."""
    landmarks = {
        "LEFT_SHOULDER":  (40.0, 100.0, 0.0),
        "RIGHT_SHOULDER": (-40.0, 100.0, 0.0),
        "LEFT_WRIST":     (100.0, 120.0, 0.0),
        "RIGHT_WRIST":    (-30.0, 105.0, 0.0),
    }
    result = classify_forehand_backhand(landmarks, ball_pos=(100.0, 120.0))
    assert result is not None
    assert result[0] == FOREHAND


def test_two_handed_result_carries_a_confidence():
    result = classify_forehand_backhand(_two_handed_scene(grip_x=60.0),
                                        ball_pos=(60.0, 120.0))
    assert result is not None
    assert 0.0 <= result[1] <= 1.0
