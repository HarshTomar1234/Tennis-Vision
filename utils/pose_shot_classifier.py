"""
utils/pose_shot_classifier.py
─────────────────────────────
Forehand vs backhand from body geometry, replacing the position-only guess.

The distinction, physically
---------------------------
A forehand and a backhand differ by exactly one thing: whether the hitting arm has
crossed the body's midline. On a forehand the hitting wrist stays on its own anatomical
side of the torso; on a backhand it crosses to the other side. Two-handed backhands still
cross (both hands travel together), so the rule holds for them too.

Three traps the naive version falls into
----------------------------------------
1. Handedness - a left-hander's forehand mirrors a right-hander's, so any rule hardcoding
   "right hand = forehand" is wrong half the time.
2. Facing direction - a player facing the camera has their anatomical left on the image's
   right, and the reverse when facing away. Image left/right flips meaning mid-rally.
3. **Side-on collapse** - a player turns side-on to hit, which is precisely when the
   shoulder axis shrinks to almost nothing in image space. Measured on the reference
   clip: shoulder width fell to 2.8-8.5px on real groundstrokes (vs 17-23px in ready
   stance), where a 2-3px landmark error flips the answer. A 2-D image-plane rule is
   therefore least reliable exactly when it is being asked the real question.

Traps 1 and 2 are avoided by working in body-relative coordinates and letting handedness
emerge from which wrist is hitting. Trap 3 is avoided by building the body axis in the
horizontal (x, z) plane using MediaPipe's depth: on the same side-on frame where shoulder
dx was 0.037, dz was 0.607. x and z are complementary - as one collapses the other grows -
so the horizontal-plane axis stays well-conditioned at every orientation. This is also the
physically correct plane: forehand vs backhand is a horizontal rotation question.
"""
from __future__ import annotations

FOREHAND = "Forehand"
BACKHAND = "Backhand"

# Maximum separation between the two wrists, as a fraction of shoulder-axis length, for
# them to count as both being on the grip. Two hands on a racket handle sit roughly a
# tenth of a shoulder width apart; anything wider is two arms that merely happen to be
# equidistant from the ball.
TWO_HANDED_MAX_GAP = 0.45

Landmark = tuple[float, float, float]   # (x_px, y_px, z_px)


def _image_dist(a: Landmark, b_xy: tuple[float, float]) -> float:
    """Distance in image space only - the ball has no depth estimate to compare against."""
    return ((a[0] - b_xy[0]) ** 2 + (a[1] - b_xy[1]) ** 2) ** 0.5


def classify_forehand_backhand(
    landmarks: dict[str, Landmark] | None,
    ball_pos: tuple[float, float] | None,
    ambiguity_ratio: float = 0.25,
    max_contact_distance: float | None = None,
) -> tuple[str, float] | None:
    """
    Decide forehand vs backhand from pose landmarks at a contact frame.

    Args:
        landmarks: from PoseEstimator.detect_in_bbox - (x_px, y_px, z_px) per name.
                   Needs both shoulders and at least one wrist.
        ball_pos:  (x, y) ball centre in image space, used to pick the hitting hand.
        ambiguity_ratio: if both wrists are within this fraction of the body-axis length
                   of each other in ball-distance, the hitting hand is genuinely unclear
                   (two-handed shot, overlapping arms) - return None rather than guess.
        max_contact_distance: if set, reject the frame when the hitting wrist is further
                   than this (in pixels) from the ball. A wrist that far away is not
                   hitting anything, so there is no shot to label. On the reference clip
                   real contacts sat 31-71px away while a ready-stance frame that the
                   trajectory detector had wrongly flagged as a shot sat at 289px, so
                   this doubles as a contact-frame validator.

    Returns:
        (label, confidence 0-1), or None when the geometry cannot be determined.
        Confidence scales with how far the wrist is from the midline relative to the
        body axis length - an arm barely off-centre is a weak signal and says so.

        Returning None is deliberate. The point of this module is to replace a
        confident-sounding guess with either a real answer or an honest absence.
    """
    if not landmarks or ball_pos is None:
        return None

    left_sh  = landmarks.get("LEFT_SHOULDER")
    right_sh = landmarks.get("RIGHT_SHOULDER")
    if left_sh is None or right_sh is None:
        return None

    wrists = {
        name: landmarks[name]
        for name in ("LEFT_WRIST", "RIGHT_WRIST")
        if name in landmarks
    }
    if not wrists:
        return None

    # Body axis in the HORIZONTAL (x, z) plane: from the player's right shoulder to
    # their left. Using z alongside x keeps this well-conditioned when the player turns
    # side-on, which is exactly when a hit happens.
    axis = (left_sh[0] - right_sh[0], left_sh[2] - right_sh[2])
    axis_len = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
    if axis_len < 1e-6:
        return None   # landmarks collapsed entirely - no usable body axis

    # The hitting hand is the wrist nearest the ball at contact (image space - the ball
    # has no depth estimate).
    hitting_name, hitting_pos = min(
        wrists.items(), key=lambda kv: _image_dist(kv[1], ball_pos)
    )

    # Contact validation: a wrist too far from the ball means this frame is not a hit.
    if max_contact_distance is not None:
        if _image_dist(hitting_pos, ball_pos) > max_contact_distance:
            return None

    # Both wrists similarly close to the ball has two very different causes, and
    # collapsing them into "unknown" was throwing away the commonest stroke in tennis.
    two_handed = False
    if len(wrists) == 2:
        d_left  = _image_dist(wrists["LEFT_WRIST"],  ball_pos)
        d_right = _image_dist(wrists["RIGHT_WRIST"], ball_pos)
        if abs(d_left - d_right) < ambiguity_ratio * axis_len:
            # Case 1: the hands are TOGETHER on the grip. That is a two-handed stroke,
            # not an absence of evidence, and the pair of wrists is the hitting hand.
            # Case 2: the hands are far apart and happen to be equidistant from the ball
            # (camera angle, landmark error). Nothing can be read from that.
            #
            # The wrist separation tells them apart: two hands on a grip sit a few
            # centimetres apart, which is a small fraction of a shoulder width.
            lw, rw = wrists["LEFT_WRIST"], wrists["RIGHT_WRIST"]
            gap = ((lw[0] - rw[0]) ** 2 + (lw[2] - rw[2]) ** 2) ** 0.5
            if gap > TWO_HANDED_MAX_GAP * axis_len:
                return None
            two_handed = True
            hitting_pos = ((lw[0] + rw[0]) / 2.0,
                           (lw[1] + rw[1]) / 2.0,
                           (lw[2] + rw[2]) / 2.0)

    # Project the wrist's horizontal offset from the torso centre onto the body axis.
    # Positive => wrist is toward the player's anatomical LEFT, independent of camera
    # facing and of how side-on the player is standing.
    centre = ((left_sh[0] + right_sh[0]) / 2.0, (left_sh[2] + right_sh[2]) / 2.0)
    rel    = (hitting_pos[0] - centre[0], hitting_pos[2] - centre[1])
    side   = (rel[0] * axis[0] + rel[1] * axis[1]) / axis_len

    if two_handed:
        # Both hands on the grip. Handedness cannot be read from a single frame, so the
        # "did the hand cross the midline" test does not apply. What does apply is that
        # the two-handed FOREHAND barely exists in tennis: a handful of professionals
        # have ever used one, while roughly 85% of the tour plays a two-handed backhand.
        # So hands-together is strong evidence of a backhand on its own.
        #
        # This is the same kind of inference as the rest of this project's shot rules:
        # a physical fact about how the sport is played, stated openly, rather than a
        # position heuristic. It is wrong for the rare two-handed forehand, and that is
        # recorded here rather than hidden.
        #
        # Confidence reports the evidence actually used, which is how firmly the hands
        # are together, NOT how far the grip sits off the midline. A two-handed backhand
        # struck close to the body is still unambiguously two-handed; scoring it by
        # midline offset would report near-zero confidence for a call the geometry
        # supports perfectly well.
        return (BACKHAND, max(0.0, 1.0 - gap / (TWO_HANDED_MAX_GAP * axis_len)))

    # Forehand = the wrist stayed on its own anatomical side.
    if hitting_name == "LEFT_WRIST":
        is_forehand = side > 0
    else:
        is_forehand = side < 0

    # Scale-free confidence: how far off the midline, in body-axis lengths.
    confidence = min(1.0, abs(side) / axis_len * 2.0)

    return (FOREHAND if is_forehand else BACKHAND, confidence)
