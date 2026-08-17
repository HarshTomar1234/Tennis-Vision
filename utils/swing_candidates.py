"""
utils/swing_candidates.py
─────────────────────────
Proposes contact frames from the PLAYER'S BODY, independently of the ball.

Why a fourth generator
----------------------
The three existing candidate generators (y-reversal, x-velocity change, and the bounce
detector) all read the ball's trajectory. They therefore share a failure mode: when the
ball is undetected, mislocalised, or interpolated across the contact, none of them can
fire, and no amount of retuning fixes an event none of them can see.

Measured on the labelled dataset with eval/event_recall_funnel.py, 15.4% of real contacts
are never proposed by any of the three. Detection reaches those frames (0% are lost to
"no ball nearby"), so the ball is there; the generators just cannot read a reversal out of
it.

A racket swing is visible in the player's arm whether or not the ball was tracked cleanly
through it, so this is orthogonal evidence rather than a better version of the same
evidence.

The signal
----------
A weighted joint speed, following the stroke-detection score in TennisTransformer
(arXiv 2606.15992):

    s(t) = 0.5 * v_wrist + 0.3 * v_elbow + 0.2 * v_shoulder

The wrist dominates because it travels furthest and fastest through a stroke; the elbow
and shoulder contribute because a swing moves the whole kinetic chain, which separates a
stroke from a wrist flick or a pose-estimation twitch.

Normalised by the player's own pixel height
-------------------------------------------
Raw pixel speed is not comparable between the near and far player: the same real swing
covers far fewer pixels at the back of the court. Dividing by that player's bounding-box
height converts pixels into body-lengths, which is depth-invariant and resolution-
invariant, so one threshold works for both players and for any video size. This is the
same normalisation trick that makes ball-to-player distance meaningful at any depth.

What this deliberately does not do
----------------------------------
It does not decide whether a candidate is a hit or a bounce, and it cannot: a bounce
involves no player at all, so this generator is silent on them by construction. That is
correct. It proposes candidates, and utils.hit_bounce_classifier decides what they are.

MEASURED AND NOT WIRED IN
-------------------------
On broadcast footage this signal does not separate a stroke from ordinary movement.
Measured on the reference clip with eval/swing_candidate_recall.py:

                        n      p10   median      p90
    at contacts         4    0.014    0.043    0.148
    elsewhere         436    0.020    0.051    0.106

    ordinary frames scoring at or above the median contact: 58.5%

The two distributions are indistinguishable, and ordinary frames score slightly HIGHER on
median than contacts do. A coverage sweep looks encouraging at a low threshold (it reaches
both contacts the ball generators miss) but that is an artifact of firing on 37 of 570
frames: at that rate it covers things by chance rather than by evidence.

The reason is that tennis players are never still. Split-steps, recovery sprints and
racket preparation all drive the same joints as hard as a stroke does, so peak joint speed
carries little information about whether a ball was struck. It might separate on close,
high-frame-rate footage where a swing is temporally resolved; it does not here.

The measurement also surfaced a bigger problem it was not looking for: only 4 of the 7
labelled contacts produced a usable pose at all. The contact frame is the worst moment to
ask for one, since the player is fully extended, often side-on, frequently occluded by
their own arm, and motion-blurred. That is a ceiling on pose-based forehand/backhand
classification, and is measured directly by eval/pose_availability_at_contacts.py.

The module and its tests are kept because the code is correct and the measurement is
reproducible, not because the signal works.
"""
from __future__ import annotations

# Weights from the stroke-detection score in TennisTransformer (arXiv 2606.15992).
# They sum to 1.0, so the score stays in body-lengths per frame.
W_WRIST, W_ELBOW, W_SHOULDER = 0.5, 0.3, 0.2

# A swing must reach at least this speed, in body-lengths per frame, to be proposed.
# Calibrate with eval/swing_candidate_recall.py rather than by eye: too low and ordinary
# running motion is proposed as a stroke, too high and gentle blocks and touch volleys
# are missed.
MIN_SWING_SPEED = 0.055

# Frames either side used to measure joint speed. A stroke's acceleration phase is short,
# and averaging over too long a window flattens the very peak being looked for.
SPEED_WINDOW = 2

# Two peaks closer together than this describe one swing. A player cannot strike twice
# within a fifth of a second.
MIN_PEAK_SEPARATION = 6


def _joint_speed(track: list, frame: int, joint: str, window: int) -> float | None:
    """Speed of one joint at `frame`, in pixels per frame, or None if unmeasurable."""
    before = track[max(0, frame - window)]
    after = track[min(len(track) - 1, frame + window)]
    if before is None or after is None:
        return None
    a, b = before.get(joint), after.get(joint)
    if a is None or b is None:
        return None
    span = min(len(track) - 1, frame + window) - max(0, frame - window)
    if span <= 0:
        return None
    return ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 / span


def swing_score(
    pose_track: list,
    heights: list,
    frame: int,
    window: int = SPEED_WINDOW,
) -> float | None:
    """
    Weighted joint speed at `frame`, in body-lengths per frame.

    Args:
        pose_track: per-frame dict of landmark name to (x, y) in frame pixels, or None
                    where pose was unavailable.
        heights:    per-frame player bounding-box height in pixels, used to normalise.
        frame:      frame to score.
        window:     frames either side used to measure speed.

    Returns:
        The score, or None when the pose or the height is missing.
    """
    if not (0 <= frame < len(pose_track)) or pose_track[frame] is None:
        return None
    height = heights[frame] if frame < len(heights) else None
    if not height or height <= 0:
        return None

    total = 0.0
    for weight, left, right in (
        (W_WRIST, "LEFT_WRIST", "RIGHT_WRIST"),
        (W_ELBOW, "LEFT_ELBOW", "RIGHT_ELBOW"),
        (W_SHOULDER, "LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ):
        speeds = [s for s in (_joint_speed(pose_track, frame, left, window),
                              _joint_speed(pose_track, frame, right, window))
                  if s is not None]
        if not speeds:
            return None
        # The faster arm is the hitting arm. A two-handed stroke moves both, and taking
        # the max is right there too: the pair move together, so max equals either.
        total += weight * max(speeds)

    return total / height


def detect_swing_candidates(
    pose_track: list,
    heights: list,
    min_speed: float = MIN_SWING_SPEED,
    min_separation: int = MIN_PEAK_SEPARATION,
    window: int = SPEED_WINDOW,
) -> list[int]:
    """
    Frames where this player's arm reaches a local speed peak consistent with a stroke.

    Peaks are found rather than thresholds crossed, because a fast swing holds high speed
    over several frames and every one of them would otherwise be proposed. Only the
    fastest frame in each burst is kept.

    Args:
        pose_track:     per-frame landmark dicts, or None where pose failed.
        heights:        per-frame player bbox height in pixels.
        min_speed:      minimum peak height, in body-lengths per frame.
        min_separation: minimum frames between two accepted peaks.
        window:         frames either side used to measure speed.

    Returns:
        Candidate contact frames, sorted.
    """
    scores = [swing_score(pose_track, heights, f, window) for f in range(len(pose_track))]

    peaks: list[tuple[float, int]] = []
    for f, s in enumerate(scores):
        if s is None or s < min_speed:
            continue
        prev_s = scores[f - 1] if f > 0 else None
        next_s = scores[f + 1] if f + 1 < len(scores) else None
        # A local maximum. The >= on the left and > on the right breaks ties on a plateau
        # by keeping its first frame, so a flat peak yields exactly one candidate.
        if (prev_s is None or s >= prev_s) and (next_s is None or s > next_s):
            peaks.append((s, f))

    # Strongest first, so when two peaks are too close the weaker one is the one dropped.
    peaks.sort(reverse=True)
    kept: list[int] = []
    for _score, frame in peaks:
        if all(abs(frame - k) >= min_separation for k in kept):
            kept.append(frame)

    return sorted(kept)
