"""
utils/bounce_candidates.py
──────────────────────────
Generates candidate BOUNCE frames from the ball trajectory.

Why this module has to exist
----------------------------
The pipeline previously fed `detect_xvelocity_candidates` (plus y-reversals) into a
hit-vs-bounce classifier and wondered why it kept returning hits. Measured on a
Wimbledon clip, every candidate showed a huge horizontal-velocity reversal:

    frame 149  vy_change  0.3   vx_change  49.3   P(hit) 1.000
    frame 231  vy_change 38.7   vx_change 152.7   P(hit) 1.000
    frame 295  vy_change 93.7   vx_change 248.3   P(hit) 1.000

The classifier was right every time. `detect_xvelocity_candidates` is a **hit
detector by construction** — a large |vx change| is precisely what distinguishes a
racket strike, because a racket reverses the ball's horizontal direction and a court
bounce does not. Feeding its output to a hit/bounce classifier can only return hits.
A 14-second rally yielded one bounce, the serve's landing was missed, and every
speed derived from a bounce was consequently wrong.

The physical signature this looks for instead
---------------------------------------------
A bounce applies a sharp **upward impulse** to a descending ball, while leaving its
horizontal direction intact — friction and restitution scrub horizontal speed but do
not reverse it. So a bounce is: descending beforehand, a sharp drop in vertical
velocity, and vx not reversed. That last clause is what stops this module re-finding
the hits, since a racket strike reverses vx.

Why it does NOT require the ball to visibly start rising
--------------------------------------------------------
The obvious formulation — vy flips from positive to negative — was measured and is
materially worse: **55.9 % bounce recall versus 80.9 %** for requiring only a sharp
drop (30 clips, 136 labelled bounces). Roughly a quarter of real bounces never show
an upward vy in the tracked trajectory at all: shallow bounces rebound at a low angle,
and perspective compresses the rebound to nearly nothing when the bounce is far from
the camera, so the ball reads as still descending afterwards. Demanding a visible
rise throws those away permanently.

The impulse is present in every case; the visible reversal is not. Measure the
impulse.

Deliberately generous: this generates *candidates*, and the downstream classifier and
the service-box / plausibility gates do the filtering. Recall matters more than
precision at this stage — a bounce never proposed here can never be recovered later.
"""
from __future__ import annotations


def _ball_positions(ball_detections: list[dict]) -> list[tuple[float, float] | None]:
    """Per-frame ball centre, None where undetected. Matches the other generators."""
    positions: list[tuple[float, float] | None] = []
    for det in ball_detections:
        bbox = det.get(1)
        if bbox is not None:
            positions.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        else:
            positions.append(None)
    return positions


def _mean_velocity(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Average per-frame velocity across a run of consecutive detections."""
    span = len(points) - 1
    return ((points[-1][0] - points[0][0]) / span,
            (points[-1][1] - points[0][1]) / span)


def detect_bounce_candidates(
    ball_detections: list[dict],
    window: int = 4,
    min_vy_drop: float = 3.0,
    max_vx_reversal_ratio: float = 1.0,
    min_spacing: int = 8,
) -> list[int]:
    """
    Candidate bounce frames: a descending ball taking a sharp upward impulse while
    continuing in the same horizontal direction.

    Args:
        ball_detections: per-frame {1: [x1, y1, x2, y2]}.
        window:          frames of context each side used to estimate velocity.
        min_vy_drop:     minimum (vy_before - vy_after) in px/frame. Swept over
                         0.5-5.0; recall is flat at 80.9 % up to 3.0 and only falls
                         beyond it, so 3.0 is the largest value that costs no recall
                         while rejecting the most jitter.
        max_vx_reversal_ratio: reject when horizontal direction reverses by more than
                         this fraction of incoming speed — a racket, not a court. At
                         1.0 this rejects only outright reversals, which measured best;
                         tightening it discarded real bounces on balls hit nearly down
                         the line, which carry little vx to preserve.
        min_spacing:     minimum frames between candidates; a ball cannot bounce twice
                         within a few frames.

    Returns:
        Sorted candidate frame indices.
    """
    positions = _ball_positions(ball_detections)
    candidates: list[int] = []

    for frame in range(window, len(positions) - window):
        if positions[frame] is None:
            continue

        before = [p for p in positions[frame - window:frame] if p is not None]
        after = [p for p in positions[frame + 1:frame + 1 + window] if p is not None]
        if len(before) < 2 or len(after) < 2:
            continue

        vx_before, vy_before = _mean_velocity(before)
        vx_after, vy_after = _mean_velocity(after)

        # Must be descending into the event; a rising ball cannot be hitting the floor.
        if vy_before <= 0:
            continue
        # The upward impulse. Deliberately NOT "vy_after < 0" — see module docstring:
        # requiring a visible rise costs 25 points of recall on shallow/distant bounces.
        if (vy_before - vy_after) < min_vy_drop:
            continue

        # Horizontal direction must survive. A racket reverses it; a court does not.
        if vx_before != 0 and (vx_after / vx_before) < -max_vx_reversal_ratio:
            continue

        candidates.append(frame)

    # Collapse runs: a single bounce spans a few frames of curvature, and the
    # steepest vertical flip within a run is the best estimate of the contact frame.
    merged: list[int] = []
    for frame in candidates:
        if merged and frame - merged[-1] < min_spacing:
            continue
        merged.append(frame)
    return merged
