"""
utils/hit_bounce_classifier.py
───────────────────────────────
Classifies a floor-level ball event (from utils.ball_state.classify_floor_level) as a
CONTACT (player hit) or BOUNCE (court), using ball-trajectory shape alone — no player
position needed.

Why this exists
----------------
The player-proximity heuristic in classify_contact_vs_bounce has a measured ~5/7 ceiling
on our own footage (docs/journal/0003) that turned out to be mostly an artifact of an
incomplete ground truth, not the heuristic itself (docs/journal/0010, 0011) — but while
investigating that, feature exploration on the real 1,034-event TrackNet ground truth
(eval/explore_hit_bounce_features.py) found a much stronger, complementary signal:

  A bounce is a court reflection — mostly preserves horizontal (x) velocity, since the
  ground doesn't impart much sideways force. A hit is a player redirecting the ball, which
  can reverse or sharply change x-direction (cross-court shots, returns). Measured: hits
  flip x-direction 71.8% of the time, bounces only 2.1% of the time.

A 3-feature logistic regression (height, |vertical-velocity change|, |horizontal-velocity
change|) trained on that data reaches 84.1% held-out accuracy on clip-level split (not
event-level, which would leak camera/lighting/player correlations) — see
eval/train_hit_bounce_classifier.py for the full methodology and eval/journal 0012 for
the honest numbers.

This is trajectory-only and needs no player detection, so it works even when player
tracking is unavailable, and combines naturally with the proximity heuristic where player
positions ARE available (main.py can use both and let them agree/disagree as a signal).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .ball_state import CONTACT, BOUNCE

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_PATH = "models/hit_bounce_classifier.json"
EVENT_WINDOW = 4   # frames before/after the event used to compute velocity — must match
                    # the WINDOW constant in eval/explore_hit_bounce_features.py, since
                    # the trained weights assume this exact window size.

_cached_weights: dict | None = None
_cached_path: str | None = None


def _load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict | None:
    global _cached_weights, _cached_path
    if _cached_weights is not None and _cached_path == path:
        return _cached_weights

    if not Path(path).exists():
        logger.warning(
            f"Hit/bounce classifier weights not found at '{path}'. "
            f"Train with: python eval/train_hit_bounce_classifier.py"
        )
        return None

    _cached_weights = json.loads(Path(path).read_text())
    _cached_path = path
    return _cached_weights


def compute_event_features(
    positions: list[tuple[float, float] | None],
    event_frame: int,
    window: int = EVENT_WINDOW,
) -> dict | None:
    """
    Extract the trajectory-shape features the classifier needs around `event_frame`.

    Args:
        positions: per-frame (x, y) ball centre, or None where undetected. Index must
                   align with `event_frame`.
        event_frame: the floor-level candidate frame (from classify_floor_level).
        window:    frames of context required on each side (default matches training).

    Returns:
        {"height_y", "vy_change_mag", "vx_change_mag"}, or None if there isn't enough
        clean position data around this frame to compute velocity on both sides.
    """
    if event_frame < 0 or event_frame >= len(positions) or positions[event_frame] is None:
        return None

    before = [p for p in positions[max(0, event_frame - window):event_frame] if p is not None]
    after  = [p for p in positions[event_frame + 1:event_frame + 1 + window] if p is not None]
    if len(before) < 2 or len(after) < 2:
        return None

    vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
    vy_before = (before[-1][1] - before[0][1]) / (len(before) - 1)
    vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
    vy_after  = (after[-1][1] - after[0][1]) / (len(after) - 1)

    return {
        "height_y": positions[event_frame][1],
        "vy_change_mag": abs(vy_after - vy_before),
        "vx_change_mag": abs(vx_after - vx_before),
    }


def classify_hit_or_bounce(
    features: dict | None,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> tuple[str, float] | None:
    """
    Classify a floor-level event as CONTACT or BOUNCE from its trajectory features.

    Args:
        features: from compute_event_features(); None passes through as None (honest
                  absence, not a guess — matches the convention in pose_shot_classifier).
        weights_path: trained weights, see eval/train_hit_bounce_classifier.py.

    Returns:
        (CONTACT | BOUNCE, probability of the returned label), or None if features are
        missing or weights aren't trained yet.
    """
    if features is None:
        return None

    weights = _load_weights(weights_path)
    if weights is None:
        return None

    x = [features[name] for name in weights["feature_names"]]
    mu, sigma, w, b = weights["mu"], weights["sigma"], weights["w"], weights["b"]

    z = b
    for xi, mui, sigi, wi in zip(x, mu, sigma, w):
        z += wi * ((xi - mui) / sigi)

    p_hit = 1.0 / (1.0 + pow(2.718281828, -z))
    if p_hit >= 0.5:
        return CONTACT, p_hit
    return BOUNCE, 1.0 - p_hit


def classify_reversals_by_trajectory(
    reversal_frames: list[int],
    ball_detections: list[dict],
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> tuple[list[int], list[int]]:
    """
    Drop-in alternative to classify_contact_vs_bounce (utils.ball_state) with the same
    (contacts, bounces) return shape, but using trajectory shape instead of player
    proximity — no player detection needed. See module docstring for why: 84.1% held-out
    accuracy on real data, vs the proximity heuristic's measured ~5/7 ceiling on our own
    footage (which turned out to be mostly an incomplete-ground-truth artifact, but the
    trajectory signal is independently strong regardless — see docs/journal/0012).

    Frames where the classifier can't reach a decision (not enough trajectory context,
    or the weights aren't trained) are dropped from both lists rather than guessed.
    """
    positions: list[tuple[float, float] | None] = []
    for det in ball_detections:
        bbox = det.get(1)
        if bbox is not None:
            positions.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        else:
            positions.append(None)

    contacts, bounces = [], []
    for frame in reversal_frames:
        features = compute_event_features(positions, frame)
        result = classify_hit_or_bounce(features, weights_path)
        if result is None:
            continue
        (contacts if result[0] == CONTACT else bounces).append(frame)

    return contacts, bounces
