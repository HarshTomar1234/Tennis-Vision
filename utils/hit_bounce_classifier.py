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


def detect_xvelocity_candidates(
    ball_detections: list[dict],
    min_delta_x: float = 5.0,
    min_spacing: int = 15,
    window: int = EVENT_WINDOW,
) -> list[int]:
    """
    Candidate contact/bounce frames from local peaks in |horizontal-velocity change|,
    complementing (not replacing) trajectory-reversal detection.

    Why this exists: journal 0014 found a structural gap in y-reversal-only detection —
    some real contacts (verified on the reference clip) don't reverse vertical direction
    at all, or reverse too shallowly to separate from noise at any threshold (swept
    min_delta_y down to 1 with no improvement). But the same clip's real shots often DO
    show a sharp change in horizontal velocity (the physical signal behind
    hit_bounce_classifier — journal 0012), even when the vertical trajectory barely moves.

    Tested standalone at dataset scale first (91 real clips): x-velocity alone actually
    recalls WORSE than y-reversal (70.3% vs 76.0% at the best threshold) — it is not a
    good replacement. But the UNION of both signals recalls 87.7%, a genuine +11.7 point
    gain, because they catch different kinds of real events (vertical-redirect shots vs
    horizontal-redirect shots). Use this alongside get_ball_shot_frames, not instead of
    it — feed the union to classify_reversals_by_trajectory, which is what actually
    filters the resulting extra candidates back down to real contacts/bounces.

    Args:
        ball_detections: per-frame {1: [x1,y1,x2,y2]} (same format as get_ball_shot_frames).
        min_delta_x:     minimum |vx change| to count as a candidate peak.
        min_spacing:     minimum frames between two candidates.
        window:          frames of context on each side used to compute velocity.

    Returns:
        Candidate frame indices, sorted, at least min_spacing apart.
    """
    positions: list[tuple[float, float] | None] = []
    for det in ball_detections:
        bbox = det.get(1)
        if bbox is not None:
            positions.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        else:
            positions.append(None)

    n = len(positions)
    scores = [0.0] * n
    for i in range(window, n - window):
        before = [p for p in positions[i - window:i] if p is not None]
        after  = [p for p in positions[i + 1:i + 1 + window] if p is not None]
        if len(before) < 2 or len(after) < 2:
            continue
        vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
        vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
        scores[i] = abs(vx_after - vx_before)

    candidates: list[int] = []
    for i in range(n):
        if scores[i] < min_delta_x:
            continue
        lo, hi = max(0, i - min_spacing // 2), min(n, i + min_spacing // 2 + 1)
        if scores[i] == max(scores[lo:hi]):
            if not candidates or i - candidates[-1] >= min_spacing:
                candidates.append(i)

    return candidates


def merge_nearby_candidates(candidates: list[int], min_gap: int = 10) -> list[int]:
    """
    Collapse candidates within min_gap frames of each other into one representative
    frame per cluster (the cluster's median).

    get_ball_shot_frames (y-reversal) and detect_xvelocity_candidates each fire near a
    real event independently, with no knowledge of each other, so their union can put
    several candidates a few frames apart around the same single real contact. On our
    own reference clip this inflated the apparent shot count from 7 (y-reversal only,
    journal 0012) to 22 (union, journal 0015) -- most of the "extra" shots turned out to
    be duplicate detections of the same handful of real events, not new false events
    (see docs/journal/0018's frame-by-frame check). It also made downstream per-frame
    work (pose classification) sensitive to *which* nearby frame got checked: contact
    happens at one instant, so a candidate a few frames off has different wrist
    positions than the true contact frame, and pose can succeed on one cluster member
    while failing on another for the same swing.

    min_gap=10 matches the tolerance used throughout this sprint's eval scripts for
    "near the same event" (e.g. TOLERANCE in retest_union_type_accuracy.py).

    Args:
        candidates: raw candidate frames (e.g. the union of get_ball_shot_frames and
                    detect_xvelocity_candidates), any order.
        min_gap:    candidates within this many frames of their cluster's last member
                    are merged into the same cluster.

    Returns:
        One representative frame per cluster, sorted.
    """
    if not candidates:
        return []

    ordered = sorted(candidates)
    clusters: list[list[int]] = [[ordered[0]]]
    for c in ordered[1:]:
        if c - clusters[-1][-1] <= min_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return [cluster[len(cluster) // 2] for cluster in clusters]


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
