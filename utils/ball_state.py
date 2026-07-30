"""
utils/ball_state.py
───────────────────
Ball state machine.

Two independent classifications, for two different jobs:

1. FLOOR_LEVEL vs IN_FLIGHT  — for geometry (Phase 1's actual deliverable).
   Every trajectory reversal (a y-direction flip) is a floor-level event: either a
   player hit the ball or it bounced off the court. The floor homography is valid at
   *both*, so this split needs no further disambiguation — it only needs to separate
   "ball is at floor level, project it" from "ball is airborne, interpolate between
   anchors instead." See docs/journal/0003 for why contact-vs-bounce is NOT required
   for this and why three tuning attempts at splitting them were abandoned.

2. CONTACT vs BOUNCE (secondary, best-effort) — for shot counting / stats, where a
   reversal caused by a player hitting the ball is meaningfully different from one
   caused by the court. This uses player-proximity as a heuristic and has a measured
   ceiling of ~5/7 correctly identified shots on the reference clip (see journal 0003)
   — it is useful but NOT a reliable ground truth. Do not tune it further hoping for a
   perfect split; the signal genuinely does not support it with position data alone.
   A better fix later is pose-based swing detection (Phase 2), not more thresholds here.
"""
from __future__ import annotations

# Primary labels — geometry (reliable)
FLOOR_LEVEL = "floor_level"
IN_FLIGHT   = "in_flight"

# Secondary labels — shot stats (best-effort, ~5/7 ceiling measured)
CONTACT = "contact"
BOUNCE  = "bounce"


def _ball_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _player_foot(bbox: list[float]) -> tuple[float, float]:
    """Foot = bottom-centre of the player box (matches the court-mapping convention)."""
    return (bbox[0] + bbox[2]) / 2.0, bbox[3]


def classify_floor_level(
    reversal_frames: list[int],
    n_frames: int,
) -> list[str]:
    """
    The geometry-relevant classification: every reversal is FLOOR_LEVEL, everything
    else is IN_FLIGHT. No proximity heuristic — both contact and bounce are valid
    homography anchors, so no further disambiguation is needed here.

    Returns a per-frame label list of length n_frames.
    """
    states = [IN_FLIGHT] * n_frames
    for f in reversal_frames:
        if 0 <= f < n_frames:
            states[f] = FLOOR_LEVEL
    return states


def classify_contact_vs_bounce(
    reversal_frames: list[int],
    ball_detections: list[dict],
    player_detections: list[dict],
    shot_player_distance_px: float = 300.0,
) -> tuple[list[int], list[int]]:
    """
    Best-effort split of reversal frames into (contacts, bounces) by player proximity.

    Measured ceiling: ~5/7 real shots correctly identified on the reference clip
    (docs/journal/0003_contact_vs_bounce_dead_end_2026-07-29.md). Proximity alone
    cannot cleanly separate contact from bounce because of the frame offset between a
    detected reversal and the true contact instant, and because players are
    continuously near the ball throughout a rally. Use for approximate shot counting,
    not as ground truth.

    Returns (contacts, bounces) — both subsets of reversal_frames.
    """
    contacts: list[int] = []
    bounces: list[int] = []

    for f in reversal_frames:
        ball_bbox = ball_detections[f].get(1) if f < len(ball_detections) else None
        if ball_bbox is None:
            continue

        bx, by = _ball_center(ball_bbox)
        players = player_detections[f] if f < len(player_detections) else {}

        nearest = float("inf")
        for pbbox in players.values():
            fx, fy = _player_foot(pbbox)
            d = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5
            nearest = min(nearest, d)

        if nearest <= shot_player_distance_px:
            contacts.append(f)
        else:
            bounces.append(f)

    return contacts, bounces
