"""
utils/serve_detector.py
───────────────────────
Identifies which contact frames are actually serves, from physical evidence.

What this replaces
------------------
`ShotClassifier._determine_shot_type` labelled a shot "Serve" purely because it was
first in the sequence:

    # First shot in sequence is always a serve
    if is_first_shot:
        return self.SHOT_TYPES['SERVE']

That premise only holds if a clip begins exactly at the start of a point. Ours are
cut from mid-match, so the first detected shot is usually a mid-rally groundstroke —
meaning every "Serve" the pipeline had ever reported was this heuristic firing rather
than a serve being recognised. Worse, `classify_shots` skips shot frames with missing
data, so when index 0 was skipped no shot got the flag at all and rallies came back
with no serve whatsoever (measured on six Wimbledon clips, all of which contain real
serves).

The evidence this uses instead
------------------------------
A serve is the only shot in tennis that is simultaneously:

  1. **Struck above the player's head.** The toss puts the ball well over the server —
     contact is around 2.7 m while the player is under 2 m. In image space the ball
     sits above the top edge of the player's bounding box. This alone also catches
     smashes, which share the trait.

  2. **Struck from the baseline or behind it.** This is what separates a serve from a
     smash: smashes are put away near the net, serves are struck from behind the
     baseline. Checked in mini-court space, where the two baselines are known
     geometry rather than image guesswork.

Both must hold. Each is independently measurable, and `explain=True` reports which
one rejected a frame so failures can be diagnosed rather than guessed at.

Deliberately NOT used: "is the first shot", ball direction, or rally position — those
are the assumptions that produced the phantom serves in the first place.
"""
from __future__ import annotations

from utils.bbox_utils import get_center_of_bbox

# How far beyond a baseline a server may stand, as a fraction of court length.
# Servers stand on or just behind the line; 12 % of court length (~3 m) is generous
# enough for a deep stance without reaching into rally territory.
BASELINE_TOLERANCE_FRAC = 0.12

# The ball must clear the top of the player's box by this fraction of the box height.
# Kept at 0 (simply "above the box") because a serving player's box often already
# includes the raised hitting arm, which pushes its top edge up.
HEAD_CLEARANCE_FRAC = 0.0


def _hitting_player(frame_players: dict, ball_center) -> int | None:
    """The player whose box centre is horizontally nearest the ball."""
    if not frame_players:
        return None
    return min(
        frame_players,
        key=lambda pid: abs(get_center_of_bbox(frame_players[pid])[0] - ball_center[0]),
    )


def is_serve(
    frame: int,
    ball_detections: list[dict],
    player_detections: list[dict],
    player_mini_court: dict,
    far_baseline_y: float,
    near_baseline_y: float,
    explain: bool = False,
):
    """
    Decide whether the contact at `frame` is a serve.

    Args:
        frame:             contact frame index.
        ball_detections:   per-frame {ball_id: bbox} in image space.
        player_detections: per-frame {player_id: bbox} in image space.
        player_mini_court: {frame: {player_id: (x, y)}} in mini-court space.
        far_baseline_y:    mini-court y of the far baseline.
        near_baseline_y:   mini-court y of the near baseline.
        explain:           also return a short reason string.

    Returns:
        bool, or (bool, reason) when `explain` is True.
    """
    def result(verdict: bool, reason: str):
        return (verdict, reason) if explain else verdict

    if frame >= len(ball_detections) or frame >= len(player_detections):
        return result(False, "frame out of range")

    ball_box = ball_detections[frame].get(1)
    if not ball_box:
        return result(False, "no ball detection at contact")

    frame_players = player_detections[frame]
    ball_centre = get_center_of_bbox(ball_box)
    player_id = _hitting_player(frame_players, ball_centre)
    if player_id is None:
        return result(False, "no player detected at contact")

    # ── Evidence 1: ball above the player's head ──────────────────────────────
    px1, py1, px2, py2 = frame_players[player_id]
    box_height = py2 - py1
    if ball_centre[1] > py1 + HEAD_CLEARANCE_FRAC * box_height:
        return result(False, "ball not above player's head")

    # ── Evidence 2: struck from the baseline or behind it ─────────────────────
    position = player_mini_court.get(frame, {}).get(player_id)
    if position is None:
        return result(False, "no mini-court position for hitter")

    court_length = abs(near_baseline_y - far_baseline_y)
    tolerance = BASELINE_TOLERANCE_FRAC * court_length
    behind_far = position[1] <= far_baseline_y + tolerance
    behind_near = position[1] >= near_baseline_y - tolerance
    if not (behind_far or behind_near):
        return result(False, "hitter is not at a baseline (smash, not serve)")

    return result(True, f"serve by player {player_id}")


def detect_serve_frames(
    contact_frames: list[int],
    ball_detections: list[dict],
    player_detections: list[dict],
    player_mini_court: dict,
    far_baseline_y: float,
    near_baseline_y: float,
) -> list[int]:
    """Subset of `contact_frames` that carry serve evidence, in order."""
    return [
        frame for frame in sorted(contact_frames)
        if is_serve(frame, ball_detections, player_detections, player_mini_court,
                    far_baseline_y, near_baseline_y)
    ]
