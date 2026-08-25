"""
utils/player_selection.py
─────────────────────────
Narrows raw person detections to the two players, with stable ids 1 and 2.

Why this is shared rather than inline
-------------------------------------
main.py did this itself and the evals did not, so the evals fed every detected person
into the pipeline: on the reference clip that is fourteen people, and the "player" ids
reaching downstream logic included spectators and ball kids. That is invisible while
nothing depends on WHICH player was involved, and it stops being invisible the moment
something does. The rally grammar in utils.rally_decode depends on exactly that: its
strongest rule is that a player cannot hit twice in succession, and a spectator id
between two contacts by one player makes an impossible rally look legal.

So the eval was grading a worse input than the product ships, in a way that made a new
feature look harmful. The function lives here so that cannot drift apart again.
"""
from __future__ import annotations


def select_two_players(
    player_tracker,
    player_detections: list[dict],
    court_keypoints,
) -> tuple[list[dict], dict]:
    """
    Filter to the two players and renumber them 1 and 2.

    Args:
        player_tracker:    supplies `choose_and_filter_players` (the 6-criteria scoring).
        player_detections: per-frame {track_id: bbox} for every detected person.
        court_keypoints:   court keypoints used by the selection scoring.

    Returns:
        (detections, id_map). `detections` holds only the chosen players under ids 1 and
        2; `id_map` records the original track ids they came from, for logging.
    """
    chosen = player_tracker.choose_and_filter_players(player_detections, court_keypoints)

    # Build the id map from every frame, not frame 0: selection already narrowed this to
    # the chosen players, but a player can be absent from the opening frame (replay wipe,
    # off-screen at serve) and reading only frame 0 would silently drop them.
    chosen_ids = sorted({track_id for frame in chosen for track_id in frame})
    id_map = {orig: new for new, orig in enumerate(chosen_ids[:2], start=1)}

    normalized = [
        {id_map[k]: v for k, v in frame.items() if k in id_map} for frame in chosen
    ]
    return normalized, id_map
