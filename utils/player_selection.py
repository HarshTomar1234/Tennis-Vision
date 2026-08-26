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


# ── Was the selection any good? ────────────────────────────────────────────────

from dataclasses import dataclass

SELECTION_OK = "ok"
SELECTION_DEGRADED = "degraded"
SELECTION_FAILED = "failed"

# Below this share of frames, shot attribution starts failing: the rally grammar's
# same-player rule needs a player to exist at the contact frame, and a player who is
# absent cannot be credited with the shot. Not tuned, it is the level at which the
# downstream consumer breaks.
MIN_COVERAGE = 0.80

# A continuous absence longer than this share of the clip is a different failure from
# scattered misses with the same average: it is a stretch of the rally with nobody to
# attribute shots to.
MAX_GAP_FRACTION = 0.10


@dataclass(frozen=True)
class SelectionQuality:
    """Whether the two selected tracks could plausibly be the two players."""

    status: str
    reason: str
    coverage: tuple[float, float]
    longest_gaps: tuple[int, int]
    opposite_sides: bool
    people_detected: int

    @property
    def is_ok(self) -> bool:
        return self.status == SELECTION_OK

    def as_dict(self) -> dict:
        # Every value is coerced to a plain Python type. Bounding boxes and frame counts
        # arrive as numpy scalars, and json.dump does not serialise those: it writes the
        # file as far as the first one and then raises, leaving a truncated summary.
        payload = {
            "status": self.status,
            "player_1_coverage": round(float(self.coverage[0]), 3),
            "player_2_coverage": round(float(self.coverage[1]), 3),
            "player_1_longest_gap_frames": int(self.longest_gaps[0]),
            "player_2_longest_gap_frames": int(self.longest_gaps[1]),
            "players_on_opposite_sides": bool(self.opposite_sides),
            "people_detected": int(self.people_detected),
        }
        if self.status != SELECTION_OK:
            payload["warning"] = self.reason
        return payload


def assess_selection(
    selected: list[dict],
    net_y: float,
    people_detected: int,
) -> SelectionQuality:
    """
    Judge a completed selection without any ground truth.

    Why this exists
    ---------------
    The court-validity gate stops a clip whose COURT was fitted to the crowd. Nothing
    stopped a clip whose PLAYERS were. Measured across the nine evaluation clips
    (eval/player_selection_sanity.py), input_video_11 passes the court gate comfortably
    at 0.327 line support and still selects two tracks that sit on the same side of the
    net, one of them present for 40% of frames with a 199-frame hole in the middle. The
    pipeline reported confident numbers on it.

    The check needs no labels because the sport supplies the constraint: singles is
    played across the net, so two tracks on the same half cannot both be players. That is
    the same kind of reasoning the rally grammar and the court gate already use.

    This is a sanity check, not an accuracy measurement. Precision, recall, IDF1 and ID
    switches need labelled boxes that do not exist for this footage.

    Args:
        selected:        per-frame {1|2: bbox}, the output of select_two_players.
        net_y:           image y of the net line, midway between the two baselines.
        people_detected: how many distinct tracks the detector found, for context.
    """
    total = len(selected) or 1
    coverage, gaps, medians = [], [], []

    for pid in (1, 2):
        present = [pid in frame for frame in selected]
        coverage.append(sum(present) / total)

        worst = run = 0
        for is_present in present:
            run = 0 if is_present else run + 1
            worst = max(worst, run)
        gaps.append(worst)

        feet = sorted(frame[pid][3] for frame in selected if pid in frame)
        medians.append(feet[len(feet) // 2] if feet else None)

    near, far = medians
    # bool(), not the bare comparison. Bounding boxes arrive as numpy scalars, so this
    # expression evaluates to numpy.bool_, which json.dump cannot serialise: the summary
    # was written as far as this key and then truncated mid-file.
    opposite = bool(near is not None and far is not None
                    and (near - net_y) * (far - net_y) < 0)

    problems = []
    if not opposite:
        problems.append(
            "both selected tracks sit on the same side of the net, so at least one of "
            "them is not a player. Singles is played across the net, so this is not a "
            "close call"
        )
    for pid, (cov, gap) in enumerate(zip(coverage, gaps), start=1):
        if cov < MIN_COVERAGE:
            problems.append(f"player {pid} is present on only {cov:.0%} of frames")
        if gap > total * MAX_GAP_FRACTION:
            problems.append(f"player {pid} is missing for {gap} consecutive frames")

    if not problems:
        status, reason = SELECTION_OK, "two players, opposite sides, tracked throughout"
    elif not opposite:
        # The decisive one. Everything downstream that names a player is wrong.
        status = SELECTION_FAILED
        reason = ("Player selection failed: " + "; ".join(problems)
                  + ". Shot counts, per-player statistics and shot attribution from this "
                    "run are not measurements.")
    else:
        status = SELECTION_DEGRADED
        reason = ("Player tracking is incomplete: " + "; ".join(problems)
                  + ". Shots played during those stretches cannot be attributed, so the "
                    "per-player counts are an under-count.")

    return SelectionQuality(
        status=status, reason=reason,
        coverage=(coverage[0], coverage[1]), longest_gaps=(gaps[0], gaps[1]),
        opposite_sides=opposite, people_detected=people_detected,
    )
