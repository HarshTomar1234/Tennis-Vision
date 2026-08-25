"""
tests/test_mini_court_no_fabrication.py
────────────────────────────────────────
Guards MiniCourt against inventing a coordinate, and against substituting a different
mapping algorithm without saying so.

Two defects, both silent, both of the class this project exists to refuse.

1. **A fabricated position.** When a bounding box could not be mapped, the player was
   placed at the CENTRE OF THE COURT. That is not a degraded measurement, it is an
   invented one, and it flowed straight into distance covered and player speed as
   though it had been observed. A wrong number that looks measured is worse than a
   missing one, because nothing downstream can tell the difference.

2. **A silent algorithm swap.** When `cv2.findHomography` could not fit a court, the
   code fell back to `_fallback_mapping`, the nearest-keypoint approximation that the
   README describes as the old and wrong method. Nothing in the log, the summary JSON
   or the HUD said it had happened, so a run could quietly degrade to a materially
   different algorithm and still report its numbers with full confidence.

The fix for the first is omission; for the second, it is bookkeeping the pipeline can
report. These tests pin both.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mini_visual_court import MiniCourt


def _mini_court() -> MiniCourt:
    return MiniCourt(np.zeros((720, 1280, 3), dtype=np.uint8))


def _court_centre(mini: MiniCourt) -> tuple[float, float]:
    """The coordinate the old code invented, so a test can assert nothing equals it."""
    return (mini.start_x + mini.mini_court_width // 2,
            mini.start_y + mini.mini_court_height // 2)


# ── 1. No fabricated coordinates ────────────────────────────────────────────────

def test_unmappable_player_is_omitted_not_placed_at_court_centre():
    mini = _mini_court()
    kp = mini.drawing_key_points

    # A bbox of the wrong shape raises inside the mapping. Previously that landed the
    # player at the centre of the court; now the id should simply not be present.
    players = [{7: ["not", "a", "bbox"]}]
    balls = [{}]

    out_players, _ = mini.convert_bounding_boxes_to_mini_court_coordinates(
        players, balls, kp, use_homography=True
    )

    assert 7 not in out_players[0], (
        "an unmappable player must be absent, not given an invented coordinate"
    )
    assert mini.unmappable_positions == 1, "the omission has to be counted, not hidden"


def test_unmappable_ball_is_omitted_not_placed_at_court_centre():
    mini = _mini_court()
    kp = mini.drawing_key_points

    out_players, out_ball = mini.convert_bounding_boxes_to_mini_court_coordinates(
        [{}], [{1: None}], kp, use_homography=True
    )

    assert 1 not in out_ball[0]
    assert mini.unmappable_positions == 1


def test_no_output_position_equals_the_invented_centre():
    """
    The regression stated as the property that matters.

    Written as "nothing lands exactly on the centre" rather than as a check on one
    code path, because the old behaviour would satisfy any narrower assertion by
    producing a plausible-looking coordinate.
    """
    mini = _mini_court()
    kp = mini.drawing_key_points
    centre = _court_centre(mini)

    players = [{1: [0, 0, 0]}, {2: []}]          # both unmappable, different shapes
    balls = [{1: "nonsense"}, {}]

    out_players, out_ball = mini.convert_bounding_boxes_to_mini_court_coordinates(
        players, balls, kp, use_homography=True
    )

    produced = [p for frame in out_players.values() for p in frame.values()]
    produced += [b for frame in out_ball.values() for b in frame.values()]

    assert produced == [], "nothing was mappable, so nothing should have been produced"
    for value in produced:
        assert tuple(value) != centre
    assert mini.unmappable_positions == 3


def test_mappable_positions_still_come_through():
    """The guard above must not be satisfiable by dropping everything."""
    mini = _mini_court()
    kp = mini.drawing_key_points

    out_players, _ = mini.convert_bounding_boxes_to_mini_court_coordinates(
        [{1: [600.0, 300.0, 680.0, 500.0]}], [{}], kp, use_homography=True
    )

    assert 1 in out_players[0]
    assert mini.unmappable_positions == 0


# ── 2. A fallback mapping is never silent ───────────────────────────────────────

def test_homography_failure_is_recorded():
    mini = _mini_court()

    # Degenerate keypoints: every point identical, so no homography can be fitted and
    # the nearest-keypoint fallback is used instead.
    degenerate = [100.0, 100.0] * 14

    out_players, _ = mini.convert_bounding_boxes_to_mini_court_coordinates(
        [{1: [600.0, 300.0, 680.0, 500.0]}], [{}], degenerate, use_homography=True
    )

    assert mini.homography_failed_frames == {0}, (
        "a frame that fell back to the approximation must be recorded, so the pipeline "
        "can report that those coordinates are not homography-mapped"
    )
    assert mini.calibration_is_approximate is True


def test_successful_homography_records_no_failure():
    mini = _mini_court()
    kp = mini.drawing_key_points

    mini.convert_bounding_boxes_to_mini_court_coordinates(
        [{1: [600.0, 300.0, 680.0, 500.0]}], [{}], kp, use_homography=True
    )

    assert mini.homography_failed_frames == set()
    assert mini.calibration_is_approximate is False


def test_deliberately_disabling_homography_is_not_a_failure():
    """
    use_homography=False is a configuration choice the pipeline already reports, not a
    fallback. Counting it here would make the warning fire on every run of that mode and
    train the reader to ignore it.
    """
    mini = _mini_court()
    kp = mini.drawing_key_points

    mini.convert_bounding_boxes_to_mini_court_coordinates(
        [{1: [600.0, 300.0, 680.0, 500.0]}], [{}], kp, use_homography=False
    )

    assert mini.homography_failed_frames == set()
    assert mini.calibration_is_approximate is False


def test_ball_anchor_projection_also_records_homography_failure():
    """The second conversion method needs the same bookkeeping as the first."""
    from utils.ball_state import FLOOR_LEVEL

    mini = _mini_court()
    degenerate = [100.0, 100.0] * 14

    mini.convert_ball_to_mini_court_coordinates(
        [{1: [600.0, 300.0, 620.0, 320.0]}], degenerate, [FLOOR_LEVEL],
        use_homography=True,
    )

    assert mini.homography_failed_frames == {0}
