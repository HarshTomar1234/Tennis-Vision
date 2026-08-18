"""
Tests for utils/trajectory_3d.py.

The reconstruction is closed-form, so it can be checked against physics that is known
independently rather than against its own output. That matters here: this module's
numbers are meant to replace the wrong speeds the pipeline currently reports, so
"it runs" is not evidence of anything.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.trajectory_3d import (
    GRAVITY,
    Trajectory3D,
    estimate_contact_height,
    reconstruct_rally,
    reconstruct_segment,
)


class TestReconstructSegment:
    def test_ball_thrown_straight_up_returns_to_start(self):
        """
        Classic projectile check. Launch and land at the same point after T seconds:
        the vertical launch velocity must be gT/2, and the apex must be at gT²/8.
        Both are standard results, so any error in the solver shows up here.
        """
        duration = 2.0
        segment = reconstruct_segment((0.0, 0.0), (0.0, 0.0), 0.0, 0.0, duration)

        assert segment is not None
        assert segment.velocity[0] == pytest.approx(0.0)
        assert segment.velocity[1] == pytest.approx(0.0)
        assert segment.velocity[2] == pytest.approx(GRAVITY * duration / 2)
        assert segment.apex_height_m == pytest.approx(GRAVITY * duration**2 / 8, rel=1e-2)

    def test_horizontal_velocity_is_distance_over_time(self):
        segment = reconstruct_segment((0.0, 0.0), (10.0, 0.0), 0.0, 0.0, 2.0)
        assert segment.velocity[0] == pytest.approx(5.0)
        assert segment.horizontal_distance_m == pytest.approx(10.0)

    def test_speed_includes_the_vertical_component(self):
        """
        The whole point of reconstructing in 3-D: a floor projection sees only the
        horizontal component, so it always under-reports. Speed here must exceed the
        purely horizontal figure.
        """
        duration = 1.0
        segment = reconstruct_segment((0.0, 0.0), (20.0, 0.0), 1.0, 0.0, duration)

        horizontal_only_kmh = (20.0 / duration) * 3.6
        assert segment.speed_kmh > horizontal_only_kmh

    def test_endpoints_are_reproduced_exactly(self):
        segment = reconstruct_segment((1.0, 2.0), (7.0, -3.0), 2.5, 0.0, 1.4)
        first, last = segment.points[0], segment.points[-1]

        assert first == pytest.approx((1.0, 2.0, 2.5))
        assert last == pytest.approx((7.0, -3.0, 0.0), abs=1e-6)

    def test_trajectory_arcs_above_a_straight_line(self):
        """A real flight bulges upward; a straight line between the endpoints does not."""
        segment = reconstruct_segment((0.0, 0.0), (10.0, 0.0), 1.0, 0.0, 1.0)
        midpoint_height = segment.height_at(0.5)
        straight_line_height = (1.0 + 0.0) / 2
        assert midpoint_height > straight_line_height

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_refuses_non_positive_duration(self, duration):
        assert reconstruct_segment((0.0, 0.0), (1.0, 1.0), 0.0, 0.0, duration) is None

    def test_refuses_missing_endpoint(self):
        assert reconstruct_segment(None, (1.0, 1.0), 0.0, 0.0, 1.0) is None


class TestContactHeight:
    def test_serve_is_struck_above_a_groundstroke(self):
        assert estimate_contact_height("Serve") > estimate_contact_height("Forehand")

    def test_unknown_and_missing_fall_back_to_groundstroke(self):
        groundstroke = estimate_contact_height("Forehand")
        assert estimate_contact_height(None) == groundstroke
        assert estimate_contact_height("Wibble") == groundstroke

    def test_is_case_insensitive(self):
        assert estimate_contact_height("serve") == estimate_contact_height("SERVE")


class TestReconstructRally:
    def test_builds_one_segment_between_consecutive_events(self):
        trajectories = reconstruct_rally(
            event_frames=[0, 25, 50],
            ball_positions_m={0: (0.0, 0.0), 25: (5.0, 5.0), 50: (10.0, 0.0)},
            shot_types={0: "Forehand"},
            bounce_frames={25},
            fps=25.0,
        )
        assert len(trajectories) == 2
        assert trajectories[0].start_frame == 0 and trajectories[0].end_frame == 25
        # A bounce endpoint sits on the floor; a strike does not.
        assert trajectories[0].end[2] == 0.0
        assert trajectories[0].start[2] > 0.0

    def test_drops_segments_longer_than_a_plausible_flight(self):
        """
        A long gap means an event between the two was missed. Joining them would
        invent a flight that never happened -- and that is exactly how serve speed
        previously produced 37.9 km/h for a 182 km/h serve.
        """
        trajectories = reconstruct_rally(
            event_frames=[0, 250],
            ball_positions_m={0: (0.0, 0.0), 250: (10.0, 0.0)},
            shot_types={}, bounce_frames=set(), fps=25.0, max_flight_s=3.0,
        )
        assert trajectories == []

    def test_skips_events_with_no_known_position(self):
        trajectories = reconstruct_rally(
            event_frames=[0, 25, 50],
            ball_positions_m={0: (0.0, 0.0), 50: (10.0, 0.0)},   # 25 missing
            shot_types={}, bounce_frames=set(), fps=25.0,
        )
        assert len(trajectories) == 0

    def test_refuses_invalid_fps(self):
        assert reconstruct_rally([0, 10], {0: (0.0, 0.0), 10: (1.0, 1.0)},
                                 {}, set(), fps=0) == []


def test_realistic_serve_produces_a_plausible_speed():
    """
    End-to-end sanity: a serve struck at 2.6 m landing 18 m away in 0.45 s should come
    out in the professional range, and above the horizontal-only figure the old
    floor-projected method would have produced.
    """
    segment = reconstruct_segment((0.0, 0.0), (18.0, 0.0), 2.6, 0.0, 0.45)

    assert 120 < segment.speed_kmh < 260, f"implausible serve speed {segment.speed_kmh}"
    assert segment.speed_kmh > (18.0 / 0.45) * 3.6
