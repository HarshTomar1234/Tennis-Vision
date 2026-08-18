"""
Tests for utils/serve_landing.py.

The projection function is injected, so these tests define an exact synthetic court
and can assert the constraint logic directly rather than inferring it from pipeline
output. Every case here corresponds to a failure mode observed on real footage.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.serve_landing import MAX_FLIGHT_S, MIN_FLIGHT_S, find_serve_landing

FPS = 25.0
NET_Y = 11.9              # metres, mid-court
SERVICE_FAR = 5.5         # far service line
SERVICE_NEAR = 18.3       # near service line
SERVER_Y = 23.0           # server stands at the near baseline


def ball(frames: dict[int, tuple[float, float]], total: int = 100) -> list[dict]:
    """Build a per-frame detection list from {frame: (x, y)} image points."""
    out = []
    for i in range(total):
        if i in frames:
            x, y = frames[i]
            out.append({1: [x - 2, y - 2, x + 2, y + 2]})
        else:
            out.append({})
    return out


def court_map(mapping: dict[int, tuple[float, float]]):
    """Projection stub: frame -> court position, ignoring the image point."""
    return lambda frame, _point: mapping.get(frame)


class TestFindsTheLanding:
    def test_picks_the_lowest_valid_frame(self):
        """
        Three frames all project inside the service box; the landing is the one lowest
        on screen. This is the real-world case that broke the generic detector: it
        picked a later frame where the ball had already bounced and risen.
        """
        detections = ball({20: (600, 300), 22: (610, 480), 24: (620, 350)})
        projections = court_map({20: (5.0, 9.0), 22: (5.2, 8.5), 24: (5.4, 8.0)})

        result = find_serve_landing(10, detections, projections, SERVER_Y, NET_Y,
                                    SERVICE_FAR, SERVICE_NEAR, FPS)
        assert result is not None
        landing_frame, court = result
        assert landing_frame == 22          # image y = 480 is lowest on screen
        assert court == (5.2, 8.5)

    def test_rejects_frames_projecting_outside_the_service_box(self):
        """
        A frame lower on screen but projecting past the baseline must lose to a
        higher-but-valid one. This is exactly the -31 m apex projection that produced
        a 366 km/h serve reading.
        """
        detections = ball({20: (600, 300), 22: (610, 900)})
        projections = court_map({20: (5.0, 9.0), 22: (3.6, -31.0)})

        result = find_serve_landing(10, detections, projections, SERVER_Y, NET_Y,
                                    SERVICE_FAR, SERVICE_NEAR, FPS)
        assert result is not None
        assert result[0] == 20

    def test_returns_none_when_nothing_lands_in_the_box(self):
        """No observed landing means no landing reported - never a guess."""
        detections = ball({20: (600, 300), 22: (610, 900)})
        projections = court_map({20: (3.0, -20.0), 22: (3.6, -31.0)})

        assert find_serve_landing(10, detections, projections, SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, FPS) is None


class TestFlightWindow:
    def test_ignores_frames_too_soon_after_contact(self):
        """A ball still at the racket cannot be the landing."""
        too_soon = 10 + int(MIN_FLIGHT_S * FPS) - 2
        detections = ball({too_soon: (600, 900)})
        projections = court_map({too_soon: (5.0, 9.0)})

        assert find_serve_landing(10, detections, projections, SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, FPS) is None

    def test_ignores_frames_beyond_a_plausible_flight(self):
        """
        A later rally bounce must not be adopted as the serve's landing - the failure
        that paired a serve with an event 1.78 s later and reported 37.9 km/h.
        """
        too_late = 10 + int(MAX_FLIGHT_S * FPS) + 5
        detections = ball({too_late: (600, 900)})
        projections = court_map({too_late: (5.0, 9.0)})

        assert find_serve_landing(10, detections, projections, SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, FPS) is None


class TestServeDirection:
    def test_far_server_lands_in_the_near_box(self):
        """A serve from the far end must land on the near side of the net."""
        detections = ball({20: (600, 500)})
        projections = court_map({20: (5.0, 15.0)})       # near half

        result = find_serve_landing(10, detections, projections,
                                    server_y_m=1.0,       # far baseline
                                    net_y_m=NET_Y, service_line_far_m=SERVICE_FAR,
                                    service_line_near_m=SERVICE_NEAR, fps=FPS)
        assert result is not None and result[0] == 20

    def test_far_server_does_not_accept_a_landing_behind_itself(self):
        detections = ball({20: (600, 500)})
        projections = court_map({20: (5.0, 8.0)})        # far half, behind the server

        assert find_serve_landing(10, detections, projections,
                                  server_y_m=1.0, net_y_m=NET_Y,
                                  service_line_far_m=SERVICE_FAR,
                                  service_line_near_m=SERVICE_NEAR, fps=FPS) is None


class TestDegenerateInputs:
    @pytest.mark.parametrize("fps", [0, -1])
    def test_refuses_invalid_fps(self, fps):
        assert find_serve_landing(10, ball({}), court_map({}), SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, fps) is None

    def test_handles_no_detections(self):
        assert find_serve_landing(10, ball({}), court_map({}), SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, FPS) is None

    def test_handles_contact_near_end_of_clip(self):
        assert find_serve_landing(95, ball({}, total=100), court_map({}), SERVER_Y,
                                  NET_Y, SERVICE_FAR, SERVICE_NEAR, FPS) is None

    def test_handles_projection_failures(self):
        detections = ball({20: (600, 500)})
        assert find_serve_landing(10, detections, lambda f, p: None, SERVER_Y, NET_Y,
                                  SERVICE_FAR, SERVICE_NEAR, FPS) is None
