"""
utils/serve_landing.py
──────────────────────
Finds where a serve actually lands, using the serve's own physics as the search.

Why the generic bounce detector is not enough here
--------------------------------------------------
`detect_bounce_candidates` looks for a vertical-velocity impulse anywhere in the
trajectory. That works across a rally, but on a serve it routinely lands a few frames
late - and a few frames is the whole problem. Measured on a Wimbledon clip whose
broadcast radar read 215.6 km/h: the candidate after the serve contact sat at f261,
by which point the ball had already bounced and was rising again. Its floor
projection came out at y = -31 m on a 23.7 m court, because the camera ray through a
risen ball passes near the horizon and meets the ground far beyond the real landing.
Fed to the speed calculation that produced 366 km/h.

The generic detector answers "was there an impulse near here?". For a serve we can
ask a much stronger question, because a legal serve's landing is heavily constrained:

  1. it happens 0.25-0.90 s after contact (a serve crosses ~18 m of court);
  2. it lands inside the service box diagonally opposite the server;
  3. at the landing instant the ball is at its lowest point on screen.

Searching for a frame that satisfies all three is both more accurate and
self-validating: if no frame qualifies, the honest answer is that the landing was not
observed, and no speed should be reported. That is the same refuse-don't-guess rule
used by the court-validity gate and the serve-speed physical gates.

This is the confidence-gated principle applied to detection rather than to output:
the constraint does not merely filter a result, it *defines* what counts as one.
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# A served ball crosses ~18-20 m before landing. At professional pace that takes about
# 0.30 s; a slow second serve worth reporting takes about 0.65 s. The window is wider
# than both so it never excludes a real serve, and far tighter than "any later bounce".
MIN_FLIGHT_S = 0.25
MAX_FLIGHT_S = 0.90

# How far outside the service box a landing may fall and still be attributed to the
# serve. A fault lands beyond the line but near it; anything further is a different
# event. Metres.
SERVICE_BOX_MARGIN_M = 1.5


def find_serve_landing(
    contact_frame: int,
    ball_detections: list[dict],
    project_to_court: Callable[[int, tuple[float, float]], tuple[float, float] | None],
    server_y_m: float,
    net_y_m: float,
    service_line_far_m: float,
    service_line_near_m: float,
    fps: float,
) -> tuple[int, tuple[float, float]] | None:
    """
    Locate the frame where a serve lands, and its court position.

    Args:
        contact_frame:      frame of the serve strike.
        ball_detections:    per-frame {1: [x1, y1, x2, y2]} in image space.
        project_to_court:   (frame, image_point) -> court (x, y) in metres, or None.
                            Passed in rather than imported so this module stays
                            testable without a homography, and so it always uses the
                            caller's per-frame court fit.
        server_y_m:         server's court y, used to decide which half they serve to.
        net_y_m:            court y of the net.
        service_line_far_m / service_line_near_m: court y of each service line.
        fps:                video frame rate.

    Returns:
        (landing_frame, (x_m, y_m)), or None when no frame in the window satisfies the
        constraints - meaning the landing was not observed and nothing should be
        reported for this serve.
    """
    if fps <= 0 or contact_frame < 0:
        return None

    first = contact_frame + max(1, int(MIN_FLIGHT_S * fps))
    last = min(len(ball_detections) - 1, contact_frame + int(MAX_FLIGHT_S * fps))
    if first > last:
        return None

    # A serve is struck into the diagonally opposite box, so the valid landing band
    # runs from the net to the far side's service line.
    if server_y_m > net_y_m:
        low, high = service_line_far_m, net_y_m
    else:
        low, high = net_y_m, service_line_near_m
    low, high = min(low, high) - SERVICE_BOX_MARGIN_M, max(low, high) + SERVICE_BOX_MARGIN_M

    best_frame: int | None = None
    best_image_y = float("-inf")
    best_court: tuple[float, float] | None = None
    seen = projected = 0

    for frame in range(first, last + 1):
        bbox = ball_detections[frame].get(1)
        if bbox is None:
            continue
        seen += 1
        image_point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

        court = project_to_court(frame, image_point)
        if court is None:
            continue
        projected += 1
        logger.debug(f"      f{frame} ball -> court y={court[1]:6.1f} m "
                     f"(band {low:.1f}..{high:.1f})")
        if not (low <= court[1] <= high):
            continue

        # Among physically valid candidates the landing is the lowest point on screen
        # (image y grows downward), i.e. the instant closest to the court surface.
        if image_point[1] > best_image_y:
            best_image_y = image_point[1]
            best_frame = frame
            best_court = court

    if best_frame is None or best_court is None:
        logger.debug(f"      no landing: {seen} frames with a ball, "
                     f"{projected} projected, none inside the service box")
        return None
    return best_frame, best_court
