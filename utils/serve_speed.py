"""
utils/serve_speed.py
────────────────────
Serve speed measured from floor-valid geometry only.

The problem this solves
-----------------------
Every other speed in this pipeline is derived from the ball's position after it has
been projected through the *floor* homography. That projection is only correct when
the ball is actually touching the floor. A tennis ball is airborne for roughly 90 %
of its flight, so its floor projection is the point where the camera ray through the
ball meets the ground — not where the ball is. Speeds computed from that projection
are wrong by an amount that depends on ball height and camera geometry, and measured
against broadcast radar on our own clips they came out roughly half the true value.

The fix, and why it is exact rather than another approximation
--------------------------------------------------------------
Two things in a serve ARE reliably on the floor:

  1. the server's feet at the moment of contact, and
  2. the ball's first bounce.

Both project through the floor homography correctly. The horizontal distance between
them is therefore a real measurement in metres, and dividing by the flight time gives
a real speed — no airborne projection involved anywhere.

Ignoring the ball's vertical drop costs almost nothing: a serve struck at ~2.7 m and
landing ~18 m away travels sqrt(18² + 2.7²) = 18.2 m, so treating the path as
horizontal understates it by ~1 %.

What this number is, precisely
------------------------------
It is the **average speed over the flight**, whereas a broadcast radar gun reports
the speed **at contact**. Air drag means the ball is always slower later in flight,
so this figure reads systematically below the televised one — expect roughly 10-15 %
for a flat serve. That is a known, explainable offset, not an error, and it must be
reported as "average flight speed" rather than dressed up as a radar-equivalent
number. `eval/serve_speed_accuracy.py` measures the real offset against broadcast
ground truth.
"""
from __future__ import annotations

import math


def bounce_is_in_service_box(
    contact_pos: tuple[float, float],
    bounce_pos: tuple[float, float],
    net_y: float,
    far_service_y: float,
    near_service_y: float,
    margin_frac: float = 0.15,
) -> bool:
    """
    Check the landing is where a legal serve must land: the service box opposite
    the server, between the net and that side's service line.

    Why this check is load-bearing rather than cosmetic
    ---------------------------------------------------
    The whole method rests on the bounce being a real floor contact, because only
    then is its floor projection meaningful. The hit/bounce classifier is ~84 %
    accurate, so roughly one in six "bounces" is not one — and when a mid-flight
    point is used instead, the ball is metres in the air and its floor projection
    lands far away. Measured on a Wimbledon clip: a serve contact projected to
    mini-court y = -16.7, i.e. *outside the court*, from a server standing at
    y = 552. That single bad point inflated the flight distance to 26.6 m when a
    serve travels ~18 m, which is the whole of the 31 % error against radar.

    A serve that lands anywhere else is not a serve landing, so the measurement is
    refused rather than reported.
    """
    if contact_pos is None or bounce_pos is None:
        return False

    server_is_near = contact_pos[1] > net_y
    span = abs(near_service_y - far_service_y)
    margin = margin_frac * span

    if server_is_near:
        # Serving towards the far end: land between the far service line and the net.
        low, high = far_service_y - margin, net_y + margin
    else:
        low, high = net_y - margin, near_service_y + margin

    return low <= bounce_pos[1] <= high


def serve_speed_kmh(
    contact_pos: tuple[float, float],
    bounce_pos: tuple[float, float],
    contact_frame: int,
    bounce_frame: int,
    px_to_m_scale: float,
    fps: float,
    max_realistic_kmh: float | None = None,
) -> float:
    """
    Average flight speed of a serve, in km/h, from two floor-anchored points.

    Args:
        contact_pos:   server's foot position at contact, mini-court pixels.
        bounce_pos:    ball's first bounce, mini-court pixels.
        contact_frame: frame index of the serve contact.
        bounce_frame:  frame index of the bounce.
        px_to_m_scale: metres per mini-court pixel.
        fps:           video frame rate.
        max_realistic_kmh: reject anything above this as a tracking artefact rather
                       than reporting it (returns 0.0, the same "no measurement"
                       sentinel used elsewhere in the pipeline).

    Returns:
        Average flight speed in km/h, or 0.0 when it cannot be measured.
    """
    if contact_pos is None or bounce_pos is None:
        return 0.0

    frames = bounce_frame - contact_frame
    if frames <= 0 or fps <= 0:
        return 0.0

    seconds = frames / fps
    distance_m = math.dist(contact_pos, bounce_pos) * px_to_m_scale
    if distance_m <= 0:
        return 0.0

    kmh = (distance_m / seconds) * 3.6
    if max_realistic_kmh is not None and kmh > max_realistic_kmh:
        return 0.0
    return kmh


# A served ball crosses ~18-20 m of court before landing. At the slowest club-level
# serve worth reporting (~110 km/h average over the flight) that takes ~0.65 s; at
# professional pace (~200 km/h) it takes ~0.33 s. The window below is deliberately
# wider than both, but far tighter than "any bounce later in the rally".
MIN_SERVE_FLIGHT_S = 0.25
MAX_SERVE_FLIGHT_S = 0.90


def find_serve_and_bounce(
    shot_classifications: dict,
    bounce_frames: list[int],
    fps: float,
) -> tuple[int, int] | None:
    """
    Locate the serve contact frame and the bounce that is physically its landing.

    The candidate is the first bounce after contact, but it is only accepted if the
    interval is a plausible serve flight time. Without that check the pairing silently
    latches onto a later rally bounce when the serve's own landing was missed by the
    detector — measured on our own clip 5, which paired a contact with a bounce 46
    frames (1.78 s) later and reported 37.9 km/h for a serve the broadcast radar
    clocked at 182 km/h. A wrong number presented confidently is worse than no number,
    so an implausible interval returns None.

    Returns (contact_frame, bounce_frame), or None when no valid pair exists.
    """
    if fps <= 0:
        return None

    serve_frames = [
        frame for frame, info in shot_classifications.items()
        if str(info.get("shot_type", "")).lower() == "serve"
    ]
    if not serve_frames:
        return None

    contact_frame = min(serve_frames)
    for bounce in sorted(bounce_frames):
        if bounce <= contact_frame:
            continue
        flight_s = (bounce - contact_frame) / fps
        if flight_s < MIN_SERVE_FLIGHT_S:
            continue          # too soon to be the landing — likely detector noise
        if flight_s > MAX_SERVE_FLIGHT_S:
            return None       # the real landing was missed; do not pair with a later bounce
        return contact_frame, bounce
    return None
