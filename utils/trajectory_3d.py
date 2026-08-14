"""
utils/trajectory_3d.py
──────────────────────
Reconstructs the ball's 3-D flight path between floor-anchored events.

Why this exists — it is the speed fix, not a visualisation feature
------------------------------------------------------------------
Every speed this pipeline reports is derived from the ball's position after
projection through the *floor* homography. That projection is only valid while the
ball touches the floor, and a tennis ball is airborne for roughly 90 % of its flight.
The consequence is measurable: on a Wimbledon clip a serve contact projected to
mini-court y = -16.7 — outside the court entirely — because the camera ray through a
ball 2.7 m in the air meets the ground 25 m away. Rally speeds come out
systematically low, and no amount of threshold tuning can fix a geometry error.

Reconstructing the trajectory in 3-D removes the error at its source. The 3-D rally
viewer is a by-product of the same computation, not a separate feature.

The method, and why it needs no optimiser
-----------------------------------------
Between two events the ball is in free flight, so its horizontal motion is constant
and its vertical motion is parabolic. If both endpoints and the flight time are
known, the trajectory is *fully determined* — it is a two-point boundary value
problem with a closed-form solution:

    vx  = (x1 - x0) / T
    vy  = (y1 - y0) / T
    vz0 = (z1 - z0 + ½·g·T²) / T          from z(T) = z0 + vz0·T - ½·g·T²

No fitting, no initial guess, no convergence risk. The endpoints come from the
homography at moments when it is *valid* — a bounce is on the floor by definition,
and a player's feet are on the floor at contact — so the inputs are exactly the
measurements this pipeline can trust.

`speed_kmh` is then the true 3-D speed at the start of the segment, including the
vertical component the floor projection discarded.

Honest limits
-------------
- **Drag is not modelled.** A real ball decelerates through flight, so the constant
  horizontal velocity here is an average over the segment rather than the speed at
  contact. Expect readings below a radar gun's, which measures at contact. This is
  the same caveat that applies to `utils/serve_speed.py`, and for the same reason.
- **Spin (the Magnus effect) is not modelled.** Heavy topspin bends a trajectory
  downward faster than gravity alone; this reconstruction will place the apex slightly
  high on such shots.
- Contact height is estimated, not measured (see `estimate_contact_height`), because
  a racket strike is the one endpoint that is genuinely not on the floor.

Both simplifications are documented rather than hidden, and both are strict
improvements on projecting an airborne ball through a floor homography.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

GRAVITY = 9.81  # m/s²

# Typical contact heights in metres. Used only when a pose-derived height is
# unavailable; a serve is struck overhead, a groundstroke around waist/chest.
CONTACT_HEIGHT_SERVE = 2.6
CONTACT_HEIGHT_GROUNDSTROKE = 0.9
CONTACT_HEIGHT_VOLLEY = 1.1
BOUNCE_HEIGHT = 0.0

# Longest credible single free flight, in seconds. A groundstroke's contact-to-bounce
# flight runs about 0.4-1.0 s; a high defensive lob is the extreme case and still lands
# inside ~1.5 s. Anything longer is not one flight — it is two or more with the events
# between them missed, or a dead-ball period between points that the ball interpolator
# bridged. A parabola stretched across several flights is both slower and taller than
# any of them, so admitting one corrupts speed and apex together.
#
# Measured on input_video_2 (30 fps, 570 frames): with no effective cap the segment
# durations ran to 1.73 s and mean speed was 57 km/h; capping at 1.5 s left 18 segments
# with durations 0.40-1.47 s and mean speed 60 km/h. Most segments (10 of 18) are now
# under 0.85 s, which is the regime a real flight occupies.
#
# 1.5 s rather than something tighter because a genuine lob does reach it — the cap is
# meant to reject stitched-together flights, not real high balls.
MAX_PLAUSIBLE_FLIGHT_S = 1.5


@dataclass
class Trajectory3D:
    """One free-flight segment of the ball, reconstructed in court coordinates.

    Court frame: x across the court, y along it, z up, all in metres, origin at the
    mini-court's coordinate origin projected to the floor.
    """

    start_frame: int
    end_frame: int
    duration_s: float
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    velocity: tuple[float, float, float]
    points: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def speed_kmh(self) -> float:
        """True 3-D speed at the start of the segment."""
        vx, vy, vz = self.velocity
        return math.sqrt(vx * vx + vy * vy + vz * vz) * 3.6

    @property
    def apex_height_m(self) -> float:
        """Highest point reached, in metres. Below the launch height if never rising."""
        return max(z for _, _, z in self.points) if self.points else self.start[2]

    @property
    def horizontal_distance_m(self) -> float:
        return math.dist(self.start[:2], self.end[:2])

    def height_at(self, fraction: float) -> float:
        """Height at a fraction (0..1) through the flight — used for net clearance."""
        t = max(0.0, min(1.0, fraction)) * self.duration_s
        return self.start[2] + self.velocity[2] * t - 0.5 * GRAVITY * t * t


def estimate_contact_height(shot_type: str | None) -> float:
    """
    Height of the racket strike, in metres.

    A contact is the one endpoint not on the floor, so its height cannot come from
    the homography. These are population averages by shot type — deliberately coarse,
    and the reconstruction is not very sensitive to them: a 20 cm error over an 18 m
    flight changes the launch angle by well under a degree.
    """
    if not shot_type:
        return CONTACT_HEIGHT_GROUNDSTROKE
    kind = str(shot_type).strip().lower()
    if kind in ("serve", "smash"):
        return CONTACT_HEIGHT_SERVE
    if kind == "volley":
        return CONTACT_HEIGHT_VOLLEY
    return CONTACT_HEIGHT_GROUNDSTROKE


def reconstruct_segment(
    start_xy_m: tuple[float, float],
    end_xy_m: tuple[float, float],
    start_z_m: float,
    end_z_m: float,
    duration_s: float,
    samples: int = 24,
) -> Trajectory3D | None:
    """
    Closed-form free-flight reconstruction between two known points.

    Args:
        start_xy_m / end_xy_m: floor positions in metres.
        start_z_m / end_z_m:   heights in metres (0.0 for a bounce).
        duration_s:            flight time between the two events.
        samples:               points to emit along the arc, for drawing.

    Returns:
        A Trajectory3D, or None when the inputs cannot describe a flight (non-positive
        duration, missing endpoints). Returning None rather than a degenerate arc keeps
        the "refuse instead of guess" convention used throughout this pipeline.
    """
    if start_xy_m is None or end_xy_m is None or duration_s <= 0:
        return None
    samples = max(1, samples)

    vx = (end_xy_m[0] - start_xy_m[0]) / duration_s
    vy = (end_xy_m[1] - start_xy_m[1]) / duration_s
    # From z(T) = z0 + vz0·T − ½gT², solved for the launch vertical velocity.
    vz = (end_z_m - start_z_m + 0.5 * GRAVITY * duration_s ** 2) / duration_s

    points: list[tuple[float, float, float]] = []
    for i in range(samples + 1):
        t = duration_s * i / samples
        points.append((
            start_xy_m[0] + vx * t,
            start_xy_m[1] + vy * t,
            start_z_m + vz * t - 0.5 * GRAVITY * t * t,
        ))

    return Trajectory3D(
        start_frame=0, end_frame=0, duration_s=duration_s,
        start=(start_xy_m[0], start_xy_m[1], start_z_m),
        end=(end_xy_m[0], end_xy_m[1], end_z_m),
        velocity=(vx, vy, vz),
        points=points,
    )


def reconstruct_rally(
    event_frames: list[int],
    ball_positions_m: dict[int, tuple[float, float]],
    shot_types: dict[int, str],
    bounce_frames: set[int],
    fps: float,
    max_flight_s: float = MAX_PLAUSIBLE_FLIGHT_S,
) -> list[Trajectory3D]:
    """
    Reconstruct every free-flight segment of a rally.

    Args:
        event_frames:     contact and bounce frames, in order.
        ball_positions_m: floor position in METRES per event frame.
        shot_types:       frame -> shot type, used to estimate contact height.
        bounce_frames:    which event frames are bounces (height 0) rather than strikes.
        fps:              video frame rate.
        max_flight_s:     segments longer than this are dropped. A gap that long means
                          an event between them was missed, and joining the two ends
                          would invent a flight that never happened.

    Returns:
        Reconstructed segments, in order. Segments that cannot be reconstructed are
        omitted rather than approximated.
    """
    if fps <= 0:
        return []

    trajectories: list[Trajectory3D] = []
    ordered = sorted(event_frames)

    for start_frame, end_frame in zip(ordered, ordered[1:]):
        duration = (end_frame - start_frame) / fps
        if duration <= 0 or duration > max_flight_s:
            continue

        start_xy = ball_positions_m.get(start_frame)
        end_xy = ball_positions_m.get(end_frame)
        if start_xy is None or end_xy is None:
            continue

        start_z = (BOUNCE_HEIGHT if start_frame in bounce_frames
                   else estimate_contact_height(shot_types.get(start_frame)))
        end_z = (BOUNCE_HEIGHT if end_frame in bounce_frames
                 else estimate_contact_height(shot_types.get(end_frame)))

        segment = reconstruct_segment(start_xy, end_xy, start_z, end_z, duration)
        if segment is None:
            continue
        segment.start_frame = start_frame
        segment.end_frame = end_frame
        trajectories.append(segment)

    return trajectories
