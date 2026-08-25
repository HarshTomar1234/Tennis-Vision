"""
utils/trajectory_3d.py
──────────────────────
Reconstructs the ball's 3-D flight path between floor-anchored events.

Why this exists - it is the speed fix, not a visualisation feature
------------------------------------------------------------------
Every speed this pipeline reports is derived from the ball's position after
projection through the *floor* homography. That projection is only valid while the
ball touches the floor, and a tennis ball is airborne for roughly 90 % of its flight.
The consequence is measurable: on a Wimbledon clip a serve contact projected to
mini-court y = -16.7 - outside the court entirely - because the camera ray through a
ball 2.7 m in the air meets the ground 25 m away. Rally speeds come out
systematically low, and no amount of threshold tuning can fix a geometry error.

Reconstructing the trajectory in 3-D removes the error at its source. The 3-D rally
viewer is a by-product of the same computation, not a separate feature.

The method, and why it needs no optimiser
-----------------------------------------
Between two events the ball is in free flight, so its horizontal motion is constant
and its vertical motion is parabolic. If both endpoints and the flight time are
known, the trajectory is *fully determined* - it is a two-point boundary value
problem with a closed-form solution:

    vx  = (x1 - x0) / T
    vy  = (y1 - y0) / T
    vz0 = (z1 - z0 + ½·g·T²) / T          from z(T) = z0 + vz0·T - ½·g·T²

No fitting, no initial guess, no convergence risk. The endpoints come from the
homography at moments when it is *valid* - a bounce is on the floor by definition,
and a player's feet are on the floor at contact - so the inputs are exactly the
measurements this pipeline can trust.

`speed_kmh` is then the true 3-D speed at the start of the segment, including the
vertical component the floor projection discarded.

Honest limits, ordered by measured size
---------------------------------------
This list used to name drag, spin and contact height, and to omit the term that turns
out to dominate all of them. `eval/speed_timing_sensitivity.py` perturbs each source by
its own measured uncertainty on the reference clip's 25 real segments:

    source of error                     mean    median   worst
    event timing (+/- 2.4 frames)      12.5%    12.2%    28.0%
    contact height (+/- 0.20 m)         0.7%     0.2%     3.5%
    ball localization (+/- 0.09 m)      0.1%     0.1%     0.7%

- **Event timing dominates, by roughly 18x over contact height.** `T` comes from event
  FRAMES, and the measured mean event offset is 2.4 frames (README, 25-clip sample).
  Speed is (distance / T), so that error passes straight through and scales as 1/T:
  flights under 0.5 s average 23.9% sensitivity, flights over 1.0 s average 6.5%. A
  reconstructed speed is therefore about as accurate as the event detector is punctual,
  and no amount of better geometry improves it.
- **Contact height is nearly free.** The +/- 0.20 m tolerance this module already called
  negligible is negligible: 0.2% median. That claim is now measured rather than asserted.
- **Ball localization is nearly free** for speed, at 0.1% median. It still bounds landing
  positions, which is a different question.
- **Drag is not modelled.** A real ball decelerates through flight, so the constant
  horizontal velocity here is an average over the segment rather than the speed at
  contact. Expect readings below a radar gun's, which measures at contact. This is
  the same caveat that applies to `utils/serve_speed.py`, and for the same reason.
- **Spin (the Magnus effect) is not modelled.** Heavy topspin bends a trajectory
  downward faster than gravity alone; this reconstruction will place the apex slightly
  high on such shots.

The ordering matters for what to fix next. Adding drag or Magnus terms to a speed whose
dominant error is a 2.4-frame timing offset would be modelling the small terms while the
large one goes unaddressed.

Not every segment is a shot
---------------------------
A rally alternates contact, bounce, contact. Only a segment that BEGINS at a racket
contact is a ball leaving a racket. A segment that begins at a bounce is the post-bounce
leg travelling to the receiver: a real part of the ball's path, correctly reconstructed,
and not a shot. See `classify_segment_speed`.
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
# inside ~1.5 s. Anything longer is not one flight - it is two or more with the events
# between them missed, or a dead-ball period between points that the ball interpolator
# bridged. A parabola stretched across several flights is both slower and taller than
# any of them, so admitting one corrupts speed and apex together.
#
# Measured on input_video_2 (30 fps, 570 frames): with no effective cap the segment
# durations ran to 1.73 s and mean speed was 57 km/h; capping at 1.5 s left 18 segments
# with durations 0.40-1.47 s and mean speed 60 km/h. Most segments (10 of 18) are now
# under 0.85 s, which is the regime a real flight occupies.
#
# 1.5 s rather than something tighter because a genuine lob does reach it - the cap is
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
        """Height at a fraction (0..1) through the flight - used for net clearance."""
        t = max(0.0, min(1.0, fraction)) * self.duration_s
        return self.start[2] + self.velocity[2] * t - 0.5 * GRAVITY * t * t


def estimate_contact_height(shot_type: str | None) -> float:
    """
    Height of the racket strike, in metres.

    A contact is the one endpoint not on the floor, so its height cannot come from
    the homography. These are population averages by shot type - deliberately coarse,
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


def crosses_net(start_xy_m, end_xy_m, net_y_m: float) -> bool:
    """
    Whether a flight passes from one side of the net to the other.

    Two contacts made by different players MUST cross the net, because the players
    stand on opposite sides. A segment joining them that stays on one side therefore
    proves an event between them was missed - the opponent's shot went undetected and
    two same-side events were joined into a flight that never happened.

    Measured on input_video_2: 7 of 16 reconstructed segments never crossed the net,
    and they clustered at 28-45 km/h while genuine crossing flights ran 65-112 km/h.
    That bimodal split is the signature of stitched-together non-flights.

    Not every same-side segment is wrong: a ball that bounces on the receiver's side
    and is then struck by the receiver legitimately stays on one side, as does a ball
    hit into the net. So this is applied only where the physics is unambiguous.
    """
    if start_xy_m is None or end_xy_m is None:
        return False
    return (start_xy_m[1] - net_y_m) * (end_xy_m[1] - net_y_m) < 0


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


# ── Speed validity ─────────────────────────────────────────────────────────────

VALID = "valid"
PLAUSIBLE_BUT_UNCERTAIN = "plausible_but_uncertain"
OUTLIER = "outlier"
NOT_A_SHOT = "not_a_shot"

# Below this flight duration the measured timing sensitivity exceeds about 20%, so the
# speed is real but should not be read to the nearest km/h. Taken from
# eval/speed_timing_sensitivity.py: flights under 0.5 s average 23.9% sensitivity to a
# 2.4-frame event offset, flights over 1.0 s average 6.5%. This is not a plausibility
# threshold on the speed itself, which would be inventing a number; it is the point where
# this pipeline's own event-timing error stops being a rounding detail.
SHORT_FLIGHT_S = 0.5


def classify_segment_speed(
    starts_at_contact: bool,
    ends_at_bounce: bool,
    crosses_the_net: bool,
    duration_s: float,
) -> tuple[str, str]:
    """
    Whether a reconstructed segment's speed is a shot speed, and how much to trust it.

    Deliberately not a threshold on km/h. A speed bound would have been the easy answer
    to a 17.6 km/h reading on the reference clip, and it would have been the wrong one:
    the reading is not a physics failure, it is a labelling failure. Three explanations
    were tested against the data before writing this (see the commit that added it):

    1. "Endpoints outside the baseline mark a bad segment." REFUTED. 18 of 25 segments
       have one, including the three fastest, because a contact endpoint is the player's
       FEET and players stand behind the baseline constantly.
    2. "Slow segments are the ones that never cross the net." True but not a defect. A
       ball that bounces on the receiver's side and is then struck by the receiver
       legitimately stays on one side, and that leg is short and slow by nature.
    3. "Shot speed is aggregating legs that are not shots." This is the real one. Of 25
       segments, only 14 begin at a racket contact. The other 11 are post-bounce legs,
       correctly reconstructed and not shots, and averaging them into a figure labelled
       "shot speed" is what produced the odd number.

    Args:
        starts_at_contact: the segment begins at a racket strike rather than a bounce.
        ends_at_bounce:    the segment ends at a floor bounce rather than a strike.
        crosses_the_net:   the flight passes from one side of the net to the other.
        duration_s:        flight time.

    Returns:
        (status, reason). Callers should present an OUTLIER or NOT_A_SHOT segment
        without a speed rather than with one.
    """
    if not starts_at_contact:
        return NOT_A_SHOT, (
            "begins at a bounce, so it is the ball travelling from the bounce to the "
            "receiver rather than a ball leaving a racket"
        )

    if ends_at_bounce and not crosses_the_net:
        # Struck, and landing on the striker's own side without crossing. In tennis that
        # is a ball into the net or a mishit, and in either case the free-flight
        # assumption is broken: the reconstruction models an uninterrupted parabola.
        # More often on real footage it means the contact was misdetected. Same physics
        # as the existing contact-to-contact net-crossing rule, one step further.
        return OUTLIER, (
            "struck and landing on the striker's own side without crossing the net, so "
            "either the ball did not complete a free flight or the contact was "
            "misdetected"
        )

    if duration_s < SHORT_FLIGHT_S:
        return PLAUSIBLE_BUT_UNCERTAIN, (
            f"flight of {duration_s:.2f} s is short enough that this pipeline's measured "
            f"2.4-frame event-timing offset moves the speed by roughly 20% or more"
        )

    return VALID, "free flight between a racket contact and a known endpoint"
