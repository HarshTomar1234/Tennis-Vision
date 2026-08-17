"""
utils/kalman_smoother.py
────────────────────────
Constant-velocity Kalman filter for 2-D mini-court positions (player and ball).

Two jobs, both from the same filter:

1. Jitter reduction - a stationary or steadily-moving point stops visibly wobbling
   frame to frame (João's feedback: "denoise the projections to real world coordinates
   using something like a Kalman filter").

2. Instantaneous velocity - the filter's [vx, vy] state gives continuous ball/player
   speed without depending on correctly identifying discrete shot events. This turned
   out to matter more than expected: speed_accuracy.py was failing because the
   distance-between-two-shot-frames approach is only as good as shot-event detection
   (measured ceiling ~5/7 - see docs/journal/0003 and 0004). Kalman velocity sidesteps
   that entirely by estimating speed continuously from the whole trajectory.

Uses cv2.KalmanFilter (already a dependency via opencv-python) rather than adding
filterpy for one class's worth of functionality.
"""
from __future__ import annotations

import cv2
import numpy as np


class PositionKalmanFilter:
    """
    Constant-velocity Kalman filter over one 2-D trajectory.

    State:       [x, y, vx, vy]   (position in px, velocity in px/frame)
    Measurement: [x, y]
    """

    def __init__(self, process_noise: float = 1e-2, measurement_noise: float = 1e-1):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], dtype=np.float32)
        kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self._kf = kf
        self._initialized = False

    def update(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Predict + correct with a new measurement. Returns (x, y, vx, vy) smoothed."""
        if not self._initialized:
            # Column vector, not a flat array: OpenCV 5 enforces the (4, 1) state shape
            # that OpenCV 4 accepted loosely. mediapipe pulls in opencv 5, so this must
            # be explicit.
            self._kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self._initialized = True

        self._kf.predict()
        corrected = self._kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        cx, cy, vx, vy = corrected.flatten()
        return float(cx), float(cy), float(vx), float(vy)

    def predict_only(self) -> tuple[float, float, float, float]:
        """Advance one frame with no measurement (e.g. a gap in detections)."""
        state = self._kf.predict()
        x, y, vx, vy = state.flatten()
        return float(x), float(y), float(vx), float(vy)


def smooth_trajectories(
    positions: dict[int, dict[int, tuple[float, float]]] | list[dict[int, tuple[float, float]]],
    process_noise: float = 1e-2,
    measurement_noise: float = 1e-1,
) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, dict[int, tuple[float, float]]]]:
    """
    Smooth a {frame_num: {entity_id: (x, y)}} sequence - one independent Kalman filter
    per entity id (each player, or the ball), since they move independently.

    Accepts either a dict keyed by frame number (what MiniCourt's coordinate-conversion
    methods return) or a plain list indexed by frame - both conventions exist in this
    codebase. Frames are always processed in increasing frame-number order.

    Args:
        positions: per-frame positions, e.g. player_mini_court or ball_mini_court.
        process_noise, measurement_noise: filter tuning (see PositionKalmanFilter).

    Returns:
        (smoothed_positions, velocities) - dicts keyed by frame number, same shape as
        the input. velocities values are (vx, vy) in px/frame (mini-court space).
    """
    if isinstance(positions, list):
        positions = dict(enumerate(positions))

    filters: dict[int, PositionKalmanFilter] = {}
    smoothed:   dict[int, dict[int, tuple[float, float]]] = {}
    velocities: dict[int, dict[int, tuple[float, float]]] = {}

    for frame_num in sorted(positions):
        frame_positions = positions[frame_num]
        frame_smoothed: dict[int, tuple[float, float]] = {}
        frame_velocity: dict[int, tuple[float, float]] = {}

        for entity_id, pos in frame_positions.items():
            if entity_id not in filters:
                filters[entity_id] = PositionKalmanFilter(process_noise, measurement_noise)

            x, y = float(pos[0]), float(pos[1])
            sx, sy, vx, vy = filters[entity_id].update(x, y)
            frame_smoothed[entity_id] = (sx, sy)
            frame_velocity[entity_id] = (vx, vy)

        smoothed[frame_num]   = frame_smoothed
        velocities[frame_num] = frame_velocity

    return smoothed, velocities


def peak_speed_kmh_near_frame(
    velocities: dict[int, dict[int, tuple[float, float]]] | list[dict[int, tuple[float, float]]],
    frame: int,
    entity_id: int,
    window: int,
    px_to_m_scale: float,
    fps: float,
    max_realistic_kmh: float | None = None,
) -> float:
    """
    Peak ball/player speed (km/h) in a small window around `frame`.

    Contact/bounce frames are the exact instant the velocity direction reverses, so the
    Kalman velocity *at* that frame is mid-transition. The ball's real travel speed is
    better represented by the peak speed in a small surrounding window (it was
    approaching at that speed just before contact).

    Taking a bare max over the window is vulnerable to a single noisy raw detection
    (tracker jitter, not real motion) producing a physically impossible reading -- found
    on our own clip: one candidate frame sat in a jittery stretch of raw TrackNet
    detections and reported 371.7 km/h for a groundstroke. This isn't specific to that
    clip or to the candidate that triggered it: raw per-frame detection noise near an
    event frame can happen on any input, so `max_realistic_kmh` lets a caller reject the
    result rather than pass tracking noise off as a measurement (see docs/journal/0015,
    same "drop don't guess" convention as classify_reversals_by_trajectory).

    Args:
        velocities:    output of smooth_trajectories (px/frame, mini-court space);
                       dict keyed by frame number, or a plain list.
        frame:         the contact/bounce frame to inspect.
        entity_id:     which tracked id (e.g. 1 for the ball).
        window:        frames to check on each side of `frame`.
        px_to_m_scale: metres per mini-court pixel (DOUBLE_LINE_WIDTH / court_drawing_width).
        fps:           video frame rate.
        max_realistic_kmh: if given, treat a peak above this as tracking noise, not a
                       measurement (returns 0.0, same sentinel as "no data").

    Returns:
        Peak speed in km/h over the window (0.0 if no data in range, or if the peak
        exceeds max_realistic_kmh).
    """
    if isinstance(velocities, list):
        velocities = dict(enumerate(velocities))

    peak_px_per_frame = 0.0
    for f in range(frame - window, frame + window + 1):
        vx, vy = velocities.get(f, {}).get(entity_id, (0.0, 0.0))
        speed = (vx ** 2 + vy ** 2) ** 0.5
        peak_px_per_frame = max(peak_px_per_frame, speed)

    m_per_frame = peak_px_per_frame * px_to_m_scale
    kmh = m_per_frame * fps * 3.6

    if max_realistic_kmh is not None and kmh > max_realistic_kmh:
        return 0.0
    return kmh


# ── Fixed-interval (Rauch-Tung-Striebel) smoothing ───────────────────────────
#
# The filter above is causal: at frame k it has only seen frames up to k. That is the
# right constraint for live tracking and the wrong one here, because the whole video is
# on disk before analysis starts. A gap in detections is the clearest case - a forward
# filter entering one can only coast in a straight line at the last known velocity,
# while the frames AFTER the gap say exactly where the ball came out.
#
# The RTS smoother adds a backward pass that redistributes that future information into
# the past. It is the standard fixed-interval smoother for a linear Gaussian model and
# it is exact for one: no extra tuning, no new dependency, ~40 lines.
#
# Implemented directly rather than through cv2.KalmanFilter because the recursion needs
# the predicted covariance at each step (P_{k+1|k}), and driving OpenCV's filter while
# harvesting its internals is both fragile and harder to read than the equations.
#
# NOT enabled in the pipeline, because measurement says it does not earn its place yet.
# Against the TrackNet dataset's own labels (8 clips, 966 frames, via
# eval/ball_localization_accuracy.py --smooth):
#
#     configuration                coverage   recall@5px   median err
#     raw detections                  89.8%       36.9%        6.3px
#     RTS across the whole clip      100.0%       38.1%        6.7px
#     RTS per free-flight span       100.0%       38.4%        6.6px
#
# Two findings worth keeping.
#
# Smoothing ACROSS a contact is measurably worse than smoothing between contacts, at
# every setting tried. A racket hit or bounce changes the velocity discontinuously, so a
# constant-velocity smoother run through one blends the incoming and outgoing velocities
# and pulls the estimate off the truth on both sides. Within the sweep, the least
# aggressive setting always won, which is the same fact seen from another angle.
#
# But even applied per span it does not improve median error, and the reason is that
# TrackNet's error is not Gaussian: median 5.8px against a 90th percentile of 19.4px is a
# heavy tail of gross mislocalizations, usually the detector locking onto the wrong
# object. A Kalman smoother assumes zero-mean Gaussian noise, so it spreads those
# outliers into neighbouring good frames rather than rejecting them. The missing step is
# an outlier gate before smoothing (a chi-square test on the innovation against the
# predicted covariance), which would drop those measurements instead of averaging them in.
# Until that exists, smoothing buys complete coverage and about 1.5 points of recall for
# 4% worse median error, which is not a trade worth making by default.

# Constant-velocity model, shared by the forward and backward passes.
_A = np.array([[1.0, 0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0, 1.0],
               [0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 1.0]])
_H = np.array([[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0]])


def rts_smooth(
    measurements: list[tuple[float, float] | None],
    process_noise: float = 1e-2,
    measurement_noise: float = 1e-1,
) -> list[tuple[float, float, float, float]]:
    """
    Forward-backward smooth a 2-D trajectory that may have gaps.

    Args:
        measurements:       per-frame (x, y), or None where nothing was detected. Gaps are
                            handled by predicting through them, so the smoothed result is
                            informed by the frames on BOTH sides of a gap.
        process_noise:      how much the constant-velocity assumption is trusted. Larger
                            follows the measurements more closely.
        measurement_noise:  assumed detector error. Larger smooths harder.

    Returns:
        One (x, y, vx, vy) per input frame, including for frames that had no measurement.
        Returns an empty list for empty input, and passes a single measurement through
        with zero velocity (a smoother needs at least two frames to say anything).
    """
    n = len(measurements)
    if n == 0:
        return []

    Q = np.eye(4) * float(process_noise)
    R = np.eye(2) * float(measurement_noise)

    # Start from the first real measurement, at rest. Without a real starting point the
    # filter spends the first frames dragging itself in from the origin.
    first = next((m for m in measurements if m is not None), None)
    if first is None:
        return [(0.0, 0.0, 0.0, 0.0)] * n

    x = np.array([first[0], first[1], 0.0, 0.0])
    P = np.eye(4) * 1e3          # near-total ignorance about the initial velocity

    # ── forward pass, keeping what the backward pass needs ──────────────────
    x_filt, P_filt, x_pred, P_pred = [], [], [], []
    for z in measurements:
        xp = _A @ x
        Pp = _A @ P @ _A.T + Q
        x_pred.append(xp)
        P_pred.append(Pp)

        if z is None:
            # No measurement: the prediction IS the estimate, and the covariance grows.
            x, P = xp, Pp
        else:
            S = _H @ Pp @ _H.T + R
            K = Pp @ _H.T @ np.linalg.inv(S)
            x = xp + K @ (np.asarray(z, dtype=float) - _H @ xp)
            P = Pp - K @ _H @ Pp

        x_filt.append(x)
        P_filt.append(P)

    # ── backward pass ───────────────────────────────────────────────────────
    # The last frame has no future to learn from, so it starts as the filtered estimate
    # and every earlier frame is corrected towards what the frames after it implied.
    xs = [None] * n
    Ps = [None] * n
    xs[-1], Ps[-1] = x_filt[-1], P_filt[-1]

    for k in range(n - 2, -1, -1):
        # Gain of the smoother: how much the next frame's correction propagates back.
        # pinv rather than inv because a long detection gap can leave P_pred
        # ill-conditioned, and a LinAlgError here would lose the whole trajectory.
        C = P_filt[k] @ _A.T @ np.linalg.pinv(P_pred[k + 1])
        xs[k] = x_filt[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P_filt[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T

    return [(float(s[0]), float(s[1]), float(s[2]), float(s[3])) for s in xs]
