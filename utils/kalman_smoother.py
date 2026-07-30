"""
utils/kalman_smoother.py
────────────────────────
Constant-velocity Kalman filter for 2-D mini-court positions (player and ball).

Two jobs, both from the same filter:

1. Jitter reduction — a stationary or steadily-moving point stops visibly wobbling
   frame to frame (João's feedback: "denoise the projections to real world coordinates
   using something like a Kalman filter").

2. Instantaneous velocity — the filter's [vx, vy] state gives continuous ball/player
   speed without depending on correctly identifying discrete shot events. This turned
   out to matter more than expected: speed_accuracy.py was failing because the
   distance-between-two-shot-frames approach is only as good as shot-event detection
   (measured ceiling ~5/7 — see docs/journal/0003 and 0004). Kalman velocity sidesteps
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
    Smooth a {frame_num: {entity_id: (x, y)}} sequence — one independent Kalman filter
    per entity id (each player, or the ball), since they move independently.

    Accepts either a dict keyed by frame number (what MiniCourt's coordinate-conversion
    methods return) or a plain list indexed by frame — both conventions exist in this
    codebase. Frames are always processed in increasing frame-number order.

    Args:
        positions: per-frame positions, e.g. player_mini_court or ball_mini_court.
        process_noise, measurement_noise: filter tuning (see PositionKalmanFilter).

    Returns:
        (smoothed_positions, velocities) — dicts keyed by frame number, same shape as
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
) -> float:
    """
    Peak ball/player speed (km/h) in a small window around `frame`.

    Contact/bounce frames are the exact instant the velocity direction reverses, so the
    Kalman velocity *at* that frame is mid-transition. The ball's real travel speed is
    better represented by the peak speed in a small surrounding window (it was
    approaching at that speed just before contact).

    Args:
        velocities:    output of smooth_trajectories (px/frame, mini-court space);
                       dict keyed by frame number, or a plain list.
        frame:         the contact/bounce frame to inspect.
        entity_id:     which tracked id (e.g. 1 for the ball).
        window:        frames to check on each side of `frame`.
        px_to_m_scale: metres per mini-court pixel (DOUBLE_LINE_WIDTH / court_drawing_width).
        fps:           video frame rate.

    Returns:
        Peak speed in km/h over the window (0.0 if no data in range).
    """
    if isinstance(velocities, list):
        velocities = dict(enumerate(velocities))

    peak_px_per_frame = 0.0
    for f in range(frame - window, frame + window + 1):
        vx, vy = velocities.get(f, {}).get(entity_id, (0.0, 0.0))
        speed = (vx ** 2 + vy ** 2) ** 0.5
        peak_px_per_frame = max(peak_px_per_frame, speed)

    m_per_frame = peak_px_per_frame * px_to_m_scale
    return m_per_frame * fps * 3.6
