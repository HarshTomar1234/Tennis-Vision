"""
tests/test_kalman_smoother.py
Unit tests for the constant-velocity Kalman smoother.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

from utils.kalman_smoother import (
    PositionKalmanFilter,
    smooth_trajectories,
    peak_speed_kmh_near_frame,
)


def test_converges_to_true_velocity_on_constant_motion():
    kf = PositionKalmanFilter(process_noise=1e-2, measurement_noise=1e-1)
    vx_true, vy_true = 2.0, 3.0
    x, y = 0.0, 0.0

    vx_est = vy_est = 0.0
    for _ in range(60):
        x += vx_true
        y += vy_true
        _, _, vx_est, vy_est = kf.update(x, y)

    assert abs(vx_est - vx_true) < 0.3
    assert abs(vy_est - vy_true) < 0.3


def test_smooths_noisy_measurements_around_a_line():
    random.seed(0)
    kf = PositionKalmanFilter(process_noise=1e-2, measurement_noise=2.0)

    true_pts = [(float(i), float(i) * 2) for i in range(40)]
    noisy_pts = [(x + random.uniform(-3, 3), y + random.uniform(-3, 3)) for x, y in true_pts]

    smoothed = [kf.update(x, y)[:2] for x, y in noisy_pts]

    def mse(pts):
        return sum((px - tx) ** 2 + (py - ty) ** 2 for (px, py), (tx, ty) in zip(pts, true_pts)) / len(pts)

    # skip the first few frames (filter still converging) for a fair comparison
    assert mse(smoothed[10:]) < mse(noisy_pts[10:])


def test_smooth_trajectories_keeps_entities_independent():
    positions = [
        {1: (0.0, 0.0), 2: (100.0, 100.0)},
        {1: (1.0, 1.0), 2: (99.0, 101.0)},
        {1: (2.0, 2.0), 2: (98.0, 102.0)},
    ]
    smoothed, velocities = smooth_trajectories(positions)

    assert len(smoothed) == 3
    assert set(smoothed[0].keys()) == {1, 2}
    # player 2 moving in the opposite direction shouldn't leak into player 1's velocity
    assert velocities[2][1][0] > 0    # entity 1 moving in +x
    assert velocities[2][2][0] < 0    # entity 2 moving in -x


def test_smooth_trajectories_handles_missing_entity_in_some_frames():
    positions = [
        {1: (0.0, 0.0)},
        {},                    # entity 1 not detected this frame
        {1: (2.0, 2.0)},
    ]
    smoothed, velocities = smooth_trajectories(positions)
    assert smoothed[1] == {}   # no measurement -> no output for that frame
    assert 1 in smoothed[2]


def test_peak_speed_kmh_near_frame_picks_the_local_max():
    # constant scale: 1 px = 0.01 m ; fps = 30 -> easy numbers to check by hand
    velocities = [
        {1: (0.0, 0.0)},
        {1: (10.0, 0.0)},   # 10 px/frame
        {1: (5.0, 0.0)},
        {1: (2.0, 0.0)},
    ]
    speed = peak_speed_kmh_near_frame(
        velocities, frame=2, entity_id=1, window=2,
        px_to_m_scale=0.01, fps=30,
    )
    # peak in [0,4] window is 10 px/frame -> 0.1 m/frame -> 3 m/s -> 10.8 km/h
    assert abs(speed - 10.8) < 0.1


def test_peak_speed_returns_zero_when_no_data():
    speed = peak_speed_kmh_near_frame([{}], frame=0, entity_id=1, window=1,
                                       px_to_m_scale=0.01, fps=30)
    assert speed == 0.0


def test_peak_speed_rejects_implausible_reading():
    # same setup as the local-max test, but scaled up to a physically impossible speed
    velocities = [{1: (0.0, 0.0)}, {1: (500.0, 0.0)}, {1: (5.0, 0.0)}]
    speed = peak_speed_kmh_near_frame(
        velocities, frame=1, entity_id=1, window=1,
        px_to_m_scale=0.01, fps=30, max_realistic_kmh=260.0,
    )
    assert speed == 0.0


def test_peak_speed_keeps_plausible_reading_with_gate_set():
    velocities = [{1: (0.0, 0.0)}, {1: (10.0, 0.0)}, {1: (5.0, 0.0)}]
    speed = peak_speed_kmh_near_frame(
        velocities, frame=1, entity_id=1, window=1,
        px_to_m_scale=0.01, fps=30, max_realistic_kmh=260.0,
    )
    assert speed > 0
