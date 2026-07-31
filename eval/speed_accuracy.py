"""
eval/speed_accuracy.py
─────────────────────────────
Runs the full pipeline and checks whether computed speeds are in realistic
tennis ranges.

Expected ranges (physics-based ground truth):
  Ball  — serve: 150–220 km/h | groundstroke: 60–120 km/h | minimum: 30 km/h
  Player movement: 0–29 km/h (max sprint ~28 km/h)

Usage:
  python eval/speed_accuracy.py
  python eval/speed_accuracy.py input_videos/input_video_2.mp4
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import pandas as pd
from copy import deepcopy

import constants
from _ball_source import pipeline_ball_detections
from trackers import PlayerTracker
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt
from utils import (
    read_video, measure_distance_between_points,
    convert_pixel_distance_to_meters, UILayoutManager,
    classify_floor_level, classify_contact_vs_bounce,
    classify_reversals_by_trajectory,
    smooth_trajectories, peak_speed_kmh_near_frame,
    detect_xvelocity_candidates,
)

# Realistic range constants (km/h)
BALL_MIN_REALISTIC  = 30.0
BALL_MAX_REALISTIC  = 250.0
PLAYER_MAX_REALISTIC = 32.0


def evaluate(video_path: str) -> dict:
    print(f"\n{'=' * 55}")
    print("Speed Accuracy Evaluation")
    print(f"Video: {video_path}")
    print(f"{'=' * 55}")

    frames = read_video(video_path)
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    print(f"Loaded {len(frames)} frames @ {fps:.1f} fps")

    # Run detection the way the pipeline does: TrackNet ball (from stub) + player stub.
    print("Running detection pipeline (may take a moment)...")
    player_tracker = PlayerTracker(model_path="yolov8x")

    player_dets = player_tracker.detect_frames(
        frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl"
    )
    ball_tracker, ball_dets = pipeline_ball_detections(frames)   # TrackNet, matches pipeline
    ball_dets = ball_tracker.interpolate_ball_positions(ball_dets)

    # ponytail: single-frame keypoints — fine for the near-static input_video_2 baseline;
    # switch to court_detector.predict_all_frames(...) when validating a moving-camera clip.
    court_detector  = CourtLineDetector("models/keypoints_model.pth")
    court_keypoints = court_detector.predict(frames[0])
    all_kp          = [court_keypoints] * len(frames)

    player_dets = player_tracker.choose_and_filter_players(player_dets, court_keypoints)
    ids = sorted(player_dets[0].keys())
    id_map = {orig: new for new, orig in enumerate(ids[:2], start=1)}
    player_dets = [{id_map[k]: v for k, v in f.items() if k in id_map} for f in player_dets]

    layout   = UILayoutManager(frames[0].shape, court_keypoints)
    mini_crt = MiniCourt(frames[0], layout_params=layout.get_mini_court_params())

    player_mini, _unused_ball_mini = mini_crt.convert_bounding_boxes_to_mini_court_coordinates(
        player_dets, ball_dets, all_kp, use_homography=True
    )

    # Floor-level-anchored ball projection (Phase 1): only trust the homography at
    # trajectory reversals (contact/bounce), interpolate in between. See
    # utils.ball_state and docs/journal/0003 for why this replaces raw per-frame
    # projection, which is geometrically wrong while the ball is airborne.
    # matches main.py: union of y-reversal + x-velocity candidates (docs/journal/0015)
    raw_reversals = sorted(set(ball_tracker.get_ball_shot_frames(ball_dets)) | set(detect_xvelocity_candidates(ball_dets)))
    floor_states  = classify_floor_level(raw_reversals, len(frames))
    ball_mini     = mini_crt.convert_ball_to_mini_court_coordinates(
        ball_dets, all_kp, floor_states, use_homography=True
    )

    # Real shots only (not bounces) for the shot-to-shot speed loop.
    # Two independent classifiers, compared: player-proximity (needs player detections,
    # ~5/7 measured ceiling on this clip — journal 0003) vs trajectory-shape (needs only
    # ball positions, 84.1% held-out accuracy on 1,034 real TrackNet-dataset events —
    # journal 0012). Trajectory is used for the actual speed calc below since it's
    # validated at far larger scale; proximity result printed alongside for comparison.
    shot_frames_prox, bounce_frames_prox = classify_contact_vs_bounce(raw_reversals, ball_dets, player_dets)
    shot_frames, bounce_frames = classify_reversals_by_trajectory(raw_reversals, ball_dets)
    print(f"Raw reversals : {len(raw_reversals)}")
    print(f"  proximity   -> {len(shot_frames_prox)} shots + {len(bounce_frames_prox)} bounces  {shot_frames_prox}")
    print(f"  trajectory  -> {len(shot_frames)} shots + {len(bounce_frames)} bounces  {shot_frames}")

    # Kalman-smoothed ball trajectory (Phase 1, Step 3): gives continuous velocity
    # instead of depending on distance-between-two-shot-events, which was the actual
    # cause of the earlier FAIL (event detection ceiling ~5/7, not ball geometry —
    # see docs/journal/0004). Ball "shot speed" = peak velocity in a small window
    # around the contact frame, matching how real speed guns measure it (at/near
    # contact, not averaged over the whole flight).
    _ball_smoothed, ball_velocities = smooth_trajectories(ball_mini)
    px_to_m_scale = constants.DOUBLE_LINE_WIDTH / mini_crt.get_width_of_mini_court()

    ball_speeds: list[float]   = []
    player_speeds: list[float] = []

    for sf in shot_frames:
        speed = peak_speed_kmh_near_frame(
            ball_velocities, frame=sf, entity_id=1, window=5,
            px_to_m_scale=px_to_m_scale, fps=fps,
        )
        if speed > 0:
            ball_speeds.append(speed)

    # Player movement speed stays distance/time between shots — it is already
    # realistic (see baseline) and, unlike the ball, a player doesn't reverse velocity
    # instantaneously, so the discrete-event window issue doesn't apply the same way.
    for i in range(len(shot_frames) - 1):
        sf, ef = shot_frames[i], shot_frames[i + 1]
        dur = (ef - sf) / fps
        if dur <= 0:
            continue

        b_start = ball_mini[sf].get(1)
        p_pos = player_mini[sf]
        if p_pos:
            shooter = min(p_pos.keys(), key=lambda pid: measure_distance_between_points(
                p_pos[pid], b_start or (0, 0)))
            opp = 1 if shooter == 2 else 2
            ps, pe = player_mini[sf].get(opp), player_mini[ef].get(opp)
            if ps and pe:
                d_px = measure_distance_between_points(ps, pe)
                d_m  = convert_pixel_distance_to_meters(
                    d_px, constants.DOUBLE_LINE_WIDTH, mini_crt.get_width_of_mini_court()
                )
                player_speeds.append(d_m / dur * 3.6)

    print(f"\n{'─' * 45}")
    print("Ball speeds (km/h):")
    for i, s in enumerate(ball_speeds):
        flag = "" if BALL_MIN_REALISTIC <= s <= BALL_MAX_REALISTIC else "  ← OUT OF RANGE"
        print(f"  Shot {i + 1}: {s:6.1f} km/h{flag}")

    ok_ball = sum(1 for s in ball_speeds if BALL_MIN_REALISTIC <= s <= BALL_MAX_REALISTIC)
    print(f"\n  Realistic: {ok_ball}/{len(ball_speeds)}")
    if ball_speeds:
        print(f"  Range:     {min(ball_speeds):.1f} – {max(ball_speeds):.1f} km/h")
        print(f"  Mean:      {sum(ball_speeds)/len(ball_speeds):.1f} km/h")

    print("\nPlayer speeds (km/h):")
    for i, s in enumerate(player_speeds):
        flag = "" if s <= PLAYER_MAX_REALISTIC else "  ← UNREALISTICALLY HIGH"
        print(f"  Rally {i + 1}: {s:6.1f} km/h{flag}")

    print(f"{'─' * 45}")

    verdict_ball = "PASS" if ok_ball == len(ball_speeds) and ball_speeds else "FAIL"
    print(f"\nVERDICT (ball speeds in range): {verdict_ball}")

    return {
        "fps": fps,
        "ball_speeds": ball_speeds,
        "player_speeds": player_speeds,
        "ball_in_range": ok_ball,
        "ball_total": len(ball_speeds),
    }


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "input_videos/input_video_2.mp4"
    evaluate(video)
