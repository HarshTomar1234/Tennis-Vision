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
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt
from utils import (
    read_video, measure_distance_between_points,
    convert_pixel_distance_to_meters, UILayoutManager,
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

    # Run full detection pipeline (using stubs where available)
    print("Running detection pipeline (may take a moment)...")
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker   = BallTracker(model_path="models/last.pt")

    player_dets = player_tracker.detect_frames(
        frames, read_from_stub=True, stub_path="tracker_stubs/player_detections.pkl"
    )
    ball_dets = ball_tracker.detect_frames_with_tracking(
        frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections_tracked.pkl"
    )
    ball_dets = ball_tracker.interpolate_ball_positions(ball_dets)

    court_detector  = CourtLineDetector("models/keypoints_model.pth")
    court_keypoints = court_detector.predict(frames[0])
    all_kp          = [court_keypoints] * len(frames)

    player_dets = player_tracker.choose_and_filter_players(player_dets, court_keypoints)
    ids = sorted(player_dets[0].keys())
    id_map = {orig: new for new, orig in enumerate(ids[:2], start=1)}
    player_dets = [{id_map[k]: v for k, v in f.items() if k in id_map} for f in player_dets]

    layout   = UILayoutManager(frames[0].shape, court_keypoints)
    mini_crt = MiniCourt(frames[0], layout_params=layout.get_mini_court_params())

    player_mini, ball_mini = mini_crt.convert_bounding_boxes_to_mini_court_coordinates(
        player_dets, ball_dets, all_kp, use_homography=True
    )

    shot_frames = ball_tracker.get_ball_shot_frames(ball_dets)
    print(f"Shot frames: {shot_frames}")

    ball_speeds: list[float]   = []
    player_speeds: list[float] = []

    for i in range(len(shot_frames) - 1):
        sf, ef = shot_frames[i], shot_frames[i + 1]
        dur = (ef - sf) / fps
        if dur <= 0:
            continue

        b_start = ball_mini[sf].get(1)
        b_end   = ball_mini[ef].get(1)
        if b_start and b_end:
            d_px = measure_distance_between_points(b_start, b_end)
            d_m  = convert_pixel_distance_to_meters(
                d_px, constants.DOUBLE_LINE_WIDTH, mini_crt.get_width_of_mini_court()
            )
            ball_speeds.append(d_m / dur * 3.6)

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
