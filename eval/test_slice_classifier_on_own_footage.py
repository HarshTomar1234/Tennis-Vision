"""
eval/test_slice_classifier_on_own_footage.py
────────────────────────────────────────────────
Applies the THETIS-trained slice classifier (models/slice_classifier.json,
eval/train_slice_classifier.py) to real shots on our own footage -- both input videos -- to see
whether the domain gap flagged in 0024 is really structural, or was partly an artifact
of the earlier sanity check's own bug (that check reused ONE frame's player bbox across
a 40-frame window instead of tracking the player per-frame, which would starve pose
coverage regardless of any real domain gap).

video_2 (input_video_2.mp4): uses the 7 hand-labeled real shot frames from
shot_frame_accuracy.py's BUILTIN_GT, with per-frame player bboxes from the cached
tracker stub -- shooter picked as whichever player is closer to the ball at the shot
frame (mirrors main.py's own logic).

video (input_video.mp4): no cached stubs exist. Runs YOLOv8x fresh (single video, ~1-2
min) and uses shot frames visually identified during earlier frame-by-frame inspection
of this clip (frames ~120 and ~160 showed clear groundstroke swings on inspection).
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from utils import PoseEstimator, measure_distance_between_points
from eval.explore_slice_features import hitting_wrist_trace, linreg_slope

WEIGHTS_PATH = "models/slice_classifier.json"
WINDOW = 20


def classify_window(video_path, player_bbox_by_frame, shot_frame, est, window=WINDOW):
    cap = cv2.VideoCapture(video_path)
    lo, hi = max(0, shot_frame - window), shot_frame + window
    sequence = []
    coverage_frames = 0
    for f in range(lo, hi):
        bbox = player_bbox_by_frame.get(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret or bbox is None:
            sequence.append(None)
            continue
        landmarks = est.detect_in_bbox(frame, bbox)
        sequence.append(landmarks)
        if landmarks:
            coverage_frames += 1
    cap.release()

    trace, _ = hitting_wrist_trace(sequence)
    pts = [(i, p) for i, p in enumerate(trace) if p] if trace else []
    result = {
        "shot_frame": shot_frame,
        "window_frames": hi - lo,
        "pose_coverage": coverage_frames,
        "usable_trace_points": len(pts),
    }
    if len(pts) < 10:
        result["prediction"] = None
        result["reason"] = "too few usable trace points (<10)"
        return result

    ys = [p[1] for _, p in pts]
    n = len(pts)
    last_third = ys[int(n * 2 / 3):]
    slope = linreg_slope(last_third) if len(last_third) >= 2 else None
    if slope is None:
        result["prediction"] = None
        result["reason"] = "last third too short"
        return result

    weights = json.loads(Path(WEIGHTS_PATH).read_text())
    is_slice = slope <= weights["threshold"]
    result["last_third_slope"] = slope
    result["threshold"] = weights["threshold"]
    result["prediction"] = "SLICE" if is_slice else "NOT SLICE (flat/topspin)"
    return result


def test_video_2(est):
    print("=" * 60)
    print("input_video_2.mp4 -- 7 hand-labeled real shots")
    print("=" * 60)

    with open("tracker_stubs/player_detections.pkl", "rb") as f:
        pdets = pickle.load(f)
    with open("tracker_stubs/ball_detections_tracknet.pkl", "rb") as f:
        bdets_raw = pickle.load(f)

    from trackers.tracknet_ball_tracker import TrackNetBallTracker
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    bdets = tracker.interpolate_ball_positions(bdets_raw)

    gt_shots = [27, 110, 209, 282, 384, 429, 482]

    for shot_frame in gt_shots:
        ball_bbox = bdets[shot_frame].get(1)
        if ball_bbox is None:
            print(f"frame {shot_frame}: no ball position, skipping")
            continue
        ball_xy = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)

        p1_bbox = pdets[shot_frame].get(1)
        p2_bbox = pdets[shot_frame].get(2)
        d1 = measure_distance_between_points(
            ((p1_bbox[0] + p1_bbox[2]) / 2, (p1_bbox[1] + p1_bbox[3]) / 2), ball_xy
        ) if p1_bbox else float("inf")
        d2 = measure_distance_between_points(
            ((p2_bbox[0] + p2_bbox[2]) / 2, (p2_bbox[1] + p2_bbox[3]) / 2), ball_xy
        ) if p2_bbox else float("inf")
        shooter = 1 if d1 <= d2 else 2

        player_bbox_by_frame = {f: pdets[f].get(shooter) for f in range(len(pdets))}
        result = classify_window("input_videos/input_video_2.mp4", player_bbox_by_frame, shot_frame, est)
        result["shooter"] = shooter
        print(f"frame {shot_frame} (player {shooter}): {result}")


def test_video_1(est):
    print()
    print("=" * 60)
    print("input_video.mp4 -- visually-identified groundstroke swings")
    print("=" * 60)

    from utils import read_video
    from trackers import PlayerTracker

    frames = read_video("input_videos/input_video.mp4")
    tracker = PlayerTracker(model_path="yolov8x")
    pdets = tracker.detect_frames(frames, read_from_stub=False, stub_path=None)

    # frame 120: bottom player mid-run into a forehand (visually confirmed, journal 0021 grid images)
    # frame 160: bottom player at contact, forehand groundstroke (visually confirmed)
    for shot_frame in (120, 160):
        # bottom player = larger y-center, consistent through this clip
        candidates = pdets[shot_frame]
        if not candidates:
            print(f"frame {shot_frame}: no detections")
            continue
        bottom_id = max(candidates, key=lambda tid: (candidates[tid][1] + candidates[tid][3]) / 2)

        player_bbox_by_frame = {f: pdets[f].get(bottom_id) for f in range(len(pdets))}
        result = classify_window("input_videos/input_video.mp4", player_bbox_by_frame, shot_frame, est)
        result["track_id"] = bottom_id
        print(f"frame {shot_frame} (track_id {bottom_id}): {result}")


def main():
    est = PoseEstimator("models/pose_landmarker_lite.task", bbox_padding=0.45)
    if not est.available:
        print("Pose model unavailable, aborting.")
        return

    test_video_2(est)
    test_video_1(est)
    est.close()


if __name__ == "__main__":
    main()
