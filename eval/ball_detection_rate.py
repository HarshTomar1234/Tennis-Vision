"""
eval/ball_detection_rate.py
─────────────────────────────
Measures raw (pre-interpolation) ball detection rate for a video.
Run before and after switching to TrackNet to confirm improvement.

Usage:
  python eval/ball_detection_rate.py                        # YOLO baseline
  python eval/ball_detection_rate.py --tracker tracknet     # TrackNet
  python eval/ball_detection_rate.py input_videos/clip.mp4  # custom video

Targets:
  YOLO baseline  : ~16.5%
  TrackNet target: >60%
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_video


def _build_tracker(tracker: str, model_path: str):
    if tracker == "tracknet":
        from trackers.tracknet_ball_tracker import TrackNetBallTracker
        return TrackNetBallTracker(model_path=model_path)
    from trackers import BallTracker
    return BallTracker(model_path=model_path)


def evaluate(video_path: str, tracker: str = "yolo",
             model_path: str | None = None) -> dict:
    defaults = {
        "yolo":     "models/last.pt",
        "tracknet": "models/tracknet.pt",
    }
    model_path = model_path or defaults.get(tracker, defaults["yolo"])

    print(f"\n{'=' * 55}")
    print(f"Ball Detection Rate Evaluation")
    print(f"Video   : {video_path}")
    print(f"Tracker : {tracker.upper()}  ({model_path})")
    print(f"{'=' * 55}")

    print("Loading video...")
    frames = read_video(video_path)
    total  = len(frames)
    print(f"Frames  : {total}")

    print(f"Running {tracker.upper()} detection (no stubs, no interpolation)...")
    tracker_obj = _build_tracker(tracker, model_path)
    raw_detections = tracker_obj.detect_frames(frames, read_from_stub=False, stub_path=None)

    detected = sum(1 for d in raw_detections if d.get(1))
    missing  = total - detected
    det_pct  = 100 * detected / total
    miss_pct = 100 * missing  / total

    print(f"\n{'─' * 45}")
    print(f"  Detected (real)    : {detected:4d} / {total}  ({det_pct:.1f}%)")
    print(f"  Missing/guessed    : {missing:4d} / {total}  ({miss_pct:.1f}%)")

    # Consecutive-detection streaks
    streaks: list[int] = []
    current = 0
    for d in raw_detections:
        if d.get(1):
            current += 1
        else:
            if current:
                streaks.append(current)
            current = 0
    if current:
        streaks.append(current)

    if streaks:
        avg_streak = sum(streaks) / len(streaks)
        print(f"  Detection streaks  : {len(streaks)} bursts, avg {avg_streak:.1f} frames")
        print(f"  Longest streak     : {max(streaks)} frames")
    else:
        print("  No detection streaks found")

    # Shot frame coverage
    interp      = tracker_obj.interpolate_ball_positions(raw_detections)
    shot_frames = tracker_obj.get_ball_shot_frames(interp)
    real_at_shots = sum(1 for sf in shot_frames if raw_detections[sf].get(1))
    print(f"\n  Shot frames        : {len(shot_frames)} detected")
    print(f"  Real ball at shot  : {real_at_shots}/{len(shot_frames)} "
          f"({100*real_at_shots/max(len(shot_frames), 1):.0f}%)")
    print(f"{'─' * 45}")

    if det_pct >= 60:
        print("VERDICT: GOOD - >60% real detections (acceptable for speed calc)")
    elif det_pct >= 30:
        print("VERDICT: MARGINAL - 30–60% (speeds partially guessed)")
    else:
        print("VERDICT: POOR - <30% real detections (speeds unreliable)")

    return {
        "tracker":       tracker,
        "total_frames":  total,
        "detected":      detected,
        "detection_pct": round(det_pct, 1),
        "shot_frames":   shot_frames,
        "real_at_shots": real_at_shots,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ball detection rate benchmark")
    parser.add_argument("video",   nargs="?",
                        default="input_videos/input_video_2.mp4",
                        help="Path to input video")
    parser.add_argument("--tracker", choices=["yolo", "tracknet"],
                        default="yolo",
                        help="Which tracker to benchmark (default: yolo)")
    parser.add_argument("--model",   default=None,
                        help="Override model path")
    args = parser.parse_args()

    evaluate(args.video, tracker=args.tracker, model_path=args.model)
