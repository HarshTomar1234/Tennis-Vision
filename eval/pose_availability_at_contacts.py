"""
eval/pose_availability_at_contacts.py
─────────────────────────────────────
How often is pose actually usable at the moment a shot is struck?

Why this is the number that matters for shot classification
-----------------------------------------------------------
Forehand versus backhand has no physical rule. Unlike the serve (ball above the head, at a
baseline), unlike the volley (no bounce since the last contact), unlike the smash (above
the head, not at a baseline), there is nothing in the ball's trajectory that says which
side of the body a stroke came off. It has to be read from the player, which in this
pipeline means MediaPipe pose at the contact frame.

That makes pose availability a hard ceiling. If pose is unavailable on half the contacts,
then a pose-based forehand/backhand classifier cannot label more than half the shots
however good it is, and the rest fall back to a position rule that has never had ground
truth.

This was noticed while calibrating the swing generator
(eval/swing_candidate_recall.py): of 7 hand-labelled contacts on the reference clip, only
4 produced a computable pose. That is a small enough sample to be luck, so this measures
it across the eval suite.

The contact frame is the worst possible moment to ask for a pose. The player is fully
extended, often rotated side-on, frequently occluded by their own racket arm, and moving
fast enough to motion-blur at broadcast shutter speeds. A window is therefore reported
alongside the exact frame: if pose is available a few frames either side, a classifier can
use the nearest usable frame instead of giving up.

Usage
-----
    python eval/pose_availability_at_contacts.py
    python eval/pose_availability_at_contacts.py --clips "datasets/eval_clips/*.mp4"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court_line_detector import CourtLineDetector
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils import PoseEstimator, derive_shot_frames, read_video, stub_path_for_video
from utils.bbox_utils import get_center_of_bbox, measure_distance_between_points

# Frames either side of a contact to look for a usable pose when the exact frame fails.
FALLBACK_WINDOW = 4


def nearest_player(boxes: dict, ball_box):
    """The player whose centre is nearest the ball, which is the one who hit it."""
    if not boxes:
        return None
    if ball_box is None:
        return next(iter(boxes.values()))
    ball_centre = get_center_of_bbox(ball_box)
    pid = min(boxes, key=lambda p: measure_distance_between_points(
        ball_centre, get_center_of_bbox(boxes[p])))
    return boxes[pid]


def analyse(video_path: str, estimator: PoseEstimator) -> dict:
    frames = read_video(video_path)
    if len(frames) < 10:
        return {}

    ball_tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    interpolated = ball_tracker.interpolate_ball_positions(
        ball_tracker.detect_frames(frames)
    )
    player_tracker = PlayerTracker(model_path="yolov8x")
    players = player_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path=stub_path_for_video("tracker_stubs/player_detections.pkl", video_path),
    )
    court = CourtLineDetector("models/keypoints_model_geoaug.pth")
    players = player_tracker.choose_and_filter_players(players, court.predict(frames[0]))

    shots, _bounces, _raw, _flips = derive_shot_frames(ball_tracker, interpolated, players)

    exact = window = 0
    for frame in shots:
        if frame >= len(frames):
            continue
        box = nearest_player(players[frame], interpolated[frame].get(1))
        if box is not None and estimator.detect_in_bbox(frames[frame], box) is not None:
            exact += 1
            window += 1
            continue
        # Fall back to the nearest frame either side that does produce a pose.
        for offset in range(1, FALLBACK_WINDOW + 1):
            found = False
            for f in (frame - offset, frame + offset):
                if not (0 <= f < len(frames)):
                    continue
                b = nearest_player(players[f], interpolated[f].get(1))
                if b is not None and estimator.detect_in_bbox(frames[f], b) is not None:
                    found = True
                    break
            if found:
                window += 1
                break

    return {"clip": os.path.basename(video_path), "shots": len(shots),
            "exact": exact, "window": window}


def main():
    p = argparse.ArgumentParser(description="Pose availability at contact frames")
    p.add_argument("--clips", default="datasets/eval_clips/*.mp4")
    args = p.parse_args()

    paths = sorted(glob.glob(args.clips))
    if not paths:
        print(f"No clips matched {args.clips}")
        sys.exit(1)

    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Run: tennis-vision download-models")
        sys.exit(1)

    rows = [r for path in paths if (r := analyse(path, estimator))]
    estimator.close()

    print(f"\n{'clip':<26} {'shots':>6} {'pose at frame':>14} {'within ±4':>11}")
    print("-" * 62)
    total_shots = total_exact = total_window = 0
    for r in rows:
        total_shots += r["shots"]
        total_exact += r["exact"]
        total_window += r["window"]
        e = r["exact"] / r["shots"] if r["shots"] else 0.0
        w = r["window"] / r["shots"] if r["shots"] else 0.0
        print(f"{r['clip']:<26} {r['shots']:>6} {e:>13.0%} {w:>10.0%}")

    print("-" * 62)
    if total_shots:
        print(f"{'TOTAL':<26} {total_shots:>6} "
              f"{total_exact / total_shots:>13.0%} {total_window / total_shots:>10.0%}")
    print("\n'pose at frame' is the ceiling on any classifier that reads the contact frame")
    print("itself. 'within ±4' is the ceiling if it may use the nearest usable frame,")
    print("which is the cheaper of the two fixes if the gap between them is large.")


if __name__ == "__main__":
    main()
