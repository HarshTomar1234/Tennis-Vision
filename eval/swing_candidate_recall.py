"""
eval/swing_candidate_recall.py
──────────────────────────────
Calibrates the body-based swing generator, and tests whether it is actually orthogonal.

The question that matters
-------------------------
Adding a fourth candidate generator is only worth its compute if it proposes contacts the
existing three miss. If it fires on the same events they already cover, it adds cost and
false positives for nothing.

eval/event_recall_funnel.py measures that 15.4% of real contacts are never proposed by any
ball-trajectory generator, while 0% are lost for want of a detected ball. Those are the
events this generator exists for, so the headline number here is not its own recall but
its recall ON THE CONTACTS THE BALL GENERATORS MISS.

Also calibrates MIN_SWING_SPEED, which is a guess until run against real pose. Too low and
running between shots is proposed as a stroke; too high and gentle blocks and touch volleys
are missed.

Pose runs on every frame here, unlike in the pipeline where it runs only on known contact
frames. That is the cost of using the body to FIND contacts rather than to describe them,
and this script is where that cost gets justified or rejected.

Usage
-----
    python eval/swing_candidate_recall.py
    python eval/swing_candidate_recall.py --video datasets/eval_clips/input_video_5.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court_line_detector import CourtLineDetector
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils import (
    PoseEstimator,
    derive_shot_frames,
    detect_xvelocity_candidates,
    merge_nearby_candidates,
    read_video,
    stub_path_for_video,
)
from utils.bounce_candidates import detect_bounce_candidates
from utils.swing_candidates import detect_swing_candidates, swing_score

# Hand-labelled contact frames for the reference clip (audit session 2026-05-06).
BUILTIN_GT = {
    "input_videos/input_video_2.mp4": [27, 110, 209, 282, 384, 429, 482],
}

TOLERANCE = 10
SWEEP = [0.02, 0.03, 0.04, 0.055, 0.07, 0.09, 0.12]


def covered(frames, target, tol=TOLERANCE) -> bool:
    return any(abs(f - target) <= tol for f in frames)


def build_pose_tracks(frames, player_detections):
    """Per-player pose track and bbox height, one entry per frame."""
    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Fetch it with: tennis-vision download-models")
        sys.exit(1)

    player_ids = sorted({pid for f in player_detections for pid in f})
    tracks = {pid: [] for pid in player_ids}
    heights = {pid: [] for pid in player_ids}

    for i, frame in enumerate(frames):
        boxes = player_detections[i] if i < len(player_detections) else {}
        for pid in player_ids:
            bbox = boxes.get(pid)
            if bbox is None:
                tracks[pid].append(None)
                heights[pid].append(0.0)
                continue
            landmarks = estimator.detect_in_bbox(frame, bbox)
            tracks[pid].append(landmarks)
            heights[pid].append(float(bbox[3] - bbox[1]))
        if (i + 1) % 100 == 0:
            print(f"    pose {i + 1}/{len(frames)} frames")

    estimator.close()
    return tracks, heights


def main():
    p = argparse.ArgumentParser(description="Calibrate the swing candidate generator")
    p.add_argument("--video", default="input_videos/input_video_2.mp4")
    args = p.parse_args()

    gt = BUILTIN_GT.get(args.video)
    if gt is None:
        print(f"No hand-labelled contacts for {args.video}.")
        print("Label it with tools/label_shots.py, then add it to BUILTIN_GT.")
        sys.exit(1)

    print(f"\nVideo        : {args.video}")
    print(f"Ground truth : {gt}\n")

    frames = read_video(args.video)

    ball_tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    interpolated = ball_tracker.interpolate_ball_positions(
        ball_tracker.detect_frames(frames)
    )
    player_tracker = PlayerTracker(model_path="yolov8x")
    players = player_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path=stub_path_for_video("tracker_stubs/player_detections.pkl", args.video),
    )

    # Narrow to the two actual players before running pose. YOLO finds 11 to 14 people
    # per frame on broadcast footage (line judges, ball kids, the umpire, front rows of
    # crowd), and posing all of them would be an order of magnitude more work for
    # landmarks belonging to people who never hit anything.
    court_detector = CourtLineDetector("models/keypoints_model_geoaug.pth")
    court_keypoints = court_detector.predict(frames[0])
    players = player_tracker.choose_and_filter_players(players, court_keypoints)
    n_tracked = len({pid for fr in players for pid in fr})
    print(f"Players selected for pose: {n_tracked}")

    # What the three ball-trajectory generators propose, before any classification.
    ball_union = merge_nearby_candidates(sorted(
        set(ball_tracker.get_ball_shot_frames(interpolated))
        | set(detect_xvelocity_candidates(interpolated))
        | set(detect_bounce_candidates(interpolated))
    ))
    missed_by_ball = [t for t in gt if not covered(ball_union, t)]

    print(f"Ball generators propose {len(ball_union)} candidates, "
          f"covering {len(gt) - len(missed_by_ball)}/{len(gt)} labelled contacts.")
    print(f"Contacts they MISS: {missed_by_ball or 'none'}\n")

    print("Running pose on every frame (this is the cost being justified)...")
    tracks, heights = build_pose_tracks(frames, players)

    print(f"\n{'min_speed':>10} {'candidates':>11} {'GT covered':>11} "
          f"{'of ball-missed':>15}")
    print("-" * 54)

    for threshold in SWEEP:
        proposed: set[int] = set()
        for pid in tracks:
            proposed.update(
                detect_swing_candidates(tracks[pid], heights[pid], min_speed=threshold)
            )
        merged = merge_nearby_candidates(sorted(proposed))

        hit_gt = sum(1 for t in gt if covered(merged, t))
        hit_missed = sum(1 for t in missed_by_ball if covered(merged, t))
        of_missed = f"{hit_missed}/{len(missed_by_ball)}" if missed_by_ball else "n/a"

        print(f"{threshold:>10.3f} {len(merged):>11} {hit_gt:>8}/{len(gt)} "
              f"{of_missed:>15}")

    print("-" * 54)
    print("The rightmost column is the point of this generator. Covering contacts the")
    print("ball generators already find adds candidates without adding information.")

    # Coverage alone does not say whether the signal SEPARATES. A generator that fires
    # everywhere covers everything. What decides whether this is usable is how the score
    # at real contacts compares with the score at every other frame.
    def best_score(frame):
        vals = [v for pid in tracks
                if (v := swing_score(tracks[pid], heights[pid], frame)) is not None]
        return max(vals) if vals else None

    near_contact = set()
    for t in gt:
        near_contact.update(range(t - 3, t + 4))

    at_contacts = sorted(v for t in gt if (v := best_score(t)) is not None)
    elsewhere = sorted(v for fr in range(len(frames))
                       if fr not in near_contact and (v := best_score(fr)) is not None)

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")

    print("\n" + "=" * 62)
    print("Does the swing score separate contacts from everything else?")
    print("=" * 62)
    print(f"{'':<14} {'n':>6} {'p10':>8} {'median':>8} {'p90':>8}")
    print(f"{'at contacts':<14} {len(at_contacts):>6} {pct(at_contacts, .1):>8.3f} "
          f"{pct(at_contacts, .5):>8.3f} {pct(at_contacts, .9):>8.3f}")
    print(f"{'elsewhere':<14} {len(elsewhere):>6} {pct(elsewhere, .1):>8.3f} "
          f"{pct(elsewhere, .5):>8.3f} {pct(elsewhere, .9):>8.3f}")

    # The share of ordinary frames outscoring the median contact is the cleanest single
    # statement of whether any threshold can separate the two.
    med = pct(at_contacts, .5)
    if elsewhere:
        above = sum(1 for v in elsewhere if v >= med) / len(elsewhere)
        print(f"\nOrdinary frames scoring at or above the median contact: {above:.1%}")
        print("If that share is large, no threshold separates them and the signal is not")
        print("discriminative here, whatever the coverage table above suggests.")


if __name__ == "__main__":
    main()
