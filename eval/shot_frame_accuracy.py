"""
eval/shot_frame_accuracy.py
─────────────────────────────
Compares detected shot frames against a manually labeled ground truth CSV.

Ground truth format (create manually):
  shot_num, true_frame
  1, 27
  2, 110
  ...

Usage:
  python eval/shot_frame_accuracy.py                        # uses built-in reference for input_video_2
  python eval/shot_frame_accuracy.py --gt my_labels.csv --video my_clip.mp4

Metric: mean absolute frame offset between detected and ground truth shot frames.
Acceptable: <= 5 frames.  Poor: > 15 frames.
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from _ball_source import pipeline_ball_detections
from utils import read_video, derive_shot_frames, stub_path_for_video
from trackers import PlayerTracker

# Hand-labeled ground truth for input_video_2.mp4
# Captured during audit session 2026-05-06 from audit_frames/ analysis
BUILTIN_GT = {
    "input_videos/input_video_2.mp4": [27, 110, 209, 282, 384, 429, 482],
}


def load_gt_csv(path: str) -> list[int]:
    frames = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["true_frame"]))
    return sorted(frames)


def match_detections(detected: list[int], ground_truth: list[int],
                     tolerance: int = 30) -> list[dict]:
    """
    Greedy nearest-neighbour matching.
    Each GT frame is matched to the closest detected frame within tolerance.
    """
    results = []
    used = set()

    for gt_frame in ground_truth:
        best_det, best_gap = None, float("inf")
        for det_frame in detected:
            if det_frame in used:
                continue
            gap = abs(det_frame - gt_frame)
            if gap < best_gap and gap <= tolerance:
                best_det, best_gap = det_frame, gap
        if best_det is not None:
            used.add(best_det)
            results.append({"gt": gt_frame, "detected": best_det, "offset": best_gap, "matched": True})
        else:
            results.append({"gt": gt_frame, "detected": None, "offset": None, "matched": False})

    return results


def evaluate(video_path: str, gt_frames: list[int]) -> dict:
    print(f"\n{'=' * 55}")
    print("Shot Frame Accuracy Evaluation")
    print(f"Video         : {video_path}")
    print(f"Ground truth  : {gt_frames}")
    print(f"{'=' * 55}")

    frames          = read_video(video_path)
    tracker, ball_det = pipeline_ball_detections(frames, video_path)   # TrackNet, matches pipeline
    ball_det        = tracker.interpolate_ball_positions(ball_det)

    # Grade the frames the pipeline reports as SHOTS, not the raw candidate union.
    # This eval used to stop at the merged candidates, which include every bounce in
    # the rally: 25 candidates against 7 real shots, scored as 18 false-positive shots
    # and 28% precision for a stage main.py never emits. `derive_shot_frames` is the
    # same function main.py calls, so this now measures the shipped behaviour.
    player_tracker = PlayerTracker(model_path="yolov8x")
    player_det = player_tracker.detect_frames(
        frames,
        read_from_stub=True,
        stub_path=stub_path_for_video("tracker_stubs/player_detections.pkl", video_path),
    )
    detected, bounces, raw = derive_shot_frames(tracker, ball_det, player_det)
    print(f"Candidates  : {len(raw)} raw reversals → {len(detected)} shots, {len(bounces)} bounces")

    print(f"GT shots    : {len(gt_frames)}")
    print(f"Detected    : {len(detected)} at frames {detected}")

    matches = match_detections(detected, gt_frames)

    print(f"\n{'─' * 45}")
    print(f"{'GT Frame':>10}  {'Detected':>10}  {'Offset':>8}  Status")
    for m in matches:
        det   = str(m["detected"]) if m["detected"] is not None else "MISSED"
        off   = str(m["offset"])   if m["offset"]   is not None else "-"
        status = "✓" if m["matched"] else "✗ MISS"
        print(f"{m['gt']:>10}  {det:>10}  {off:>8}  {status}")

    matched    = [m for m in matches if m["matched"]]
    missed     = [m for m in matches if not m["matched"]]
    false_pos  = len(detected) - len(matched)

    offsets = [m["offset"] for m in matched]
    mae     = sum(offsets) / len(offsets) if offsets else float("inf")

    print(f"{'─' * 45}")
    print(f"Matched     : {len(matched)} / {len(gt_frames)}")
    print(f"Missed      : {len(missed)}")
    print(f"False pos.  : {false_pos}")
    print(f"Mean offset : {mae:.1f} frames")

    # The verdict must account for recall and false positives, not only the timing of
    # whichever shots happened to match. It previously used `mae` alone, which made it
    # structurally incapable of failing: a run that matched 1 of 7 shots with 3 false
    # positives still printed "EXCELLENT", because the one match was well timed. That
    # word was quoted verbatim in the README as a measured result.
    recall = len(matched) / len(gt_frames) if gt_frames else 0.0
    precision = len(matched) / len(detected) if detected else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0

    print(f"Recall      : {recall:.1%}")
    print(f"Precision   : {precision:.1%}")
    print(f"F1          : {f1:.3f}")

    if f1 >= 0.9 and mae <= 5:
        verdict = f"EXCELLENT (F1 {f1:.2f}, mean offset {mae:.1f} frames)"
    elif f1 >= 0.75 and mae <= 15:
        verdict = f"ACCEPTABLE (F1 {f1:.2f}, mean offset {mae:.1f} frames)"
    elif f1 < 0.75:
        verdict = (f"POOR - F1 {f1:.2f}: {len(missed)} shot(s) missed, "
                   f"{false_pos} false positive(s). Timing of matched shots is not "
                   f"the problem here.")
    else:
        verdict = f"POOR - mean offset {mae:.1f} frames is too large"

    print(f"VERDICT     : {verdict}")

    return {
        "matched": len(matched),
        "missed": len(missed),
        "false_positives": false_pos,
        "mae_frames": round(mae, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate shot-frame detection accuracy")
    parser.add_argument("--video", default="input_videos/input_video_2.mp4")
    parser.add_argument("--gt",    default=None,
                        help="CSV with columns: shot_num, true_frame")
    args = parser.parse_args()

    if args.gt:
        gt_frames = load_gt_csv(args.gt)
    elif args.video in BUILTIN_GT:
        gt_frames = BUILTIN_GT[args.video]
    else:
        print(f"No ground truth for {args.video}. Provide --gt <csv>.")
        sys.exit(1)

    evaluate(args.video, gt_frames)


if __name__ == "__main__":
    main()
