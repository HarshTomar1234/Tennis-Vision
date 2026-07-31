"""
eval/retest_reversal_detector_on_tracknet_dataset.py
─────────────────────────────────────────────────────
Re-tests the production reversal detector (TrackNetBallTracker.get_ball_shot_frames)
against real ground truth from the original TrackNet dataset (19,835 frames, 95 clips,
confirmed hit/bounce labels — see datasets/README.md), instead of the 7 hand-labeled
shots on one clip that capped every attempt so far this sprint (journal 0003, 0007).

This directly tests Phase 1's core assumption in utils/ball_state.py: "every trajectory
reversal is a floor-level event (hit or bounce)". At dataset scale, not clip scale.

Usage:
  python eval/retest_reversal_detector_on_tracknet_dataset.py
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trackers.tracknet_ball_tracker import TrackNetBallTracker

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"
TOLERANCE = 10   # frames — matches the tolerance already used in shot_frame_accuracy.py


def load_clip(zf: zipfile.ZipFile, label_path: str) -> list[dict]:
    """Build (frame -> {1: [x1,y1,x2,y2]}) from a Label.csv, plus true hit/bounce frames."""
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))

    detections: list[dict] = []
    true_hits, true_bounces = [], []

    for i, row in enumerate(rows):
        vis = row["visibility"]
        if vis != "0" and row["x-coordinate"] and row["y-coordinate"]:
            x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
            detections.append({1: [x - 5, y - 5, x + 5, y + 5]})
        else:
            detections.append({})

        if row["status"] == "1":
            true_hits.append(i)
        elif row["status"] == "2":
            true_bounces.append(i)

    return detections, true_hits, true_bounces


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    label_paths = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))

    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")

    total_hits = total_bounces = 0
    matched_hits = matched_bounces = 0
    total_detected = matched_detected = 0
    clips_skipped = 0

    hit_ys, bounce_ys = [], []

    for label_path in label_paths:
        detections, true_hits, true_bounces = load_clip(zf, label_path)
        n = len(detections)

        if n < 40:   # too short for the reversal detector's windowing
            clips_skipped += 1
            continue

        detected = tracker.get_ball_shot_frames(detections)

        true_events = sorted(true_hits + true_bounces)
        total_hits += len(true_hits)
        total_bounces += len(true_bounces)
        total_detected += len(detected)

        for h in true_hits:
            if any(abs(h - d) <= TOLERANCE for d in detected):
                matched_hits += 1
        for b in true_bounces:
            if any(abs(b - d) <= TOLERANCE for d in detected):
                matched_bounces += 1
        for d in detected:
            if any(abs(d - e) <= TOLERANCE for e in true_events):
                matched_detected += 1

        for h in true_hits:
            if detections[h]:
                hit_ys.append((detections[h][1][1] + detections[h][1][3]) / 2)
        for b in true_bounces:
            if detections[b]:
                bounce_ys.append((detections[b][1][1] + detections[b][1][3]) / 2)

    total_events = total_hits + total_bounces
    matched_events = matched_hits + matched_bounces

    print("=" * 60)
    print(f"Re-test: reversal detector vs {len(label_paths) - clips_skipped} real clips "
          f"({clips_skipped} skipped, too short)")
    print(f"Tolerance: ±{TOLERANCE} frames")
    print("=" * 60)
    print(f"True hits    : {total_hits}")
    print(f"True bounces : {total_bounces}")
    print(f"True events  : {total_events}")
    print(f"Detected reversals: {total_detected}")
    print()
    print(f"RECALL  (true events found by a reversal): "
          f"{matched_events}/{total_events} = {100*matched_events/total_events:.1f}%")
    print(f"  - hits    matched: {matched_hits}/{total_hits} = {100*matched_hits/total_hits:.1f}%")
    print(f"  - bounces matched: {matched_bounces}/{total_bounces} = {100*matched_bounces/total_bounces:.1f}%")
    print(f"PRECISION (detected reversals that are real): "
          f"{matched_detected}/{total_detected} = {100*matched_detected/total_detected:.1f}%")
    print()
    print("Ball height (y-px) at TRUE events — hit vs bounce (higher y = lower on screen):")
    print(f"  hit    n={len(hit_ys)}  mean={sum(hit_ys)/len(hit_ys):.1f}")
    print(f"  bounce n={len(bounce_ys)}  mean={sum(bounce_ys)/len(bounce_ys):.1f}")


if __name__ == "__main__":
    main()
