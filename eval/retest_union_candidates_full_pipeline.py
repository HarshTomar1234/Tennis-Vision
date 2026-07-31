"""
eval/retest_union_candidates_full_pipeline.py
────────────────────────────────────────────────
Decisive test for the union candidate generator (journal 0015): does combining
y-reversal + x-velocity candidates, then filtering through the trained hit/bounce
classifier, actually improve real recall/precision — not just raw candidate recall,
which was already shown to not be the whole story (journal 0014's coincidental count
match).
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils.hit_bounce_classifier import detect_xvelocity_candidates, classify_reversals_by_trajectory

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"
TOLERANCE = 10


def load_clip(zf, label_path):
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))
    detections = []
    for r in rows:
        if r["visibility"] != "0" and r["x-coordinate"]:
            x, y = float(r["x-coordinate"]), float(r["y-coordinate"])
            detections.append({1: [x - 5, y - 5, x + 5, y + 5]})
        else:
            detections.append({})
    true_events = sorted(i for i, r in enumerate(rows) if r["status"] in ("1", "2"))
    return detections, true_events


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    label_paths = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")

    for label, use_union in [("y-reversal only (baseline)", False), ("union (y-reversal + x-velocity)", True)]:
        total_true = matched_true = total_classified = matched_classified = 0

        for lp in label_paths:
            detections, true_events = load_clip(zf, lp)
            n = len(detections)
            if n < 40:
                continue

            yrev = tracker.get_ball_shot_frames(detections)
            if use_union:
                xvel = detect_xvelocity_candidates(detections)
                raw_candidates = sorted(set(yrev) | set(xvel))
            else:
                raw_candidates = yrev

            contacts, bounces = classify_reversals_by_trajectory(raw_candidates, detections)
            classified = sorted(contacts + bounces)

            total_true += len(true_events)
            matched_true += sum(1 for e in true_events if any(abs(e - c) <= TOLERANCE for c in classified))
            total_classified += len(classified)
            matched_classified += sum(1 for c in classified if any(abs(c - e) <= TOLERANCE for e in true_events))

        recall = 100 * matched_true / total_true if total_true else 0
        precision = 100 * matched_classified / total_classified if total_classified else 0
        print(f"{label}:")
        print(f"  {total_classified} classified events, recall={recall:.1f}%  precision={precision:.1f}%")
        print()


if __name__ == "__main__":
    main()
