"""
eval/retest_union_type_accuracy.py
────────────────────────────────────────
Closes the methodology gap found while investigating the Phase 2 pose/backhand
question: the "decisive" union-candidate test in retest_union_candidates_full_pipeline.py
measures recall/precision against ANY true event (hit or bounce), not against the
CORRECT type. A frame classified CONTACT that lands near a real BOUNCE counts as a true
positive there -- which hid a real regression: on our own clip, shot count inflated from
7 (y-reversal only, journal 0012) to 22 (union, journal 0015), because the x-velocity
candidates are disproportionately landing on bounces, not hits, and the trajectory
classifier is calling some of them CONTACT anyway.

This measures TYPE-specific accuracy: of frames classified CONTACT, how many are near a
real HIT specifically (status=="1"), not just near any event. Same for BOUNCE vs
status=="2". Compares y-reversal-only against the union so we can see exactly what the
union costs in type precision, not just what it gains in any-event recall.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils.hit_bounce_classifier import (
    detect_xvelocity_candidates,
    classify_reversals_by_trajectory,
    merge_nearby_candidates,
)

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
    true_hits    = sorted(i for i, r in enumerate(rows) if r["status"] == "1")
    true_bounces = sorted(i for i, r in enumerate(rows) if r["status"] == "2")
    return detections, true_hits, true_bounces


def near_any(frame, targets, tol=TOLERANCE):
    return any(abs(frame - t) <= tol for t in targets)


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    label_paths = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")

    configs = [
        ("y-reversal only (baseline)", False, False),
        ("union (y-reversal + x-velocity)", True, False),
        ("union + merged nearby candidates", True, True),
    ]
    for label, use_union, use_merge in configs:
        n_contacts = n_contact_tp = 0
        n_bounces  = n_bounce_tp  = 0
        total_true_hits = total_hit_recalled = 0

        for lp in label_paths:
            detections, true_hits, true_bounces = load_clip(zf, lp)
            if len(detections) < 40:
                continue

            yrev = tracker.get_ball_shot_frames(detections)
            if use_union:
                xvel = detect_xvelocity_candidates(detections)
                raw_candidates = sorted(set(yrev) | set(xvel))
            else:
                raw_candidates = yrev
            if use_merge:
                raw_candidates = merge_nearby_candidates(raw_candidates)

            contacts, bounces = classify_reversals_by_trajectory(raw_candidates, detections)

            n_contacts += len(contacts)
            n_contact_tp += sum(1 for c in contacts if near_any(c, true_hits))
            n_bounces += len(bounces)
            n_bounce_tp += sum(1 for b in bounces if near_any(b, true_bounces))

            total_true_hits += len(true_hits)
            total_hit_recalled += sum(1 for h in true_hits if near_any(h, contacts))

        contact_type_precision = 100 * n_contact_tp / n_contacts if n_contacts else 0
        bounce_type_precision  = 100 * n_bounce_tp / n_bounces if n_bounces else 0
        hit_type_recall        = 100 * total_hit_recalled / total_true_hits if total_true_hits else 0

        print(f"{label}:")
        print(f"  CONTACT frames: {n_contacts}, type precision (near a real HIT): {contact_type_precision:.1f}%")
        print(f"  BOUNCE frames:  {n_bounces}, type precision (near a real BOUNCE): {bounce_type_precision:.1f}%")
        print(f"  Real-hit recall (a real hit gets a CONTACT nearby): {hit_type_recall:.1f}%")
        print()


if __name__ == "__main__":
    main()
