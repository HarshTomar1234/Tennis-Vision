"""
eval/extract_thetis_slice_features.py
────────────────────────────────────────
Runs our own MediaPipe PoseEstimator on downloaded THETIS clips (datasets/external/
thetis/VIDEO_RGB/) and saves per-frame upper-body landmark sequences to JSON, for
training a slice-vs-flat/topspin classifier (
"Beyond rules" gap).

Why our own extractor, not THETIS's shipped "skeleton" videos: THETIS's VIDEO_Skelet2D/
VIDEO_Skelet3D are rendered overlay videos, not raw numeric keypoints, and use Kinect's
joint definitions rather than MediaPipe's. Extracting with our own PoseEstimator keeps
the feature format identical to what main.py's pose pipeline already produces.

Each clip is a single unobstructed subject filmed head-on in an empty gym (no tennis
court, no ball) -- full frame is passed as the "bbox" since there's no player-selection
problem here, unlike our broadcast footage.
"""
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import PoseEstimator

DATASET_ROOT = Path("datasets/external/thetis/VIDEO_RGB")

# THETIS folder name -> (is_slice, family). family lets us keep forehand/backhand
# separate if useful later; is_slice is the binary label this build targets.
CATEGORIES = {
    "forehand_slice":      (True,  "forehand"),
    "forehand_flat":       (False, "forehand"),
    "forehand_openstands": (False, "forehand"),
    "backhand_slice":      (True,  "backhand"),
    "backhand":            (False, "backhand"),
    "backhand2hands":      (False, "backhand"),
}

FILENAME_RE = re.compile(r"^(p\d+)_")


def main():
    out_path = Path("datasets/external/thetis_slice_features.json")
    records = []

    est = PoseEstimator("models/pose_landmarker_lite.task", bbox_padding=0.0)
    if not est.available:
        print("Pose model unavailable, aborting.")
        return

    for category, (is_slice, family) in CATEGORIES.items():
        clip_dir = DATASET_ROOT / category
        if not clip_dir.exists():
            print(f"  (skipping {category}, not downloaded)")
            continue
        clips = sorted(clip_dir.glob("*.avi"))
        print(f"{category}: {len(clips)} clips")

        for clip_path in clips:
            m = FILENAME_RE.match(clip_path.name)
            subject = m.group(1) if m else "unknown"

            cap = cv2.VideoCapture(str(clip_path))
            sequence = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                landmarks = est.detect_in_bbox(frame, [0, 0, w, h])
                if landmarks:
                    sequence.append({k: list(v) for k, v in landmarks.items()})
                else:
                    sequence.append(None)
            cap.release()

            records.append({
                "clip": clip_path.name,
                "category": category,
                "subject": subject,
                "is_slice": is_slice,
                "family": family,
                "n_frames": len(sequence),
                "sequence": sequence,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)

    est.close()
    print(f"\nSaved {len(records)} clips -> {out_path}")
    slice_count = sum(1 for r in records if r["is_slice"])
    print(f"  slice: {slice_count}, not-slice: {len(records) - slice_count}")
    subjects = sorted(set(r["subject"] for r in records))
    print(f"  subjects: {len(subjects)} -> {subjects}")


if __name__ == "__main__":
    main()
