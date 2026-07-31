"""
eval/retest_real_model_noise.py
────────────────────────────────
The rigorous version of the noise check in journal 0010: instead of synthetic Gaussian
jitter, run our ACTUAL TrackNet model on real frames from the labeled dataset (extracted
straight from the zip, no re-download) and compare the reversal detector's behavior on
real detection noise against the same clips' true hit/bounce labels.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trackers.tracknet_ball_tracker import TrackNetBallTracker

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"
CLIPS = ["Dataset/game1/Clip1", "Dataset/game2/Clip6", "Dataset/game5/Clip6"]
TOLERANCE = 10


def load_clip_frames_and_labels(zf: zipfile.ZipFile, clip_dir: str):
    label_path = f"{clip_dir}/Label.csv"
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))

    frames = []
    for row in rows:
        img_path = f"{clip_dir}/{row['file name']}"
        data = np.frombuffer(zf.read(img_path), dtype=np.uint8)
        frames.append(cv2.imdecode(data, cv2.IMREAD_COLOR))

    true_hits    = [i for i, r in enumerate(rows) if r["status"] == "1"]
    true_bounces = [i for i, r in enumerate(rows) if r["status"] == "2"]
    return frames, true_hits, true_bounces


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")

    total_hits = total_bounces = matched_hits = matched_bounces = 0
    total_det = matched_det = 0

    for clip_dir in CLIPS:
        print(f"\n--- {clip_dir} ---")
        frames, true_hits, true_bounces = load_clip_frames_and_labels(zf, clip_dir)
        n = len(frames)
        print(f"  {n} frames, {len(true_hits)} true hits, {len(true_bounces)} true bounces")

        raw = tracker.detect_frames(frames)
        detected_rate = sum(1 for d in raw if d.get(1)) / n
        print(f"  real model detection rate: {100*detected_rate:.1f}%")

        interpolated = tracker.interpolate_ball_positions(raw)
        detected = tracker.get_ball_shot_frames(interpolated)
        events = sorted(true_hits + true_bounces)

        mh = sum(1 for h in true_hits if any(abs(h - d) <= TOLERANCE for d in detected))
        mb = sum(1 for b in true_bounces if any(abs(b - d) <= TOLERANCE for d in detected))
        md = sum(1 for d in detected if any(abs(d - e) <= TOLERANCE for e in events))

        print(f"  detected reversals: {len(detected)}  matched: {md}  "
              f"(precision {100*md/len(detected) if detected else 0:.1f}%)")
        print(f"  hits matched: {mh}/{len(true_hits)}  bounces matched: {mb}/{len(true_bounces)}")

        total_hits += len(true_hits); total_bounces += len(true_bounces)
        matched_hits += mh; matched_bounces += mb
        total_det += len(detected); matched_det += md

    total_events  = total_hits + total_bounces
    matched_events = matched_hits + matched_bounces
    print("\n" + "=" * 60)
    print(f"TOTAL (real model, real frames, {len(CLIPS)} clips):")
    print(f"  recall    = {matched_events}/{total_events} = {100*matched_events/total_events:.1f}%")
    print(f"  precision = {matched_det}/{total_det} = {100*matched_det/total_det if total_det else 0:.1f}%")


if __name__ == "__main__":
    main()
