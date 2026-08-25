"""
eval/event_detection_on_real_detections.py
──────────────────────────────────────────
Contact/bounce detection measured on REAL TrackNet detections, at dataset scale.

Why this exists
---------------
`eval/retest_union_candidates_full_pipeline.py` reports 87.6% recall / 90.3% precision for
the union candidate generator across 91 clips, and those numbers are in the README. They
are measured by feeding the dataset's own hand-labelled ball coordinates into the
candidate generators. That isolates the generators, which is a fair thing to want, but it
means the figures describe the pipeline it would be if ball detection were perfect. They
are an upper bound, not shipped behaviour.

The pipeline runs on TrackNet output, whose median localization error is 5.8px with a
19.4px 90th percentile (see eval/ball_localization_accuracy.py). A candidate generator
looking for velocity reversals is directly exposed to that: position noise manufactures
reversals, and real reversals get buried in it.

This script closes the gap by running detection for real and scoring the resulting events
against the labelled contacts. It exists because a change to the detector's postprocessing
(taking the largest connected response instead of the mean of all responding pixels)
improved localization on 16 clips while making shot F1 worse on the single 7-shot
reference clip. Seven events cannot adjudicate that; this can.

Player detections are deliberately not supplied, so the proximity fallback in
derive_shot_frames stays out of the way and what is measured is the trajectory path alone.

Usage
-----
    python eval/event_detection_on_real_detections.py --clips 12
    python eval/event_detection_on_real_detections.py --clips 12 --compare

Needs the dataset (7+ GB, not redistributable). See datasets/README.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ball_localization_accuracy import DATASET_ZIP, heatmap_to_centre, load_clip
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils import derive_shot_frames

# A detected event counts as matching a labelled contact within this many frames. Matches
# the tolerance used by eval/retest_union_candidates_full_pipeline.py so the two are
# directly comparable.
TOLERANCE = 10


def score(detected: list[int], truth: list[int]) -> tuple[int, int, int]:
    """(matched, missed, false_positives) by greedy nearest matching within TOLERANCE."""
    used: set[int] = set()
    matched = 0
    for t in truth:
        best, best_gap = None, TOLERANCE + 1
        for d in detected:
            if d in used:
                continue
            gap = abs(d - t)
            if gap < best_gap:
                best, best_gap = d, gap
        if best is not None:
            used.add(best)
            matched += 1
    return matched, len(truth) - matched, len(detected) - matched


def run(n_clips: int, max_frames: int, largest_blob: bool) -> dict:
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    if tracker.model is None:
        print("TrackNet weights not loaded. See README 'Models'.")
        sys.exit(1)

    H, W = tracker.INPUT_HEIGHT, tracker.INPUT_WIDTH
    zf = zipfile.ZipFile(DATASET_ZIP)
    labels = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))[:n_clips]

    total_matched = total_missed = total_fp = total_truth = 0
    offsets: list[int] = []

    for ci, label_path in enumerate(labels, 1):
        frames, _truth_pos, contacts = load_clip(zf, label_path, max_frames)
        if len(frames) < 10 or not contacts:
            continue

        orig_h, orig_w = frames[0].shape[:2]
        detections: list[dict] = []

        with torch.no_grad():
            for i in range(len(frames)):
                prev_f = frames[max(0, i - 1)]
                next_f = frames[min(len(frames) - 1, i + 1)]
                out = tracker.model(tracker._preprocess([prev_f, frames[i], next_f]))
                intensity = out.argmax(dim=1)[0].cpu().numpy().reshape(H, W)
                centre = heatmap_to_centre(intensity, 0, 5, largest_blob=largest_blob)
                if centre is None:
                    detections.append({})
                else:
                    # Back to original image coordinates, as a box, which is the shape the
                    # candidate generators expect.
                    cx = centre[0] * orig_w / W
                    cy = centre[1] * orig_h / H
                    detections.append({1: [cx - 12, cy - 12, cx + 12, cy + 12]})

        interpolated = tracker.interpolate_ball_positions(detections)
        # No player detections: the proximity fallback stays inert and the trajectory
        # classifier alone decides, which is what this measures.
        shots, bounces, _raw, _flips = derive_shot_frames(tracker, interpolated, [{}] * len(frames))
        events = sorted(set(shots) | set(bounces))

        m, mi, fp = score(events, contacts)
        total_matched += m
        total_missed += mi
        total_fp += fp
        total_truth += len(contacts)

        for t in contacts:
            near = [abs(d - t) for d in events if abs(d - t) <= TOLERANCE]
            if near:
                offsets.append(min(near))

        print(f"  [{ci}/{len(labels)}] {label_path.rsplit('/', 2)[-2]}: "
              f"{len(contacts)} labelled, {len(events)} detected, {m} matched")

    recall = total_matched / total_truth if total_truth else 0.0
    n_det = total_matched + total_fp
    precision = total_matched / n_det if n_det else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {
        "recall": recall, "precision": precision, "f1": f1,
        "matched": total_matched, "missed": total_missed,
        "false_positives": total_fp, "truth": total_truth,
        "mean_offset": float(np.mean(offsets)) if offsets else float("nan"),
    }


def show(name: str, r: dict) -> None:
    print(f"\n{name}")
    print(f"  labelled contacts : {r['truth']}")
    print(f"  matched / missed  : {r['matched']} / {r['missed']}")
    print(f"  false positives   : {r['false_positives']}")
    print(f"  recall            : {r['recall']:.1%}")
    print(f"  precision         : {r['precision']:.1%}")
    print(f"  F1                : {r['f1']:.3f}")
    print(f"  mean offset       : {r['mean_offset']:.1f} frames")


def main():
    p = argparse.ArgumentParser(description="Event detection on real TrackNet detections")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--compare", action="store_true",
                   help="measure both postprocessing modes side by side")
    args = p.parse_args()

    if not Path(DATASET_ZIP).exists():
        print(f"Dataset not found at {DATASET_ZIP}. See datasets/README.md.")
        sys.exit(1)

    print("=== largest connected component (current) ===")
    blob = run(args.clips, args.max_frames, largest_blob=True)

    if args.compare:
        print("\n=== mean of all responding pixels (previous) ===")
        mean = run(args.clips, args.max_frames, largest_blob=False)
        show("mean of all responding pixels (previous)", mean)
        show("largest connected component (current)", blob)
        print(f"\nF1 change: {mean['f1']:.3f} -> {blob['f1']:.3f} "
              f"({(blob['f1'] - mean['f1']) / mean['f1'] * 100:+.1f}%)")
    else:
        show("largest connected component", blob)


if __name__ == "__main__":
    main()
