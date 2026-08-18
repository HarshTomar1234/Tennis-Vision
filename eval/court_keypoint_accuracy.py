"""
eval/court_keypoint_accuracy.py
───────────────────────────────
First real measurement of court keypoint accuracy - the component the README has
always listed as "unmeasured".

Measured against the held-out validation split of the TennisCourtDetector dataset
(2,211 images from 442 distinct source videos, 14 annotated keypoints each). This is
the same dataset lineage our weights came from, so treat the headline number as an
in-distribution ceiling, not evidence of generalisation. The eval-suite clips in
`datasets/evail_clips/` are the out-of-distribution check, and they already tell a
harsher story (see docs/specs/05_clip_suite_hardening.md).

Surface breakdown
-----------------
The dataset carries no surface label, so surface is inferred from the median court
colour inside the annotated court quad, then voted per source video (all frames of
one video share a surface). Reported as colour families rather than surface names,
because a green hard court and a grass court are genuinely not separable by colour
alone:

    clay   orange/red hue          - unambiguous
    green  grass OR green hard     - AMBIGUOUS, do not read as "grass"
    blue   blue hard court         - unambiguous

Usage:
    python eval/court_keypoint_accuracy.py [--limit N] [--tolerance PX]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from court_line_detector import CourtLineDetector

DATA_DIR = Path("datasets/external/court_dataset/data")
COURT_MODEL = "models/keypoints_model.pth"

# Outer court corners in the 14-point convention: far-left, far-right,
# near-right, near-left (ordered so the polygon is traced without crossing).
OUTER_CORNERS = (0, 1, 3, 2)


def classify_surface(image: np.ndarray, kps: np.ndarray) -> str:
    """
    Infer surface colour family from the median hue inside the annotated court quad.

    Samples a grid of interior points rather than a single centre pixel: the centre
    of a tennis court is the net, and players/shadows would skew a small patch.
    """
    quad = np.array([kps[i] for i in OUTER_CORNERS], dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 255)
    # Erode so we sample court surface, not the painted boundary lines.
    mask = cv2.erode(mask, np.ones((25, 25), np.uint8))

    pixels = image[mask > 0]
    if len(pixels) < 50:
        return "unknown"

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hue, sat = float(np.median(hsv[:, 0])), float(np.median(hsv[:, 1]))

    if sat < 40:
        return "unknown"        # washed out / greyscale, no reliable colour
    if hue < 25 or hue > 170:
        return "clay"
    if 35 <= hue <= 85:
        return "green"          # grass OR green hard court - not separable here
    if 86 <= hue <= 135:
        return "blue"
    return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N images")
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="px tolerance for the per-keypoint hit rate")
    ap.add_argument("--model", default=COURT_MODEL,
                    help="keypoint weights to evaluate (compare fine-tuned vs shipped)")
    args = ap.parse_args()

    val_path = DATA_DIR / "data_val.json"
    if not val_path.exists():
        sys.exit(f"Missing {val_path} - extract the dataset first.")

    samples = json.load(open(val_path))
    if args.limit:
        samples = samples[:args.limit]

    detector = CourtLineDetector(model_path=args.model)
    print(f"\nEvaluating {len(samples)} val images on device={detector.device}")
    print(f"Weights: {args.model}\n")

    per_kp_errors = defaultdict(list)     # keypoint index -> [errors]
    all_errors: list[float] = []
    per_surface: dict[str, list[float]] = defaultdict(list)
    video_surface_votes: dict[str, Counter] = defaultdict(Counter)
    image_max_error: list[tuple[str, float]] = []
    skipped = 0

    for n, sample in enumerate(samples):
        img_path = DATA_DIR / "images" / f"{sample['id']}.png"
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        gt = np.array(sample["kps"], dtype=np.float32)
        pred = np.array(detector.predict(image), dtype=np.float32).reshape(-1, 2)

        errors = np.linalg.norm(pred - gt, axis=1)
        all_errors.extend(errors.tolist())
        for i, e in enumerate(errors):
            per_kp_errors[i].append(float(e))

        video_id = sample["id"].rsplit("_", 1)[0]
        video_surface_votes[video_id][classify_surface(image, gt)] += 1
        per_surface[video_id].extend(errors.tolist())   # keyed by video for now
        image_max_error.append((sample["id"], float(errors.max())))

        if (n + 1) % 250 == 0:
            print(f"  {n + 1}/{len(samples)} ...")

    if not all_errors:
        sys.exit("No images could be read - check the extraction path.")

    errs = np.array(all_errors)
    tol = args.tolerance
    print("\n" + "=" * 70)
    print("OVERALL (in-distribution: same dataset lineage as our weights)")
    print("=" * 70)
    print(f"  images evaluated : {len(image_max_error)}  (skipped {skipped})")
    print(f"  mean error       : {errs.mean():.2f} px")
    print(f"  median error     : {np.median(errs):.2f} px")
    print(f"  90th percentile  : {np.percentile(errs, 90):.2f} px")
    print(f"  within {tol:.0f} px      : {100 * (errs <= tol).mean():.1f} % of keypoints")
    print(f"  within 25 px     : {100 * (errs <= 25).mean():.1f} % of keypoints")

    usable = sum(1 for _, mx in image_max_error if mx <= 25)
    print(f"  images with ALL 14 keypoints within 25 px: "
          f"{100 * usable / len(image_max_error):.1f} %")

    print("\nPER-KEYPOINT median error (px)")
    names = ["far-L outer", "far-R outer", "near-L outer", "near-R outer",
             "far-L singles", "near-L singles", "far-R singles", "near-R singles",
             "far svc L", "far svc R", "near svc L", "near svc R",
             "centre svc far", "centre svc near"]
    for i in sorted(per_kp_errors):
        label = names[i] if i < len(names) else f"kp{i}"
        print(f"  {i:2d} {label:16s} {np.median(per_kp_errors[i]):6.2f}")

    # Resolve each video to one surface by majority vote, then pool its errors.
    surface_errors: dict[str, list[float]] = defaultdict(list)
    surface_videos: Counter = Counter()
    for video_id, votes in video_surface_votes.items():
        surface = votes.most_common(1)[0][0]
        surface_errors[surface].extend(per_surface[video_id])
        surface_videos[surface] += 1

    print("\nBY SURFACE COLOUR FAMILY  ('green' = grass OR green hard, ambiguous)")
    print(f"  {'surface':9s} {'videos':>7s} {'keypoints':>10s} {'median px':>10s} "
          f"{'within 25px':>12s}")
    for surface in sorted(surface_errors, key=lambda s: -len(surface_errors[s])):
        e = np.array(surface_errors[surface])
        print(f"  {surface:9s} {surface_videos[surface]:7d} {len(e):10d} "
              f"{np.median(e):10.2f} {100 * (e <= 25).mean():11.1f} %")

    print("\nWorst 10 images by max keypoint error:")
    for image_id, mx in sorted(image_max_error, key=lambda t: -t[1])[:10]:
        print(f"  {image_id:28s} {mx:8.1f} px")
    print()


if __name__ == "__main__":
    main()
