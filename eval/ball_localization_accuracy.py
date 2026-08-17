"""
eval/ball_localization_accuracy.py
──────────────────────────────────
Measures whether TrackNet puts the ball in the RIGHT PLACE, against ground truth.

Why this exists
---------------
`eval/ball_detection_rate.py` reports the fraction of frames where the detector returned
something. That is not an accuracy claim, and the README's "82.5% detection rate" was
being read as one. A detector that emits a confident box on the wrong object scores
identically to one that finds the ball.

That distinction is not academic here. Every downstream number is computed from these
positions: velocities, bounce and contact frames, 3-D reconstruction, speeds. A wrongly
placed ball produces a plausible trajectory that is simply not the ball's.

This script uses the original TrackNet dataset (95 clips, 19,835 hand-labelled frames
with per-frame ball coordinates and a visibility flag) and reports three separate things:

  detection rate  - frames where the detector output a position at all
  recall          - frames with a visible ball that were located within tolerance
  precision       - of the positions output, the fraction within tolerance

Tolerance follows the TrackNet paper's convention: a prediction counts as correct if it
falls within a few pixels of the labelled centre, at the network's own 360x640 working
resolution.

The postprocessing sweep
------------------------
The network outputs 256 intensity levels per pixel, and the shipped postprocess keeps any
pixel whose argmax intensity is non-zero, then requires a 5-pixel cluster. Both are
tunable and neither was ever measured, so this sweeps them: a higher intensity threshold
should trade recall for precision, and the cluster size sets the smallest ball the
detector will accept. One forward pass per frame feeds every variant, so the sweep costs
almost nothing beyond a single run.

Usage
-----
    python eval/ball_localization_accuracy.py                    # 10 clips, shipped settings
    python eval/ball_localization_accuracy.py --clips 25         # more clips
    python eval/ball_localization_accuracy.py --sweep            # sweep the knobs

Needs the dataset (7+ GB, not redistributable). See datasets/README.md.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from trackers.tracknet_ball_tracker import TrackNetBallTracker

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"

# Correct-within tolerance, in pixels at the network's 360x640 working resolution. The
# TrackNet paper uses a small fixed radius; 5px there is roughly a ball's width.
TOLERANCE_PX = 5.0

# The shipped postprocess, as (intensity_threshold, min_cluster_px).
SHIPPED = (0, 5)

# Variants worth measuring. Threshold 0 is the most permissive the network allows, so the
# question is whether raising it removes noise faster than it removes real detections.
SWEEP = [(0, 5), (0, 3), (0, 10), (64, 5), (128, 5), (128, 3), (192, 5)]


def load_clip(zf: zipfile.ZipFile, label_path: str, max_frames: int):
    """Frames and ground-truth positions for one clip. Positions are None when the
    label marks the ball as not visible, which must not be counted as a miss."""
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))

    clip_dir = label_path.rsplit("/", 1)[0]
    frames, truth = [], []
    for i, r in enumerate(rows[:max_frames]):
        name = f"{clip_dir}/{i:04d}.jpg"
        try:
            buf = zf.read(name)
        except KeyError:
            break
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            break
        frames.append(img)
        visible = r.get("visibility") not in ("0", "", None) and r.get("x-coordinate")
        truth.append((float(r["x-coordinate"]), float(r["y-coordinate"])) if visible else None)
    return frames, truth


def heatmap_to_centre(intensity: np.ndarray, threshold: int, min_cluster: int):
    """Ball centre in network coordinates from one 360x640 intensity map, or None."""
    ys, xs = np.where(intensity > threshold)
    if len(xs) < min_cluster:
        return None
    return float(xs.mean()), float(ys.mean())


def evaluate(n_clips: int, max_frames: int, variants: list[tuple[int, int]]) -> dict:
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    if tracker.model is None:
        print("TrackNet weights not loaded. Fetch them first (see README 'Models').")
        sys.exit(1)

    H, W = tracker.INPUT_HEIGHT, tracker.INPUT_WIDTH
    zf = zipfile.ZipFile(DATASET_ZIP)
    labels = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))[:n_clips]

    # stats[variant] = [n_output, n_correct, n_visible, [errors...]]
    stats = {v: [0, 0, 0, []] for v in variants}
    total_frames = 0

    for ci, label_path in enumerate(labels, 1):
        frames, truth = load_clip(zf, label_path, max_frames)
        if len(frames) < 3:
            continue
        print(f"  [{ci}/{len(labels)}] {label_path.rsplit('/', 2)[-2]}: {len(frames)} frames")

        orig_h, orig_w = frames[0].shape[:2]
        sx, sy = W / orig_w, H / orig_h   # ground truth is in original coords

        with torch.no_grad():
            for i in range(len(frames)):
                prev_f = frames[max(0, i - 1)]
                next_f = frames[min(len(frames) - 1, i + 1)]
                tensor = tracker._preprocess([prev_f, frames[i], next_f])
                out = tracker.model(tensor)
                intensity = out.argmax(dim=1)[0].cpu().numpy().reshape(H, W)

                total_frames += 1
                gt = truth[i] if i < len(truth) else None
                gt_net = (gt[0] * sx, gt[1] * sy) if gt else None

                for v in variants:
                    n_out, n_ok, n_vis, errs = stats[v]
                    if gt_net is not None:
                        n_vis += 1
                    centre = heatmap_to_centre(intensity, v[0], v[1])
                    if centre is not None:
                        n_out += 1
                        if gt_net is not None:
                            err = float(np.hypot(centre[0] - gt_net[0], centre[1] - gt_net[1]))
                            errs.append(err)
                            if err <= TOLERANCE_PX:
                                n_ok += 1
                    stats[v] = [n_out, n_ok, n_vis, errs]

    return {"total": total_frames, "stats": stats}


def report(result: dict) -> None:
    total = result["total"]
    print(f"\n{'=' * 78}")
    print(f"Ball localization accuracy - {total} frames, tolerance {TOLERANCE_PX:.0f}px "
          f"at 360x640")
    print(f"{'=' * 78}")
    print(f"{'thresh':>7} {'minpx':>6} {'det rate':>10} {'recall':>9} {'precision':>11} "
          f"{'median err':>12}")
    print("-" * 78)

    for v, (n_out, n_ok, n_vis, errs) in result["stats"].items():
        det = n_out / total if total else 0.0
        rec = n_ok / n_vis if n_vis else 0.0
        prec = n_ok / n_out if n_out else 0.0
        med = float(np.median(errs)) if errs else float("nan")
        mark = "  <- shipped" if v == SHIPPED else ""
        print(f"{v[0]:>7} {v[1]:>6} {det:>9.1%} {rec:>8.1%} {prec:>10.1%} "
              f"{med:>11.1f}px{mark}")

    print("-" * 78)
    print("det rate  = frames where a position was output (what the README called 82.5%)")
    print("recall    = visible-ball frames located within tolerance")
    print("precision = of positions output, the fraction within tolerance")

    # A single tolerance overstates or understates depending on where it sits relative to
    # the error distribution, so report the whole curve. 5px at 360x640 is roughly a ball
    # width; the same error is about 3x larger in pixels at 1080p, and what matters
    # downstream is whether the trajectory is right, not whether the centre is exact.
    print(f"\n{'=' * 78}")
    print("Error distribution, shipped settings (the tolerance choice matters this much)")
    print(f"{'=' * 78}")
    n_out, n_ok, n_vis, errs = result["stats"][SHIPPED]
    if errs:
        errs_arr = np.array(errs)
        print(f"{'within':>9} {'of outputs':>12} {'of visible frames':>19}")
        print("-" * 45)
        for tol in (2, 5, 10, 20, 40):
            hit = int((errs_arr <= tol).sum())
            print(f"{tol:>7}px {hit / len(errs_arr):>11.1%} {hit / n_vis:>18.1%}")
        print("-" * 45)
        for q in (50, 75, 90):
            print(f"  {q}th percentile error: {np.percentile(errs_arr, q):.1f}px")


def main():
    p = argparse.ArgumentParser(description="Measure ball localization against ground truth")
    p.add_argument("--clips", type=int, default=10, help="clips to evaluate")
    p.add_argument("--max-frames", type=int, default=200, help="frames per clip cap")
    p.add_argument("--sweep", action="store_true", help="sweep postprocessing settings")
    args = p.parse_args()

    if not Path(DATASET_ZIP).exists():
        print(f"Dataset not found at {DATASET_ZIP}. See datasets/README.md.")
        sys.exit(1)

    variants = SWEEP if args.sweep else [SHIPPED]
    report(evaluate(args.clips, args.max_frames, variants))


if __name__ == "__main__":
    main()
