"""
eval/court_validity_calibration.py
──────────────────────────────────
Measures court-fit quality across the eval clip suite so the validity threshold is
picked from data rather than guessed.

Why this exists
---------------
The court keypoint model is a plain regression head: it emits 14 points for any
image and has no way to say "this camera angle is outside my training
distribution". On a ground-level clip it happily returns points scattered across
the stands, and every downstream metric (homography, speeds, mini-court positions)
is then computed from nonsense while still looking confident.

We need a signal that separates a good fit from a bad one. This script measures
three candidate signals per clip and prints them side by side:

  reproj_rmse_px  Round-trip error: fit H (video -> mini-court), invert it, map the
                  template points back into video space, compare against the
                  detected points. Low error means the 14 detected points really do
                  form a projective image of a tennis court.
  ransac_inliers  How many of the 14 points RANSAC kept. A scattered set loses points.
  area_frac       Fraction of the frame covered by the detected court quad. Catches
                  fits that are self-consistent but absurdly small or large.

IMPORTANT caveat, stated up front: reprojection error measures the *self-consistency*
of the fit, not its correctness. A set of points forming a clean quadrilateral in
the wrong place (say, on the stands) can fit with low error. That is exactly why
this script reports area_frac and inliers too, and why the gate built from it is a
"probably usable" filter rather than a correctness proof.

Usage:
    python eval/court_validity_calibration.py [--frames N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt

CLIP_DIR = Path("datasets/evail_clips")
COURT_MODEL = "models/keypoints_model.pth"

# Clips whose court fit we already judged by eye, used to check the numbers agree
# with what we can see. Everything else is unlabelled.
KNOWN = {
    "input_video_3":  "good (dots on lines)",
    "input_video_6":  "BAD (ground-level, dots on stands)",
    "input_video_8":  "good (dots on lines)",
}


def sample_frame_indices(total: int, n: int) -> list[int]:
    """Evenly spaced sample, avoiding the first/last frame (often replay wipes)."""
    if total <= 2:
        return [0]
    n = min(n, total - 2)
    return list(np.linspace(1, total - 2, n, dtype=int))


def court_quad_area_fraction(keypoints: np.ndarray, frame_w: int, frame_h: int) -> float:
    """
    Fraction of the frame covered by the convex hull of the detected keypoints.

    Uses the hull rather than a fixed corner ordering because a bad detection has no
    meaningful ordering — the hull is well defined either way.
    """
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    hull = cv2.convexHull(pts)
    return float(cv2.contourArea(hull) / (frame_w * frame_h))


def reprojection_rmse_px(mini_court: MiniCourt, keypoints: np.ndarray) -> tuple[float, int]:
    """
    Round-trip RMSE in video pixels, plus the RANSAC inlier count.

    Returns (inf, 0) when no homography can be fitted at all.
    """
    src = np.asarray(keypoints, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(mini_court.get_court_drawing_keypoints(), dtype=np.float32).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
    if H is None:
        return float("inf"), 0
    inliers = int(mask.sum()) if mask is not None else 0

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf"), inliers

    # Map the template back into video space and compare with what was detected.
    back = cv2.perspectiveTransform(dst, H_inv).reshape(-1, 2)
    detected = src.reshape(-1, 2)
    err = np.linalg.norm(back - detected, axis=1)
    return float(np.sqrt((err ** 2).mean())), inliers


def line_support_score(
    frame: np.ndarray,
    keypoints: np.ndarray,
    lines: list[tuple[int, int]],
    samples: int = 25,
    offset_px: int = 6,
    margin: int = 8,
) -> float:
    """
    Fraction of points sampled along the predicted court lines that actually sit on
    something line-like in the image.

    This is the signal reprojection error cannot provide. A painted court line is
    brighter than the surface immediately either side of it, so at each sample we
    compare the pixel on the predicted line against the pixels `offset_px` away
    along the line's perpendicular. A predicted line lying on the stands, the crowd,
    or bare court surface fails this test; a line lying on real paint passes it.

    Returns a value in [0, 1] — the fraction of all samples across all 9 court lines
    that look like paint.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = gray.shape
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)

    def brightness(x: float, y: float) -> float | None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            return float(gray[yi, xi])
        return None

    hits = total = 0
    for a, b in lines:
        if a >= len(kp) or b >= len(kp):
            continue
        p0, p1 = kp[a], kp[b]
        seg = p1 - p0
        length = float(np.hypot(*seg))
        if length < 1.0:
            continue
        # Unit perpendicular, used to sample the surface either side of the line.
        perp = np.array([-seg[1], seg[0]], dtype=np.float32) / length

        for t in np.linspace(0.1, 0.9, samples):   # skip endpoints (corners are noisy)
            point = p0 + seg * t
            centre = brightness(*point)
            side_a = brightness(*(point + perp * offset_px))
            side_b = brightness(*(point - perp * offset_px))
            if centre is None or side_a is None or side_b is None:
                continue
            total += 1
            if centre > max(side_a, side_b) + margin:
                hits += 1

    return hits / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=8, help="frames sampled per clip")
    ap.add_argument("--model", default=COURT_MODEL,
                    help="keypoint weights to evaluate (compare fine-tuned vs shipped)")
    args = ap.parse_args()

    clips = sorted(CLIP_DIR.glob("*.mp4"))
    if not clips:
        sys.exit(f"No clips found in {CLIP_DIR}")

    detector = CourtLineDetector(model_path=args.model)

    print(f"\nSampling {args.frames} frames per clip from {CLIP_DIR}")
    print(f"Weights: {args.model}\n")
    print(f"{'clip':22s} {'reproj_rmse_px':>15s} {'inliers':>8s} {'area_frac':>10s} "
          f"{'line_support':>13s}   note")
    print("-" * 100)

    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in sample_frame_indices(total, args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()

        if not frames:
            print(f"{clip.stem:22s} {'--- unreadable ---':>16s}")
            continue

        mini_court = MiniCourt(frames[0])
        rmses, inliers, areas, supports = [], [], [], []
        h, w = frames[0].shape[:2]

        for frame in frames:
            kp = detector.predict(frame)
            rmse, n_in = reprojection_rmse_px(mini_court, kp)
            rmses.append(rmse)
            inliers.append(n_in)
            areas.append(court_quad_area_fraction(kp, w, h))
            supports.append(line_support_score(frame, kp, mini_court.lines))

        finite = [r for r in rmses if np.isfinite(r)]
        med_rmse = float(np.median(finite)) if finite else float("inf")
        print(f"{clip.stem:22s} {med_rmse:15.2f} {np.mean(inliers):8.1f} "
              f"{np.mean(areas):10.3f} {np.median(supports):13.3f}   "
              f"{KNOWN.get(clip.stem, '')}")

    print("\nRead the two known-BAD vs known-good rows: the gate threshold belongs")
    print("in the gap between them. If there is no gap, this signal does not work")
    print("and the gate needs a different one — say so rather than picking a number.\n")


if __name__ == "__main__":
    main()
