"""
utils/court_validity.py
───────────────────────
Decides whether a detected court is trustworthy enough to compute metrics from.

Why this module exists
----------------------
The court keypoint model is a plain regression head: given any image it returns 14
points, and it has no way to signal "this camera angle is outside my training
distribution". On footage it was not trained for it returns points that still form a
tidy quadrilateral - just not one lying on the actual court. Everything downstream
(homography, real-world speeds, mini-court positions) is then computed from that
wrong quadrilateral and reported with full confidence.

That failure is worse than a crash, because the output looks plausible. On our own
eval suite it produced 79–89 km/h rally speeds that read as believable and were
nonetheless derived from a court fitted to the wrong part of the frame.

What does NOT work (measured, not assumed)
------------------------------------------
Homography reprojection error is useless here. Measured across 9 clips it ranged
1.40–1.88 px on correct fits and 2.13 px on a visibly wrong one, with 14/14 RANSAC
inliers in every case. It measures whether the 14 points are *self-consistent* - and
a tidy quadrilateral on the stands is perfectly self-consistent. See
`eval/court_validity_calibration.py` for the full table.

What does work
--------------
Image evidence: a real court line is painted brighter than the surface right beside
it, so we sample along every predicted line and ask whether the pixels there are
actually brighter than the pixels a few px to either side. A predicted line lying on
the crowd, the stands, or bare court surface fails; one lying on real paint passes.

Calibrated on the 9-clip eval suite against by-eye verification of each court fit:

    verified correct fits : 0.262, 0.384   (clips 3, 8)
    verified wrong fits   : 0.053, 0.067, 0.080, 0.187   (clips 11, 6, 5, 10)

MIN_LINE_SUPPORT sits in that gap. It is a small sample and the threshold should be
re-checked whenever the clip suite grows - it is a calibrated heuristic, not a
proof of correctness.
"""
from __future__ import annotations

import cv2
import numpy as np

# Pairs of keypoint indices that form the painted lines of a tennis court, in the
# 14-point convention the keypoint model was trained on.
COURT_LINES: tuple[tuple[int, int], ...] = (
    (0, 2),    # left outer sideline
    (1, 3),    # right outer sideline
    (0, 1),    # far baseline
    (2, 3),    # near baseline
    (4, 5),    # left singles sideline
    (6, 7),    # right singles sideline
    (8, 9),    # far service line
    (10, 11),  # near service line
    (12, 13),  # centre service line
)

# Below this share of line samples landing on paint, treat the fit as unusable.
# See the calibration table in this module's docstring.
MIN_LINE_SUPPORT = 0.22


def line_support_score(
    frame: np.ndarray,
    keypoints,
    lines: tuple[tuple[int, int], ...] = COURT_LINES,
    samples: int = 25,
    offset_px: int = 6,
    margin: int = 8,
) -> float:
    """
    Fraction of points sampled along the predicted court lines that lie on paint.

    Args:
        frame:      BGR video frame the keypoints were detected in.
        keypoints:  flat sequence of 28 values (x0, y0, x1, y1, ...).
        lines:      keypoint index pairs defining each line segment.
        samples:    sample points per line segment.
        offset_px:  perpendicular distance at which the surface is sampled.
        margin:     how much brighter than the surface a line pixel must be.

    Returns:
        Value in [0, 1]. Returns 0.0 when nothing could be sampled (keypoints
        entirely outside the frame, degenerate geometry).
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
        perp = np.array([-seg[1], seg[0]], dtype=np.float32) / length

        # Endpoints are skipped: court corners are the noisiest part of the
        # prediction and often sit where two lines meet, which confuses the
        # brighter-than-both-sides test.
        for t in np.linspace(0.1, 0.9, samples):
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


def assess_court_fit(
    frames: list[np.ndarray],
    keypoints_per_frame: list,
    sample_count: int = 12,
) -> tuple[bool, float]:
    """
    Judge a clip's court fit from a sample of its frames.

    The median is used rather than the mean so that a handful of replay frames or
    occluded moments cannot drag an otherwise good clip below the threshold.

    Args:
        frames:              the clip's frames.
        keypoints_per_frame: keypoints aligned with `frames`; a single shared
                             keypoint set is also accepted (single-frame mode).
        sample_count:        how many evenly spaced frames to score.

    Returns:
        (is_valid, median_line_support)
    """
    if not frames:
        return False, 0.0

    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)

    scores = []
    for i in indices:
        kp = (keypoints_per_frame[min(i, len(keypoints_per_frame) - 1)]
              if isinstance(keypoints_per_frame, list) and keypoints_per_frame
              and np.ndim(keypoints_per_frame[0]) > 0
              else keypoints_per_frame)
        scores.append(line_support_score(frames[i], kp))

    median = float(np.median(scores)) if scores else 0.0
    return median >= MIN_LINE_SUPPORT, median
