"""
eval/test_xvelocity_candidate_generator.py
────────────────────────────────────────────
Tests whether an x-velocity-change-based candidate generator recovers real contacts that
the existing y-reversal detector structurally misses (journal 0014: some real shots don't
reverse vertical direction at all, or reverse too shallowly — not a threshold problem,
verified by sweeping min_delta_y down to 1 with no change).

Motivation: journal 0012 established that |vx change| is what physically distinguishes a
hit (player redirects the ball, can reverse x-direction) from a bounce or from noise
(court/trajectory mostly preserves x-velocity). If that's true, an x-velocity-change local
maximum should be a candidate-generation signal in its own right, not just a downstream
classifier feature — and might catch real contacts a pure y-reversal misses.

Tested at dataset scale (91 real clips), not just our own clip — the count-match mistake
from journal 0014 was exactly a result of not doing this the first time.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATASET_ZIP = "datasets/external/tracknet_original/Dataset.zip"
TOLERANCE = 10
WINDOW = 4   # matches EVENT_WINDOW in utils/hit_bounce_classifier.py


def load_clip(zf, label_path):
    with zf.open(label_path) as f:
        rows = list(csv.DictReader(io.StringIO(f.read().decode("utf-8"))))
    positions = []
    for r in rows:
        if r["visibility"] != "0" and r["x-coordinate"]:
            positions.append((float(r["x-coordinate"]), float(r["y-coordinate"])))
        else:
            positions.append(None)
    true_events = [i for i, r in enumerate(rows) if r["status"] in ("1", "2")]
    return positions, true_events


def xvelocity_candidates(positions: list, min_delta_x: float = 5.0, min_spacing: int = 15) -> list[int]:
    """Local maxima in |vx change| over a sliding window — the same feature that made the
    hit/bounce classifier work, used here as a candidate generator instead of a filter."""
    n = len(positions)
    scores = [0.0] * n
    for i in range(WINDOW, n - WINDOW):
        before = [p for p in positions[i - WINDOW:i] if p is not None]
        after  = [p for p in positions[i + 1:i + 1 + WINDOW] if p is not None]
        if len(before) < 2 or len(after) < 2:
            continue
        vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
        vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
        scores[i] = abs(vx_after - vx_before)

    candidates = []
    for i in range(n):
        if scores[i] < min_delta_x:
            continue
        lo, hi = max(0, i - min_spacing // 2), min(n, i + min_spacing // 2 + 1)
        if scores[i] == max(scores[lo:hi]):   # local peak within its own neighborhood
            if not candidates or i - candidates[-1] >= min_spacing:
                candidates.append(i)
    return candidates


def main():
    zf = zipfile.ZipFile(DATASET_ZIP)
    label_paths = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))

    from trackers.tracknet_ball_tracker import TrackNetBallTracker
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")

    for min_dx in (3.0, 5.0, 8.0):
        total_true = total_yrev_matched = total_xvel_matched = 0
        total_yrev_det = total_xvel_det = 0
        skipped = 0

        for lp in label_paths:
            positions, true_events = load_clip(zf, lp)
            n = len(positions)
            if n < 40:
                skipped += 1
                continue

            det_list = [{1: [p[0]-5, p[1]-5, p[0]+5, p[1]+5]} if p else {} for p in positions]
            yrev = tracker.get_ball_shot_frames(det_list)
            xvel = xvelocity_candidates(positions, min_delta_x=min_dx)

            total_true += len(true_events)
            total_yrev_det += len(yrev)
            total_xvel_det += len(xvel)
            total_yrev_matched += sum(1 for e in true_events if any(abs(e-d) <= TOLERANCE for d in yrev))
            total_xvel_matched += sum(1 for e in true_events if any(abs(e-d) <= TOLERANCE for d in xvel))

        print(f"--- min_delta_x={min_dx} ({len(label_paths)-skipped} clips, {total_true} true events) ---")
        print(f"  y-reversal : {total_yrev_det:>4} candidates, recall {100*total_yrev_matched/total_true:.1f}%")
        print(f"  x-velocity : {total_xvel_det:>4} candidates, recall {100*total_xvel_matched/total_true:.1f}%")
        print()


if __name__ == "__main__":
    main()
