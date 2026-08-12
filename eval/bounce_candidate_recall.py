"""
eval/bounce_candidate_recall.py
───────────────────────────────
Measures how many real bounces the candidate generators actually propose.

Why recall is the metric that matters here
------------------------------------------
Candidate generation is the first stage of the event pipeline. A bounce never
proposed at this stage cannot be recovered by any downstream classifier, however
good — which is exactly the failure that broke serve speed: the serve's landing was
never a candidate, so a later rally event was paired with the serve instead and the
measured distance came out at 26.6 m for an ~18 m serve.

Precision matters much less, because `classify_reversals_by_trajectory` and the
physical gates downstream exist to reject bad candidates. Over-proposing costs a
little compute; under-proposing loses the event permanently.

Ground truth
------------
The original TrackNet dataset, already on disk. Each clip's `Label.csv` carries a
per-frame `status`: 0 = flying, 1 = hit, 2 = bounce (encoding confirmed against the
paper and the upstream training code — see datasets/README.md). We compare against
status == 2.

A candidate counts as recalling a bounce if it lands within `--tolerance` frames of
it; the trajectory around a bounce curves over several frames, so demanding the exact
frame would measure frame-alignment rather than detection.

Usage:
    python eval/bounce_candidate_recall.py
    python eval/bounce_candidate_recall.py --clips 40 --tolerance 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.bounce_candidates import detect_bounce_candidates
from utils.hit_bounce_classifier import detect_xvelocity_candidates

DATASET = Path("datasets/external/tracknet_original/Dataset")


def load_clip(label_path: Path):
    """Returns (ball_detections, bounce_frames, hit_frames) for one clip."""
    detections, bounces, hits = [], [], []
    with open(label_path, encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            try:
                x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
                # The generators take bboxes; a 2px box around the point is enough
                # since they immediately reduce it back to a centre.
                detections.append({1: [x - 1, y - 1, x + 1, y + 1]})
            except (ValueError, KeyError):
                detections.append({})
            status = (row.get("status") or "").strip()
            if status == "2":
                bounces.append(index)
            elif status == "1":
                hits.append(index)
    return detections, bounces, hits


def simulate_production_input(label_path: Path) -> list[dict]:
    """
    Rebuild a clip's detections the way the pipeline actually sees them.

    The headline 83.9 % recall was measured on the dataset's own ground-truth
    coordinates: every frame present, no noise, no gaps. Production never sees that.
    It sees TrackNet output, which misses roughly 17.5 % of frames, passed through
    `interpolate_ball_positions` — linear interpolation plus a 3-frame rolling median.

    That matters specifically for bounce detection. The signal is a sharp change in
    vertical velocity, and the ball is hardest to detect exactly at a bounce, where it
    is fastest and lowest against the court. If the bounce frames themselves are the
    missing ones, interpolation draws a straight line straight through the event and
    the median filter smooths what remains — erasing the discontinuity we key on.

    Simulation: treat only clearly-visible frames (visibility == 1) as detected, which
    is the pessimistic-but-principled stand-in for a detector that struggles with fast,
    blurred, occluded balls. Then apply the pipeline's own interpolation, so the
    smoothing is identical rather than merely similar.
    """
    import pandas as pd

    rows, n = [], 0
    with open(label_path, encoding="utf-8") as f:
        for index, row in enumerate(csv.DictReader(f)):
            n = index + 1
            visible = (row.get("visibility") or "").strip() == "1"
            try:
                x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
            except (ValueError, KeyError):
                visible = False
                x = y = float("nan")
            rows.append({"frame": index,
                         "x": x if visible else float("nan"),
                         "y": y if visible else float("nan")})

    if not rows:
        return []

    df = pd.DataFrame(rows).set_index("frame")
    detected = int(df["x"].notna().sum())
    df[["x", "y"]] = df[["x", "y"]].interpolate().bfill().ffill()
    # Identical to trackers/tracknet_ball_tracker.interpolate_ball_positions.
    df["x"] = df["x"].rolling(3, center=True, min_periods=1).median()
    df["y"] = df["y"].rolling(3, center=True, min_periods=1).median()

    simulate_production_input.last_detection_rate = detected / max(n, 1)
    return [{1: [r.x - 1, r.y - 1, r.x + 1, r.y + 1]} for r in df.itertuples()]


def recall_of(candidates: list[int], truth: list[int], tolerance: int) -> int:
    """How many ground-truth events have a candidate within `tolerance` frames."""
    return sum(
        1 for t in truth
        if any(abs(c - t) <= tolerance for c in candidates)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=0, help="limit number of clips")
    ap.add_argument("--tolerance", type=int, default=4, help="frame tolerance")
    ap.add_argument("--simulate-production", action="store_true",
                    help="feed detections through the gaps + interpolation the pipeline "
                         "actually sees, instead of perfect ground-truth coordinates")
    args = ap.parse_args()

    label_files = sorted(DATASET.glob("game*/Clip*/Label.csv"))
    if not label_files:
        sys.exit(f"No Label.csv found under {DATASET}")
    if args.clips:
        label_files = label_files[:args.clips]

    totals = {
        "bounces": 0, "hits": 0,
        "bounce_gen_hits_bounce": 0, "bounce_gen_hits_hit": 0, "bounce_gen_count": 0,
        "xvel_gen_hits_bounce": 0, "xvel_gen_hits_hit": 0, "xvel_gen_count": 0,
    }

    detection_rates: list[float] = []

    for path in label_files:
        detections, bounces, hits = load_clip(path)
        if not detections:
            continue

        if args.simulate_production:
            simulated = simulate_production_input(path)
            if simulated:
                detections = simulated
                detection_rates.append(simulate_production_input.last_detection_rate)

        bounce_candidates = detect_bounce_candidates(detections)
        xvel_candidates = detect_xvelocity_candidates(detections)

        totals["bounces"] += len(bounces)
        totals["hits"] += len(hits)
        totals["bounce_gen_count"] += len(bounce_candidates)
        totals["xvel_gen_count"] += len(xvel_candidates)
        totals["bounce_gen_hits_bounce"] += recall_of(bounce_candidates, bounces, args.tolerance)
        totals["bounce_gen_hits_hit"] += recall_of(bounce_candidates, hits, args.tolerance)
        totals["xvel_gen_hits_bounce"] += recall_of(xvel_candidates, bounces, args.tolerance)
        totals["xvel_gen_hits_hit"] += recall_of(xvel_candidates, hits, args.tolerance)

    n_bounce, n_hit = totals["bounces"], totals["hits"]
    mode = ("PRODUCTION-SIMULATED input (gaps + interpolation + median filter)"
            if args.simulate_production else "GROUND-TRUTH coordinates (ideal input)")
    print(f"\n{len(label_files)} clips | {n_bounce} labelled bounces | {n_hit} labelled hits"
          f" | tolerance ±{args.tolerance} frames")
    print(f"Input: {mode}")
    if detection_rates:
        print(f"Simulated detection rate: {sum(detection_rates)/len(detection_rates):.1%} "
              f"of frames (pipeline measures ~82.5% with TrackNet)")
    print()

    print(f"  {'generator':>22s} {'bounce recall':>15s} {'hit recall':>12s} {'candidates':>12s}")
    print("  " + "-" * 64)
    print(f"  {'detect_bounce_candidates':>22s} "
          f"{totals['bounce_gen_hits_bounce'] / max(n_bounce, 1):14.1%} "
          f"{totals['bounce_gen_hits_hit'] / max(n_hit, 1):11.1%} "
          f"{totals['bounce_gen_count']:12d}")
    print(f"  {'detect_xvelocity (old)':>22s} "
          f"{totals['xvel_gen_hits_bounce'] / max(n_bounce, 1):14.1%} "
          f"{totals['xvel_gen_hits_hit'] / max(n_hit, 1):11.1%} "
          f"{totals['xvel_gen_count']:12d}")

    print("\n  Read this as: does the new generator find bounces the x-velocity one")
    print("  structurally cannot? A high bounce recall with a LOWER hit recall is the")
    print("  goal — it means the two generators are complementary rather than")
    print("  duplicating each other, which is what the union in main.py needs.\n")


if __name__ == "__main__":
    main()
