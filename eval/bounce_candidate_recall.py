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

    for path in label_files:
        detections, bounces, hits = load_clip(path)
        if not detections:
            continue
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
    print(f"\n{len(label_files)} clips | {n_bounce} labelled bounces | {n_hit} labelled hits"
          f" | tolerance ±{args.tolerance} frames\n")

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
