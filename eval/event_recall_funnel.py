"""
eval/event_recall_funnel.py
───────────────────────────
Where do the missing contacts go?

Event recall on real detections is 51.3%, against 87.6% given perfect ball positions
(see eval/event_detection_on_real_detections.py). So roughly half the contacts in a rally
are lost somewhere between the video and the reported event list, and "ball detection
noise" is a diagnosis at the wrong resolution to act on.

There are four places a labelled contact can disappear, and they need different fixes:

  1. NO BALL NEARBY      the detector found nothing within a few frames of the contact,
                         so no generator could possibly fire. Fix: detection.
  2. NO CANDIDATE        the ball was detected but no generator proposed this frame.
                         Fix: generator thresholds, or a generator blind to this shape.
  3. LOST IN MERGING     a candidate existed but merge_nearby_candidates collapsed it
                         into a neighbour. Fix: the merge window.
  4. CLASSIFIED AWAY     a merged candidate survived to the classifier, which either
                         rejected it or could not decide. Fix: the classifier.

This script attributes every miss to exactly one of those, so the next piece of work is
chosen by size rather than by guess.

Usage
-----
    python eval/event_recall_funnel.py --clips 12

Needs the dataset (7+ GB, not redistributable). See datasets/README.md.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ball_localization_accuracy import DATASET_ZIP, heatmap_to_centre, load_clip
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils import detect_xvelocity_candidates, merge_nearby_candidates
from utils.bounce_candidates import detect_bounce_candidates
from utils.hit_bounce_classifier import classify_reversals_by_trajectory

# A stage "covers" a labelled contact if it produced a frame within this many frames of it.
# Matches the tolerance used by the other event evals so numbers stay comparable.
TOLERANCE = 10

# How far either side of a contact a real detection must exist for the contact to be
# reachable at all. The generators read velocity over a window, so an isolated detection
# exactly on the contact frame is not enough.
DETECTION_WINDOW = 4


def match_one_to_one(stage_frames: set[int] | list[int], contacts: list[int]) -> set[int]:
    """
    Which contacts this stage covers, matching each proposed frame to at most one contact.

    One-to-one matters more than it looks. A naive "is any proposed frame within tolerance"
    test lets a single detection satisfy several nearby labelled contacts, and with a 10
    frame window and contacts sometimes 15 frames apart that inflates coverage badly: it
    reported 90% where strict matching reports 51%. The other event evals in this directory
    match greedily and one-to-one, so this has to as well or the funnel does not describe
    the same pipeline they measure.
    """
    available = sorted(stage_frames)
    used: set[int] = set()
    matched: set[int] = set()

    # Nearest-first over contacts, so a contact with an exact proposal is not robbed of it
    # by a neighbour that had a worse but still in-tolerance alternative.
    pairs = []
    for t in contacts:
        for f in available:
            gap = abs(f - t)
            if gap <= TOLERANCE:
                pairs.append((gap, t, f))
    pairs.sort()

    for _gap, t, f in pairs:
        if t in matched or f in used:
            continue
        matched.add(t)
        used.add(f)
    return matched


def sweep_min_gap(n_clips: int, max_frames: int, gaps: list[int]) -> None:
    """
    Recall and precision against merge_nearby_candidates' min_gap.

    The funnel attributes most missed contacts to merging, so this is the knob under
    suspicion. Detection runs once per clip and every gap is scored against the same
    cached candidates, so the sweep costs one pass.
    """
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    if tracker.model is None:
        print("TrackNet weights not loaded. See README 'Models'.")
        sys.exit(1)

    H, W = tracker.INPUT_HEIGHT, tracker.INPUT_WIDTH
    zf = zipfile.ZipFile(DATASET_ZIP)
    labels = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))[:n_clips]

    cached = []   # (raw_union, interpolated, contacts) per clip
    for ci, label_path in enumerate(labels, 1):
        frames, _pos, contacts = load_clip(zf, label_path, max_frames)
        if len(frames) < 10 or not contacts:
            continue
        orig_h, orig_w = frames[0].shape[:2]
        detections = []
        with torch.no_grad():
            for i in range(len(frames)):
                prev_f = frames[max(0, i - 1)]
                next_f = frames[min(len(frames) - 1, i + 1)]
                out = tracker.model(tracker._preprocess([prev_f, frames[i], next_f]))
                intensity = out.argmax(dim=1)[0].cpu().numpy().reshape(H, W)
                centre = heatmap_to_centre(intensity, 0, 5, largest_blob=True)
                if centre is None:
                    detections.append({})
                else:
                    cx = centre[0] * orig_w / W
                    cy = centre[1] * orig_h / H
                    detections.append({1: [cx - 12, cy - 12, cx + 12, cy + 12]})
        interp = tracker.interpolate_ball_positions(detections)
        union = (set(tracker.get_ball_shot_frames(interp))
                 | set(detect_xvelocity_candidates(interp))
                 | set(detect_bounce_candidates(interp)))
        cached.append((sorted(union), interp, contacts))
        print(f"  [{ci}/{len(labels)}] {label_path.rsplit('/', 2)[-2]}: {len(contacts)} labelled")

    print()
    print("=" * 66)
    print("merge_nearby_candidates min_gap sweep")
    print(f"{'=' * 66}")
    print(f"{'min_gap':>8} {'linkage':>9} {'events':>7} {'recall':>9} {'precision':>11} {'F1':>7}")
    print("-" * 66)
    def merge_bounded(candidates, min_gap):
        """Same clustering, but a cluster may not grow beyond min_gap frames wide.

        The shipped version compares each candidate to the cluster's LAST member, so a
        chain of candidates each just under min_gap apart merges transitively into one
        arbitrarily wide cluster. Comparing to the cluster's FIRST member bounds the span
        at min_gap, which is what "these are the same event" was meant to express."""
        if not candidates:
            return []
        ordered = sorted(candidates)
        clusters = [[ordered[0]]]
        for c in ordered[1:]:
            if c - clusters[-1][0] <= min_gap:
                clusters[-1].append(c)
            else:
                clusters.append([c])
        return [cl[len(cl) // 2] for cl in clusters]

    for gap, bounded in [(g, b) for g in gaps for b in (False, True)]:
        total_truth = total_matched = total_events = 0
        for union, interp, contacts in cached:
            merged = (merge_bounded(union, gap) if bounded
                      else merge_nearby_candidates(union, min_gap=gap))
            hits, bounces = classify_reversals_by_trajectory(merged, interp)
            events = sorted(set(hits) | set(bounces))
            total_events += len(events)
            total_truth += len(contacts)
            total_matched += len(match_one_to_one(events, contacts))
        rec = total_matched / total_truth if total_truth else 0.0
        prec = total_matched / total_events if total_events else 0.0
        f1 = 2 * rec * prec / (rec + prec) if (rec + prec) else 0.0
        mark = "  <- shipped" if (gap == 10 and not bounded) else ""
        kind = "bounded" if bounded else "chained"
        print(f"{gap:>8} {kind:>9} {total_events:>7} {rec:>8.1%} {prec:>10.1%} "
              f"{f1:>7.3f}{mark}")
    print("-" * 66)


def run(n_clips: int, max_frames: int) -> Counter:
    tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    if tracker.model is None:
        print("TrackNet weights not loaded. See README 'Models'.")
        sys.exit(1)

    H, W = tracker.INPUT_HEIGHT, tracker.INPUT_WIDTH
    zf = zipfile.ZipFile(DATASET_ZIP)
    labels = sorted(n for n in zf.namelist() if n.endswith("Label.csv"))[:n_clips]

    verdicts: Counter = Counter()

    for ci, label_path in enumerate(labels, 1):
        frames, _pos, contacts = load_clip(zf, label_path, max_frames)
        if len(frames) < 10 or not contacts:
            continue

        orig_h, orig_w = frames[0].shape[:2]
        detections: list[dict] = []
        real_frames: set[int] = set()

        with torch.no_grad():
            for i in range(len(frames)):
                prev_f = frames[max(0, i - 1)]
                next_f = frames[min(len(frames) - 1, i + 1)]
                out = tracker.model(tracker._preprocess([prev_f, frames[i], next_f]))
                intensity = out.argmax(dim=1)[0].cpu().numpy().reshape(H, W)
                centre = heatmap_to_centre(intensity, 0, 5, largest_blob=True)
                if centre is None:
                    detections.append({})
                else:
                    cx = centre[0] * orig_w / W
                    cy = centre[1] * orig_h / H
                    detections.append({1: [cx - 12, cy - 12, cx + 12, cy + 12]})
                    real_frames.add(i)

        interpolated = tracker.interpolate_ball_positions(detections)

        # Stage by stage, exactly as derive_shot_frames composes them.
        yrev = set(tracker.get_ball_shot_frames(interpolated))
        xvel = set(detect_xvelocity_candidates(interpolated))
        bnce = set(detect_bounce_candidates(interpolated))
        raw_union = yrev | xvel | bnce
        merged = merge_nearby_candidates(sorted(raw_union))
        hits, bounces = classify_reversals_by_trajectory(merged, interpolated)
        classified = set(hits) | set(bounces)

        m_classified = match_one_to_one(classified, contacts)
        m_merged = match_one_to_one(merged, contacts)
        m_union = match_one_to_one(raw_union, contacts)

        for t in contacts:
            # Was the ball actually seen around this contact?
            window = range(max(0, t - DETECTION_WINDOW),
                           min(len(frames), t + DETECTION_WINDOW + 1))
            n_real = sum(1 for f in window if f in real_frames)

            if t in m_classified:
                verdicts["reported"] += 1
            elif t not in m_union:
                # Nothing reached this contact. Distinguish "could not have" from
                # "should have": with almost no real detections nearby, no generator
                # had the velocity context to fire in the first place.
                verdicts["1. no ball nearby" if n_real < 2 else "2. no candidate proposed"] += 1
            elif t not in m_merged:
                verdicts["3. lost in merging"] += 1
            else:
                verdicts["4. classified away"] += 1

        print(f"  [{ci}/{len(labels)}] {label_path.rsplit('/', 2)[-2]}: "
              f"{len(contacts)} labelled")

    return verdicts


def main():
    p = argparse.ArgumentParser(description="Attribute missed contacts to a pipeline stage")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--sweep-gap", action="store_true",
                   help="sweep merge_nearby_candidates min_gap instead of the funnel")
    args = p.parse_args()

    if not Path(DATASET_ZIP).exists():
        print(f"Dataset not found at {DATASET_ZIP}. See datasets/README.md.")
        sys.exit(1)

    if args.sweep_gap:
        sweep_min_gap(args.clips, args.max_frames, [3, 4, 6, 10])
        return

    v = run(args.clips, args.max_frames)
    total = sum(v.values())
    if not total:
        print("No labelled contacts found.")
        return

    print(f"\n{'=' * 62}")
    print(f"Where the contacts go  ({total} labelled contacts)")
    print(f"{'=' * 62}")
    order = ["reported", "1. no ball nearby", "2. no candidate proposed",
             "3. lost in merging", "4. classified away"]
    for key in order:
        n = v.get(key, 0)
        bar = "#" * int(round(40 * n / total))
        print(f"  {key:<26} {n:>4}  {n / total:>6.1%}  {bar}")
    print(f"{'=' * 62}")

    missed = total - v.get("reported", 0)
    if missed:
        print(f"\nOf the {missed} missed ({missed / total:.1%} of all contacts):")
        for key in order[1:]:
            n = v.get(key, 0)
            if n:
                print(f"  {n / missed:>6.1%}  {key}")
        print("\nThe largest bucket is where the next piece of work belongs.")


if __name__ == "__main__":
    main()
