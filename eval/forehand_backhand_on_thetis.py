"""
eval/forehand_backhand_on_thetis.py
───────────────────────────────────
Is the forehand/backhand geometry biased, or is our reference clip just forehand-heavy?

The question
------------
On the reference clip the pipeline reports 10 forehands to 2 backhands. That looks wrong,
because real rallies run closer to even, but "looks wrong" is a prior, not a measurement,
and the clip has no shot-type labels to check against. It is entirely possible one player
attacked the other's forehand all rally.

THETIS has ground truth. Six of its twelve classes are forehand or backhand variants,
performed by 55 subjects, so the classifier can be tested away from our pipeline entirely.
If it is biased here, the bias is in the geometry. If it is accurate here, then the
pipeline's contact frames or that particular clip are responsible.

What this does and does not test
--------------------------------
It tests the SIDE PROJECTION: given a pose, does the hitting wrist read as being on the
player's own side (forehand) or across the midline (backhand).

It does NOT test the ball-based hand selection, because THETIS clips have no ball
annotation. In the pipeline the hitting hand is the wrist nearest the ball; here it is
approximated by the wrist furthest from the torso centre, which is the extended arm. That
is a fair proxy for a single-stroke clip and it is stated rather than hidden, because it
means a good score here does not clear the hand-selection logic.

THETIS is indoor Kinect footage. A model that works here may still fail on broadcast, so
this bounds the geometry, not the product.

Usage
-----
    python eval/forehand_backhand_on_thetis.py --per-class 25
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from utils import PoseEstimator
from utils.pose_shot_classifier import BACKHAND, FOREHAND, classify_forehand_backhand

THETIS = Path("datasets/external/thetis/VIDEO_RGB")

# Ground truth: which THETIS classes are forehands and which are backhands.
CLASS_TRUTH = {
    "forehand_flat": FOREHAND,
    "forehand_openstands": FOREHAND,
    "forehand_slice": FOREHAND,
    "forehand_volley": FOREHAND,
    "backhand": BACKHAND,
    "backhand_slice": BACKHAND,
    "backhand2hands": BACKHAND,
    "backhand_volley": BACKHAND,
}

# Frames around the clip's midpoint to try. THETIS clips hold one stroke, so the contact
# is near the middle; the exact frame is unknown, so the most confident call in the window
# is taken rather than an arbitrary single frame.
WINDOW = 6


def extended_wrist(landmarks):
    """
    Stand-in for the ball: the wrist furthest from the torso centre.

    The pipeline picks the hitting hand as the wrist nearest the ball. THETIS has no ball,
    so the extended arm is used instead, which for a single-stroke clip is the striking
    one. Passing its own position as `ball_pos` makes the classifier select that wrist.
    """
    ls, rs = landmarks.get("LEFT_SHOULDER"), landmarks.get("RIGHT_SHOULDER")
    if ls is None or rs is None:
        return None
    cx, cy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    best, best_d = None, -1.0
    for name in ("LEFT_WRIST", "RIGHT_WRIST"):
        w = landmarks.get(name)
        if w is None:
            continue
        d = ((w[0] - cx) ** 2 + (w[1] - cy) ** 2) ** 0.5
        if d > best_d:
            best, best_d = (w[0], w[1]), d
    return best


def classify_with_hand(landmarks, wrist_name):
    """Force the classifier to treat `wrist_name` as the hitting hand."""
    w = landmarks.get(wrist_name)
    if w is None:
        return None
    # Passing the wrist's own position as the ball makes the nearest-wrist selection
    # inside the classifier resolve to this hand.
    return classify_forehand_backhand(landmarks, (w[0], w[1]), ambiguity_ratio=0.0)


def classify_clip(path: Path, estimator: PoseEstimator):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 5:
        return None

    mid = len(frames) // 2
    best = None
    reachable = set()          # labels obtainable from EITHER hand, for the oracle
    for offset in range(-WINDOW, WINDOW + 1):
        i = mid + offset
        if not (0 <= i < len(frames)):
            continue
        h, w = frames[i].shape[:2]
        landmarks = estimator.detect_in_bbox(frames[i], [0, 0, w, h])
        if not landmarks:
            continue

        ball = extended_wrist(landmarks)
        if ball is not None:
            result = classify_forehand_backhand(landmarks, ball)
            if result is not None and (best is None or result[1] > best[1]):
                best = result

        # What the side projection COULD say if the correct hand were known. This
        # separates a broken projection from a broken choice of hand.
        for name in ("LEFT_WRIST", "RIGHT_WRIST"):
            forced = classify_with_hand(landmarks, name)
            if forced is not None:
                reachable.add(forced[0])

    return best, reachable


def main():
    ap = argparse.ArgumentParser(description="Forehand/backhand geometry on THETIS")
    ap.add_argument("--per-class", type=int, default=25, help="clips per class")
    args = ap.parse_args()

    if not THETIS.is_dir():
        print(f"THETIS not found at {THETIS}. Run scripts/download_thetis.py")
        sys.exit(1)

    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Run: tennis-vision download-models")
        sys.exit(1)

    print(f"\n{'class':<22} {'n':>4} {'correct':>8} {'accuracy':>9} {'undecided':>10}")
    print("-" * 60)

    predictions = Counter()
    truths = Counter()
    total_ok = total_n = total_undecided = total_oracle = total_clips = 0

    for name, truth in CLASS_TRUTH.items():
        folder = THETIS / name
        if not folder.is_dir():
            continue
        clips = sorted(folder.glob("*.avi"))[:args.per_class]
        ok = n = undecided = oracle_ok = 0
        for clip in clips:
            result, reachable = classify_clip(clip, estimator)
            if truth in reachable:
                oracle_ok += 1
            if result is None:
                undecided += 1
                continue
            n += 1
            predictions[result[0]] += 1
            truths[truth] += 1
            if result[0] == truth:
                ok += 1
        total_ok += ok
        total_n += n
        total_undecided += undecided
        total_oracle += oracle_ok
        total_clips += len(clips)
        acc = ok / n if n else 0.0
        orc = oracle_ok / len(clips) if clips else 0.0
        print(f"{name:<22} {n:>4} {ok:>8} {acc:>8.0%} {undecided:>10} {orc:>9.0%}")

    estimator.close()
    print("-" * 70)
    if total_clips:
        print()
        print(f"Oracle (correct label reachable from EITHER hand): "
              f"{total_oracle}/{total_clips} = {total_oracle / total_clips:.0%}")
        print("A high oracle with low accuracy means the side projection works and the")
        print("choice of hitting hand is what fails. A low oracle means the projection")
        print("itself cannot express the right answer.")
    if total_n:
        print(f"{'TOTAL':<22} {total_n:>4} {total_ok:>8} {total_ok / total_n:>8.0%} "
              f"{total_undecided:>10}")
        print(f"\nPredicted: {dict(predictions)}")
        print(f"Truth:     {dict(truths)}")
        fh = predictions[FOREHAND] / total_n
        print(f"\nPredicted forehand share: {fh:.0%}. Ground truth is 50% by construction,")
        print("so a share far from half is bias in the geometry rather than in the footage.")


if __name__ == "__main__":
    main()
