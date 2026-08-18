"""
eval/extract_thetis_pose_features.py
────────────────────────────────────
Extracts named pose features from THETIS clips for forehand/backhand training.

Why a learned classifier replaces the geometric rule
----------------------------------------------------
The hand-crafted side projection in utils/pose_shot_classifier.py measures 54% against
THETIS ground truth on a balanced two-class problem, predicting forehand 89% of the time
(eval/forehand_backhand_on_thetis.py). Its ceiling is structural, not a tuning issue: even
given the correct hitting hand it tops out at 78%, and on volleys it reaches 27%, because
a volley is blocked with the body square to the net and the wrist never crosses the
shoulder midline the whole test depends on.

No rule covers both cases without becoming a classifier, so this builds the input for one.

Why named summary features rather than raw sequences
----------------------------------------------------
The first version of this script emitted a 16-frame sequence as 384 raw numbers. With the
data available that was more than twice as many features as samples, which fits noise and
produces an accuracy that means nothing.

These are ~20 named quantities instead, each one a thing a coach would actually point at:
how fast each wrist travelled, which side of the body it was on at the moment of peak
speed, how far apart the hands were, how much the shoulders rotated. That keeps the model
small enough to train honestly, and it keeps the learned weights readable, so a wrong
prediction can be traced to a feature rather than to a black box. It is the same approach
as utils/hit_bounce_classifier, which uses four named features and is interpretable for
exactly this reason.

Everything is measured in a body-relative frame
-----------------------------------------------
Positions are relative to the torso centre and divided by shoulder width, so features do
not change with where the player stands, how far away they are, or the video resolution.
Speeds are in shoulder-widths per frame for the same reason.

The moment that matters is peak wrist speed, which is the closest available proxy for
contact in a clip that carries no ball annotation.

Splitting
---------
THETIS subject ids are p1 to p55 (p1-p31 beginners, p32-p55 experts) and each subject
performs every stroke several times. Splitting by clip would put the same persons other
takes in both train and test and overstate accuracy badly, so the subject is recorded on
every sample and training must group on it.

Usage
-----
    python eval/extract_thetis_pose_features.py
    python eval/extract_thetis_pose_features.py --per-class 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from utils import PoseEstimator

THETIS = Path("datasets/external/thetis/VIDEO_RGB")
DEFAULT_OUT = "datasets/external/thetis_pose_features.json"

# Forehand and backhand variants only. Services and the smash are handled by physical
# rules that already work (ball above the head, at or away from a baseline), so they are
# not part of the problem this classifier exists to solve.
CLASS_TRUTH = {
    "forehand_flat": "forehand",
    "forehand_openstands": "forehand",
    "forehand_slice": "forehand",
    "forehand_volley": "forehand",
    "backhand": "backhand",
    "backhand_slice": "backhand",
    "backhand2hands": "backhand",
    "backhand_volley": "backhand",
}

# Frames sampled across the middle half of each clip. THETIS pads every recording with the
# player walking into and out of position, which carries no stroke information.
SEQ_FRAMES = 20

SUBJECT_RE = re.compile(r"^(p\d+)_")

FEATURE_NAMES = [
    # Per wrist, at the frame of that wrist's peak speed.
    "left_peak_speed", "right_peak_speed",
    "left_side_at_peak", "right_side_at_peak",
    "left_reach_at_peak", "right_reach_at_peak",
    "left_height_at_peak", "right_height_at_peak",
    # Extremes of the side projection across the whole stroke.
    "left_side_max", "left_side_min",
    "right_side_max", "right_side_min",
    # Two-handedness and which arm led.
    "wrist_gap_at_peak", "wrist_gap_min",
    "faster_wrist_is_right", "peak_speed_ratio",
    # Torso rotation: a backhand turns the shoulders further than a forehand.
    "shoulder_angle_range", "shoulder_angle_at_peak",
    # Elbow extension distinguishes a driven stroke from a blocked one.
    "left_elbow_extension", "right_elbow_extension",
]


def frame_geometry(landmarks):
    """
    Body-relative geometry for one frame, or None when the shoulders are unusable.

    Shoulder width is the only available scale reference, so it is the one quantity that
    cannot be defaulted: without it nothing here is comparable between players or clips.
    """
    ls, rs = landmarks.get("LEFT_SHOULDER"), landmarks.get("RIGHT_SHOULDER")
    if ls is None or rs is None:
        return None
    width = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if width < 1e-6:
        return None

    cx, cy = (ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0
    # Unit vector from the right shoulder to the left: the players own left is positive.
    ax, ay = (ls[0] - rs[0]) / width, (ls[1] - rs[1]) / width

    out = {
        "width": width,
        "shoulder_angle": math.atan2(ay, ax),
    }
    for side, wrist, elbow in (("left", "LEFT_WRIST", "LEFT_ELBOW"),
                               ("right", "RIGHT_WRIST", "RIGHT_ELBOW")):
        w = landmarks.get(wrist)
        if w is None:
            return None
        dx, dy = (w[0] - cx) / width, (w[1] - cy) / width
        out[f"{side}_pos"] = (dx, dy)
        # Signed projection onto the shoulder axis: positive means the players own left.
        out[f"{side}_side"] = dx * ax + dy * ay
        out[f"{side}_reach"] = math.hypot(dx, dy)
        # Image y grows downward, so negate to make "higher" positive.
        out[f"{side}_height"] = -dy

        e = landmarks.get(elbow)
        if e is not None:
            out[f"{side}_elbow_extension"] = math.hypot(w[0] - e[0], w[1] - e[1]) / width
        else:
            out[f"{side}_elbow_extension"] = 0.0
    return out


def clip_features(path: Path, estimator: PoseEstimator):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < SEQ_FRAMES:
        return None

    lo, hi = len(frames) // 4, len(frames) * 3 // 4
    indices = [lo + (hi - lo) * i // max(1, SEQ_FRAMES - 1) for i in range(SEQ_FRAMES)]

    geo = []
    for i in indices:
        h, w = frames[i].shape[:2]
        landmarks = estimator.detect_in_bbox(frames[i], [0, 0, w, h])
        geo.append(frame_geometry(landmarks) if landmarks else None)

    usable = [g for g in geo if g is not None]
    # Needs enough of the stroke to measure motion at all. Half is generous, and clips
    # below it are reported rather than silently filled in.
    if len(usable) < SEQ_FRAMES // 2:
        return None

    # Speeds between consecutive usable frames, in shoulder-widths per frame.
    speeds = {"left": [0.0] * len(usable), "right": [0.0] * len(usable)}
    for i in range(1, len(usable)):
        for side in ("left", "right"):
            a, b = usable[i - 1][f"{side}_pos"], usable[i][f"{side}_pos"]
            speeds[side][i] = math.hypot(b[0] - a[0], b[1] - a[1])

    feats = {}
    peak_index = {}
    for side in ("left", "right"):
        peak_i = max(range(len(usable)), key=lambda i: speeds[side][i])
        peak_index[side] = peak_i
        g = usable[peak_i]
        feats[f"{side}_peak_speed"] = speeds[side][peak_i]
        feats[f"{side}_side_at_peak"] = g[f"{side}_side"]
        feats[f"{side}_reach_at_peak"] = g[f"{side}_reach"]
        feats[f"{side}_height_at_peak"] = g[f"{side}_height"]
        feats[f"{side}_side_max"] = max(u[f"{side}_side"] for u in usable)
        feats[f"{side}_side_min"] = min(u[f"{side}_side"] for u in usable)
        feats[f"{side}_elbow_extension"] = g[f"{side}_elbow_extension"]

    # The faster wrist is the hitting one, and which it is matters: a backhand puts the
    # dominant hand across the body while a forehand keeps it on its own side.
    faster = "right" if feats["right_peak_speed"] >= feats["left_peak_speed"] else "left"
    slower = "left" if faster == "right" else "right"
    feats["faster_wrist_is_right"] = 1.0 if faster == "right" else 0.0
    feats["peak_speed_ratio"] = (feats[f"{slower}_peak_speed"]
                                 / max(feats[f"{faster}_peak_speed"], 1e-6))

    # Hand separation: both hands on the grip is a two-handed stroke, which in tennis is
    # overwhelmingly a backhand.
    gaps = [math.hypot(u["left_pos"][0] - u["right_pos"][0],
                       u["left_pos"][1] - u["right_pos"][1]) for u in usable]
    feats["wrist_gap_at_peak"] = gaps[peak_index[faster]]
    feats["wrist_gap_min"] = min(gaps)

    angles = [u["shoulder_angle"] for u in usable]
    feats["shoulder_angle_range"] = max(angles) - min(angles)
    feats["shoulder_angle_at_peak"] = angles[peak_index[faster]]

    return [round(float(feats[name]), 5) for name in FEATURE_NAMES]


def main():
    ap = argparse.ArgumentParser(description="Extract THETIS pose features")
    ap.add_argument("--per-class", type=int, default=165)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not THETIS.is_dir():
        print(f"THETIS not found at {THETIS}. Run scripts/download_thetis.py")
        sys.exit(1)

    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Run: tennis-vision download-models")
        sys.exit(1)

    samples = []
    print()
    for cls, label in CLASS_TRUTH.items():
        folder = THETIS / cls
        if not folder.is_dir():
            print(f"  [skip] {cls}: not downloaded")
            continue
        clips = sorted(folder.glob("*.avi"))[:args.per_class]
        kept = 0
        for clip in clips:
            feats = clip_features(clip, estimator)
            if feats is None:
                continue
            subject = SUBJECT_RE.match(clip.name)
            samples.append({
                "clip": clip.name,
                "source_class": cls,
                "label": label,
                "subject": subject.group(1) if subject else "unknown",
                "features": feats,
            })
            kept += 1
        rate = kept / len(clips) if clips else 0.0
        print(f"  {cls:<22} {kept:>4}/{len(clips):<4} usable ({rate:.0%})")

    estimator.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "samples": samples,
    }), encoding="utf-8")

    subjects = {s["subject"] for s in samples}
    forehands = sum(1 for s in samples if s["label"] == "forehand")
    print()
    print(f"Wrote {len(samples)} samples to {args.out}")
    print(f"  forehand {forehands}, backhand {len(samples) - forehands}")
    print(f"  {len(subjects)} subjects, {len(FEATURE_NAMES)} named features")

    if forehands and len(samples) - forehands:
        skew = forehands / len(samples)
        if abs(skew - 0.5) > 0.08:
            print()
            print(f"  NOTE: the usable set is {skew:.0%} forehand although the source is")
            print("  balanced. Pose extraction fails more often on backhands, because the")
            print("  body turns away from the camera and occludes the landmarks. Training")
            print("  must account for this or it inherits the same bias it is meant to fix.")


if __name__ == "__main__":
    main()
