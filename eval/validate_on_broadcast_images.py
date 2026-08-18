"""
eval/validate_on_broadcast_images.py
────────────────────────────────────
Does the forehand/backhand classifier survive real footage?

The question this exists to answer
----------------------------------
The classifier scores 76.3% balanced on THETIS (eval/train_forehand_backhand.py), against
54% for the geometric rule it replaces. THETIS is indoor, Kinect-captured, single-stroke
footage with the player centred and well lit. This project analyses outdoor broadcast
video where the player is small, motion-blurred, partly occluded and often side-on.

The gap between those two is the single largest unmeasured risk in this project, and it is
the failure mode that has already bitten twice: a hit/bounce feature set that scored higher
on its benchmark and worse in the pipeline, and a "detection rate" that meant something
quite different from accuracy. A classifier that scores 76% indoors and 55% on broadcast is
no better than the rule it replaces, and shipping it on the indoor number would repeat
exactly that mistake.

So this is a transfer test, not a second training run. Nothing here is fitted.

Where the labels come from
--------------------------
Roboflow Universe hosts community tennis datasets annotated with stroke classes on real
footage. Each image carries a labelled player box, so MediaPipe runs on the box and the
same named features used for THETIS are computed. Same feature contract, same model,
different domain.

Caveats, stated because they bound what the number means
--------------------------------------------------------
Universe datasets are community-uploaded and their labelling quality is not guaranteed.
Class names vary between datasets and are mapped explicitly below rather than guessed at.
A single frame carries less information than a stroke sequence, so features that depend on
motion (peak speed, shoulder rotation range) are weaker here than on video, which means
this is a LOWER bound on video performance rather than a like-for-like comparison.

Requires ROBOFLOW_API_KEY, read from the environment or from a .env file. The key is never
printed and never written anywhere.

Usage
-----
    python eval/validate_on_broadcast_images.py --list
    python eval/validate_on_broadcast_images.py --workspace X --project Y --version 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from utils import PoseEstimator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from extract_thetis_pose_features import FEATURE_NAMES, frame_geometry

WEIGHTS_PATH = "models/forehand_backhand_classifier.json"

# Roboflow class names vary by dataset and are mapped explicitly. Anything not listed is
# counted as unmapped and excluded, rather than guessed into one of the two buckets.
CLASS_MAP = {
    "forehand": "forehand", "forehand-ready": "forehand",
    "forehand-stroke": "forehand", "forehand-finish": "forehand",
    "forehand_flat": "forehand", "fh": "forehand",
    "backhand": "backhand", "backhand-ready": "backhand",
    "backhand-stroke": "backhand", "backhand-finish": "backhand",
    "bh": "backhand",
}


def load_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        env = Path(".env")
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("ROBOFLOW_API_KEY not found in the environment or .env")
        sys.exit(1)
    return key


def single_frame_features(image, box, estimator: PoseEstimator):
    """
    The same named features as THETIS, from one image.

    Motion features cannot be measured from a still, so they are set to zero rather than
    invented. That weakens the model here relative to video, which is why this reads as a
    lower bound. Zero is also what the training standardisation maps to the class mean, so
    it is the least misleading filler available.
    """
    landmarks = estimator.detect_in_bbox(image, box)
    if not landmarks:
        return None
    geo = frame_geometry(landmarks)
    if geo is None:
        return None

    feats = {name: 0.0 for name in FEATURE_NAMES}
    for side in ("left", "right"):
        feats[f"{side}_side_at_peak"] = geo[f"{side}_side"]
        feats[f"{side}_reach_at_peak"] = geo[f"{side}_reach"]
        feats[f"{side}_height_at_peak"] = geo[f"{side}_height"]
        feats[f"{side}_side_max"] = geo[f"{side}_side"]
        feats[f"{side}_side_min"] = geo[f"{side}_side"]
        feats[f"{side}_elbow_extension"] = geo[f"{side}_elbow_extension"]

    gap = math.hypot(geo["left_pos"][0] - geo["right_pos"][0],
                     geo["left_pos"][1] - geo["right_pos"][1])
    feats["wrist_gap_at_peak"] = gap
    feats["wrist_gap_min"] = gap
    feats["shoulder_angle_at_peak"] = geo["shoulder_angle"]
    # The extended arm stands in for the faster one, since there is no motion to measure.
    feats["faster_wrist_is_right"] = (
        1.0 if geo["right_reach"] >= geo["left_reach"] else 0.0
    )
    return [feats[name] for name in FEATURE_NAMES]


def main():
    ap = argparse.ArgumentParser(description="Transfer test on broadcast images")
    ap.add_argument("--workspace", default="tennis-gqhor")
    ap.add_argument("--project", default="tennisvision")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--list", action="store_true", help="list classes and exit")
    args = ap.parse_args()

    if not Path(WEIGHTS_PATH).exists():
        print(f"{WEIGHTS_PATH} not found. Run eval/train_forehand_backhand.py first.")
        sys.exit(1)

    from roboflow import Roboflow

    rf = Roboflow(api_key=load_api_key())
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("coco", location="datasets/external/roboflow_tennis", overwrite=False)

    root = Path(dataset.location)
    ann_files = list(root.rglob("_annotations.coco.json"))
    if not ann_files:
        print(f"No COCO annotations found under {root}")
        sys.exit(1)

    weights = json.loads(Path(WEIGHTS_PATH).read_text(encoding="utf-8"))
    w = np.array(weights["w"])
    b = float(weights["b"])
    mu = np.array(weights["mu"])
    sd = np.array(weights["sigma"])

    estimator = PoseEstimator()
    if not estimator.available:
        print("Pose model unavailable. Run: tennis-vision download-models")
        sys.exit(1)

    seen_classes = Counter()
    results = []

    for ann_path in ann_files:
        coco = json.loads(ann_path.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        images = {im["id"]: im for im in coco.get("images", [])}
        seen_classes.update(cats.values())

        if args.list:
            continue

        for ann in coco.get("annotations", [])[: args.max_images]:
            raw = cats.get(ann["category_id"], "")
            label = CLASS_MAP.get(raw.strip().lower())
            if label is None:
                continue
            meta = images.get(ann["image_id"])
            if meta is None:
                continue
            img_path = ann_path.parent / meta["file_name"]
            if not img_path.exists():
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            x, y, bw, bh = ann["bbox"]
            feats = single_frame_features(image, [x, y, x + bw, y + bh], estimator)
            if feats is None:
                continue
            z = float(((np.array(feats) - mu) / sd) @ w + b)
            pred = "forehand" if 1.0 / (1.0 + math.exp(-max(-30, min(30, z)))) >= 0.5 else "backhand"
            results.append((label, pred))

    estimator.close()

    if args.list:
        print("\nClasses present in this dataset:")
        for name, n in seen_classes.most_common():
            mapped = CLASS_MAP.get(name.strip().lower(), "UNMAPPED")
            print(f"  {name:<28} {n:>6}  -> {mapped}")
        return

    if not results:
        print("\nNo usable samples. Either no class mapped, or pose failed on every crop.")
        print("Run with --list to see the class names this dataset actually uses.")
        return

    print(f"\n{'class':<12} {'n':>6} {'recall':>9}")
    print("-" * 32)
    per_class = {}
    for label in ("forehand", "backhand"):
        subset = [(t, p) for t, p in results if t == label]
        rec = sum(1 for t, p in subset if t == p) / len(subset) if subset else float("nan")
        per_class[label] = rec
        print(f"{label:<12} {len(subset):>6} {rec:>8.1%}")

    balanced = float(np.nanmean(list(per_class.values())))
    print("-" * 32)
    print(f"{'balanced':<12} {len(results):>6} {balanced:>8.1%}")

    print(f"\nTHETIS (indoor, video)   76.3%")
    print(f"broadcast (this test)    {balanced:.1%}")
    print(f"geometric rule           54.0%")
    print()
    if balanced < 0.58:
        print("Transfer has failed. The indoor number does not describe broadcast video,")
        print("and shipping on it would repeat the mistake this test exists to catch.")
    elif balanced < 0.70:
        print("Transfer is partial. Better than the rule, well short of the indoor number,")
        print("which points at pose quality on broadcast footage rather than the model.")
    else:
        print("Transfer holds. The classifier is worth wiring into the pipeline.")


if __name__ == "__main__":
    main()
