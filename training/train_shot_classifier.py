"""
training/train_shot_classifier.py
─────────────────────────────────
Trains ONE temporal model over pose sequences to classify all shot types, replacing
the pile of hand-written geometric rules in ShotClassifier.

Why a learned temporal model rather than more rules
---------------------------------------------------
Shot types differ by *body mechanics unfolding over time*, not by any single-frame
geometric fact. A slice and a topspin forehand can occupy identical positions at the
contact frame and differ entirely in the swing path before and after it. Rules that
look at one frame - which is what the current classifier does - cannot see that, and
in practice they produced 6 backhands out of 8 shots on one clip and phantom serves
on most others. One model over a window of frames replaces N rules with N classes.

Architecture: a small 1D CNN over the time axis. Deliberately small - 243 clips is a
tiny dataset and a transformer would memorise it. Two conv blocks plus global pooling
has enough capacity for swing-shape discrimination and little enough to be trainable
here.

Honesty constraints baked into this script
------------------------------------------
1. **Subject-wise splits, never random.** THETIS has 16 subjects, each performing
   every shot type several times. A random split puts the same person's forehand in
   both train and test, and the model then recognises the *person*, not the *shot*.
   That inflates accuracy dramatically and is the single easiest way to produce a
   fake result here. Folds are grouped by subject.
2. **Body-relative normalisation.** Coordinates arrive in pixels. Each frame is
   centred on the hip midpoint and scaled by shoulder width, so the model cannot
   cheat off where the player stands or how large they appear.
3. THETIS is indoor gym demonstration footage, not broadcast tennis. Whatever
   accuracy this reports is accuracy *on THETIS*. It is not evidence about our own
   clips, and must not be quoted as such.

Usage:
    python training/train_shot_classifier.py
    python training/train_shot_classifier.py --epochs 60 --folds 4
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parent.parent))

FEATURES_PATH = Path("datasets/external/thetis_slice_features.json")
OUT_WEIGHTS = Path("models/shot_classifier_temporal.pt")
OUT_META = Path("models/shot_classifier_temporal.json")

LANDMARKS = (
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP",
)
SEQ_LEN = 32                      # frames every clip is resampled to
N_FEATURES = len(LANDMARKS) * 3   # x, y, z per landmark


def normalise_frame(frame: dict) -> np.ndarray | None:
    """
    One frame of pose -> a body-relative feature vector, or None if unusable.

    Centred on the hip midpoint and scaled by shoulder width so the result is
    invariant to the player's position in frame and their apparent size. Without
    this the model would learn court position, which is exactly the shortcut that
    makes a classifier look good on THETIS and fail on broadcast footage.
    """
    # Frames where pose detection failed arrive as None, and partial detections are
    # missing individual landmarks. Both are dropped rather than interpolated: a
    # guessed wrist produces a guessed swing shape.
    if not frame or any(name not in frame for name in LANDMARKS):
        return None

    points = {name: np.asarray(frame[name], dtype=np.float32) for name in LANDMARKS}
    hip_centre = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2.0
    shoulder_width = float(np.linalg.norm(
        points["LEFT_SHOULDER"][:2] - points["RIGHT_SHOULDER"][:2]
    ))
    # A near-zero shoulder width means the player is edge-on and the pose has
    # collapsed; scaling by it would explode the features.
    if shoulder_width < 1e-3:
        return None

    return np.concatenate([
        (points[name] - hip_centre) / shoulder_width for name in LANDMARKS
    ])


def resample(sequence: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Linearly resample a (T, F) sequence to (length, F) along time."""
    if len(sequence) == length:
        return sequence
    source = np.linspace(0.0, 1.0, len(sequence))
    target = np.linspace(0.0, 1.0, length)
    return np.stack([np.interp(target, source, sequence[:, f])
                     for f in range(sequence.shape[1])], axis=1)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns X (N, SEQ_LEN, F), y (N,), subjects (N,), class names."""
    raw = json.load(open(FEATURES_PATH, encoding="utf-8"))
    classes = sorted({item["category"] for item in raw})
    class_index = {name: i for i, name in enumerate(classes)}

    X, y, subjects, dropped = [], [], [], 0
    for item in raw:
        frames = [normalise_frame(f) for f in item["sequence"]]
        frames = [f for f in frames if f is not None]
        # Too few usable frames to describe a swing; dropping beats padding noise.
        if len(frames) < 8:
            dropped += 1
            continue
        X.append(resample(np.stack(frames)))
        y.append(class_index[item["category"]])
        subjects.append(item["subject"])

    print(f"  loaded {len(X)} clips ({dropped} dropped for insufficient pose)")
    print(f"  classes: {dict(Counter(classes[i] for i in y))}")
    return np.stack(X), np.array(y), np.array(subjects), classes


class ShotNet(nn.Module):
    """Small 1D CNN over the time axis; sized for a few hundred training clips."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):                 # x: (B, T, F)
        return self.head(self.net(x.transpose(1, 2)).squeeze(-1))


def run_fold(X_tr, y_tr, X_te, y_te, n_classes, epochs, device):
    model = ShotNet(N_FEATURES, n_classes).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        for start in range(0, len(perm), 32):
            batch = perm[start:start + 32]
            optimiser.zero_grad()
            loss = criterion(model(X_tr_t[batch]), y_tr_t[batch])
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_te_t).argmax(1).cpu().numpy()
    return model, predictions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    X, y, subjects, classes = load_dataset()

    unique_subjects = sorted(set(subjects))
    print(f"  {len(unique_subjects)} subjects -> {args.folds}-fold GROUPED cross-validation")
    print("  (grouped by subject: the same person never appears in train and test)\n")

    rng = np.random.default_rng(0)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)
    folds = np.array_split(np.array(shuffled), args.folds)

    all_true, all_pred = [], []
    for i, held_out in enumerate(folds):
        test_mask = np.isin(subjects, held_out)
        model, predictions = run_fold(
            X[~test_mask], y[~test_mask], X[test_mask], y[test_mask],
            len(classes), args.epochs, device,
        )
        accuracy = float((predictions == y[test_mask]).mean())
        all_true.extend(y[test_mask].tolist())
        all_pred.extend(predictions.tolist())
        print(f"  fold {i + 1}: held-out subjects {list(held_out)} | "
              f"n={test_mask.sum():3d} | accuracy {accuracy:.3f}")

    all_true, all_pred = np.array(all_true), np.array(all_pred)
    overall = float((all_true == all_pred).mean())
    baseline = float(Counter(all_true.tolist()).most_common(1)[0][1] / len(all_true))

    print(f"\n  POOLED ACCURACY (unseen subjects): {overall:.3f}")
    print(f"  majority-class baseline           : {baseline:.3f}")
    print(f"  random baseline ({len(classes)} classes)         : {1 / len(classes):.3f}")

    print("\n  per class (recall on unseen subjects):")
    for i, name in enumerate(classes):
        mask = all_true == i
        if mask.sum():
            print(f"    {name:22s} {(all_pred[mask] == i).mean():.3f}  (n={mask.sum()})")

    print("\n  confusion matrix (rows = truth):")
    print("    " + " ".join(f"{n[:8]:>9s}" for n in classes))
    for i, name in enumerate(classes):
        row = [int(((all_true == i) & (all_pred == j)).sum()) for j in range(len(classes))]
        print(f"    {name[:20]:20s} " + " ".join(f"{v:9d}" for v in row))

    # Final model on all data, for use in the pipeline.
    model, _ = run_fold(X, y, X[:1], y[:1], len(classes), args.epochs, device)
    torch.save(model.state_dict(), OUT_WEIGHTS)
    OUT_META.write_text(json.dumps({
        "classes": classes,
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "landmarks": list(LANDMARKS),
        "grouped_cv_accuracy": round(overall, 4),
        "folds": args.folds,
        "n_clips": int(len(X)),
        "n_subjects": len(unique_subjects),
        "trained_on": "THETIS (indoor demonstration footage, NOT broadcast tennis)",
        "caveat": ("Accuracy is measured on held-out THETIS subjects. It is not "
                   "evidence of accuracy on broadcast footage and must not be "
                   "quoted as such."),
    }, indent=2), encoding="utf-8")
    print(f"\n  saved {OUT_WEIGHTS} and {OUT_META}\n")


if __name__ == "__main__":
    main()
