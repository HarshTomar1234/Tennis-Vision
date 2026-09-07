"""
eval/train_slice_classifier.py
─────────────────────────────────
Trains and honestly evaluates a slice-vs-flat/topspin classifier on the THETIS-derived
feature set from explore_slice_features.py. Same ladder-thinking as
train_hit_bounce_classifier.py, but with the opposite result:

  1. Single strongest feature (last_third_slope, Cohen's d=1.468 in exploration) with a
     plain threshold -- 83.3% held-out.
  2. 3-feature logistic regression (last_third_slope, overall_slope, y_range) -- 77.8%
     held-out, WORSE than the single feature. With only 12 train / 4 test subjects, the
     extra two features add overfitting risk rather than real signal -- the opposite of
     what happened with hit/bounce (where 3 features beat 1). The single-feature
     threshold is what actually gets saved and used; the logistic regression is kept in
     this script only as the honest comparison that justifies not using it.

Split by SUBJECT, not by clip: THETIS has 2-3 repetitions per subject per action, and
clips from the same subject share body proportions and swing habits, so a clip-level
split would leak information the same way an event-level split did for hit/bounce.

Domain caveat: THETIS is isolated indoor-gym demonstration footage,
not broadcast match play -- this measures whether the feature/model works on THETIS's
own held-out subjects, not whether it transfers to our own footage. That transfer is
unverified and should be tested before this is wired into main.py.

Saves weights to models/slice_classifier.json for a future
utils/slice_classifier.py to load, mirroring the hit_bounce_classifier convention.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

FEATURES_PATH = "datasets/external/thetis_slice_summary_features.json"
WEIGHTS_PATH  = "models/slice_classifier.json"
FEATURE_NAMES = ["last_third_slope", "overall_slope", "y_range"]


def subject_split(rows: list[dict], test_frac: float = 0.25, seed: int = 42):
    subjects = sorted(set(r["subject"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_frac))
    test_subjects = set(subjects[:n_test])
    train = [r for r in rows if r["subject"] not in test_subjects]
    test  = [r for r in rows if r["subject"] in test_subjects]
    return train, test, len(subjects) - n_test, n_test


def best_threshold(train: list[dict], feature: str) -> float:
    values = sorted(set(r[feature] for r in train))
    best_t, best_acc = values[0], 0.0
    for t in values:
        correct = sum(
            1 for r in train
            if (r[feature] <= t and r["is_slice"])
            or (r[feature] > t and not r["is_slice"])
        )
        acc = correct / len(train)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def evaluate_threshold(rows: list[dict], feature: str, threshold: float) -> float:
    correct = sum(
        1 for r in rows
        if (r[feature] <= threshold) == r["is_slice"]
    )
    return correct / len(rows)


def to_xy(subset: list[dict]):
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in subset], dtype=float)
    y = np.array([1.0 if r["is_slice"] else 0.0 for r in subset])
    return X, y


def train_logreg(Xtr_n: np.ndarray, ytr: np.ndarray, lr: float = 0.1, steps: int = 2000):
    w = np.zeros(Xtr_n.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Xtr_n @ w + b)))
        w -= lr * (Xtr_n.T @ (p - ytr) / len(ytr))
        b -= lr * (p - ytr).mean()
    return w, b


def logreg_accuracy(X_n: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    p = 1 / (1 + np.exp(-(X_n @ w + b)))
    return ((p >= 0.5).astype(float) == y).mean()


def main():
    rows = json.loads(Path(FEATURES_PATH).read_text())
    train, test, n_train_subj, n_test_subj = subject_split(rows)
    n_slice_train = sum(1 for r in train if r["is_slice"])
    n_slice_test  = sum(1 for r in test if r["is_slice"])
    print(f"Split: {n_train_subj} train subjects ({len(train)} clips, {n_slice_train} slice), "
          f"{n_test_subj} test subjects ({len(test)} clips, {n_slice_test} slice)\n")

    print("--- Step 1: single-feature threshold (last_third_slope) ---")
    t = best_threshold(train, "last_third_slope")
    tr_acc = evaluate_threshold(train, "last_third_slope", t)
    te_acc = evaluate_threshold(test, "last_third_slope", t)
    print(f"  threshold={t:.4f}  train={100*tr_acc:.1f}%  test={100*te_acc:.1f}%")

    print("\n--- Step 2: 3-feature logistic regression ---")
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    mu, sigma = Xtr.mean(0), Xtr.std(0)
    Xtr_n, Xte_n = (Xtr - mu) / sigma, (Xte - mu) / sigma

    w, b = train_logreg(Xtr_n, ytr)
    tr_acc_lr = logreg_accuracy(Xtr_n, ytr, w, b)
    te_acc_lr = logreg_accuracy(Xte_n, yte, w, b)
    print(f"  features: {FEATURE_NAMES}")
    print(f"  weights (standardized): {dict(zip(FEATURE_NAMES, w.round(3)))}")
    print(f"  train={100*tr_acc_lr:.1f}%  test={100*te_acc_lr:.1f}%")

    print(f"\n--- Decision ---")
    if te_acc >= te_acc_lr:
        print(f"  single-feature threshold wins ({100*te_acc:.1f}% vs {100*te_acc_lr:.1f}%) -- saving that")
        model = {
            "model_type": "single_threshold",
            "feature_name": "last_third_slope",
            "threshold": t,
            "predict_slice_if": "<=",
            "held_out_test_accuracy": float(te_acc),
        }
    else:
        print(f"  logistic regression wins ({100*te_acc_lr:.1f}% vs {100*te_acc:.1f}%) -- saving that")
        model = {
            "model_type": "logistic_regression",
            "feature_names": FEATURE_NAMES,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "w": w.tolist(),
            "b": float(b),
            "held_out_test_accuracy": float(te_acc_lr),
        }

    Path("models").mkdir(parents=True, exist_ok=True)
    model.update({
        "trained_on_clips": len(train),
        "tested_on_clips": len(test),
        "domain": "THETIS indoor-gym demonstration footage -- NOT validated on broadcast match play",
    })
    Path(WEIGHTS_PATH).write_text(json.dumps(model, indent=2))
    print(f"\nSaved weights -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
