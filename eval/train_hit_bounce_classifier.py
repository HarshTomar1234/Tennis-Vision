"""
eval/train_hit_bounce_classifier.py
─────────────────────────────────────
Trains and honestly evaluates a hit-vs-bounce classifier on the real 1,034-event feature
set from explore_hit_bounce_features.py.

Ladder-thinking, in order:
  1. Single strongest feature (|vx change|) with a plain threshold - 80.4% held-out.
     (height_y alone: 55.1% - barely above the class baseline, confirms the earlier
     coarse mean/std comparison was too optimistic about height on its own.)
  2. 3-feature logistic regression (height_y, vy_change_mag, vx_change_mag) - 84.1%
     held-out, train/test nearly identical (84.0%/84.1%, so not overfitting). The
     ~3.7-point gain over the single feature earns the extra (still small, interpretable)
     complexity. Stopping here - more features would need justifying against this bar too.

Split by CLIP, not by event, throughout: events from the same clip share camera,
lighting, and players, so an event-level split would leak information and overstate
accuracy.

Saves the trained logistic-regression weights to models/hit_bounce_classifier.json for
utils/hit_bounce_classifier.py to load - kept as data, not hardcoded, so retraining on
more data later doesn't need a code change.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

FEATURES_PATH = "datasets/external/hit_bounce_features.json"
WEIGHTS_PATH  = "models/hit_bounce_classifier.json"
FEATURE_NAMES = ["height_y", "vy_change_mag", "vx_change_mag"]


def clip_split(events: list[dict], test_frac: float = 0.25, seed: int = 42):
    clips = sorted(set(e["clip"] for e in events))
    rng = random.Random(seed)
    rng.shuffle(clips)
    n_test = max(1, int(len(clips) * test_frac))
    test_clips = set(clips[:n_test])
    train = [e for e in events if e["clip"] not in test_clips]
    test  = [e for e in events if e["clip"] in test_clips]
    return train, test, len(clips) - n_test, n_test


def best_threshold(train: list[dict], feature: str) -> float:
    values = sorted(set(e[feature] for e in train))
    best_t, best_acc = values[0], 0.0
    for t in values:
        correct = sum(
            1 for e in train
            if (e[feature] >= t and e["label"] == "hit")
            or (e[feature] < t and e["label"] == "bounce")
        )
        acc = correct / len(train)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


def evaluate_threshold(events: list[dict], feature: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for e in events:
        predicted_hit = e[feature] >= threshold
        actual_hit = e["label"] == "hit"
        if predicted_hit and actual_hit: tp += 1
        elif predicted_hit and not actual_hit: fp += 1
        elif not predicted_hit and actual_hit: fn += 1
        else: tn += 1
    n = len(events)
    return {"n": n, "accuracy": (tp + tn) / n}


def to_xy(subset: list[dict]):
    X = np.array([[e[f] for f in FEATURE_NAMES] for e in subset], dtype=float)
    y = np.array([1.0 if e["label"] == "hit" else 0.0 for e in subset])
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
    events = json.loads(Path(FEATURES_PATH).read_text())
    train, test, n_train_clips, n_test_clips = clip_split(events)
    print(f"Split: {n_train_clips} train clips ({len(train)} events), "
          f"{n_test_clips} test clips ({len(test)} events)\n")

    print("--- Step 1: single-feature thresholds ---")
    for feature in ("height_y", "vx_change_mag"):
        t = best_threshold(train, feature)
        tr_acc = evaluate_threshold(train, feature, t)["accuracy"]
        te_acc = evaluate_threshold(test, feature, t)["accuracy"]
        print(f"  {feature:<16} threshold={t:8.2f}  train={100*tr_acc:.1f}%  test={100*te_acc:.1f}%")

    print("\n--- Step 2: 3-feature logistic regression ---")
    Xtr, ytr = to_xy(train)
    Xte, yte = to_xy(test)
    mu, sigma = Xtr.mean(0), Xtr.std(0)
    Xtr_n, Xte_n = (Xtr - mu) / sigma, (Xte - mu) / sigma

    w, b = train_logreg(Xtr_n, ytr)
    tr_acc = logreg_accuracy(Xtr_n, ytr, w, b)
    te_acc = logreg_accuracy(Xte_n, yte, w, b)
    print(f"  features: {FEATURE_NAMES}")
    print(f"  weights (standardized): {dict(zip(FEATURE_NAMES, w.round(3)))}")
    print(f"  train={100*tr_acc:.1f}%  test={100*te_acc:.1f}%")

    Path("models").mkdir(exist_ok=True)
    weights = {
        "feature_names": FEATURE_NAMES,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "w": w.tolist(),
        "b": float(b),
        "held_out_test_accuracy": float(te_acc),
        "trained_on_events": len(train),
        "tested_on_events": len(test),
    }
    Path(WEIGHTS_PATH).write_text(json.dumps(weights, indent=2))
    print(f"\nSaved weights -> {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
