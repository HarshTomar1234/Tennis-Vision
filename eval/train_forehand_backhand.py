"""
eval/train_forehand_backhand.py
───────────────────────────────
Trains and honestly evaluates a forehand/backhand classifier on THETIS pose features.

The baseline it has to beat
---------------------------
The hand-crafted geometry in utils/pose_shot_classifier.py scores 54% on this task,
predicting forehand 89% of the time. On a balanced two-class problem that is barely above
chance, and it is not fixable by tuning: even given the correct hitting hand it tops out
at 78%, and on volleys it reaches 27% because a volley is played with the body square to
the net and the wrist never crosses the shoulder midline the rule depends on.

Anything below about 60% here is not worth shipping over the rule it replaces.

Why the split is by subject
---------------------------
THETIS has 55 subjects, each performing every stroke several times. A random split puts
the same persons other takes on both sides and measures memorisation of that person rather
than knowledge of the stroke. Grouping by subject is the difference between a number that
transfers and a number that flatters.

Repeated splits, not one
------------------------
A single split over 55 subjects is noisy enough to pick a winner by luck. Every
configuration is scored over many subject-grouped splits and reported with its spread, so
a difference smaller than the spread is correctly read as no difference.

Class balance
-------------
The source classes are balanced, but the USABLE set is not: pose extraction fails more
often on backhands, because the body turns away from the camera. Training on that as-is
would bake in the very bias this classifier exists to remove, so classes are reweighted to
equal influence and both balanced accuracy and per-class recall are reported. Overall
accuracy alone would let a forehand-always model look respectable.

Usage
-----
    python eval/train_forehand_backhand.py
    python eval/train_forehand_backhand.py --splits 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES_PATH = "datasets/external/thetis_pose_features.json"
WEIGHTS_PATH = "models/forehand_backhand_classifier.json"

# The geometric rule this must beat, measured by eval/forehand_backhand_on_thetis.py.
RULE_ACCURACY = 0.54


def load(path: str = FEATURES_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    names = data["feature_names"]
    samples = data["samples"]
    backend = data.get("backend", "mediapipe")
    X = np.array([s["features"] for s in samples], dtype=float)
    y = np.array([1.0 if s["label"] == "forehand" else 0.0 for s in samples])
    groups = np.array([s["subject"] for s in samples])
    classes = np.array([s["source_class"] for s in samples])
    return names, X, y, groups, classes, backend


def train_logreg(X, y, weights, l2=1e-3, lr=0.15, steps=4000):
    """
    Logistic regression with per-sample weights and L2, by plain gradient descent.

    Written out rather than pulled from scikit-learn because it is a dozen lines, the
    project already trains its hit/bounce model this way, and adding a dependency for one
    estimator is not worth the install cost for anyone reproducing these numbers.
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    wsum = weights.sum()
    for _ in range(steps):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) * weights
        w -= lr * ((X.T @ err) / wsum + l2 * w)
        b -= lr * (err.sum() / wsum)
    return w, b


def predict(X, w, b):
    return (1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))) >= 0.5


def balanced_scores(y_true, y_pred):
    """Per-class recall and their mean. Overall accuracy hides a one-class predictor."""
    out = {}
    for label, name in ((1.0, "forehand"), (0.0, "backhand")):
        mask = y_true == label
        out[name] = float((y_pred[mask] == label).mean()) if mask.any() else float("nan")
    out["balanced"] = float(np.nanmean([out["forehand"], out["backhand"]]))
    out["overall"] = float((y_pred == y_true).mean())
    return out


def run_splits(X, y, groups, n_splits, seed=0, feature_subset=None):
    Xs = X if feature_subset is None else X[:, feature_subset]
    subjects = np.unique(groups)
    rng = np.random.default_rng(seed)
    results = []

    for split in range(n_splits):
        rng.shuffle(subjects)
        cut = max(1, int(len(subjects) * 0.7))
        train_subjects = set(subjects[:cut])
        tr = np.array([g in train_subjects for g in groups])
        te = ~tr
        if tr.sum() < 20 or te.sum() < 10 or len(np.unique(y[te])) < 2:
            continue

        mu, sd = Xs[tr].mean(0), Xs[tr].std(0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (Xs[tr] - mu) / sd, (Xs[te] - mu) / sd

        # Equal influence per class, so the model cannot win by predicting the majority.
        weights = np.ones(tr.sum())
        for label in (0.0, 1.0):
            mask = y[tr] == label
            if mask.any():
                weights[mask] = len(weights) / (2.0 * mask.sum())

        w, b = train_logreg(Xtr, y[tr], weights)
        results.append(balanced_scores(y[te], predict(Xte, w, b)))

    return results


def summarise(name, results):
    if not results:
        print(f"{name:<28} no valid splits")
        return 0.0
    bal = np.array([r["balanced"] for r in results])
    fh = np.array([r["forehand"] for r in results])
    bh = np.array([r["backhand"] for r in results])
    print(f"{name:<28} {bal.mean():>7.1%} +/- {bal.std():<6.1%} "
          f"{fh.mean():>8.1%} {bh.mean():>10.1%}")
    return float(bal.mean())


def main():
    ap = argparse.ArgumentParser(description="Train forehand/backhand from pose features")
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--features", default=FEATURES_PATH,
                    help="feature file, so a pose backend can be swapped and compared")
    ap.add_argument("--no-save", action="store_true",
                    help="measure only; do not write weights")
    args = ap.parse_args()

    if not Path(args.features).exists():
        print(f"{args.features} not found. Run eval/extract_thetis_pose_features.py")
        sys.exit(1)
    names, X, y, groups, classes, backend = load(args.features)
    print(f"pose backend: {backend}")
    print(f"\n{len(y)} samples, {len(np.unique(groups))} subjects, {X.shape[1]} features")
    print(f"forehand {int(y.sum())}, backhand {int((1 - y).sum())}")

    print(f"\n{'configuration':<28} {'balanced':>15} {'forehand':>8} {'backhand':>10}")
    print("-" * 66)

    print(f"{'geometric rule (baseline)':<28} {RULE_ACCURACY:>7.1%} {'':<10} "
          f"{'~95%':>8} {'~15%':>10}")

    all_score = summarise("all features", run_splits(X, y, groups, args.splits))

    # Ablations, to show which evidence is doing the work rather than asserting it.
    groups_of = {
        "wrist side only": [i for i, n in enumerate(names) if "side" in n],
        "speed + reach only": [i for i, n in enumerate(names)
                               if "speed" in n or "reach" in n],
        "no two-handed cues": [i for i, n in enumerate(names) if "gap" not in n],
        "no shoulder rotation": [i for i, n in enumerate(names) if "shoulder" not in n],
    }
    for label, subset in groups_of.items():
        if subset:
            summarise(label, run_splits(X, y, groups, args.splits, feature_subset=subset))

    print("-" * 66)
    print("balanced = mean of per-class recall, so predicting one class always scores 50%")

    # Per source class, to expose whether volleys remain the weak point.
    print(f"\n{'source class':<24} {'n':>5} {'recall':>8}")
    print("-" * 40)
    results = run_splits(X, y, groups, args.splits)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    weights = np.ones(len(y))
    for label in (0.0, 1.0):
        mask = y == label
        weights[mask] = len(weights) / (2.0 * mask.sum())
    w, b = train_logreg((X - mu) / sd, y, weights)
    pred_all = predict((X - mu) / sd, w, b)
    for cls in sorted(set(classes)):
        mask = classes == cls
        rec = float((pred_all[mask] == y[mask]).mean())
        print(f"{cls:<24} {int(mask.sum()):>5} {rec:>7.0%}")
    print("-" * 40)
    print("(this last table is in-sample and only shows which strokes remain hard)")

    if all_score > RULE_ACCURACY + 0.05 and not args.no_save:
        Path("models").mkdir(exist_ok=True)
        Path(WEIGHTS_PATH).write_text(json.dumps({
            "feature_names": names,
            "mu": mu.tolist(), "sigma": sd.tolist(),
            "w": w.tolist(), "b": float(b),
            "balanced_accuracy": all_score,
            "n_samples": int(len(y)),
            "n_subjects": int(len(np.unique(groups))),
            "splits": args.splits,
            "beats_geometric_rule": True,
        }, indent=2), encoding="utf-8")
        print(f"\nSaved weights -> {WEIGHTS_PATH}")
        print(f"Balanced accuracy {all_score:.1%} against the rule's {RULE_ACCURACY:.0%}.")
    else:
        print(f"\nNOT saved: {all_score:.1%} does not clearly beat the geometric rule's "
              f"{RULE_ACCURACY:.0%}.")
        print("Shipping it would trade one unreliable classifier for another.")


if __name__ == "__main__":
    main()
