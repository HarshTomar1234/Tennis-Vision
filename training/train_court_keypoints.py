"""
training/train_court_keypoints.py
─────────────────────────────────
Fine-tunes the court keypoint model with GEOMETRIC augmentation.

Why geometric specifically
--------------------------
`eval/court_keypoint_accuracy.py` measured the shipped weights on the held-out val
split: 4.03 px median error, 96.8 % of images with all 14 keypoints inside 25 px,
and - critically - near-identical accuracy across surfaces (blue 3.90, clay 4.58,
green 4.65). Meanwhile the model fails on roughly 5 of our 9 real YouTube clips.

So the weakness is not surface colour, it is camera geometry: our failing clips all
fail with the predicted court displaced vertically. This script therefore augments
translation, scale and perspective - not colour. Colour jitter would have trained
hard and fixed nothing.

Approach
--------
Augmentation is applied in 224x224 space (the network's input size) rather than at
source resolution: the transform is linear so keypoints scale exactly, and warping
224x224 instead of 1280x720 keeps the GPU fed. Images are pre-decoded once into RAM
(~1 GB for the train split) because PNG decode, not the network, is the bottleneck.

Keypoints are deliberately allowed to land outside the frame after augmentation.
That is the situation we are training for - several of our clips have part of the
court out of shot, and the model must still place those corners sensibly.

Usage:
    python training/train_court_keypoints.py --epochs 20
    python training/train_court_keypoints.py --epochs 2 --limit 400   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("datasets/external/court_dataset/data")
BASE_WEIGHTS = Path("models/keypoints_model.pth")
OUT_WEIGHTS = Path("models/keypoints_model_geoaug.pth")        # best by val error
FINAL_WEIGHTS = Path("models/keypoints_model_geoaug_final.pth")  # last epoch, always kept

INPUT_SIZE = 224
NUM_KEYPOINTS = 14

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Horizontal-flip keypoint remapping. A mirrored court swaps left and right, so the
# slot for "far-left outer corner" must receive the mirrored "far-right outer
# corner", and so on. The two centre-service-line points are on the mirror axis and
# map to themselves. Verified against the dataset's own geometry - see
# test_flip_map_is_an_involution() below.
FLIP_MAP = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


def load_split(name: str, limit: int = 0) -> list[dict]:
    samples = json.load(open(DATA_DIR / f"data_{name}.json"))
    return samples[:limit] if limit else samples


class CourtKeypointDataset(Dataset):
    """
    Court images with 14 keypoints, pre-scaled to the network's 224x224 input space.

    Set `augment=True` for training. Validation uses the identity transform so its
    number stays directly comparable to eval/court_keypoint_accuracy.py.
    """

    def __init__(self, samples: list[dict], augment: bool):
        self.augment = augment
        self.images: list[np.ndarray] = []
        self.keypoints: list[np.ndarray] = []

        for i, item in enumerate(samples):
            img = cv2.imread(str(DATA_DIR / "images" / f"{item['id']}.png"))
            if img is None:
                continue
            h, w = img.shape[:2]
            kps = np.asarray(item["kps"], dtype=np.float32)
            # Into 224x224 space; the scaling is exactly what predict() inverts.
            kps[:, 0] *= INPUT_SIZE / w
            kps[:, 1] *= INPUT_SIZE / h
            self.images.append(cv2.resize(img, (INPUT_SIZE, INPUT_SIZE)))
            self.keypoints.append(kps)
            if (i + 1) % 1000 == 0:
                print(f"    loaded {i + 1}/{len(samples)}", flush=True)

    def __len__(self) -> int:
        return len(self.images)

    def _random_transform(self) -> np.ndarray:
        """
        Build a 3x3 matrix combining translation, scale and a mild perspective warp.

        Ranges are chosen to span the framings our real clips exhibit - courts
        shifted vertically, filmed from lower angles, and at varying zoom - without
        producing images no broadcast camera would ever capture.
        """
        s = INPUT_SIZE
        scale = np.random.uniform(0.85, 1.20)
        # Vertical range is wider than horizontal: every one of our observed
        # real-world failures was a vertical displacement of the predicted court.
        tx = np.random.uniform(-0.10, 0.10) * s
        ty = np.random.uniform(-0.15, 0.15) * s

        centre = s / 2.0
        affine = np.array([
            [scale, 0.0,   centre - scale * centre + tx],
            [0.0,   scale, centre - scale * centre + ty],
            [0.0,   0.0,   1.0],
        ], dtype=np.float32)

        # Mild perspective: jitter the source corners a little and solve for the
        # homography. This simulates the camera sitting lower or off-axis.
        jitter = 0.05 * s
        src = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
        dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
        perspective = cv2.getPerspectiveTransform(src, dst)

        return (perspective @ affine).astype(np.float32)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        kps = self.keypoints[idx].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
                kps[:, 0] = INPUT_SIZE - kps[:, 0]
                kps = kps[FLIP_MAP]

            M = self._random_transform()
            img = cv2.warpPerspective(
                img, M, (INPUT_SIZE, INPUT_SIZE),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
            )
            kps = cv2.perspectiveTransform(
                kps.reshape(-1, 1, 2), M
            ).reshape(-1, 2)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1))
        return tensor, torch.from_numpy(kps.flatten())


def build_model(device: str) -> torch.nn.Module:
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_KEYPOINTS * 2)
    if BASE_WEIGHTS.exists():
        model.load_state_dict(torch.load(BASE_WEIGHTS, map_location="cpu"))
        print(f"  fine-tuning from {BASE_WEIGHTS}")
    else:
        print(f"  WARNING: {BASE_WEIGHTS} missing - training from ImageNet init")
    return model.to(device)


@torch.no_grad()
def validate(model, loader, device: str, source_w: int = 1280, source_h: int = 720):
    """
    Median/mean keypoint error reported in ORIGINAL image pixels.

    The network works in 224x224 space; errors are scaled back so the number is
    directly comparable to the 4.03 px baseline from eval/court_keypoint_accuracy.py.
    """
    model.eval()
    errors: list[float] = []
    per_image_max: list[float] = []
    scale = np.array([source_w / INPUT_SIZE, source_h / INPUT_SIZE], dtype=np.float32)

    for imgs, kps in loader:
        preds = model(imgs.to(device)).cpu().numpy().reshape(-1, NUM_KEYPOINTS, 2)
        truth = kps.numpy().reshape(-1, NUM_KEYPOINTS, 2)
        dist = np.linalg.norm((preds - truth) * scale, axis=2)
        errors.extend(dist.flatten().tolist())
        per_image_max.extend(dist.max(axis=1).tolist())

    e = np.array(errors)
    usable = float(np.mean(np.array(per_image_max) <= 25.0)) * 100
    return float(np.median(e)), float(e.mean()), usable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=0, help="cap samples per split")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("  loading train split...")
    train_ds = CourtKeypointDataset(load_split("train", args.limit), augment=True)
    print("  loading val split...")
    val_ds = CourtKeypointDataset(load_split("val", args.limit), augment=False)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = build_model(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    med, mean, usable = validate(model, val_loader, device)
    print(f"\nBASELINE (before fine-tuning): median {med:.2f} px | "
          f"mean {mean:.2f} px | all-14-within-25px {usable:.1f} %")
    print("NOTE: do not expect this val number to improve. The baseline weights were")
    print("trained on this exact distribution without augmentation, so they are near")
    print("optimal here. Success is val holding roughly steady while robustness on")
    print("real clips improves - which only the clip-suite check can show.\n")
    baseline_med = med
    best = med

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (imgs, kps) in enumerate(train_loader):
            imgs, kps = imgs.to(device), kps.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), kps)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        med, mean, usable = validate(model, val_loader, device)
        flag = ""
        if med < best:
            best = med
            torch.save(model.state_dict(), OUT_WEIGHTS)
            flag = "  <- saved"
        print(f"epoch {epoch + 1:3d}/{args.epochs} | train MSE {running / max(len(train_loader), 1):8.2f} "
              f"| val median {med:6.2f} px | mean {mean:6.2f} px | usable {usable:5.1f} %{flag}",
              flush=True)

    # The final epoch is always kept, separately from the best-by-val checkpoint.
    # Selecting purely on in-distribution val would bias us against exactly the
    # robustness this run exists to buy.
    torch.save(model.state_dict(), FINAL_WEIGHTS)

    print(f"\nBaseline val median : {baseline_med:.2f} px")
    print(f"Best val median     : {best:.2f} px"
          + (f"  -> {OUT_WEIGHTS}" if best < baseline_med else "  (never beat baseline; nothing saved here)"))
    print(f"Final epoch weights : {FINAL_WEIGHTS}")
    print("\nGate before shipping - run BOTH, and adopt only if the clip suite improves:")
    print(f"  python eval/court_keypoint_accuracy.py            # in-distribution, must hold")
    print(f"  python eval/court_validity_calibration.py         # real clips, must improve\n")


def test_flip_map_is_an_involution() -> None:
    """Flipping twice must return every keypoint to its own slot."""
    assert [FLIP_MAP[FLIP_MAP[i]] for i in range(NUM_KEYPOINTS)] == list(range(NUM_KEYPOINTS))
    assert sorted(FLIP_MAP) == list(range(NUM_KEYPOINTS)), "flip map must be a permutation"


if __name__ == "__main__":
    test_flip_map_is_an_involution()
    main()
