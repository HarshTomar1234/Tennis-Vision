"""
scripts/download_models.py
──────────────────────────
Fetches every model weight the pipeline needs into models/.

Run once after cloning:

    python scripts/download_models.py

Why the sources differ
----------------------
Only one of these weights is ours to redistribute. The court model is a derivative we
fine-tuned and published, so it comes from our Hugging Face repo. The pose model has a
stable official URL from Google. The ball tracker's weights belong to the upstream
TrackNet author and are fetched from that project's own release rather than rehosted
here - their licence is not explicitly stated and the dataset was released for research
reproduction, so redistributing them is not ours to do.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Direct-download weights: (filename, url, approx MB, what it does)
DIRECT_DOWNLOADS = [
    (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        6,
        "MediaPipe pose - forehand/backhand from body geometry",
    ),
]

# Hugging Face weights: (repo_id, filename, approx MB, what it does)
HF_DOWNLOADS = [
    (
        "Coddieharsh/tennis-court-keypoints",
        "keypoints_model_geoaug.pth",
        95,
        "Court keypoints - every real-world measurement depends on this",
    ),
]

# Weights we deliberately do not redistribute, with instructions instead.
MANUAL = [
    (
        "tracknet.pt",
        43,
        "Ball detection (TrackNet)",
        "gdown 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O models/tracknet.pt\n"
        "      (source: https://github.com/yastrebksv/TrackNet - research use only)",
    ),
]


def download_direct(filename: str, url: str, size_mb: int, purpose: str) -> bool:
    target = MODELS_DIR / filename
    if target.exists():
        print(f"  [skip]     {filename} already present")
        return True
    print(f"  [download] {filename} (~{size_mb} MB) - {purpose}")
    try:
        urllib.request.urlretrieve(url, target)
        return True
    except Exception as exc:  # noqa: BLE001 - report and continue with the others
        print(f"  [FAILED]   {filename}: {exc}")
        target.unlink(missing_ok=True)
        return False


def download_hf(repo_id: str, filename: str, size_mb: int, purpose: str) -> bool:
    target = MODELS_DIR / filename
    if target.exists():
        print(f"  [skip]     {filename} already present")
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(f"  [FAILED]   {filename}: pip install huggingface_hub")
        return False

    print(f"  [download] {filename} (~{size_mb} MB) - {purpose}")
    try:
        cached = hf_hub_download(repo_id=repo_id, filename=filename)
        target.write_bytes(Path(cached).read_bytes())
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAILED]   {filename}: {exc}")
        return False


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading models into {MODELS_DIR}\n")

    ok = True
    for repo_id, filename, size_mb, purpose in HF_DOWNLOADS:
        ok &= download_hf(repo_id, filename, size_mb, purpose)
    for filename, url, size_mb, purpose in DIRECT_DOWNLOADS:
        ok &= download_direct(filename, url, size_mb, purpose)

    missing_manual = [m for m in MANUAL if not (MODELS_DIR / m[0]).exists()]
    if missing_manual:
        print("\n  Fetch these yourself - they are not ours to redistribute:\n")
        for filename, size_mb, purpose, instructions in missing_manual:
            print(f"    {filename} (~{size_mb} MB) - {purpose}")
            print(f"      {instructions}\n")

    # yolov8x downloads itself through ultralytics on first use, so it is not listed.
    print("  yolov8s/yolov8x (player detection) download automatically on first run.\n")

    if not ok or missing_manual:
        print("Some weights are still missing - the pipeline will fail until they are "
              "present. See README 'Models'.\n")
        return 1
    print("All models present.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
