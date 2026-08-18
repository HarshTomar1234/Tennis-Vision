"""
scripts/download_thetis.py
──────────────────────────
Fetches the THETIS tennis shot dataset (RGB videos only) into datasets/external/thetis.

What THETIS is
--------------
8,374 sequences from 55 subjects, 31 beginners and 24 experts, captured with a Kinect.
Twelve shot classes, one shot per clip:

    backhand              forehand_flat         flat_service
    backhand2hands        forehand_openstands   kick_service
    backhand_slice        forehand_slice        slice_service
    backhand_volley       forehand_volley       smash

It is the only freely available dataset this project knows of that labels volley and
smash, which the rule-based classifiers in utils/shot_physics.py have never been
validated against.

Why only RGB
------------
The full repository is about 13 GB across five modalities (RGB, depth, mask, and 2-D/3-D
skeleton). Only the RGB videos are useful here, because the pipeline consumes ordinary
video and derives pose itself. A blobless sparse checkout fetches just the requested
class directories rather than the whole thing.

The important caveat
--------------------
THETIS is indoor, Kinect-captured, single-shot footage. This project analyses outdoor
broadcast video. Transfer between the two is a real research question and it is not
answered by training accuracy on THETIS: our 6-class model scores 73.4% on held-out
THETIS subjects and has never been shown to work on a broadcast frame. Treat THETIS as
pretraining and as the only available ground truth for volley and smash, not as evidence
about broadcast performance.

Subjects are p1-p31 (beginners) and p32-p55 (experts). Split by subject, never by clip,
or the same person's other takes leak into the test set.

Usage
-----
    python scripts/download_thetis.py                    # all 12 classes
    python scripts/download_thetis.py --classes smash forehand_volley
    python scripts/download_thetis.py --missing-only     # only classes not on disk

Source: https://github.com/THETIS-dataset/dataset (freely available for research).
Cite: Gourgari et al., "THETIS: Three Dimensional Tennis Shots a human action dataset",
CVPR Workshops 2013.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/THETIS-dataset/dataset.git"
TARGET = Path("datasets/external/thetis/VIDEO_RGB")

ALL_CLASSES = [
    "backhand", "backhand2hands", "backhand_slice", "backhand_volley",
    "forehand_flat", "forehand_openstands", "forehand_slice", "forehand_volley",
    "flat_service", "kick_service", "slice_service", "smash",
]


def missing_classes() -> list[str]:
    return [c for c in ALL_CLASSES if not (TARGET / c).is_dir()]


def fetch(classes: list[str], workdir: Path) -> bool:
    """Blobless sparse checkout of the requested class directories."""
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"  cloning index (no file contents yet)")
    clone = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", REPO, str(workdir)],
        capture_output=True, text=True,
    )
    if clone.returncode != 0:
        print(f"  clone failed: {clone.stderr.strip()[:300]}")
        return False

    paths = [f"VIDEO_RGB/{c}" for c in classes]
    print(f"  fetching {len(classes)} class(es): {', '.join(classes)}")
    sparse = subprocess.run(
        ["git", "sparse-checkout", "set", *paths],
        cwd=workdir, capture_output=True, text=True,
    )
    if sparse.returncode != 0:
        print(f"  sparse-checkout failed: {sparse.stderr.strip()[:300]}")
        return False
    return True


def install(classes: list[str], workdir: Path, merge: bool = False) -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    installed = 0
    for name in classes:
        src = workdir / "VIDEO_RGB" / name
        dst = TARGET / name
        if not src.is_dir():
            print(f"  [FAILED]  {name}: not present after checkout")
            continue
        if dst.exists() and not merge:
            print(f"  [skip]    {name} already present ({len(list(dst.glob('*.avi')))} clips)")
            continue

        if dst.exists():
            # Merge mode: copy only clips not already on disk. Purely additive, because
            # an earlier attempt at completing these classes moved the existing folders
            # aside first, the fetch then failed, and the only copy of that data was
            # briefly a directory named .old_something. Never again: nothing here
            # removes or renames what is already present.
            added = 0
            for clip in src.glob("*.avi"):
                target = dst / clip.name
                if not target.exists():
                    shutil.copy2(clip, target)
                    added += 1
            total = len(list(dst.glob("*.avi")))
            print(f"  [merge]   {name}: +{added} new, {total} total")
            installed += 1 if added else 0
            continue

        shutil.copytree(src, dst)
        print(f"  [ok]      {name}: {len(list(dst.glob('*.avi')))} clips")
        installed += 1
    return installed


def main() -> int:
    p = argparse.ArgumentParser(description="Download THETIS RGB videos")
    p.add_argument("--classes", nargs="*", choices=ALL_CLASSES,
                   help="specific classes (default: all)")
    p.add_argument("--missing-only", action="store_true",
                   help="only classes not already in datasets/external/thetis")
    p.add_argument("--merge", action="store_true",
                   help="add clips missing from classes already on disk (never deletes)")
    p.add_argument("--keep-temp", action="store_true",
                   help="keep the temporary checkout for inspection")
    args = p.parse_args()

    if args.missing_only:
        classes = missing_classes()
        if not classes:
            print("All 12 classes already present.")
            return 0
    else:
        classes = args.classes or ALL_CLASSES

    print(f"\nTHETIS RGB into {TARGET}")
    print(f"Classes requested: {len(classes)}\n")

    workdir = Path("datasets/external/.thetis_checkout")
    if not fetch(classes, workdir):
        print("\nFetch failed. The dataset is a git repository of about 13 GB; this "
              "script takes only the RGB directories you ask for.")
        return 1

    print()
    installed = install(classes, workdir, merge=args.merge)

    if not args.keep_temp:
        shutil.rmtree(workdir, ignore_errors=True)

    present = [c for c in ALL_CLASSES if (TARGET / c).is_dir()]
    print(f"\n{installed} class(es) installed. {len(present)}/12 now present.")
    if len(present) < len(ALL_CLASSES):
        print(f"Still missing: {', '.join(c for c in ALL_CLASSES if c not in present)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
