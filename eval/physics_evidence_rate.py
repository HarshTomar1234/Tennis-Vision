"""
eval/physics_evidence_rate.py
──────────────────────────────
Is the physics shot layer a classifier, or a validation filter?

Why this exists
---------------
`utils/shot_physics.py` was built to replace position-guessed Volley and Smash labels
with physically evidenced ones. On the reference clip it evidences **zero** shots and
downgrades **seven** unevidenced Volley/Smash labels to "Groundstroke". On that clip it
is therefore not classifying anything: it is rejecting labels nothing supports, which is
useful and is a different thing.

One clip does not settle what to call it. This runs the real pipeline across a set of
clips and aggregates what the layer actually did, so the public wording can be chosen on
evidence. Calling a filter a classifier would overstate it; weakening the rejection
because it "never fires positively" would be the opposite mistake and is explicitly not
what this measures for.

Method
------
Runs `main.py` per clip and reads the `shot_classification` block from each run's
summary.json. It does not reimplement the physics call, so what is counted is exactly
what the product does. That is slow, and it is the only way the number means anything.

Usage
-----
    python eval/physics_evidence_rate.py
    python eval/physics_evidence_rate.py --clips "datasets/eval_clips/*.mp4"
    python eval/physics_evidence_rate.py --max-frames 200      # quicker, less complete
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_GLOB = "datasets/eval_clips/*.mp4"


def run_clip(video: str, max_frames: int) -> dict | None:
    """Run the pipeline on one clip and return its shot_classification block."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "main.py",
            "--input", video,
            "--output", str(Path(tmp) / "out.avi"),
            "--config", "configs/dev.yaml",
        ]
        if max_frames:
            cmd += ["--max-frames", str(max_frames)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"  {Path(video).name}: pipeline failed\n{result.stderr[-500:]}")
            return None

    # The pipeline writes into the configured stats dir, not the temp dir.
    summaries = sorted(glob.glob("output/stats/summary_*.json"))
    if not summaries:
        return None
    summary = json.load(open(summaries[-1], encoding="utf-8"))
    return summary.get("shot_classification")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--clips", default=DEFAULT_GLOB)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="truncate each clip; faster, and a partial rally sees fewer "
                         "shots, so the rates are less trustworthy")
    args = ap.parse_args()

    videos = sorted(glob.glob(args.clips))
    if not videos:
        print(f"No clips matched {args.clips}. See datasets/README.md.")
        return 1

    print(f"Running the real pipeline on {len(videos)} clip(s). This is slow by design: "
          f"reimplementing the physics call here would measure a copy, not the product.\n")

    totals = {"shots": 0, "physics_evidenced": 0, "physics_downgraded": 0,
              "serve_evidenced": 0}
    rows = []

    for video in videos:
        name = Path(video).name
        print(f"  {name} ...", flush=True)
        block = run_clip(video, args.max_frames)
        if not block:
            continue
        rows.append((name, block))
        totals["shots"] += block.get("shots", 0)
        totals["physics_evidenced"] += block.get("physics_evidenced", 0)
        totals["physics_downgraded"] += block.get(
            "physics_downgraded_to_groundstroke", 0)
        totals["serve_evidenced"] += block.get("serve_evidenced", 0)

    if not rows:
        print("No clip produced a shot classification block.")
        return 1

    print(f"\n{'clip':<28} {'shots':>6} {'evidenced':>10} {'downgraded':>11} {'serves':>7}")
    print("-" * 66)
    for name, b in rows:
        print(f"{name:<28} {b.get('shots', 0):>6} {b.get('physics_evidenced', 0):>10} "
              f"{b.get('physics_downgraded_to_groundstroke', 0):>11} "
              f"{b.get('serve_evidenced', 0):>7}")

    shots = totals["shots"] or 1
    print("-" * 66)
    print(f"{'TOTAL':<28} {totals['shots']:>6} {totals['physics_evidenced']:>10} "
          f"{totals['physics_downgraded']:>11} {totals['serve_evidenced']:>7}")

    print(f"\nPositive evidence rate : {totals['physics_evidenced'] / shots:.1%} of shots")
    print(f"Rejection rate         : {totals['physics_downgraded'] / shots:.1%} of shots")

    if totals["physics_evidenced"] == 0:
        print("\nVERDICT: the physics layer evidenced nothing across this sample. It is a "
              "VALIDATION FILTER, not a classifier, and public wording should say so. "
              "Its rejection logic is doing real work and must not be weakened on the "
              "strength of this result.")
    elif totals["physics_evidenced"] < totals["physics_downgraded"]:
        print("\nVERDICT: the layer rejects more often than it evidences. 'Physics "
              "validation' describes it more accurately than 'physics classifier'.")
    else:
        print("\nVERDICT: the layer positively evidences shots often enough to be "
              "described as a classifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
