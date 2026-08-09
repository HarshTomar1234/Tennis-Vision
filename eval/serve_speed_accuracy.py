"""
eval/serve_speed_accuracy.py
────────────────────────────
Measures serve speed against real broadcast radar ground truth.

Where the ground truth comes from
---------------------------------
Tournament broadcasts overlay the radar-measured speed of each serve (the IBM panel
at the Australian Open, the corner readout at Roland Garros). Those numbers were read
off the clips by eye and recorded in GROUND_TRUTH below. This is genuine external
ground truth for a metric that has never been validated in this project.

Two honest caveats, both material:

  1. The overlay reports the speed of the *most recent* serve. It is recorded here
     only for clips where the serve visibly belongs to the point being played.
  2. Radar measures speed AT CONTACT. This pipeline measures AVERAGE SPEED OVER THE
     FLIGHT (see utils/serve_speed.py). Drag makes the average lower than the contact
     speed, so a reading below ground truth is expected — the question this script
     answers is *by how much, and how consistently*. A consistent ratio is a usable,
     explainable measurement. A scattered one means the method does not work.

How this runs
-------------
It shells out to `main.py` per clip and reads `serve_avg_flight_speed_kmh` from the
summary JSON, rather than re-assembling the pipeline here. An earlier version did
rebuild the stages inline and silently diverged from production (different candidate
sources, wrong mini-court height argument), producing "no serve found" on a clip
where the real pipeline finds one. Measuring anything other than what actually ships
is worse than not measuring.

Usage:
    python eval/serve_speed_accuracy.py            # all clips with ground truth
    python eval/serve_speed_accuracy.py --clip 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Broadcast radar readings, km/h, read from the on-screen panel of each clip.
GROUND_TRUTH: dict[int, int] = {
    5:  182,
    7:  194,
    8:  136,
    9:  184,
    10: 149,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", type=int, default=0, help="evaluate a single clip")
    args = ap.parse_args()

    clips = [args.clip] if args.clip else sorted(GROUND_TRUTH)

    print("\nServe speed vs broadcast radar")
    print("(ours = average speed over flight; radar = speed at contact — ours reads")
    print(" lower by design; watch the ratio column for consistency)\n")
    print(f"  {'clip':6s} {'radar':>7s} {'ours':>8s} {'ratio':>7s}   note")
    print("-" * 60)

    ratios = []
    for clip in clips:
        result = measure_clip(clip)
        truth = GROUND_TRUTH[clip]
        if result is None:
            print(f"  {clip:<6d} {truth:7d} {'--':>8s} {'--':>7s}   no serve/bounce pair found")
            continue
        ours, note = result
        if ours <= 0:
            print(f"  {clip:<6d} {truth:7d} {'--':>8s} {'--':>7s}   {note}")
            continue
        ratio = ours / truth
        ratios.append(ratio)
        print(f"  {clip:<6d} {truth:7d} {ours:8.1f} {ratio:7.2f}   {note}")

    if len(ratios) >= 2:
        r = np.array(ratios)
        print(f"\n  mean ratio {r.mean():.2f}  |  spread {r.min():.2f}-{r.max():.2f}"
              f"  |  std {r.std():.3f}")
        print("\n  A tight spread means the method is sound and the offset is just")
        print("  drag (correctable by a documented constant). A wide spread means the")
        print("  measurement is not reliable and must not be published as a metric.\n")
    else:
        print("\n  Not enough successful measurements to judge consistency.\n")


def measure_clip(clip: int) -> tuple[float, str] | None:
    """
    Run the real pipeline on one clip and read back its measured serve speed.

    Deliberately shells out to main.py: the number reported here is then, by
    construction, the number the product produces.
    """
    import json
    import subprocess
    import sys as _sys

    path = Path(f"datasets/evail_clips/input_video_{clip}.mp4")
    if not path.exists():
        return None

    stats_dir = Path("output/stats")
    before = set(stats_dir.glob("summary_*.json")) if stats_dir.exists() else set()

    proc = subprocess.run(
        [_sys.executable, "main.py", "-i", str(path),
         "-o", f"output/clipsuite/servespeed_{clip}.avi", "--no-stubs"],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        return 0.0, "pipeline failed"

    new = sorted(set(stats_dir.glob("summary_*.json")) - before)
    if not new:
        return 0.0, "no summary written"

    data = json.loads(new[-1].read_text(encoding="utf-8"))
    speed = float(data.get("serve_avg_flight_speed_kmh", 0.0))
    note = "court fit OK" if data.get("court_calibrated", True) else "COURT FIT INVALID"
    if speed <= 0:
        note = "no serve/bounce pair found"
    return speed, note


if __name__ == "__main__":
    main()
