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

# Broadcast radar readings for the serve clips, read off the Wimbledon panel.
#
# Reading these correctly matters more than it looks. The panel shows the speed of the
# PREVIOUS serve until the current one completes, so the value visible at the contact
# frame belongs to the serve before it. Each entry below was taken from a frame well
# AFTER the detected contact, and cross-checked against a second later frame. An
# earlier version of this table used the at-contact reading and was wrong by a whole
# serve.
#
# Clip -> (radar mph, radar km/h). Wimbledon reports mph.
SERVE_GROUND_TRUTH: dict[str, tuple[int, float]] = {
    "serve_4_t288": (110, 177.0),
    "serve_5_t363": (133, 214.0),
}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", help="evaluate a single clip stem, e.g. serve_5_t363")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    stems = [args.clip] if args.clip else sorted(SERVE_GROUND_TRUTH)

    print("\nServe speed vs broadcast radar")
    print("Ours = average speed over the flight; radar = speed at contact. Drag means")
    print("ours reads at or below radar, never above - a ratio > 1 would signal a bug.\n")
    print(f"  {'clip':16s} {'radar km/h':>11s} {'ours':>8s} {'ratio':>7s}   note")
    print("  " + "-" * 62)

    ratios = []
    for stem in stems:
        if stem not in SERVE_GROUND_TRUTH:
            print(f"  {stem:16s} no ground truth recorded")
            continue
        _mph, truth = SERVE_GROUND_TRUTH[stem]
        measured, note = measure_clip(stem)
        if measured <= 0:
            print(f"  {stem:16s} {truth:11.1f} {'--':>8s} {'--':>7s}   {note}")
            continue
        ratio = measured / truth
        ratios.append(ratio)
        print(f"  {stem:16s} {truth:11.1f} {measured:8.1f} {ratio:7.2f}   {note}")

    if ratios:
        mean = sum(ratios) / len(ratios)
        print(f"\n  mean ratio {mean:.2f} over {len(ratios)} serve(s)"
              f"  |  spread {min(ratios):.2f}-{max(ratios):.2f}")
        print("\n  A tight spread at or just below 1.0 is the expected signature: the")
        print("  method is sound and the shortfall is drag. Values above 1.0, or a wide")
        print("  spread, mean the measurement is not trustworthy and must not ship.\n")
    else:
        print("\n  No serve was measurable - nothing to compare.\n")


def measure_clip(stem: str) -> tuple[float, str]:
    """
    Run the real pipeline on one clip and read back its measured serve speed.

    Deliberately shells out to main.py: the number reported here is then, by
    construction, the number the product produces.
    """
    import json
    import subprocess
    import sys as _sys

    matches = list(Path("datasets/serve_clips").glob(f"{stem}.mp4"))
    if not matches:
        return 0.0, "clip not found (build it with scripts/build_clip_suite.py)"

    stats_dir = Path("output/stats")
    before = set(stats_dir.glob("summary_*.json")) if stats_dir.exists() else set()

    proc = subprocess.run(
        [_sys.executable, "main.py", "-i", str(matches[0]),
         "-o", f"output/clipsuite/servespeed_{stem}.avi", "--no-stubs"],
        capture_output=True, text=True, timeout=2400,
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
        note = "serve landing not observed"
    return speed, note


if __name__ == "__main__":
    main()
