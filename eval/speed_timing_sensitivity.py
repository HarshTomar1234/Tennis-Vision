"""
eval/speed_timing_sensitivity.py
─────────────────────────────────
Which source of error actually dominates a reconstructed 3-D speed?

Why this exists
---------------
`utils/trajectory_3d.py` lists its honest limits as drag, spin and contact-height
uncertainty. All three are real. None of them is the biggest one.

Speed comes out of a two-point boundary value problem:

    vx, vy = (end - start) / T          T = (end_frame - start_frame) / fps
    vz     = (z1 - z0 + 0.5*g*T^2) / T

`T` is measured in EVENT FRAMES, and event frames are not exact. The README's 25-clip
sample puts the mean offset at 2.4 frames. That error enters the speed directly and
scales as 1/T, so it is worst exactly where the flights are shortest.

This script perturbs each term by its own measured uncertainty on real reconstructed
segments and compares the resulting speed change, so the module's limitations section can
be ordered by size rather than by which effect sounds most sophisticated.

Method
------
Reads a `trajectory3d_*.json` written by a pipeline run, so the segments are the ones the
product actually produced rather than synthetic ones. For each segment:

- **Event timing**: T is moved by +/- the measured 2.4-frame offset. Endpoint POSITIONS
  are held fixed, because this models the same physical events being assigned to a
  slightly different frame, which is what the offset measures.
- **Contact height**: the launch height is moved by +/- 0.20 m, the figure
  `trajectory_3d.py` describes as changing the launch angle by "well under a degree".
- **Ball localization**: one endpoint is moved by +/- 0.09 m. That is the 5.4 px median
  error at the network's 360x640 input, rescaled to source pixels and then to court
  metres for a 1292 px frame spanning the 10.97 m doubles court.

The larger absolute change of the two perturbations is reported per term.

What this does NOT measure
--------------------------
Drag and spin, which are unmodelled in the reconstruction and cannot be estimated from
its own output. The point here is only that timing is larger than the terms already
documented, not that the list is now complete.

Usage
-----
    python eval/speed_timing_sensitivity.py
    python eval/speed_timing_sensitivity.py --scene output/stats/trajectory3d_X.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.trajectory_3d import reconstruct_segment

# Measured event offset, README "Contact and bounce event detection", 25 clips.
EVENT_OFFSET_FRAMES = 2.4
# The tolerance trajectory_3d.py already calls negligible, tested here rather than assumed.
CONTACT_HEIGHT_TOLERANCE_M = 0.20
# 5.4 px median at 360x640 -> source pixels -> court metres. See the module docstring.
LOCALIZATION_TOLERANCE_M = 0.09


def _speed(start, end, z0, z1, duration_s) -> float | None:
    segment = reconstruct_segment(start, end, z0, z1, duration_s)
    return segment.speed_kmh if segment else None


def _worst_change(base: float, speeds: list[float | None]) -> float:
    """Largest percentage move away from the unperturbed speed."""
    changes = [abs(s - base) / base * 100 for s in speeds if s]
    return max(changes) if changes else 0.0


def analyse(scene_path: str, fps: float) -> list[dict]:
    segments = json.load(open(scene_path, encoding="utf-8"))["segments"]
    rows = []

    for seg in segments:
        points = seg["points"]
        start, end = tuple(points[0][:2]), tuple(points[-1][:2])
        z0, z1 = points[0][2], points[-1][2]
        duration = (seg["end_frame"] - seg["start_frame"]) / fps

        base = _speed(start, end, z0, z1, duration)
        if not base:
            continue

        timing = _worst_change(base, [
            _speed(start, end, z0, z1, duration + d / fps)
            for d in (-EVENT_OFFSET_FRAMES, EVENT_OFFSET_FRAMES)
            if duration + d / fps > 0
        ])
        height = _worst_change(base, [
            _speed(start, end, max(z0 + d, 0.0), z1, duration)
            for d in (-CONTACT_HEIGHT_TOLERANCE_M, CONTACT_HEIGHT_TOLERANCE_M)
        ])
        localization = _worst_change(base, [
            _speed((start[0] + d, start[1]), end, z0, z1, duration)
            for d in (-LOCALIZATION_TOLERANCE_M, LOCALIZATION_TOLERANCE_M)
        ])

        rows.append({
            "start_frame": seg["start_frame"], "end_frame": seg["end_frame"],
            "duration_s": duration, "speed_kmh": base,
            "timing_pct": timing, "height_pct": height, "localization_pct": localization,
        })

    return rows


def report(rows: list[dict]) -> None:
    print(f"\n{'segment':>12} {'T (s)':>6} {'km/h':>7} | "
          f"{'timing':>7} {'height':>7} {'localiz':>8}")
    print("-" * 60)
    for r in rows:
        print(f"{r['start_frame']:>4}->{r['end_frame']:<6} {r['duration_s']:>6.2f} "
              f"{r['speed_kmh']:>7.1f} | {r['timing_pct']:>6.1f}% "
              f"{r['height_pct']:>6.1f}% {r['localization_pct']:>7.1f}%")

    print(f"\n{'source of error':<38} {'mean':>7} {'median':>7} {'worst':>7}")
    print("-" * 62)
    for label, key in (
        (f"event timing (+/- {EVENT_OFFSET_FRAMES} frames)", "timing_pct"),
        (f"contact height (+/- {CONTACT_HEIGHT_TOLERANCE_M} m)", "height_pct"),
        (f"ball localization (+/- {LOCALIZATION_TOLERANCE_M} m)", "localization_pct"),
    ):
        values = sorted(r[key] for r in rows)
        mean = sum(values) / len(values)
        print(f"{label:<38} {mean:>6.1f}% {values[len(values)//2]:>6.1f}% "
              f"{values[-1]:>6.1f}%")

    short = [r for r in rows if r["duration_s"] < 0.5]
    long_ = [r for r in rows if r["duration_s"] >= 1.0]
    print(f"\nTiming sensitivity scales as 1/T, so it is worst on the shortest flights:")
    if short:
        print(f"  flights under 0.5 s ({len(short)}): "
              f"{sum(r['timing_pct'] for r in short) / len(short):.1f}% mean")
    if long_:
        print(f"  flights over 1.0 s ({len(long_)}): "
              f"{sum(r['timing_pct'] for r in long_) / len(long_):.1f}% mean")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default=None,
                    help="trajectory3d_*.json from a run (default: the richest one in "
                         "output/stats)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="frame rate of the clip the scene came from")
    args = ap.parse_args()

    scene = args.scene
    if scene is None:
        candidates = glob.glob("output/stats/trajectory3d_*.json")
        if not candidates:
            print("No trajectory3d_*.json in output/stats. Run the pipeline first:")
            print("  python main.py --input input_videos/input_video_2.mp4")
            return 1
        # The richest scene, not the newest: a --max-frames run leaves a one-segment
        # file behind and would otherwise be picked up and reported as the answer.
        scene = max(candidates,
                    key=lambda f: len(json.load(open(f, encoding="utf-8"))["segments"]))

    rows = analyse(scene, args.fps)
    if not rows:
        print(f"No reconstructable segments in {scene}")
        return 1

    print(f"source: {os.path.basename(scene)}   {len(rows)} segments   {args.fps:.0f} fps")
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
