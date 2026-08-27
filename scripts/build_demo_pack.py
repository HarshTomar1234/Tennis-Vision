"""
scripts/build_demo_pack.py
───────────────────────────
Assembles the public demonstration assets from a real pipeline run.

Everything here is extracted from output the pipeline actually produced. Nothing is drawn,
composited or relabelled for presentation. If a number appears on a demo frame it appears
there because the pipeline rendered it, and if a stage produced nothing the demo says so
rather than substituting a better-looking example.

That constraint is the point. A demo that shows a capability the shipped code does not
have is the same defect class as a report that publishes a speed it could not measure.

What it produces
----------------
    output/demo/
      frames/          key frames from the annotated video, one per pipeline stage
      viewer.html      the interactive 3-D reconstruction, self-contained
      summary.json     the run's own evidence output, copied verbatim
      DEMO_NOTES.md    what each asset shows and which run it came from

Usage
-----
    python main.py --input input_videos/input_video_2.mp4 -o output/demo/demo.avi
    python scripts/build_demo_pack.py
"""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import cv2

DEMO = Path("output/demo")
FRAMES = DEMO / "frames"


def latest_summary() -> tuple[Path, dict]:
    """The most recent VALID summary. Malformed ones are skipped, not repaired."""
    for path in sorted(glob.glob("output/stats/summary_*.json"), reverse=True):
        try:
            return Path(path), json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    raise SystemExit("No valid summary_*.json. Run the pipeline first.")


def pick_frames(video: Path, summary: dict) -> list[tuple[int, str]]:
    """
    Frames chosen by what the pipeline reported, not by eye.

    The serve and the contacts come from the run's own event list, so the demo cannot
    show a moment the pipeline did not actually identify.
    """
    total = summary.get("calibration", {}).get("frames_total", 0)
    picks = [(int(total * 0.05), "opening: court keypoints, player tracking, mini-court")]

    scene = sorted(glob.glob("output/stats/trajectory3d_*.json"), reverse=True)
    if scene:
        segs = json.load(open(scene[0], encoding="utf-8"))["segments"]
        shots = [s for s in segs
                 if s.get("speed_status") in ("valid", "plausible_but_uncertain")]
        for s in shots[:3]:
            mid = (s["start_frame"] + s["end_frame"]) // 2
            speed = s.get("speed_kmh")
            picks.append((mid, f"reconstructed flight f{s['start_frame']}-{s['end_frame']}"
                               f"{f', {speed:.0f} km/h' if speed else ''}"))
    picks.append((max(0, total - 5), "final frame: accumulated statistics panel"))
    return [(f, label) for f, label in picks if 0 <= f < total]


def main() -> int:
    videos = sorted(glob.glob("output/demo/*.avi"), reverse=True)
    if not videos:
        print("No annotated video in output/demo/. Run:")
        print("  python main.py --input input_videos/input_video_2.mp4 "
              "-o output/demo/tennis_vision_demo.avi")
        return 1
    video = Path(videos[0])

    summary_path, summary = latest_summary()
    FRAMES.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    picks = pick_frames(video, summary)
    written = []
    for frame_no, label in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            continue
        out = FRAMES / f"f{frame_no:04d}.jpg"
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written.append((out.name, frame_no, label))
        print(f"  {out.name}  {label}")
    cap.release()

    shutil.copy(summary_path, DEMO / "summary.json")
    viewer = video.with_suffix(".html")
    if viewer.exists():
        shutil.copy(viewer, DEMO / "viewer.html")

    quality = {
        "court": f"{'calibrated' if summary.get('court_calibrated') else 'FAILED'} "
                 f"(line support {summary.get('court_line_support')})",
        "frame rate": summary.get("frame_rate_support", {}).get("status"),
        "players": summary.get("player_selection", {}).get("status"),
        "ball coverage": f"{summary.get('ball', {}).get('coverage', 0):.0%}",
        "shots": summary.get("shot_classification", {}).get("shots"),
        "serves evidenced": summary.get("shot_classification", {}).get("serve_evidenced"),
    }
    speed = summary.get("shot_speed_3d_kmh", {})

    notes = [
        "# Tennis-Vision demo pack",
        "",
        f"Generated from `{video.name}` and `{summary_path.name}`. Every asset here is",
        "output the pipeline produced. Nothing was drawn, composited or relabelled.",
        "",
        "## What the run reported about itself",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    notes += [f"| {k} | {v} |" for k, v in quality.items()]
    if speed.get("status") == "unavailable":
        notes.append("| 3-D shot speed | unavailable (no segment began at a contact) |")
    else:
        notes.append(f"| 3-D shot speed | {speed.get('segments')} shot segments, "
                     f"{speed.get('min')}-{speed.get('max')} km/h, "
                     f"{speed.get('segments_excluded')} excluded |")
    notes += [
        "",
        "## Frames",
        "",
    ] + [f"- `frames/{n}` (frame {f}) - {l}" for n, f, l in written] + [
        "",
        "## Files",
        "",
        "- `viewer.html` - interactive 3-D reconstruction, self-contained, no network",
        "- `summary.json` - the run's evidence output, copied verbatim",
        "",
        "Reproduce:",
        "",
        "```bash",
        "python main.py --input input_videos/input_video_2.mp4 \\",
        "    -o output/demo/tennis_vision_demo.avi",
        "python scripts/build_demo_pack.py",
        "```",
    ]
    (DEMO / "DEMO_NOTES.md").write_text("\n".join(notes), encoding="utf-8")

    print(f"\nDemo pack -> {DEMO}")
    for k, v in quality.items():
        print(f"  {k:<16} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
