"""
eval/heldout_segments.py
─────────────────────────
The second half of the held-out benchmark: how does the pipeline do on the tennis INSIDE
those clips?

Why this exists as a separate measurement
------------------------------------------
`eval/heldout_benchmark.py` runs the frozen 15-second windows exactly as they were cut and
refuses 14 of 14. That result is real and is reported unchanged: an arbitrary window taken
from a broadcast is not a supported input, because the pipeline assumes one continuous
camera take and a broadcast window routinely contains the end of a point, a crowd shot, a
replay and the next serve.

But it answers "does the pipeline survive camera cuts", not "does the pipeline analyse
tennis". Those are different questions and merging them would hide both answers. Per-frame
court scoring shows that 7 of the 14 windows contain a contiguous run of frames that ARE a
valid court view; the other 7 never show a playable court at all.

This script locates that run and analyses it. Same frozen clips, same frozen manifest,
nothing re-cut to taste and nothing dropped for looking bad.

Is this circular?
-----------------
It would be if it then reported a court-validity number, so it does not. The court gate is
used only to LOCATE the segment. Everything reported here is downstream of it: ball
coverage, player selection, event counts, 3-D reconstruction. Those are not measurements
the segment selection can flatter.

What it still cannot measure
-----------------------------
Event precision and recall, because these clips have no frame-level ground truth. Those
numbers stay sourced from the labelled TrackNet dataset and are not restated here.

Usage
-----
    python eval/heldout_segments.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from court_line_detector import CourtLineDetector
from utils.court_validity import MIN_LINE_SUPPORT, line_support_score

MANIFEST = Path("datasets/heldout/manifest.json")
CLIP_DIR = Path("datasets/heldout/clips")
SEG_DIR = Path("datasets/heldout/segments")
RESULTS = Path("datasets/heldout/segment_results.json")

STRIDE = 15          # score every 15th frame; finer than the 12 samples the gate uses
MIN_SEGMENT_FRAMES = 100   # ~4 s at 25 fps. Below this there is no rally to reconstruct.


def find_court_segment(video: Path, court: CourtLineDetector) -> tuple[int, int] | None:
    """Longest contiguous run of frames whose court fit is valid, or None."""
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None

    marks = []   # (frame_index, passes)
    for i in range(0, len(frames), STRIDE):
        kp = court.predict(frames[i])
        marks.append((i, line_support_score(frames[i], kp) >= MIN_LINE_SUPPORT))

    best = run_start = None
    best_len = 0
    for idx, ok in marks + [(len(frames), False)]:
        if ok and run_start is None:
            run_start = idx
        elif not ok and run_start is not None:
            if idx - run_start > best_len:
                best_len, best = idx - run_start, (run_start, idx)
            run_start = None

    if best is None or best_len < MIN_SEGMENT_FRAMES:
        return None
    return best


def cut(video: Path, span: tuple[int, int], out: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, span[0])
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    for _ in range(span[1] - span[0]):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame); n += 1
    writer.release(); cap.release()
    return n


def analyse(video: Path) -> dict:
    before = set(glob.glob("output/stats/summary_*.json"))
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "main.py", "--input", str(video),
         "--output", f"output/videos/seg_{video.stem}.avi",
         "--config", "configs/config.yaml"],
        capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    if r.returncode != 0:
        return {"crashed": True, "elapsed_s": elapsed, "stderr": r.stderr[-800:]}
    new = set(glob.glob("output/stats/summary_*.json")) - before
    if not new:
        return {"crashed": False, "no_summary": True, "elapsed_s": elapsed}
    return {"crashed": False, "elapsed_s": elapsed,
            "summary": json.load(open(max(new), encoding="utf-8"))}


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[1]).parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    SEG_DIR.mkdir(parents=True, exist_ok=True)
    court = CourtLineDetector("models/keypoints_model_geoaug.pth")

    print("Locating the court-valid segment inside each frozen clip ...\n")
    results = {"manifest": manifest["name"], "segments": []}

    for entry in manifest["clips"]:
        clip = CLIP_DIR / entry["filename"]
        if not clip.exists():
            continue
        span = find_court_segment(clip, court)
        if span is None:
            print(f"  {entry['clip_id']}: no court segment of {MIN_SEGMENT_FRAMES}+ frames")
            results["segments"].append({"clip_id": entry["clip_id"],
                                        "has_segment": False})
            continue

        seg_path = SEG_DIR / f"seg_{entry['clip_id']}.mp4"
        n = cut(clip, span, seg_path)
        print(f"  {entry['clip_id']}: frames {span[0]}-{span[1]} ({n} frames, "
              f"{n/25:.1f}s) -> analysing", flush=True)
        outcome = analyse(seg_path)
        results["segments"].append({
            "clip_id": entry["clip_id"], "has_segment": True,
            "span": list(span), "frames": n,
            "elapsed_s": round(outcome["elapsed_s"], 1),
            "crashed": outcome.get("crashed", False),
            "summary": outcome.get("summary"),
        })

    json.dump(results, open(RESULTS, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {RESULTS}")
    report(results)
    return 0


def report(results: dict) -> None:
    segs = [s for s in results["segments"] if s.get("has_segment")]
    none = [s for s in results["segments"] if not s.get("has_segment")]

    print("\n" + "=" * 96)
    print("HELD-OUT BENCHMARK, PART B  ·  the court-valid segment inside each frozen clip")
    print("=" * 96)
    print(f"\n{len(segs)} of {len(results['segments'])} clips contain an analysable "
          f"segment. {len(none)} never show a playable court.\n")

    print(f"{'id':<5} {'sec':>5} {'court':>7} {'players':<9} {'ball':>6} {'shots':>6} "
          f"{'serve':>6} {'3D':>4} {'speed km/h':>12} {'rally':>7}")
    print("-" * 96)
    ok_court = ok_players = 0
    for s in segs:
        m = s.get("summary") or {}
        if not m:
            print(f"{s['clip_id']:<5}  CRASH"); continue
        sel = m.get("player_selection", {}).get("status", "-")
        sp = m.get("shot_speed_3d_kmh", {})
        rng = (f"{sp.get('min', 0):.0f}-{sp.get('max', 0):.0f}"
               if sp.get("status") != "unavailable" and sp.get("segments") else "-")
        rc = m.get("rally_decoding", {})
        ok_court += bool(m.get("court_calibrated"))
        ok_players += sel == "ok"
        print(f"{s['clip_id']:<5} {s['frames']/25:>5.1f} "
              f"{m.get('court_line_support', 0):>7.3f} {sel:<9} "
              f"{m.get('ball', {}).get('coverage', 0):>5.0%} "
              f"{m.get('shot_classification', {}).get('shots', 0):>6} "
              f"{m.get('shot_classification', {}).get('serve_evidenced', 0):>6} "
              f"{sp.get('segments', 0):>4} {rng:>12} "
              f"{rc.get('events_kept', 0):>7}")

    n = len([s for s in segs if s.get("summary")])
    if n:
        print(f"\nOn the analysable segments:")
        print(f"  court calibrated       {ok_court}/{n}")
        print(f"  player selection ok    {ok_players}/{n}")
        balls = [s["summary"].get("ball", {}).get("coverage", 0)
                 for s in segs if s.get("summary")]
        print(f"  ball coverage          median {sorted(balls)[len(balls)//2]:.0%}, "
              f"range {min(balls):.0%}-{max(balls):.0%}")
        shots = [s["summary"].get("shot_classification", {}).get("shots", 0)
                 for s in segs if s.get("summary")]
        print(f"  shots reported         {sum(shots)} across {n} segments")

    print("\nNOT measured here: event precision and recall. These clips have no "
          "frame-level\nground truth, so those numbers stay sourced from the labelled "
          "TrackNet dataset.")


if __name__ == "__main__":
    sys.exit(main())
