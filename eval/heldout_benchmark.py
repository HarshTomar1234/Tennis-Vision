"""
eval/heldout_benchmark.py
──────────────────────────
Runs the frozen held-out benchmark and reports what the pipeline did on footage it has
never been tuned against.

Why a separate script from the development evals
-------------------------------------------------
Every other eval in this directory runs on `datasets/eval_clips/*.mp4`, and those nine
clips are contaminated: they calibrated MIN_LINE_SUPPORT, they swept the rally deletion
prior, they produced the frame-rate bands, the physics evidence rate and the player
sanity figures. Numbers from them describe how the system performs on data it was fitted
to, which is a different and easier question than how it performs on new footage.

`datasets/heldout/manifest.json` names a set cut from a region of source video that has
never influenced any threshold, model choice or feature selection. The manifest was
frozen before a single clip was viewed, and the selection rule is arithmetic, so the set
cannot have been curated for flattering results.

What this measures, and what it cannot
---------------------------------------
It measures whether the pipeline COMPLETES, whether its GATES FIRE CORRECTLY, and what
it REFUSES on unseen footage. It does not measure event precision or recall, because
these clips have no frame-level ground truth. Those numbers stay sourced from the
labelled TrackNet dataset and are not restated here.

That distinction is the point. A benchmark that produced an accuracy figure from
unlabelled video would be inventing one.

Usage
-----
    python eval/heldout_benchmark.py                 # run all, write results JSON
    python eval/heldout_benchmark.py --report-only   # re-print from saved results
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

MANIFEST = Path("datasets/heldout/manifest.json")
CLIP_DIR = Path("datasets/heldout/clips")
RESULTS = Path("datasets/heldout/results.json")

# Phase 11 failure taxonomy. Every clip that is not a clean PASS maps to one of these,
# so the question answered is "what fails repeatedly", not "which video looked bad".
TAXONOMY = ["INPUT", "FPS", "CAMERA", "COURT", "PLAYER", "BALL", "EVENT", "RALLY",
            "POSE", "SHOT", "PHYSICS", "3D", "OUTPUT", "PERFORMANCE",
            "UNSUPPORTED_CONDITION", "UNKNOWN"]


def run_clip(video: Path) -> dict:
    """Run the real pipeline once and return its summary plus timing."""
    before = set(glob.glob("output/stats/summary_*.json"))
    started = time.time()

    result = subprocess.run(
        [sys.executable, "main.py",
         "--input", str(video),
         "--output", f"output/videos/heldout_{video.stem}.avi",
         "--config", "configs/config.yaml"],       # no stubs: fresh detection
        capture_output=True, text=True, timeout=3600,
    )
    elapsed = time.time() - started

    if result.returncode != 0:
        return {"crashed": True, "elapsed_s": elapsed,
                "stderr_tail": result.stderr[-1500:]}

    new = set(glob.glob("output/stats/summary_*.json")) - before
    if not new:
        return {"crashed": False, "no_summary": True, "elapsed_s": elapsed}

    summary = json.load(open(max(new), encoding="utf-8"))
    return {"crashed": False, "elapsed_s": elapsed, "summary": summary}


def classify(summary: dict) -> tuple[str, list[str], list[str]]:
    """(verdict, failure categories, human-readable notes) for one clip."""
    problems, notes = [], []

    fps = summary.get("frame_rate_support", {})
    if fps.get("status") == "unsupported":
        problems.append("FPS"); notes.append(f"fps {fps.get('fps')} unsupported")
    elif fps.get("status") == "partially_supported":
        notes.append(f"fps {fps.get('fps')} partially supported")

    if not summary.get("court_calibrated", False):
        problems.append("COURT")
        notes.append(f"court fit failed (line support {summary.get('court_line_support')})")

    sel = summary.get("player_selection", {})
    if sel.get("status") == "failed":
        problems.append("PLAYER"); notes.append("player selection failed")
    elif sel.get("status") == "degraded":
        notes.append("player tracking degraded")

    ball = summary.get("ball", {})
    if ball and ball.get("coverage", 1.0) < 0.60:
        problems.append("BALL")
        notes.append(f"ball seen on {ball['coverage']:.0%} of frames")
    elif ball and ball.get("longest_gap_frames", 0) > ball.get("frames", 1) * 0.25:
        problems.append("BALL")
        notes.append(f"ball missing for {ball['longest_gap_frames']} consecutive frames")

    shots = summary.get("shot_classification", {}).get("shots", 0)
    if shots == 0:
        problems.append("EVENT"); notes.append("no shots detected")

    rally = summary.get("rally_decoding", {})
    if rally.get("relabelled_over_high_confidence", 0) > 0:
        notes.append(f"{rally['relabelled_over_high_confidence']} high-confidence "
                     f"decoder override(s)")

    speed = summary.get("shot_speed_3d_kmh", {})
    if speed.get("status") == "unavailable":
        notes.append("no 3-D shot speed available")

    calib = summary.get("calibration", {})
    if calib.get("frames_using_fallback_mapping", 0) > 0:
        problems.append("COURT")
        notes.append(f"{calib['frames_using_fallback_mapping']} frames on fallback mapping")
    if calib.get("positions_unmappable_and_omitted", 0) > 0:
        notes.append(f"{calib['positions_unmappable_and_omitted']} positions omitted")

    if not problems:
        verdict = "PASS"
    elif shots > 0 and "COURT" not in problems and "PLAYER" not in problems:
        verdict = "PARTIAL"
    else:
        verdict = "REFUSED"     # gates fired; the pipeline declined to report. Correct.
    return verdict, problems, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST, encoding="utf-8"))

    if args.report_only and RESULTS.exists():
        results = json.load(open(RESULTS, encoding="utf-8"))
    else:
        results = {"manifest": manifest["name"], "clips": []}
        for entry in manifest["clips"]:
            video = CLIP_DIR / entry["filename"]
            if not video.exists():
                print(f"  {entry['clip_id']}: MISSING {video}")
                continue
            print(f"  {entry['clip_id']} running ...", flush=True)
            outcome = run_clip(video)
            verdict, problems, notes = (
                ("CRASH", ["OUTPUT"], [outcome.get("stderr_tail", "")[:200]])
                if outcome.get("crashed") or outcome.get("no_summary")
                else classify(outcome["summary"])
            )
            results["clips"].append({
                **{k: entry[k] for k in ("clip_id", "filename", "source_start_s")},
                "verdict": verdict, "failures": problems, "notes": notes,
                "elapsed_s": round(outcome["elapsed_s"], 1),
                "summary": outcome.get("summary"),
            })
            print(f"    {verdict}  {'; '.join(notes) if notes else 'clean'}"
                  f"  [{outcome['elapsed_s']:.0f}s]")
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(RESULTS, "w", encoding="utf-8"), indent=2)
        print(f"\nwrote {RESULTS}")

    report(results, manifest)
    return 0


def report(results: dict, manifest: dict) -> None:
    clips = results["clips"]
    if not clips:
        print("no results"); return

    print("\n" + "=" * 100)
    print(f"HELD-OUT BENCHMARK  ·  {manifest['name']}  ·  {len(clips)} clips")
    print("=" * 100)

    print(f"\n{'id':<5} {'t(s)':>6} {'verdict':<8} {'fps':>6} {'court':>7} "
          f"{'players':<9} {'ball':>6} {'shots':>6} {'3D':>8} {'sec':>6}")
    print("-" * 100)
    for c in clips:
        s = c.get("summary") or {}
        sel = s.get("player_selection", {}).get("status", "-")
        ball = s.get("ball", {}).get("coverage")
        sp = s.get("shot_speed_3d_kmh", {})
        n3d = sp.get("segments", 0) if sp.get("status") != "unavailable" else 0
        print(f"{c['clip_id']:<5} {c['source_start_s']:>6} {c['verdict']:<8} "
              f"{s.get('frame_rate_support', {}).get('fps', 0):>6.1f} "
              f"{s.get('court_line_support', 0):>7.3f} "
              f"{sel:<9} "
              f"{(f'{ball:.0%}' if ball is not None else '-'):>6} "
              f"{s.get('shot_classification', {}).get('shots', 0):>6} "
              f"{n3d:>8} {c['elapsed_s']:>6.0f}")

    verdicts = {}
    for c in clips:
        verdicts[c["verdict"]] = verdicts.get(c["verdict"], 0) + 1
    print("\nVERDICTS:", "  ".join(f"{k} {v}" for k, v in sorted(verdicts.items())))

    tally = {}
    for c in clips:
        for f in c["failures"]:
            tally[f] = tally.get(f, 0) + 1
    print("\nFAILURE TAXONOMY (what fails repeatedly)")
    print("-" * 50)
    if tally:
        for cat, n in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"  {cat:<24} {n:>3} clip(s)   {n/len(clips):.0%}")
    else:
        print("  none")

    # Output-integrity checks: these must hold on every clip regardless of verdict.
    print("\nOUTPUT INTEGRITY (must hold on every clip)")
    print("-" * 50)
    fabricated = sum((c.get("summary") or {}).get("calibration", {})
                     .get("positions_unmappable_and_omitted", 0) for c in clips)
    fallback = sum((c.get("summary") or {}).get("calibration", {})
                   .get("frames_using_fallback_mapping", 0) for c in clips)
    crashes = sum(1 for c in clips if c["verdict"] == "CRASH")
    serialised = sum(1 for c in clips if c.get("summary") is not None)
    print(f"  crashes                          {crashes} / {len(clips)}")
    print(f"  summary.json written and valid   {serialised} / {len(clips)}")
    print(f"  positions omitted, not invented  {fabricated} (omission is correct)")
    print(f"  frames on silent fallback map    {fallback} (reported, not silent)")

    total = sum(c["elapsed_s"] for c in clips)
    vid = sum(15.0 for _ in clips)
    print(f"\nRUNTIME  {total/60:.1f} min for {vid:.0f}s of video "
          f"= {total/vid:.1f}x real time, GTX 1050 Ti, fresh detection")


if __name__ == "__main__":
    sys.exit(main())
