"""
eval/serve_false_positive_check.py
──────────────────────────────────
Does the pipeline claim a serve on clips that contain none?

Why this matters more than it sounds
------------------------------------
Most footage a user brings is cut from mid-rally. It contains no serve at all, and a
system that labels the first shot it sees "Serve" is wrong on nearly every such clip
while looking confident. That was the original behaviour here:

    # First shot in sequence is always a serve
    if is_first_shot:
        return self.SHOT_TYPES['SERVE']

It was replaced by utils/serve_detector.py, which requires two independent physical
facts: the ball struck above the player's head, and the hitter at or behind a baseline.
This script checks that the replacement actually holds on real clips, because the failure
it guards against is silent.

What the numbers mean
---------------------
A serve claimed on a mid-rally clip is a false positive and is the failure under test.
A serve claimed on a clip that opens with one is correct. The script reports per-clip
counts and the reason each candidate was rejected, so a wrong call is diagnosable rather
than mysterious.

This does NOT prove serve detection is accurate. It only shows how often serves are
claimed, and where. Confirming each claim is right needs hand-labelled shot types, which
tools/label_shots.py produces and this repository does not yet have at scale.

Usage
-----
    python eval/serve_false_positive_check.py
    python eval/serve_false_positive_check.py --clips datasets/eval_clips/*.mp4
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt
from trackers import PlayerTracker
from trackers.tracknet_ball_tracker import TrackNetBallTracker
from utils import assess_court_fit, derive_shot_frames, read_video, stub_path_for_video
from utils.serve_detector import is_serve

DEFAULT_GLOB = "datasets/eval_clips/*.mp4"


def analyse(video_path: str) -> dict:
    frames = read_video(video_path)
    if len(frames) < 10:
        return {"clip": os.path.basename(video_path), "error": "too short"}

    ball_tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    detections = ball_tracker.detect_frames(frames)
    interpolated = ball_tracker.interpolate_ball_positions(detections)

    player_tracker = PlayerTracker(model_path="yolov8x")
    players = player_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path=stub_path_for_video("tracker_stubs/player_detections.pkl", video_path),
    )

    # Per-frame keypoints, matching the pipeline's default (pipeline.per_frame_keypoints).
    # Detecting once on frame 0 and reusing it looks equivalent and is not: on any clip
    # where the camera pans, the frame-0 court no longer lies on the painted lines later
    # in the clip, so the validity gate rejects a court the pipeline would have accepted.
    # That mistake made this script report 5 of 9 clips failing when the pipeline passes
    # most of them, which would have read as a pipeline regression rather than an eval bug.
    court = CourtLineDetector("models/keypoints_model_geoaug.pth")
    all_keypoints = court.predict_all_frames(frames)
    keypoints = all_keypoints[0]
    court_ok, support = assess_court_fit(frames, all_keypoints)

    shots, _bounces, _raw = derive_shot_frames(ball_tracker, interpolated, players)

    mini_court = MiniCourt(frames[0])
    if court_ok:
        player_mini, _ = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            players, interpolated, all_keypoints
        )
    else:
        # Without a trusted court there are no mini-court positions, so the baseline
        # half of the serve test cannot be evaluated and no serve can be claimed. That
        # is the correct outcome, not a gap: refusing to answer beats guessing.
        player_mini = {}

    far_y = mini_court.court_start_y
    near_y = mini_court.court_end_y

    serves, reasons = [], []
    for frame in shots:
        verdict, why = is_serve(frame, interpolated, players, player_mini,
                                far_y, near_y, explain=True)
        if verdict:
            serves.append(frame)
        else:
            reasons.append(why)

    return {
        "clip": os.path.basename(video_path),
        "frames": len(frames),
        "court_ok": court_ok,
        "support": support,
        "shots": len(shots),
        "serves": serves,
        "top_reason": max(set(reasons), key=reasons.count) if reasons else "",
    }


def main():
    p = argparse.ArgumentParser(description="Check for serves claimed on mid-rally clips")
    p.add_argument("--clips", default=DEFAULT_GLOB)
    args = p.parse_args()

    paths = sorted(glob.glob(args.clips))
    if not paths:
        print(f"No clips matched {args.clips}")
        sys.exit(1)

    # Written as well as printed: the detectors emit progress on stderr with carriage
    # returns, which shreds an interleaved table in any captured log.
    out_csv = "output/stats/serve_false_positive_check.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows = []
    total_serves = total_shots = 0
    for path in paths:
        r = analyse(path)
        if r.get("error"):
            rows.append((r["clip"], "", "", "", r["error"]))
            continue
        total_serves += len(r["serves"])
        total_shots += r["shots"]
        rows.append((r["clip"], "ok" if r["court_ok"] else "FAIL", r["shots"],
                     " ".join(str(f) for f in r["serves"]), r["top_reason"]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "court", "shots", "serve_frames", "top_rejection_reason"])
        w.writerows(rows)

    print(f"\n{'clip':<26} {'court':>6} {'shots':>6} {'serves':>7}  most common rejection")
    print("-" * 92)
    for clip, court, shots, serves, reason in rows:
        n = len(serves.split()) if serves else 0
        shown = f"{n}" + (f" @{serves}" if serves else "")
        print(f"{clip:<26} {court:>6} {str(shots):>6} {shown:>7}  {reason}")

    print("-" * 92)
    print(f"Table also written to {out_csv}")
    print(f"{total_serves} serves claimed across {total_shots} shots in {len(paths)} clips.")
    print("\nA serve claimed on a clip cut from mid-rally is the false positive this")
    print("guards against. Cross-check any clip reporting one against the footage.")


if __name__ == "__main__":
    main()
