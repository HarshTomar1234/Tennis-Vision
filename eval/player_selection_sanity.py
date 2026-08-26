"""
eval/player_selection_sanity.py
────────────────────────────────
Did the pipeline pick the two players, and did it keep hold of them?

Why this exists
---------------
The README says plainly that player detection has no ground-truth evaluation, and that is
still true: this is not one. Precision, recall, IDF1 and ID-switch counts need labelled
boxes per frame, which do not exist for this footage, and producing them is post-launch
work.

What can be checked without any labels is whether the selection produced something that
could possibly be a tennis match. Three of these are decisive on their own:

- **Two players, on opposite sides of the net.** Singles is played across the net. If both
  selected tracks sit on the same half for most of the clip, the selection has taken a
  spectator, a line judge or the umpire instead of a player. No ground truth needed: the
  geometry of the sport settles it.
- **Coverage.** The share of frames where both, one or neither player is present. Players
  are on court for essentially the whole rally, so a low figure means the tracker is
  dropping them.
- **The longest continuous gap.** Coverage averages hide the shape of the loss. Ten
  scattered missing frames and one 200-frame hole are the same average and completely
  different problems, and only the second breaks the shot attribution the rally grammar
  depends on.

Also reported, without a pass or fail, because they are context rather than verdicts:
how many distinct people YOLO found (the pool the selection chose from), and how many
separate track fragments each chosen player was assembled from.

Method
------
Reads the cached player detections and runs the real `select_two_players`, so what is
graded is the selection the product ships. Detections are checked against the clip's
frame count first: a cache written by a truncated run describes a different video.

Usage
-----
    python eval/player_selection_sanity.py
    python eval/player_selection_sanity.py --clips "datasets/eval_clips/*.mp4"
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from court_line_detector import CourtLineDetector
from trackers import PlayerTracker
from utils import read_video, select_two_players, stub_path_for_video

DEFAULT_GLOB = "datasets/eval_clips/*.mp4"
COURT_MODEL = "models/keypoints_model_geoaug.pth"

# Share of frames a player must be present for before the clip is worth trusting. Not a
# tuned threshold: it is the level below which shot attribution starts failing, because
# the rally grammar's same-player rule needs a player to exist at the contact frame.
MIN_COVERAGE = 0.80


def load_player_detections(video: str, frame_count: int):
    """Cached detections for this clip, or None if the cache does not describe it."""
    path = Path(stub_path_for_video("tracker_stubs/player_detections.pkl", video))
    if not path.exists():
        return None
    with open(path, "rb") as f:
        detections = pickle.load(f)
    if len(detections) != frame_count:
        print(f"    cache has {len(detections)} frames, clip has {frame_count}: ignoring")
        return None
    return detections


def longest_gap(present: list[bool]) -> int:
    """Longest run of consecutive frames where the player was absent."""
    worst = run = 0
    for is_present in present:
        run = 0 if is_present else run + 1
        worst = max(worst, run)
    return worst


def fragments(detections: list[dict], track_id: int) -> int:
    """How many separate continuous stretches this id was seen in."""
    count = 0
    was_present = False
    for frame in detections:
        is_present = track_id in frame
        if is_present and not was_present:
            count += 1
        was_present = is_present
    return count


def analyse(video: str, court: CourtLineDetector, tracker: PlayerTracker) -> dict | None:
    frames = read_video(video)
    if not frames:
        return None

    raw = load_player_detections(video, len(frames))
    if raw is None:
        return None

    keypoints = court.predict(frames[0])
    selected, id_map = select_two_players(tracker, raw, keypoints)

    # Net line in image space, midway between the two baselines. Keypoint 0 is the far
    # baseline's left corner and keypoint 2 the near baseline's, in the 14-point
    # convention the court model was trained on.
    net_y = (keypoints[1] + keypoints[5]) / 2.0

    people_seen = len({tid for frame in raw for tid in frame})

    stats = {
        "clip": Path(video).name,
        "frames": len(frames),
        "people_detected": people_seen,
        "players_selected": len(id_map),
        "original_track_ids": list(id_map),
    }

    both = sum(1 for f in selected if len(f) == 2)
    one = sum(1 for f in selected if len(f) == 1)
    stats["coverage_both"] = both / len(frames)
    stats["coverage_none"] = (len(frames) - both - one) / len(frames)

    sides = {}
    for pid in (1, 2):
        present = [pid in f for f in selected]
        seen = sum(present)
        stats[f"p{pid}_coverage"] = seen / len(frames)
        stats[f"p{pid}_longest_gap"] = longest_gap(present)
        stats[f"p{pid}_fragments"] = sum(
            fragments(raw, tid) for tid, new in id_map.items() if new == pid
        )
        feet = [f[pid][3] for f in selected if pid in f]     # bbox bottom = feet
        sides[pid] = (sorted(feet)[len(feet) // 2] if feet else None)

    # The decisive check. Both players on the same side of the net means the selection
    # took someone who is not a player.
    near, far = sides[1], sides[2]
    stats["opposite_sides"] = (
        near is not None and far is not None and (near - net_y) * (far - net_y) < 0
    )
    stats["median_feet_y"] = (near, far)
    stats["net_y"] = net_y
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--clips", default=DEFAULT_GLOB)
    args = ap.parse_args()

    videos = sorted(glob.glob(args.clips))
    if not videos:
        print(f"No clips matched {args.clips}. See datasets/README.md.")
        return 1

    court = CourtLineDetector(COURT_MODEL)
    tracker = PlayerTracker.__new__(PlayerTracker)   # selection only, no YOLO needed

    rows = []
    for video in videos:
        print(f"  {Path(video).name} ...", flush=True)
        result = analyse(video, court, tracker)
        if result:
            rows.append(result)

    if not rows:
        print("\nNo clip had a usable player cache. Run the pipeline on them first.")
        return 1

    print(f"\n{'clip':<24} {'ppl':>4} {'sel':>4} {'both%':>7} {'p1%':>6} {'p2%':>6} "
          f"{'gap1':>5} {'gap2':>5} {'frag':>5} {'sides':>6}")
    print("-" * 82)
    for r in rows:
        print(f"{r['clip']:<24} {r['people_detected']:>4} {r['players_selected']:>4} "
              f"{r['coverage_both']:>6.0%} {r['p1_coverage']:>5.0%} {r['p2_coverage']:>5.0%} "
              f"{r['p1_longest_gap']:>5} {r['p2_longest_gap']:>5} "
              f"{r['p1_fragments'] + r['p2_fragments']:>5} "
              f"{'ok' if r['opposite_sides'] else 'SAME':>6}")

    print("\nFailures:")
    problems = 0
    for r in rows:
        issues = []
        if r["players_selected"] < 2:
            issues.append(f"only {r['players_selected']} player selected")
        if not r["opposite_sides"]:
            issues.append("both selected tracks on the SAME side of the net, so at "
                          "least one is not a player")
        for pid in (1, 2):
            if r[f"p{pid}_coverage"] < MIN_COVERAGE:
                issues.append(f"player {pid} present on only "
                              f"{r[f'p{pid}_coverage']:.0%} of frames")
            if r[f"p{pid}_longest_gap"] > r["frames"] * 0.1:
                issues.append(f"player {pid} missing for {r[f'p{pid}_longest_gap']} "
                              f"consecutive frames")
        if issues:
            problems += 1
            print(f"  {r['clip']}: " + "; ".join(issues))
    if not problems:
        print("  none")

    print(f"\n{len(rows) - problems} of {len(rows)} clips pass every sanity check.")
    print("This is a confidence check, not an accuracy measurement. Precision, recall, "
          "IDF1 and ID switches need labelled boxes that do not exist for this footage; "
          "the README says so and that remains true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
