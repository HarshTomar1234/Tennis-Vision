"""
tools/frame_strip.py
─────────────────────
Prints a contact-sheet image of a stretch of frames, for reviewing a gap
`label_shots.py`'s candidate list skips over.

Why this exists
----------------
`n` in label_shots.py jumps between seeded candidates. Candidates come from the
pipeline's own detector, which misses roughly a quarter of contacts and more of the
bounces (see README, "Contact and bounce event detection"). Jumping candidate to
candidate never shows the frames in between, so an event the detector missed is never
seen either, not just never labelled. This dumps every frame in a range as one image so
the whole gap can be scanned by eye at once, instead of arrow-keying through it one frame
at a time inside the GUI tool.

This does not decide anything. It is a faster way to look, not a labeller. The actual
call (contact, bounce, or nothing) still belongs in label_shots.py.

Usage
-----
  # Auto mode: find every gap in the existing labels bigger than --min-gap, one image each
  python tools/frame_strip.py --video datasets/eval_clips/input_video_5.mp4

  # One specific range
  python tools/frame_strip.py --video datasets/eval_clips/input_video_5.mp4 \
      --start 267 --end 315
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from utils import read_video

THUMB_WIDTH = 220
COLUMNS = 5
MAX_THUMBS = 20


def load_labelled_frames(labels_csv: str) -> list[int]:
    if not Path(labels_csv).exists():
        return []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        return sorted(int(row["frame"]) for row in csv.DictReader(f))


def find_gaps(labelled: list[int], total_frames: int, min_gap: int) -> list[tuple[int, int, str]]:
    """(start, end, description) for every stretch nothing was labelled in."""
    gaps = []
    if not labelled:
        return [(0, total_frames - 1, "no labels yet, whole clip")]

    if labelled[0] >= min_gap:
        gaps.append((0, labelled[0], "before the first label"))
    for a, b in zip(labelled, labelled[1:]):
        if b - a >= min_gap:
            gaps.append((a, b, f"between f{a} and f{b}"))
    if total_frames - 1 - labelled[-1] >= min_gap:
        gaps.append((labelled[-1], total_frames - 1, "after the last label"))
    return gaps


def make_contact_sheet(frames: list, indices: list[int]) -> np.ndarray:
    thumbs = []
    for idx in indices:
        frame = frames[idx]
        h, w = frame.shape[:2]
        scale = THUMB_WIDTH / w
        thumb = cv2.resize(frame, (THUMB_WIDTH, int(h * scale)))
        cv2.rectangle(thumb, (0, 0), (THUMB_WIDTH, 22), (0, 0, 0), -1)
        cv2.putText(thumb, f"f{idx}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1)
        thumbs.append(thumb)

    rows = []
    for i in range(0, len(thumbs), COLUMNS):
        row = thumbs[i:i + COLUMNS]
        if len(row) < COLUMNS:
            blank = np.zeros_like(row[0])
            row += [blank] * (COLUMNS - len(row))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def save_strip(video_frames: list, start: int, end: int, out_path: str) -> int:
    span = end - start + 1
    step = max(1, span // MAX_THUMBS)
    indices = list(range(start, end + 1, step))
    if indices[-1] != end:
        indices.append(end)

    sheet = make_contact_sheet(video_frames, indices)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, sheet)
    return len(indices)


def main() -> None:
    ap = argparse.ArgumentParser(description="Contact-sheet image of a frame range")
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", default=None,
                     help="default: datasets/labels/<clip>.csv")
    ap.add_argument("--min-gap", type=int, default=40,
                     help="auto mode: only image gaps at least this many frames wide")
    ap.add_argument("--start", type=int, default=None, help="manual mode: first frame")
    ap.add_argument("--end", type=int, default=None, help="manual mode: last frame")
    ap.add_argument("--out-dir", default=None,
                     help="default: datasets/labels/_review/<clip>/")
    args = ap.parse_args()

    clip_name = Path(args.video).stem
    labels_csv = args.labels or f"datasets/labels/{clip_name}.csv"
    out_dir = args.out_dir or f"datasets/labels/_review/{clip_name}"

    print(f"Loading {args.video} ...")
    frames = read_video(args.video)
    total = len(frames)
    print(f"  {total} frames")

    if args.start is not None and args.end is not None:
        gaps = [(args.start, args.end, "manual range")]
    else:
        labelled = load_labelled_frames(labels_csv)
        gaps = find_gaps(labelled, total, args.min_gap)
        print(f"  {len(labelled)} existing labels in {labels_csv}, "
              f"{len(gaps)} gap(s) >= {args.min_gap} frames")

    if not gaps:
        print("  Nothing to image: no gap reaches --min-gap. Lower it with --min-gap N "
              "if you still want to check a smaller stretch.")
        return

    for start, end, desc in gaps:
        out_path = f"{out_dir}/gap_{start}-{end}.png"
        n = save_strip(frames, start, end, out_path)
        print(f"  f{start}-f{end} ({desc}), {n} frames shown -> {out_path}")


if __name__ == "__main__":
    main()
