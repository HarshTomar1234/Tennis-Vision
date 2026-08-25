"""
tools/label_shots.py
────────────────────
Keyboard-driven labelling tool for tennis contact frames, bounces, and shot types.

Why it works the way it does
----------------------------
Scrubbing 570 frames blind to find 7 contacts is slow and error-prone. The existing
trajectory-reversal detector is already about 5/7 right, so this tool pre-seeds its
candidate list from that detector and lets you jump straight between candidates with
n / p. You confirm, correct, or reject each one. Which candidates you *reject* is itself
useful signal - those are exactly the false positives the detector needs to learn to drop.

Everything is one keypress, no modifiers. Labelling is repetitive; chords slow it down.

Usage:
  python tools/label_shots.py                                  # default clip
  python tools/label_shots.py --video input_videos/clip.mp4
  python tools/label_shots.py --video ... --out datasets/labels/clip.csv

Controls:
  <- ->      step 1 frame              , .    step 10 frames
  n p        next / prev candidate
  space      mark CONTACT here         b      mark BOUNCE here
  1-8        shot type: 1 serve 2 forehand 3 backhand 4 volley
                        5 smash 6 slice 7 drop 8 lob
  tab        toggle player 1 / 2
  u          toggle 'unsure' on the label at this frame
  x          delete label at this frame
  s          save CSV                  q      quit
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from utils import read_video, stub_path_for_video

SHOT_TYPES = {
    ord("1"): "serve",
    ord("2"): "forehand",
    ord("3"): "backhand",
    ord("4"): "volley",
    ord("5"): "smash",
    ord("6"): "slice",
    ord("7"): "drop",
    ord("8"): "lob",
}

FIELDNAMES = ["frame", "event", "player_id", "shot_type", "confidence", "notes"]


def load_candidates(ball_stub: str) -> list[int]:
    """Seed the candidate list from the existing reversal detector, if its output exists."""
    if not Path(ball_stub).exists():
        return []
    try:
        from trackers.tracknet_ball_tracker import TrackNetBallTracker

        with open(ball_stub, "rb") as f:
            detections = pickle.load(f)
        tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
        return tracker.get_ball_shot_frames(
            tracker.interpolate_ball_positions(detections)
        )
    except Exception as exc:  # noqa: BLE001 - candidates are a convenience, not a requirement
        print(f"  (could not seed candidates: {exc})")
        return []


def load_detections(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def load_existing_labels(out_path: str) -> dict[int, dict]:
    """Resume a previous session rather than starting over."""
    labels: dict[int, dict] = {}
    if not Path(out_path).exists():
        return labels
    with open(out_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["frame"])] = {
                "event":      row["event"],
                "player_id":  row["player_id"],
                "shot_type":  row["shot_type"],
                "confidence": row.get("confidence", "sure"),
                "notes":      row.get("notes", ""),
            }
    print(f"  Resumed {len(labels)} existing labels from {out_path}")
    return labels


def save_labels(labels: dict[int, dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for frame in sorted(labels):
            writer.writerow({"frame": frame, **labels[frame]})
    print(f"  Saved {len(labels)} labels -> {out_path}")


def draw_overlay(frame, idx, total, labels, candidates, ball_dets, player_dets, active_player):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    if idx < len(player_dets):
        for pid, bbox in player_dets[idx].items():
            x1, y1, x2, y2 = (int(v) for v in bbox)
            colour = (0, 255, 0) if pid == active_player else (140, 140, 140)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(canvas, f"P{pid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

    if idx < len(ball_dets) and ball_dets[idx].get(1):
        x1, y1, x2, y2 = (int(v) for v in ball_dets[idx][1])
        cv2.circle(canvas, ((x1 + x2) // 2, (y1 + y2) // 2), 10, (0, 255, 255), 2)

    # Header bar
    cv2.rectangle(canvas, (0, 0), (w, 74), (0, 0, 0), -1)
    is_candidate = idx in candidates
    nearest = min((abs(idx - c) for c in candidates), default=None)

    line1 = f"frame {idx}/{total - 1}   active: P{active_player}"
    if is_candidate:
        line1 += "   [CANDIDATE]"
    elif nearest is not None:
        line1 += f"   nearest candidate: {nearest}f away"
    cv2.putText(canvas, line1, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    label = labels.get(idx)
    if label:
        text = f"LABELLED: {label['event']}"
        if label["shot_type"]:
            text += f" / {label['shot_type']}"
        if label["player_id"]:
            text += f" / P{label['player_id']}"
        if label["confidence"] == "unsure":
            text += "  (unsure)"
        colour = (0, 200, 255) if label["confidence"] == "unsure" else (0, 255, 0)
    else:
        text = "unlabelled"
        colour = (150, 150, 150)
    cv2.putText(canvas, text, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)

    cv2.putText(canvas, f"labels: {len(labels)}   [s]ave  [q]uit", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description="Label tennis contact frames and shot types")
    ap.add_argument("--video", default="input_videos/input_video_2.mp4")
    ap.add_argument("--out",   default=None, help="CSV path (default: datasets/labels/<clip>.csv)")
    ap.add_argument("--ball-stub",   default=None,
                     help="default: tracker_stubs/ball_detections_tracknet.pkl, keyed to --video")
    ap.add_argument("--player-stub", default=None,
                     help="default: tracker_stubs/player_detections.pkl, keyed to --video")
    args = ap.parse_args()

    clip_name = Path(args.video).stem
    out_path  = args.out or f"datasets/labels/{clip_name}.csv"

    # Keyed per clip, same convention as every eval script (utils.stub_path_for_video).
    # Unkeyed defaults here previously loaded whichever clip's cache happened to be at
    # the unversioned path, seeding candidates from the wrong video's detections.
    ball_stub = args.ball_stub or stub_path_for_video(
        "tracker_stubs/ball_detections_tracknet.pkl", args.video)
    player_stub = args.player_stub or stub_path_for_video(
        "tracker_stubs/player_detections.pkl", args.video)

    print(f"Loading {args.video} ...")
    frames = read_video(args.video)
    total  = len(frames)

    ball_dets   = load_detections(ball_stub)
    player_dets = load_detections(player_stub)
    candidates  = set(load_candidates(ball_stub))
    labels      = load_existing_labels(out_path)

    print(f"  {total} frames | {len(candidates)} detector candidates | "
          f"{len(labels)} existing labels")
    print("  Controls: n/p candidates, space=contact, b=bounce, 1-8 shot type, "
          "tab=player, u=unsure, x=delete, s=save, q=quit")

    sorted_candidates = sorted(candidates)
    idx = sorted_candidates[0] if sorted_candidates else 0
    active_player = 1
    dirty = False

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow("label", draw_overlay(frames[idx], idx, total, labels, candidates,
                                          ball_dets, player_dets, active_player))
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            if dirty:
                print("  Unsaved changes - press s to save, or q again to discard.")
                if (cv2.waitKey(0) & 0xFF) != ord("q"):
                    continue
            break

        elif key == ord("s"):
            save_labels(labels, out_path)
            dirty = False

        # Navigation
        elif key in (81, ord("a")):          idx = max(0, idx - 1)
        elif key in (83, ord("d")):          idx = min(total - 1, idx + 1)
        elif key == ord(","):                idx = max(0, idx - 10)
        elif key == ord("."):                idx = min(total - 1, idx + 10)
        elif key == ord("n"):
            nxt = [c for c in sorted_candidates if c > idx]
            if nxt: idx = nxt[0]
        elif key == ord("p"):
            prv = [c for c in sorted_candidates if c < idx]
            if prv: idx = prv[-1]

        # Labelling
        elif key == ord(" "):
            labels[idx] = {"event": "contact", "player_id": str(active_player),
                           "shot_type": "", "confidence": "sure", "notes": ""}
            dirty = True
        elif key == ord("b"):
            labels[idx] = {"event": "bounce", "player_id": "", "shot_type": "",
                           "confidence": "sure", "notes": ""}
            dirty = True
        elif key in SHOT_TYPES:
            if idx not in labels:
                labels[idx] = {"event": "contact", "player_id": str(active_player),
                               "shot_type": "", "confidence": "sure", "notes": ""}
            labels[idx]["shot_type"] = SHOT_TYPES[key]
            dirty = True
        elif key == 9:   # tab
            active_player = 2 if active_player == 1 else 1
            if idx in labels and labels[idx]["event"] == "contact":
                labels[idx]["player_id"] = str(active_player)
                dirty = True
        elif key == ord("u"):
            if idx in labels:
                labels[idx]["confidence"] = (
                    "sure" if labels[idx]["confidence"] == "unsure" else "unsure"
                )
                dirty = True
        elif key == ord("x"):
            if labels.pop(idx, None) is not None:
                dirty = True

    cv2.destroyAllWindows()
    if dirty:
        print("  Exited with unsaved changes (nothing written).")


if __name__ == "__main__":
    main()
