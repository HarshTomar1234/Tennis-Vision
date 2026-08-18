"""
eval/_ball_source.py
────────────────────
Shared helper so every eval measures the SAME ball detections the real pipeline
uses (TrackNet, loaded from the cached stub when present - matching main.py).

Previously the evals hardcoded the old YOLO BallTracker, so they graded a worse
configuration than the pipeline actually runs. This keeps them honest and in sync.
"""
from __future__ import annotations

import pickle
from pathlib import Path


def pipeline_ball_detections(
    frames: list,
    video_path: str,
    stub: str = "tracker_stubs/ball_detections_tracknet.pkl",
    model: str = "models/tracknet.pt",
):
    """
    Return (tracker, raw_detections) as the pipeline would produce them.

    Loads TrackNet detections from this clip's cached stub when available (fast, and
    exactly what the pipeline used); otherwise runs TrackNet fresh. Detections are raw
    (pre-interpolation) so callers can interpolate/shot-detect as needed.

    `video_path` is required, not optional. The stub used to be a single shared file, so
    an eval invoked with `--video clip_B.mp4` happily graded clip A's ball detections if
    A had run last: the ground truth came from B, the detections came from A, and the
    resulting accuracy number described no real configuration at all. Making the caller
    name the video is what keys the cache and makes the mismatch check possible.
    """
    from trackers.tracknet_ball_tracker import TrackNetBallTracker
    from utils.video_utils import stub_path_for_video, stub_matches_frames

    tracker = TrackNetBallTracker(model_path=model)
    stub_path = Path(stub_path_for_video(stub, video_path))
    if stub_path.exists():
        with open(stub_path, "rb") as f:
            detections = pickle.load(f)
        if stub_matches_frames(detections, frames):
            return tracker, detections
        print(f"  [stub] {stub_path.name} has {len(detections)} frames, clip has "
              f"{len(frames)} - ignoring stale cache, detecting fresh")

    detections = tracker.detect_frames(frames)
    stub_path.parent.mkdir(exist_ok=True)
    with open(stub_path, "wb") as f:
        pickle.dump(detections, f)
    return tracker, detections
