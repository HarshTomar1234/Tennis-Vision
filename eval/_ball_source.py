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
    stub: str = "tracker_stubs/ball_detections_tracknet.pkl",
    model: str = "models/tracknet.pt",
):
    """
    Return (tracker, raw_detections) as the pipeline would produce them.

    Loads TrackNet detections from the cached stub when available (fast, and exactly
    what the pipeline used); otherwise runs TrackNet fresh. Detections are raw
    (pre-interpolation) so callers can interpolate/shot-detect as needed.
    """
    from trackers.tracknet_ball_tracker import TrackNetBallTracker

    tracker = TrackNetBallTracker(model_path=model)
    stub_path = Path(stub)
    if stub_path.exists():
        with open(stub_path, "rb") as f:
            detections = pickle.load(f)
    else:
        detections = tracker.detect_frames(frames)
    return tracker, detections
