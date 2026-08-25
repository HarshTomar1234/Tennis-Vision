"""
tests/test_stub_keying.py
─────────────────────────
Guards the detection cache against serving one clip's detections to another.

The bug these cover shipped silently: `tracker_stubs/ball_detections_tracknet.pkl` was
a single shared file with no record of which video wrote it. Analysing clip A and then
clip B gave B the ball positions of A, and every number computed downstream - speeds,
bounce frames, shot frames - described the wrong video while looking entirely normal.
The evals in eval/ read the same shared stub, so a clip could be *graded* on detections
that never came from it.

Two independent guards, tested separately here because they catch different mistakes:
filename keying separates different clips, and the frame-count check catches the case
keying cannot see (same filename, different cut, e.g. a --max-frames run).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_utils import stub_path_for_video, stub_matches_frames

BASE = "tracker_stubs/ball_detections_tracknet.pkl"


def test_different_videos_get_different_stubs():
    """The whole point: two clips must never share one cache file."""
    a = stub_path_for_video(BASE, "input_videos/clip_05_wimbledon.mp4")
    b = stub_path_for_video(BASE, "input_videos/clip_08_roland_garros.mp4")
    assert a != b


def test_stub_keeps_directory_and_extension():
    """Callers still mkdir the parent and expect a .pkl, so the shape must survive."""
    path = stub_path_for_video(BASE, "input_videos/clip_05.mp4")
    assert path.startswith("tracker_stubs/")
    assert path.endswith(".pkl")


def test_stub_name_carries_the_clip_name():
    """A human debugging a stale cache needs to see which clip a file belongs to."""
    path = stub_path_for_video(BASE, "input_videos/clip_05_wimbledon.mp4")
    assert "clip_05_wimbledon" in path


def test_same_video_is_stable_across_calls_and_directories():
    """Caching is worthless if the key moves; the clip's own path prefix must not leak in."""
    assert (stub_path_for_video(BASE, "input_videos/clip_05.mp4")
            == stub_path_for_video(BASE, "datasets/eval_clips/clip_05.mp4"))


def test_video_extension_does_not_change_the_key():
    """A clip transcoded .avi to .mp4 is the same clip, and both name it identically."""
    assert (stub_path_for_video(BASE, "clips/rally.avi")
            == stub_path_for_video(BASE, "clips/rally.mp4"))


def test_frame_count_mismatch_is_rejected():
    """The guard filename keying cannot provide: same name, different cut."""
    assert not stub_matches_frames([{}] * 300, [None] * 570)


def test_matching_frame_count_is_accepted():
    assert stub_matches_frames([{}] * 570, [None] * 570)


def test_empty_stub_against_real_frames_is_rejected():
    """A truncated or half-written cache must not pass as valid."""
    assert not stub_matches_frames([], [None] * 570)


# ─────────────────────────────────────────────────────────────────
# Truncated runs must not WRITE a cache
# ─────────────────────────────────────────────────────────────────
#
# The two guards above are both on the READ side, and they were enough for every reader
# that used them. The write side had no guard at all: a --max-frames run cached its
# detections under the full clip's name, producing a file that describes 40 frames and
# claims to describe 570.
#
# That is not hypothetical. The end-to-end smoke test runs with --max-frames 40, so
# `pytest tests/` was itself sufficient to leave a truncated cache behind. Guarded
# readers recovered. tools/label_shots.py did not, and it is the tool that produces the
# ground truth every accuracy number in this project is measured against.

import pickle

import pytest

from trackers.player_tracker import PlayerTracker


def test_player_tracker_skips_caching_when_asked(tmp_path, monkeypatch):
    """save_stub=False must leave no cache file behind, however detection went."""
    tracker = PlayerTracker.__new__(PlayerTracker)
    monkeypatch.setattr(PlayerTracker, "detect_frame", lambda self, frame: {1: [0, 0, 1, 1]})

    stub = tmp_path / "player_detections.pkl"
    out = tracker.detect_frames([None, None, None], stub_path=str(stub), save_stub=False)

    assert len(out) == 3, "detection itself must still happen"
    assert not stub.exists(), (
        "a truncated run wrote a cache keyed to the whole clip, which is the defect"
    )


def test_player_tracker_caches_by_default(tmp_path, monkeypatch):
    """The opposite case, so the guard above cannot be satisfied by never caching."""
    tracker = PlayerTracker.__new__(PlayerTracker)
    monkeypatch.setattr(PlayerTracker, "detect_frame", lambda self, frame: {1: [0, 0, 1, 1]})

    stub = tmp_path / "player_detections.pkl"
    tracker.detect_frames([None, None, None], stub_path=str(stub))

    assert stub.exists()
    with open(stub, "rb") as f:
        assert len(pickle.load(f)) == 3


def test_label_tool_refuses_a_stub_of_the_wrong_length(tmp_path, capsys):
    """
    The labelling tool must not seed from a cache that does not describe this clip.

    Silence is the danger here: fewer candidates looks like a quiet clip, not like a
    wrong input, so the tool has to say so.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "tools"))
    from tools.label_shots import load_detections

    stub = tmp_path / "ball.pkl"
    with open(stub, "wb") as f:
        pickle.dump([{1: [0, 0, 1, 1]}] * 40, f)     # a --max-frames 40 run

    out = load_detections(str(stub), total_frames=570, what="Ball positions")

    assert out == [], "a 40-frame cache must not be used for a 570-frame clip"
    assert "IGNORING" in capsys.readouterr().out, "the refusal has to be visible"


def test_label_tool_accepts_a_matching_stub(tmp_path):
    from tools.label_shots import load_detections

    stub = tmp_path / "ball.pkl"
    with open(stub, "wb") as f:
        pickle.dump([{1: [0, 0, 1, 1]}] * 570, f)

    assert len(load_detections(str(stub), total_frames=570, what="Ball")) == 570
