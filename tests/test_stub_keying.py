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
