"""
Tests for utils/viewer_3d.py.

The viewer is a generated artifact, so these check the things that silently break it:
malformed embedded JSON, a missing video reference, coordinates that are not in court
metres, and — most importantly — that an uncalibrated run is not presented as if it
were measured.
"""
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.trajectory_3d import reconstruct_segment
from utils.viewer_3d import COURT_LENGTH_M, COURT_WIDTH_DOUBLES_M, build_viewer

FPS = 25.0


def make_trajectory(start_frame=10, end_frame=25):
    t = reconstruct_segment((1.0, 2.0), (6.0, 14.0), 1.0, 0.0,
                            (end_frame - start_frame) / FPS)
    t.start_frame, t.end_frame = start_frame, end_frame
    return t


def embedded_data(html: str) -> dict:
    """Pull the JSON payload back out of the generated page."""
    match = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert match, "page does not contain an embedded DATA object"
    return json.loads(match.group(1))


class TestGeneratedPage:
    def test_writes_a_self_contained_page(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        html = out.read_text(encoding="utf-8")

        assert out.exists()
        assert html.lstrip().startswith("<!doctype html>")
        # No network dependencies: the page must work offline, from a file:// URL.
        assert "http://" not in html and "https://" not in html
        assert "__DATA__" not in html, "template placeholder was not substituted"

    def test_embeds_valid_json(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        data = embedded_data(out.read_text(encoding="utf-8"))
        assert len(data["segments"]) == 1

    def test_uses_real_court_dimensions(self, tmp_path):
        """The viewer draws a tennis court, not the mini-court's pixel proxy."""
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        court = embedded_data(out.read_text(encoding="utf-8"))["court"]
        assert court["length"] == pytest.approx(COURT_LENGTH_M)
        assert court["width"] == pytest.approx(COURT_WIDTH_DOUBLES_M)
        assert court["singles_width"] < court["width"]

    def test_segment_carries_speed_apex_and_timing(self, tmp_path):
        out = build_viewer([make_trajectory(10, 25)], tmp_path / "v.html", FPS)
        seg = embedded_data(out.read_text(encoding="utf-8"))["segments"][0]

        assert seg["start_frame"] == 10 and seg["end_frame"] == 25
        assert seg["start_s"] == pytest.approx(10 / FPS, abs=1e-3)
        assert seg["speed_kmh"] > 0 and seg["apex_m"] > 0
        assert len(seg["points"]) > 2

    def test_points_are_in_court_metres(self, tmp_path):
        """
        A common failure would be emitting mini-court pixels, which look plausible but
        place the ball tens of metres off court. Points must sit within a sane envelope.
        """
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        for x, y, z in embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["points"]:
            assert -5 <= x <= COURT_WIDTH_DOUBLES_M + 5
            assert -5 <= y <= COURT_LENGTH_M + 5
            assert 0 <= z <= 15

    def test_every_segment_carries_its_evidence(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        seg = embedded_data(out.read_text(encoding="utf-8"))["segments"][0]
        assert seg["evidence"], "a displayed segment must state why it is trusted"

    def test_shot_label_is_applied(self, tmp_path):
        out = build_viewer([make_trajectory(10, 25)], tmp_path / "v.html", FPS,
                           shot_types={10: "Serve"})
        assert embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["label"] == "Serve"


class TestHonesty:
    def test_invalid_court_fit_is_stated_on_the_page(self, tmp_path):
        """
        The rendered video gets a warning banner on an uncalibrated run. A viewer that
        looked identical would undo that, so it must say so too.
        """
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS,
                           court_valid=False)
        note = embedded_data(out.read_text(encoding="utf-8"))["note"]
        assert "FAILED VALIDATION" in note
        assert "not measurements" in note

    def test_valid_run_still_states_the_speed_caveat(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        note = embedded_data(out.read_text(encoding="utf-8"))["note"]
        assert "average over the flight" in note


class TestVideoReference:
    def test_references_video_relatively(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS,
                           video_path="clip.avi")
        assert embedded_data(out.read_text(encoding="utf-8"))["video"] == "clip.avi"

    def test_absent_video_is_null_not_broken(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        assert embedded_data(out.read_text(encoding="utf-8"))["video"] is None


def test_handles_no_segments(tmp_path):
    """An empty rally must still produce a readable page, not a crash."""
    out = build_viewer([], tmp_path / "v.html", FPS)
    assert embedded_data(out.read_text(encoding="utf-8"))["segments"] == []
