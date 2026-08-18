"""
Tests for utils/viewer_3d.py.

The viewer is a generated artifact, so these check the things that silently break it:
malformed embedded JSON, a missing video reference, coordinates that are not in court
metres, and - most importantly - that an uncalibrated run is not presented as if it
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
        html = out.read_text(encoding="utf-8")
        assert embedded_data(html)["court_valid"] is False
        # The page must actually render the warning, not merely carry the flag.
        assert "FAILED" in html and "NOT measurements" in html

    def test_valid_run_still_states_the_speed_caveat(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        data = embedded_data(out.read_text(encoding="utf-8"))
        assert data["court_valid"] is True
        assert "average over the flight" in data["note"]


class TestEvidenceIsReal:
    """
    The first version emitted three identical hardcoded strings for every segment,
    including "endpoints on the floor" - which was false for 14 of 16 segments in a
    real run, because a contact endpoint is 0.9-2.6 m up by construction. Evidence
    that the payload contradicts is worse than no evidence, so these tests pin it to
    the actual numbers.
    """

    def test_evidence_reports_the_real_endpoint_heights(self, tmp_path):
        # Contact at 1.0 m down to a bounce at 0.0 m.
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS)
        evidence = " ".join(embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["evidence"])
        assert "racket contact" in evidence, "a raised endpoint must not be called floor"
        assert "on the floor" in evidence, "the bounce endpoint should be named as floor"

    def test_evidence_states_the_measured_duration(self, tmp_path):
        out = build_viewer([make_trajectory(10, 25)], tmp_path / "v.html", FPS)
        evidence = " ".join(embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["evidence"])
        assert f"{(25 - 10) / FPS:.2f} s" in evidence

    def test_evidence_differs_between_different_segments(self, tmp_path):
        """Identical evidence for every segment is the bug this class exists to stop."""
        a = make_trajectory(10, 25)
        b = reconstruct_segment((0.5, 1.0), (9.0, 20.0), 0.0, 0.0, 0.8)
        b.start_frame, b.end_frame = 40, 60
        out = build_viewer([a, b], tmp_path / "v.html", FPS)
        segs = embedded_data(out.read_text(encoding="utf-8"))["segments"]
        assert segs[0]["evidence"] != segs[1]["evidence"]

    def test_out_of_court_landing_is_labelled_out(self, tmp_path):
        t = reconstruct_segment((1.0, 2.0), (2.0, 27.0), 1.0, 0.0, 0.9)  # past baseline
        t.start_frame, t.end_frame = 10, 30
        out = build_viewer([t], tmp_path / "v.html", FPS)
        evidence = " ".join(embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["evidence"])
        assert "outside the lines" in evidence

    def test_net_crossing_is_reported(self, tmp_path):
        t = reconstruct_segment((5.0, 2.0), (5.0, 20.0), 1.0, 0.0, 0.8)   # crosses
        t.start_frame, t.end_frame = 10, 30
        out = build_viewer([t], tmp_path / "v.html", FPS)
        evidence = " ".join(embedded_data(out.read_text(encoding="utf-8"))["segments"][0]["evidence"])
        assert "crosses the net" in evidence


class TestRobustness:
    def test_payload_cannot_break_out_of_the_script_block(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS,
                           shot_types={10: "</script><script>alert(1)</script>"})
        html = out.read_text(encoding="utf-8")
        assert "</script><script>alert" not in html

    def test_absolute_video_path_is_reduced_to_a_name(self, tmp_path):
        out = build_viewer([make_trajectory()], tmp_path / "v.html", FPS,
                           video_path=r"D:\some\where\clip.mp4")
        assert embedded_data(out.read_text(encoding="utf-8"))["video"] == "clip.mp4"

    def test_nan_fails_loudly_rather_than_silently(self, tmp_path):
        """Bare NaN is invalid JSON but valid JS: the page would load and the arc
        would vanish into undefined coordinates."""
        t = make_trajectory()
        t.points[0] = (float("nan"), 0.0, 0.0)
        with pytest.raises(ValueError):
            build_viewer([t], tmp_path / "v.html", FPS)


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


# ── player ground positions ──────────────────────────────────────────────────
# Players were absent from the viewer entirely: it drew ball arcs, a court and a net,
# with nobody in it. Their FEET are the one position the floor homography places exactly,
# since a standing player is on the ground by definition, so unlike ball height these
# coordinates are measured rather than modelled.

from utils.viewer_3d import COURT_LENGTH_M, COURT_WIDTH_DOUBLES_M, players_to_metres


def test_mini_court_origin_maps_to_court_origin():
    """A player at the mini-court origin is at (0, 0) metres on the real court."""
    out = players_to_metres({0: {1: (100.0, 200.0)}}, 100.0, 200.0, 0.05)
    assert out["0"]["1"] == (0.0, 0.0)


def test_pixels_convert_to_metres_by_the_given_scale():
    out = players_to_metres({5: {2: (140.0, 260.0)}}, 100.0, 200.0, 0.05)
    x, y = out["5"]["2"]
    assert abs(x - 2.0) < 1e-6, f"x was {x}"
    assert abs(y - 3.0) < 1e-6, f"y was {y}"


def test_position_far_outside_the_court_is_dropped():
    """
    A tracking failure puts a player in the crowd, and drawing it would assert that
    someone stood there. Out-of-range positions are dropped rather than clamped.
    """
    out = players_to_metres({0: {1: (100.0, 200.0), 2: (10000.0, 200.0)}},
                            100.0, 200.0, 0.05)
    assert "1" in out["0"]
    assert "2" not in out["0"]


def test_a_little_outside_the_lines_is_kept():
    """Players legitimately stand behind the baseline and wide of the sidelines."""
    beyond = (COURT_LENGTH_M + 1.5) / 0.05 + 200.0
    out = players_to_metres({0: {1: (100.0, beyond)}}, 100.0, 200.0, 0.05)
    assert "1" in out.get("0", {}), "a player two metres behind the baseline was dropped"


def test_frames_with_no_usable_player_are_omitted():
    out = players_to_metres({0: {1: (99999.0, 99999.0)}}, 100.0, 200.0, 0.05)
    assert out == {}


def test_none_position_is_skipped_without_raising():
    out = players_to_metres({0: {1: None, 2: (100.0, 200.0)}}, 100.0, 200.0, 0.05)
    assert out["0"] == {"2": (0.0, 0.0)}


def test_empty_input():
    assert players_to_metres({}, 0.0, 0.0, 0.05) == {}
    assert players_to_metres(None, 0.0, 0.0, 0.05) == {}


def test_players_omitted_from_payload_when_the_court_failed(tmp_path):
    """
    Without a trusted court these coordinates mean nothing, so the viewer must not
    receive them. This is the same refusal the rest of the pipeline makes.
    """
    out = tmp_path / "v.html"
    build_viewer([], out, fps=30.0, court_valid=False,
                 players_m={"0": {"1": (5.0, 10.0)}})
    html = out.read_text(encoding="utf-8")
    assert '"players": null' in html or '"players":null' in html


def test_players_reach_the_payload_when_the_court_is_valid(tmp_path):
    out = tmp_path / "v.html"
    build_viewer([], out, fps=30.0, court_valid=True,
                 players_m={"0": {"1": (5.0, 10.0)}})
    html = out.read_text(encoding="utf-8")
    assert '"players"' in html
    assert "5.0" in html and "10.0" in html
