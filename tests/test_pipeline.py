"""
tests/test_pipeline.py

Unit + integration tests for Tennis-Vision.

Run:
  pytest tests/                     # all tests
  pytest tests/ -m "not slow"       # skip heavy integration tests
  pytest tests/ -v                  # verbose

Markers:
  slow  - tests that load video / run YOLO (require model files + stubs)
"""
import os
import sys
import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

# Ensure repo root is on path when pytest is run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).parent.parent

# Weights the end-to-end smoke test genuinely cannot run without. They are downloaded
# rather than committed (two of them are not ours to redistribute), so a fresh clone has
# none of them and the smoke test used to fail with a subprocess traceback that said
# nothing about the cause. It now skips with the command that fixes it.
#
# This is a skip, not a silent pass: pytest reports it by name with this reason, and CI
# deselects it explicitly with -m "not slow" rather than letting it disappear quietly.
REQUIRED_WEIGHTS = (
    "models/tracknet.pt",
    "models/keypoints_model_geoaug.pth",
    "models/pose_landmarker_lite.task",
)


def _missing_weights() -> list[str]:
    return [w for w in REQUIRED_WEIGHTS if not (REPO_ROOT / w).exists()]


# ─────────────────────────────────────────────────────────────────
# Config tests
# ─────────────────────────────────────────────────────────────────

class TestConfigLoading:
    def test_defaults_when_no_file(self, tmp_path):
        from main import load_config
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert cfg["pipeline"]["per_frame_keypoints"] is True
        assert cfg["shot_classifier"]["volley_distance_threshold"] == 40
        assert cfg["io"]["output_stats_dir"] == "output/stats"

    def test_user_overrides_defaults(self, tmp_path):
        from main import load_config
        import yaml
        override = {"shot_classifier": {"volley_distance_threshold": 60}}
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump(override))

        cfg = load_config(str(cfg_file))
        assert cfg["shot_classifier"]["volley_distance_threshold"] == 60
        # Non-overridden key should still be default
        assert cfg["pipeline"]["use_bytetrack"] is True

    def test_cli_overrides_config(self, tmp_path):
        """Simulate --no-stubs and --fast CLI flags."""
        from main import load_config
        cfg = load_config(str(tmp_path / "none.yaml"))
        cfg["stubs"]["use_player_stubs"] = False
        cfg["stubs"]["use_ball_stubs"]   = False
        cfg["pipeline"]["per_frame_keypoints"] = False
        cfg["pipeline"]["use_bytetrack"]       = False
        assert cfg["stubs"]["use_player_stubs"] is False
        assert cfg["pipeline"]["per_frame_keypoints"] is False


# ─────────────────────────────────────────────────────────────────
# ShotClassifier tests
# ─────────────────────────────────────────────────────────────────

class TestShotClassifier:
    def setup_method(self):
        from utils import ShotClassifier
        self.clf = ShotClassifier(
            volley_threshold=40, smash_height_threshold=0.7, net_y_relative=0.5
        )

    def test_first_shot_is_serve(self):
        result = self.clf._determine_shot_type(
            i=0, player_id=1, player_y=200,
            ball_trajectory_y=50, mini_court_height=400, is_first_shot=True
        )
        assert result == "Serve"

    def test_player_near_net_is_volley(self):
        # net_y = 400 * 0.5 = 200; player_y = 210 → |210-200| = 10 < 40 → Volley
        result = self.clf._determine_shot_type(
            i=1, player_id=1, player_y=210,
            ball_trajectory_y=10, mini_court_height=400, is_first_shot=False
        )
        assert result == "Volley"

    def test_player_far_from_net_is_not_volley(self):
        # player_y = 350 → |350-200| = 150 > 40 → not Volley
        result = self.clf._determine_shot_type(
            i=1, player_id=1, player_y=350,
            ball_trajectory_y=10, mini_court_height=400, is_first_shot=False
        )
        assert result != "Volley"

    def test_shot_color_title_case(self):
        """All legend lookups use Title Case - must return non-white."""
        white = (255, 255, 255)
        for shot_type in ("Serve", "Forehand", "Backhand", "Volley", "Smash"):
            color = self.clf.get_shot_color(shot_type)
            assert color != white, f"get_shot_color('{shot_type}') returned white - key mismatch"

    def test_shot_color_case_insensitive(self):
        """Lowercase lookup must also work (regression guard)."""
        for shot_type in ("serve", "forehand", "backhand", "volley", "smash"):
            color = self.clf.get_shot_color(shot_type)
            assert color != (255, 255, 255)

    # ── every detected contact must be represented ──────────────────────────────
    #
    # classify_shots iterated range(len - 1), because measuring the ball's vertical
    # travel needs the NEXT contact. The final contact of every clip was therefore never
    # classified: no shot type, no pose upgrade, no entry in the result. On the reference
    # clip that is 14 classifications for 15 detected contacts, and main.py's statistics
    # loop had the identical off-by-one, so the published shot count was one low too.

    @staticmethod
    def _positions(frames, court_height=400):
        """Two players and a ball on the mini-court at each of `frames`."""
        players = {f: {1: (100.0, 350.0), 2: (100.0, 50.0)} for f in frames}
        ball    = {f: {1: (100.0, 200.0 + 10.0 * i)} for i, f in enumerate(frames)}
        return players, ball

    def test_every_detected_shot_is_classified(self):
        frames = [10, 40, 70, 100, 130]
        players, ball = self._positions(frames)

        result = self.clf.classify_shots(players, ball, frames, 400)

        assert len(result) == len(frames), (
            f"{len(frames)} contacts detected but {len(result)} classified: the last "
            f"shot is being dropped"
        )
        assert set(result) == set(frames)

    def test_final_shot_is_classified(self):
        """Named separately: it is the one the off-by-one silently removed."""
        frames = [10, 40, 70]
        players, ball = self._positions(frames)

        result = self.clf.classify_shots(players, ball, frames, 400)

        assert frames[-1] in result, "the final contact must carry a shot type"
        assert result[frames[-1]]["shot_type"]

    def test_a_lone_contact_is_still_a_shot(self):
        """The same off-by-one from the other end: a one-contact clip reported none."""
        frames = [42]
        players, ball = self._positions(frames)

        result = self.clf.classify_shots(players, ball, frames, 400)

        assert len(result) == 1

    def test_no_contacts_classifies_nothing(self):
        assert self.clf.classify_shots({}, {}, [], 400) == {}

    def test_old_volley_threshold_would_fail(self):
        """Prove that threshold=150 (old broken value) caught mid-court as volley."""
        broken_clf = __import__("utils").ShotClassifier(volley_threshold=150)
        # player_y=300 in a 400px court → |300-200|=100 < 150 → Volley (wrong!)
        result = broken_clf._determine_shot_type(
            i=1, player_id=1, player_y=300,
            ball_trajectory_y=10, mini_court_height=400, is_first_shot=False
        )
        assert result == "Volley", "This test documents the old bug - should still fire with threshold=150"


# ─────────────────────────────────────────────────────────────────
# Homography tests
# ─────────────────────────────────────────────────────────────────

class TestHomography:
    """Test homography computation without requiring video files."""

    def _make_dummy_frame(self, w=1280, h=720):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_homography_computes_without_crash(self):
        from mini_visual_court import MiniCourt
        frame = self._make_dummy_frame()
        mc = MiniCourt(frame)

        # Synthetic keypoints that form a plausible tennis court in a 1280×720 frame
        video_kp = [
            100, 100,   # pt 0: top-left
            1180, 100,  # pt 1: top-right
            100, 620,   # pt 2: bottom-left
            1180, 620,  # pt 3: bottom-right
            300, 100, 980, 100,   # pt 4, 5
            300, 620, 980, 620,   # pt 6, 7
            300, 280, 980, 280,   # pt 8, 9
            300, 440, 980, 440,   # pt 10, 11
            640, 280,             # pt 12
            640, 440,             # pt 13
        ]
        H = mc.compute_homography(video_kp)
        # May be None if RANSAC rejects (dummy court geometry vs drawing_key_points)
        # but should not throw an exception
        assert H is None or H.shape == (3, 3)

    def test_apply_homography_returns_finite(self):
        import cv2
        from mini_visual_court import MiniCourt
        frame = self._make_dummy_frame()
        mc = MiniCourt(frame)

        # Identity homography
        H = np.eye(3, dtype=np.float32)
        x, y = mc.apply_homography(H, (640, 360))
        assert abs(x - 640) < 1
        assert abs(y - 360) < 1


# ─────────────────────────────────────────────────────────────────
# Stats save tests
# ─────────────────────────────────────────────────────────────────

class TestStatsSave:
    def test_save_creates_csv_and_json(self, tmp_path):
        import pandas as pd
        from main import save_stats
        import logging
        log = logging.getLogger("test")

        df = pd.DataFrame({
            "frame_num": range(10),
            "player_1_number_of_shots": [0] * 10,
            "player_1_total_shot_speed": [0.0] * 10,
            "player_1_last_shot_speed": [0.0] * 10,
            "player_1_total_player_speed": [0.0] * 10,
            "player_1_last_player_speed": [0.0] * 10,
            "player_2_number_of_shots": [0] * 10,
            "player_2_total_shot_speed": [0.0] * 10,
            "player_2_last_shot_speed": [0.0] * 10,
            "player_2_total_player_speed": [0.0] * 10,
            "player_2_last_player_speed": [0.0] * 10,
            "player_1_average_shot_speed": [0.0] * 10,
            "player_2_average_shot_speed": [0.0] * 10,
            "player_1_average_player_speed": [0.0] * 10,
            "player_2_average_player_speed": [0.0] * 10,
        })

        save_stats(df, str(tmp_path), log)

        csvs  = list(tmp_path.glob("stats_*.csv"))
        jsons = list(tmp_path.glob("summary_*.json"))
        assert len(csvs)  == 1, "Expected exactly one CSV file"
        assert len(jsons) == 1, "Expected exactly one JSON file"

        with open(jsons[0]) as f:
            summary = json.load(f)
        assert "total_shots_p1" in summary
        assert "avg_shot_speed_p1_kmh" in summary


# ─────────────────────────────────────────────────────────────────
# Integration test (requires model files + stubs - mark as slow)
# ─────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_full_pipeline_smoke(tmp_path):
    """
    End-to-end smoke test: runs the real pipeline on a short slice of video and
    checks that an output video is produced.

    Deliberately does NOT rely on cached detection stubs. Stubs are gitignored (they
    are per-video caches, and shipping one makes a newcomer analyse their clip with
    another clip's detections), so a test that needed them would pass only on a
    machine that had already run the pipeline - exactly the machine where a break is
    least likely to be noticed. --max-frames keeps a fresh-detection run fast enough
    to stay a smoke test.

    Requires the model weights: python scripts/download_models.py
    """
    missing = _missing_weights()
    if missing:
        pytest.skip(
            "end-to-end smoke test needs the downloadable model weights, missing: "
            + ", ".join(missing)
            + ". Fetch them with: python scripts/download_models.py"
        )

    import subprocess, shutil

    out_video = str(tmp_path / "smoke_output.avi")
    out_dir   = str(tmp_path / "stats")

    result = subprocess.run(
        [
            sys.executable, "main.py",
            "--input",  "input_videos/input_video_2.mp4",
            "--output", out_video,
            "--fast",             # first-frame keypoints, no ByteTrack
            "--no-stubs",         # fresh detection; never depend on a cache
            "--max-frames", "40",
        ],
        capture_output=True, text=True, timeout=600
    )

    # Pipeline should exit 0
    assert result.returncode == 0, f"Pipeline crashed:\n{result.stderr}"

    # Output video must exist and be > 0 bytes
    assert os.path.exists(out_video), "Output video not created"
    assert os.path.getsize(out_video) > 0, "Output video is empty"


# ─────────────────────────────────────────────────────────────────
# The stats HUD must not print a NaN at the viewer
# ─────────────────────────────────────────────────────────────────
#
# Caught on a launch demo frame: player 2 had not hit yet, so the stats frame carried NaN
# for their shot type, and formatting it straight into the panel printed the literal
# string "nan" on screen. A rendered frame is what gets screenshotted and shared, and
# "nan" reads as a broken measurement rather than an absent one. Every other absent value
# in this project is shown as absent.

@pytest.mark.parametrize("absent", [float("nan"), None, "", "nan", "NaN", "None"])
def test_absent_shot_type_renders_as_a_dash(absent):
    from utils.player_stats_drawer_utils import _shot_type_or_dash
    assert _shot_type_or_dash(absent) == "-"


@pytest.mark.parametrize("real", ["Serve", "Forehand", "Backhand", "Groundstroke"])
def test_a_real_shot_type_is_passed_through(real):
    """The guard must not swallow genuine labels."""
    from utils.player_stats_drawer_utils import _shot_type_or_dash
    assert _shot_type_or_dash(real) == real
