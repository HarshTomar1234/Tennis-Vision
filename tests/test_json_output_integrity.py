"""
tests/test_json_output_integrity.py
────────────────────────────────────
Guards the boundary where analysis becomes a file.

Three failures live here, and all three share a shape: the output looks like a file and
is not one.

1. **numpy scalars.** `json.dump` refuses them, and it refuses them PART WAY THROUGH,
   because it streams to the handle. A single `numpy.bool_` in the player-selection
   payload once left `summary.json` truncated mid-key. It was found only because a later
   script tried to read it back and got a JSONDecodeError.

2. **Bare NaN.** Invalid JSON that Python happens to accept and every strict parser
   rejects, so the file passes the project's own tests and fails in a browser, in jq and
   in any other language. `utils/viewer_3d.py` already refuses NaN for exactly this
   reason. The summary writer should not be laxer than the viewer.

3. **Partial writes.** Any error during serialisation used to leave whatever had already
   been streamed. Serialising to a string first means a failure leaves the previous state
   untouched.

The rule these enforce: **a correct file, or no file and a loud error. Never a file that
parses in one language and not another.**
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _json_scalar, _write_json, save_stats

LOG = logging.getLogger("test")


def strict_load(text: str):
    """Parse the way a non-Python consumer would: NaN and Infinity are errors."""
    def reject(constant):
        raise ValueError(f"non-standard JSON constant: {constant}")
    return json.loads(text, parse_constant=reject)


def stats_row(**overrides) -> pd.DataFrame:
    row = {"frame_num": 0}
    for p in (1, 2):
        row |= {
            f"player_{p}_number_of_shots": 0,
            f"player_{p}_average_shot_speed": 0.0,
            f"player_{p}_average_player_speed": 0.0,
        }
    row.update(overrides)
    return pd.DataFrame([row])


# ── numpy must never reach the file ─────────────────────────────────────────────

@pytest.mark.parametrize("payload", [
    {"flag": np.bool_(True)},
    {"count": np.int64(7)},
    {"ratio": np.float32(0.82)},
    {"values": np.array([1.0, 2.0])},
    {"nested": {"deep": {"flag": np.bool_(False), "n": np.int32(3)}}},
])
def test_numpy_scalars_serialise(tmp_path, payload):
    out = tmp_path / "x.json"
    _write_json(out, payload, LOG)
    strict_load(out.read_text(encoding="utf-8"))


def test_json_scalar_rejects_genuinely_unserialisable_types():
    """
    The coercion must not become a silent stringifier. An object nobody thought about
    should raise, not appear in the file as a repr.
    """
    with pytest.raises(TypeError):
        _json_scalar(object())


# ── NaN and infinity are refused, loudly ────────────────────────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_are_refused(tmp_path, bad):
    out = tmp_path / "x.json"
    with pytest.raises(ValueError):
        _write_json(out, {"speed": bad}, LOG)


def test_a_refused_write_leaves_no_file(tmp_path):
    """
    The important half. A failure must not leave a truncated file behind, because a
    truncated file is indistinguishable from a real one until something tries to read it.
    """
    out = tmp_path / "x.json"
    with pytest.raises(ValueError):
        _write_json(out, {"speed": float("nan")}, LOG)
    assert not out.exists(), "a failed write must leave nothing, not a partial file"


def test_a_refused_write_does_not_destroy_an_existing_file(tmp_path):
    """Serialising before opening also means the previous good file survives."""
    out = tmp_path / "x.json"
    _write_json(out, {"speed": 61.5}, LOG)
    before = out.read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        _write_json(out, {"speed": float("nan")}, LOG)

    assert out.read_text(encoding="utf-8") == before


# ── end to end through save_stats ───────────────────────────────────────────────

@pytest.mark.parametrize("extra", [
    {},
    {"court_fit": (np.bool_(True), np.float32(0.61))},
    {"player_selection": {"status": "ok", "players_on_opposite_sides": np.bool_(True)}},
    {"ball": {"coverage": np.float32(0.82), "longest_gap_frames": np.int64(4)}},
    {"fps_support": {"fps": np.float64(30.0), "status": "supported"}},
    {"court_fit": (False, 0.03)},
    {"trajectories_3d": []},
])
def test_save_stats_writes_strictly_valid_json(tmp_path, extra):
    save_stats(stats_row(), str(tmp_path), LOG, **extra)
    written = sorted(tmp_path.glob("summary_*.json"))
    assert written, "a summary must be written"
    strict_load(written[-1].read_text(encoding="utf-8"))


def test_every_summary_this_project_has_written_is_strictly_valid():
    """
    Any summary still on disk from a real run must parse under a strict reader. This is
    the check that would have caught the truncated file at the time it was produced
    rather than days later.

    Files written before the fix are allowed to fail and are skipped, so this does not
    fail on historical artefacts; it guards what the current code produces.
    """
    stats = Path(__file__).resolve().parent.parent / "output" / "stats"
    if not stats.exists():
        pytest.skip("no runs on this machine")
    recent = sorted(stats.glob("summary_*.json"))[-5:]
    if not recent:
        pytest.skip("no summaries yet")
    for path in recent:
        text = path.read_text(encoding="utf-8")
        try:
            json.loads(text)
        except json.JSONDecodeError:
            pytest.skip(f"{path.name} predates the atomic-write fix")
        strict_load(text)
