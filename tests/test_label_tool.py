"""
tests/test_label_tool.py
CSV round-trip and label-state tests for the labelling tool. The interactive OpenCV loop
can't run headless, but everything that touches data can — and that's the part where a
bug would silently corrupt a dataset.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.label_shots import load_existing_labels, save_labels, SHOT_TYPES, FIELDNAMES


def test_csv_round_trip_is_identical():
    labels = {
        27:  {"event": "contact", "player_id": "1", "shot_type": "serve",
              "confidence": "sure", "notes": ""},
        63:  {"event": "bounce",  "player_id": "",  "shot_type": "",
              "confidence": "sure", "notes": "near baseline"},
        110: {"event": "contact", "player_id": "2", "shot_type": "backhand",
              "confidence": "unsure", "notes": "occluded"},
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.csv")
        save_labels(labels, path)
        reloaded = load_existing_labels(path)

    assert reloaded == labels


def test_load_returns_empty_for_missing_file():
    with tempfile.TemporaryDirectory() as d:
        assert load_existing_labels(os.path.join(d, "nope.csv")) == {}


def test_save_creates_parent_directories():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "deeper", "clip.csv")
        save_labels({1: {"event": "contact", "player_id": "1", "shot_type": "forehand",
                         "confidence": "sure", "notes": ""}}, path)
        assert os.path.exists(path)


def test_frames_are_written_in_order():
    labels = {
        300: {"event": "contact", "player_id": "1", "shot_type": "forehand",
              "confidence": "sure", "notes": ""},
        27:  {"event": "contact", "player_id": "2", "shot_type": "serve",
              "confidence": "sure", "notes": ""},
        150: {"event": "bounce",  "player_id": "",  "shot_type": "",
              "confidence": "sure", "notes": ""},
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "clip.csv")
        save_labels(labels, path)
        with open(path, encoding="utf-8") as f:
            frames = [line.split(",")[0] for line in f.read().splitlines()[1:]]

    assert frames == ["27", "150", "300"]


def test_shot_type_keys_cover_the_documented_set():
    """The docstring, the schema, and the keymap must not drift apart."""
    assert set(SHOT_TYPES.values()) == {
        "serve", "forehand", "backhand", "volley", "smash", "slice", "drop", "lob"
    }
    assert FIELDNAMES == ["frame", "event", "player_id", "shot_type", "confidence", "notes"]
