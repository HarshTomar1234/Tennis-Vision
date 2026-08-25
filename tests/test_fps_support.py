"""
tests/test_fps_support.py
─────────────────────────
Guards the frame-rate support gate.

Why this matters more than it looks: every threshold in the event-detection path is
counted in FRAMES and every classifier velocity is PIXELS PER FRAME, so the same tennis
sampled at a different rate produces a different event set. The measured record for this
project sits entirely between 23.57 and 30.0 fps. Nothing outside that band has ever been
validated, and before this gate nothing said so: a 60 fps clip produced a shot count that
looked exactly like a 30 fps one.

These tests pin the classification, and they pin the two things that make it useful: that
the reason is specific enough to act on, and that a clip whose header could not be read
is not quietly treated as 30 fps.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.fps_support import (
    PARTIALLY_SUPPORTED,
    PARTIAL_MAX_FPS,
    PARTIAL_MIN_FPS,
    SUPPORTED,
    SUPPORTED_MAX_FPS,
    SUPPORTED_MIN_FPS,
    UNSUPPORTED,
    assess_fps,
)


# ── the band every published number was measured on ─────────────────────────────

@pytest.mark.parametrize("fps", [23.57, 23.64, 24.0, 25.0, 29.97, 30.0])
def test_the_measured_record_is_supported(fps):
    """
    These are real rates from this project's own clips: the nine evaluation clips run
    23.57 to 29.82, both input videos are 30.0. If any of them ever stops reading as
    supported, the published numbers no longer describe the footage they came from.
    """
    assert assess_fps(fps).status == SUPPORTED


@pytest.mark.parametrize("fps", [24.0, 25.0, 30.0])
def test_supported_rates_report_is_supported(fps):
    assert assess_fps(fps).is_supported is True


# ── outside it ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fps", [18.0, 20.0, 22.0])
def test_low_but_recoverable_is_partially_supported(fps):
    result = assess_fps(fps)
    assert result.status == PARTIALLY_SUPPORTED
    assert result.is_supported is False


@pytest.mark.parametrize("fps", [33.0, 40.0, 48.0, 50.0])
def test_high_but_recoverable_is_partially_supported(fps):
    assert assess_fps(fps).status == PARTIALLY_SUPPORTED


@pytest.mark.parametrize("fps", [5.0, 10.0, 15.0, 17.9])
def test_far_below_is_unsupported(fps):
    assert assess_fps(fps).status == UNSUPPORTED


@pytest.mark.parametrize("fps", [60.0, 59.94, 90.0, 120.0, 240.0])
def test_far_above_is_unsupported(fps):
    """
    60 fps is ordinary footage and it is genuinely not supported: resampling the
    reference clip to it yields 64% more events per second and a contact/bounce split of
    11:35 on a rally with roughly 14 of each. Saying so is the point of this gate.
    """
    assert assess_fps(fps).status == UNSUPPORTED


# ── boundaries, at the values the table actually measured ───────────────────────

def test_boundaries_are_inclusive():
    assert assess_fps(SUPPORTED_MIN_FPS).status == SUPPORTED
    assert assess_fps(SUPPORTED_MAX_FPS).status == SUPPORTED
    assert assess_fps(PARTIAL_MIN_FPS).status == PARTIALLY_SUPPORTED
    assert assess_fps(PARTIAL_MAX_FPS).status == PARTIALLY_SUPPORTED


def test_just_outside_each_boundary_steps_down_one_level():
    assert assess_fps(SUPPORTED_MIN_FPS - 0.1).status == PARTIALLY_SUPPORTED
    assert assess_fps(SUPPORTED_MAX_FPS + 0.1).status == PARTIALLY_SUPPORTED
    assert assess_fps(PARTIAL_MIN_FPS - 0.1).status == UNSUPPORTED
    assert assess_fps(PARTIAL_MAX_FPS + 0.1).status == UNSUPPORTED


# ── an unreadable header is not 30 fps ──────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, 0, 0.0, -1.0])
def test_unreadable_frame_rate_is_unsupported_not_assumed(bad):
    """
    main.py defaults to 30 fps so the pipeline can still run. That default must not also
    make the clip read as supported, because the thresholds it is about to apply are
    counted in frames and nobody knows the real rate.
    """
    result = assess_fps(bad)
    assert result.status == UNSUPPORTED
    assert "header" in result.reason.lower()


# ── the reason has to be actionable ─────────────────────────────────────────────

def test_reason_says_which_direction_the_error_goes():
    """A caveat that does not say what to expect is decoration."""
    low = assess_fps(20.0).reason.lower()
    high = assess_fps(50.0).reason.lower()

    assert "fewer" in low, "a slow clip under-counts events; the reason should say so"
    assert "more" in high, "a fast clip over-counts events; the reason should say so"


def test_high_rate_reason_admits_it_is_a_lower_bound():
    """
    The high-rate rows come from resampling a 30 fps clip, which invents intermediate
    points a real camera would have measured. That caveat has to survive into the
    user-facing text, or the number reads as an estimate rather than a floor.
    """
    assert "lower bound" in assess_fps(50.0).reason.lower()


# ── the shape written into summary.json ─────────────────────────────────────────

def test_as_dict_carries_what_a_consumer_needs():
    payload = assess_fps(60.0).as_dict()

    assert payload["status"] == UNSUPPORTED
    assert payload["fps"] == 60.0
    assert payload["measured_range_fps"] == [SUPPORTED_MIN_FPS, SUPPORTED_MAX_FPS]
    assert payload["reason"]


def test_as_dict_is_json_serialisable():
    import json

    json.dumps(assess_fps(29.97).as_dict())


# ── never refuses the clip ──────────────────────────────────────────────────────

@pytest.mark.parametrize("fps", [None, 0, 1.0, 23.0, 30.0, 1000.0])
def test_assess_never_raises(fps):
    """
    The gate reports; the caller decides. Refusing to process an unusual clip would be
    worse than processing it with the caveat attached, so this must always return.
    """
    assert assess_fps(fps).status in {SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED}
