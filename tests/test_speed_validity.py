"""
tests/test_speed_validity.py
─────────────────────────────
Guards the 3-D speed validity classification.

The problem it solves: a reference-clip run published `shot_speed_3d_kmh` with a range of
18 to 170 km/h. 18 km/h is not a struck tennis ball, and the obvious fix was a minimum
speed threshold. That would have been wrong. Three explanations were tested against the
25 real segments first:

1. "Endpoints outside the baseline mark a bad segment." REFUTED: 18 of 25 have one,
   including the three fastest, because a contact endpoint is the player's FEET.
2. "The slow ones never cross the net." True, and not a defect: a ball that bounces on
   the receiver's side and is then struck by the receiver legitimately stays on one side.
3. "Shot speed is aggregating legs that are not shots." The real cause. Only 14 of 25
   segments begin at a racket contact; the other 11 are post-bounce legs.

So the classification is about what KIND of measurement a segment is, not about whether
its number looks plausible. These tests pin that distinction, because a future speed
threshold would quietly satisfy the symptom and lose the reasoning.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from utils.trajectory_3d import (
    NOT_A_SHOT,
    OUTLIER,
    PLAUSIBLE_BUT_UNCERTAIN,
    SHORT_FLIGHT_S,
    VALID,
    classify_segment_speed,
)


def classify(starts_at_contact=True, ends_at_bounce=True, crosses=True, duration=0.8):
    return classify_segment_speed(starts_at_contact, ends_at_bounce, crosses, duration)


# ── a shot is a ball leaving a racket ───────────────────────────────────────────

def test_contact_to_bounce_across_the_net_is_a_valid_shot():
    status, reason = classify()
    assert status == VALID
    assert reason


def test_a_segment_starting_at_a_bounce_is_not_a_shot():
    """
    The post-bounce leg: real geometry, correctly reconstructed, and not a ball leaving
    a racket. Averaging these into "shot speed" is what produced the 17.6 km/h reading.
    """
    status, reason = classify(starts_at_contact=False)
    assert status == NOT_A_SHOT
    assert "bounce" in reason


def test_not_a_shot_takes_precedence_over_flight_length():
    """A short post-bounce leg is still not a shot, not an uncertain shot."""
    assert classify(starts_at_contact=False, duration=0.2)[0] == NOT_A_SHOT


def test_contact_to_contact_across_the_net_is_valid():
    """A volley taken before the bounce: no bounce endpoint, and it crossed."""
    assert classify(ends_at_bounce=False)[0] == VALID


# ── the one genuine outlier ─────────────────────────────────────────────────────

def test_struck_and_landing_on_the_strikers_own_side_is_an_outlier():
    """
    Struck, and bouncing on the striker's own side without crossing the net. In tennis
    that is a ball into the net or a mishit, and either way the free-flight assumption
    the reconstruction depends on is broken. On real footage it usually means the contact
    was misdetected. This is the single segment flagged on the reference clip.
    """
    status, reason = classify(crosses=False)
    assert status == OUTLIER
    assert "net" in reason


def test_a_non_crossing_segment_that_does_not_end_in_a_bounce_is_not_an_outlier():
    """
    The rule is specifically about landing on your own side. A contact-to-contact
    same-side pair is already rejected upstream by main.py's net-crossing gate, so this
    must not duplicate that judgement with different reasoning.
    """
    assert classify(ends_at_bounce=False, crosses=False)[0] != OUTLIER


# ── uncertainty comes from measured timing sensitivity ──────────────────────────

def test_a_short_flight_is_marked_uncertain():
    """
    Not a plausibility judgement on the speed. eval/speed_timing_sensitivity.py measures
    that a 2.4-frame event offset moves speed by 23.9% on flights under 0.5 s against
    6.5% over 1.0 s, so a short flight's speed is real but must not read as precise.
    """
    status, reason = classify(duration=SHORT_FLIGHT_S - 0.01)
    assert status == PLAUSIBLE_BUT_UNCERTAIN
    assert "2.4-frame" in reason


def test_a_long_flight_is_valid():
    assert classify(duration=SHORT_FLIGHT_S + 0.01)[0] == VALID


def test_short_flight_threshold_matches_the_measured_regime():
    """
    The boundary is the one the sensitivity measurement reports on, not a round number
    picked afterwards. If this moves, the reason string stops describing the evidence.
    """
    assert SHORT_FLIGHT_S == 0.5


# ── the classification is exhaustive and stable ─────────────────────────────────

@pytest.mark.parametrize("starts", [True, False])
@pytest.mark.parametrize("ends", [True, False])
@pytest.mark.parametrize("crosses", [True, False])
@pytest.mark.parametrize("duration", [0.1, 0.49, 0.5, 1.5])
def test_every_combination_returns_a_known_status_and_a_reason(
    starts, ends, crosses, duration
):
    status, reason = classify_segment_speed(starts, ends, crosses, duration)
    assert status in {VALID, PLAUSIBLE_BUT_UNCERTAIN, OUTLIER, NOT_A_SHOT}
    assert reason, "every status must explain itself; a bare status is not evidence"


def test_only_valid_and_uncertain_should_carry_a_published_speed():
    """
    Documents the contract main.py and viewer_3d.py both rely on: these two statuses mean
    "this number is a shot speed", the others mean "do not print a number here".
    """
    publishable = {VALID, PLAUSIBLE_BUT_UNCERTAIN}
    assert classify()[0] in publishable
    assert classify(duration=0.2)[0] in publishable
    assert classify(starts_at_contact=False)[0] not in publishable
    assert classify(crosses=False)[0] not in publishable
