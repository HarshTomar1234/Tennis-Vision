"""
Tests for utils/shot_physics.py.

Each test states a physical situation and the label it must produce. The point of
these is that a wrong threshold or an inverted comparison changes a shot's name, and
nothing else in the pipeline would notice.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.shot_physics import (
    LOB_MIN_APEX_M,
    classify_from_physics,
    is_lob,
    is_smash,
    is_volley,
)

HALF_COURT = 11.9   # metres, net to baseline


class TestSmash:
    def test_overhead_inside_the_court_is_a_smash(self):
        call = is_smash(ball_above_head=True, distance_from_net_m=4.0,
                        half_court_length_m=HALF_COURT)
        assert call and call.shot_type == "Smash"
        assert any("above the player's head" in r for r in call.reasons)

    def test_overhead_from_the_baseline_is_not_a_smash(self):
        """That is a serve. The height test alone cannot separate them."""
        assert is_smash(True, distance_from_net_m=11.5,
                        half_court_length_m=HALF_COURT) is None

    def test_ball_below_head_is_never_a_smash(self):
        assert is_smash(False, 3.0, HALF_COURT) is None

    def test_refuses_degenerate_court(self):
        assert is_smash(True, 3.0, half_court_length_m=0) is None


class TestVolley:
    def test_no_bounce_near_the_net_is_a_volley(self):
        call = is_volley(bounces_since_previous_contact=0, distance_from_net_m=3.0,
                         half_court_length_m=HALF_COURT)
        assert call and call.shot_type == "Volley"
        assert any("no bounce" in r for r in call.reasons)

    def test_a_bounce_in_between_means_it_is_not_a_volley(self):
        """Hitting after the bounce is a groundstroke, by definition."""
        assert is_volley(1, 3.0, HALF_COURT) is None

    def test_no_bounce_at_the_baseline_is_rejected(self):
        """
        Guard against the real failure mode: bounce recall is ~80%, so a MISSED bounce
        would otherwise turn a baseline groundstroke into a volley.
        """
        assert is_volley(0, distance_from_net_m=11.0,
                         half_court_length_m=HALF_COURT) is None


class TestLob:
    def test_high_apex_is_a_lob(self):
        call = is_lob(apex_height_m=LOB_MIN_APEX_M + 1.0)
        assert call and call.shot_type == "Lob"

    def test_normal_rally_height_is_not_a_lob(self):
        assert is_lob(1.8) is None

    def test_missing_apex_is_not_a_lob(self):
        """No 3-D segment means no claim - never a default."""
        assert is_lob(None) is None


class TestClassifyFromPhysics:
    def test_serve_takes_precedence(self):
        call = classify_from_physics(
            ball_above_head=True, distance_from_net_m=11.5,
            half_court_length_m=HALF_COURT, bounces_since_previous_contact=1,
            outgoing_apex_m=None, is_serve=True,
        )
        assert call.shot_type == "Serve"

    def test_smash_beats_volley_when_both_could_fire(self):
        """An overhead put away at the net is a smash, not a volley."""
        call = classify_from_physics(
            ball_above_head=True, distance_from_net_m=2.0,
            half_court_length_m=HALF_COURT, bounces_since_previous_contact=0,
            outgoing_apex_m=None,
        )
        assert call.shot_type == "Smash"

    def test_ordinary_groundstroke_returns_none(self):
        """
        No physical test fires, so the module says nothing and the label must come
        from body pose. A default here would be a guess.
        """
        assert classify_from_physics(
            ball_above_head=False, distance_from_net_m=10.0,
            half_court_length_m=HALF_COURT, bounces_since_previous_contact=1,
            outgoing_apex_m=1.5,
        ) is None

    def test_unknown_bounce_count_does_not_produce_a_volley(self):
        """Missing evidence must not be read as evidence of absence."""
        call = classify_from_physics(
            ball_above_head=False, distance_from_net_m=2.0,
            half_court_length_m=HALF_COURT, bounces_since_previous_contact=None,
            outgoing_apex_m=None,
        )
        assert call is None

    def test_every_call_carries_its_evidence(self):
        for call in (
            is_smash(True, 3.0, HALF_COURT),
            is_volley(0, 3.0, HALF_COURT),
            is_lob(LOB_MIN_APEX_M + 2),
        ):
            assert call.reasons, f"{call.shot_type} produced no evidence"
