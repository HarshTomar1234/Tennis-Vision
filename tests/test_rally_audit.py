"""
tests/test_rally_audit.py
─────────────────────────
Tests for the no-ground-truth rally audit.

Every other accuracy number in this project comes from a labelled dataset, which tells a
reader how the pipeline performs on average and nothing about the clip they just uploaded.
This audit answers the second question by using the fact that a rally has impossible
orderings, not merely unlikely ones.

The property under test throughout: it must count only what the sequence PROVES is absent,
never what merely looks odd, because a self-report that cries wolf is worse than none.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rally_audit import audit_rally


def test_clean_alternating_rally_reports_nothing_missing():
    """hit, bounce, hit, bounce, hit by alternating players is a valid rally."""
    contacts = [10, 50, 90]
    bounces = [30, 70]
    hitters = {10: 1, 50: 2, 90: 1}
    a = audit_rally(contacts, bounces, hitter_by_frame=hitters)
    assert a.implied_missing == 0
    assert a.completeness == 1.0
    assert "self-consistent" in a.summary()


def test_same_player_twice_proves_a_missed_contact():
    """The strongest constraint: in singles the ball must go over and come back."""
    contacts = [10, 50]
    bounces = [30]
    a = audit_rally(contacts, bounces, hitter_by_frame={10: 1, 50: 1})
    assert a.missed_contacts == 1
    assert "twice in succession" in a.findings[0]


def test_alternating_players_are_not_flagged():
    contacts = [10, 50]
    bounces = [30]
    a = audit_rally(contacts, bounces, hitter_by_frame={10: 1, 50: 2})
    assert a.implied_missing == 0


def test_two_bounces_in_a_row_prove_a_missed_contact():
    """A second bounce ends the point, so play continuing means a contact was missed."""
    a = audit_rally([10], [30, 50])
    assert a.missed_contacts == 1
    assert "two bounces" in a.findings[0].lower()


def test_two_contacts_with_no_bounce_from_deep_is_a_missed_bounce():
    """A groundstroke follows a bounce; struck from the baseline it cannot be a volley."""
    a = audit_rally([10, 50], [], hitter_by_frame={10: 1, 50: 2},
                    player_side={10: 0.0, 50: 0.0}, net_y=12.0, half_court=12.0)
    assert a.missed_bounces == 1
    assert "too deep to be a volley" in a.findings[0]


def test_two_contacts_with_no_bounce_near_the_net_is_a_volley_and_allowed():
    """
    The check that must not cry wolf. A volley is struck before the bounce by
    definition, so a no-bounce contact near the net is correct play, not a miss.
    """
    a = audit_rally([10, 50], [], hitter_by_frame={10: 1, 50: 2},
                    player_side={10: 11.0, 50: 12.5}, net_y=12.0, half_court=12.0)
    assert a.missed_bounces == 0, a.findings


def test_without_hitter_information_the_strongest_check_is_skipped_not_guessed():
    """No hitter ids means the same-player check cannot run. It must not be faked."""
    a = audit_rally([10, 50], [30])
    assert a.missed_contacts == 0


def test_completeness_is_a_ratio_of_detected_to_implied():
    a = audit_rally([10, 50], [30], hitter_by_frame={10: 1, 50: 1})
    # 3 events detected, 1 proven missing, so 3 of 4.
    assert abs(a.completeness - 0.75) < 1e-9


def test_empty_and_single_event_clips_are_safe():
    empty = audit_rally([], [])
    assert empty.events == 0 and empty.implied_missing == 0
    assert "nothing can be audited" in empty.summary()
    single = audit_rally([10], [])
    assert single.implied_missing == 0


def test_findings_name_the_frames_so_a_person_can_check_them():
    """A self-report that cannot be verified against the video is not worth much."""
    a = audit_rally([10, 50], [], hitter_by_frame={10: 1, 50: 1})
    assert a.findings
    assert "f10" in a.findings[0] and "f50" in a.findings[0]


def test_summary_states_the_count_is_a_lower_bound():
    a = audit_rally([10, 50], [30], hitter_by_frame={10: 1, 50: 1})
    assert "lower bound" in a.summary()


def test_two_adjacent_misses_can_hide_from_this_audit():
    """
    The honest limitation, asserted so nobody later reads completeness as exact.

    Player 1 hits, then player 2's contact AND the bounce after it are both missed, so
    the surviving sequence still alternates and nothing is provable.
    """
    a = audit_rally([10, 90], [30], hitter_by_frame={10: 1, 90: 2})
    assert a.implied_missing == 0
    assert a.completeness == 1.0
