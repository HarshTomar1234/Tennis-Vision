"""
tests/test_rally_decode.py
──────────────────────────
Tests for constrained rally decoding.

The hit/bounce classifier is 86.4% held out, so about one event in seven is wrong. Alone
that is respectable; in a sequence it produces rallies that cannot happen, because the
errors compound into orderings tennis forbids. This decoder chooses the most likely
labelling that obeys the rules.

The properties that matter: it must repair an impossible ordering, it must NOT overrule
the classifier when the sequence is already legal, and it must never invent an event.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.rally_decode import BOUNCE, CONTACT, decode_rally


def test_legal_rally_is_left_alone():
    """The decoder must not meddle when the classifier already agrees with the rules."""
    events = [
        (10, 0.95, "far"),     # contact by far
        (30, 0.05, "far"),     # bounce
        (50, 0.95, "near"),    # contact by near
        (70, 0.05, "near"),    # bounce
        (90, 0.95, "far"),     # contact by far
    ]
    r = decode_rally(events)
    assert r.contacts == [10, 50, 90]
    assert r.bounces == [30, 70]
    assert r.flips == []
    assert "already a legal rally" in r.summary()


def test_same_player_twice_is_repaired_by_relabelling_the_weaker_event():
    """
    The case that motivated this.

    Two contacts by the far player in a row cannot happen. The second is only 60%
    confident, so relabelling it a bounce is both legal and more likely than the
    impossible alternative.
    """
    events = [
        (10, 0.97, "far"),
        (30, 0.60, "far"),     # classifier says contact, but that is impossible here
        (50, 0.97, "near"),
    ]
    r = decode_rally(events)
    assert r.contacts == [10, 50]
    assert r.bounces == [30]
    assert len(r.flips) == 1
    assert "f30" in r.flips[0]


def test_a_bounce_between_does_not_make_the_same_player_legal():
    """
    hit, bounce, hit by ONE player means the ball never crossed the net.

    The constraint has to survive across the bounce, which is why the decoder carries
    the last striking side through bounce states.
    """
    events = [
        (10, 0.97, "far"),
        (30, 0.02, "far"),     # a confident bounce
        (50, 0.65, "far"),     # far again: impossible, however the bounce is labelled
    ]
    r = decode_rally(events)
    assert 50 in r.bounces, f"impossible second contact was allowed: {r.contacts}"


def test_two_bounces_in_a_row_are_repaired():
    """A second bounce ends the point, so it cannot be followed by more play."""
    events = [
        (10, 0.95, "far"),
        (30, 0.05, "far"),
        (50, 0.40, "near"),    # classifier leans bounce, but bounce-bounce is illegal
        (70, 0.95, "far"),
    ]
    r = decode_rally(events)
    assert 50 in r.contacts
    assert len(r.flips) == 1


def test_volley_is_allowed_two_contacts_with_no_bounce():
    """Different players may hit in succession: that is a volley, not an error."""
    events = [
        (10, 0.95, "far"),
        (30, 0.95, "near"),    # volley, no bounce between
        (50, 0.95, "far"),
    ]
    r = decode_rally(events)
    assert r.contacts == [10, 30, 50]
    assert r.bounces == []
    assert r.flips == []


def test_no_event_is_ever_invented_or_dropped():
    """Decoding relabels; it must not change how many events exist."""
    events = [(10, 0.9, "far"), (30, 0.3, "far"), (50, 0.8, "far"), (70, 0.2, "near")]
    r = decode_rally(events)
    assert r.events == len(events)
    assert sorted(r.contacts + r.bounces) == [10, 30, 50, 70]


def test_unknown_side_does_not_trigger_the_same_player_rule():
    """
    When the side could not be determined, the constraint is skipped rather than guessed.

    Two confident contacts with no side information must both survive.
    """
    events = [(10, 0.95, None), (30, 0.95, None)]
    r = decode_rally(events)
    assert r.contacts == [10, 30]


def test_flips_record_the_confidence_that_was_overruled():
    """A decoder fighting a confident classifier must be visible, not silent."""
    events = [(10, 0.99, "far"), (30, 0.88, "far"), (50, 0.99, "near")]
    r = decode_rally(events)
    assert r.flips
    assert "88%" in r.flips[0]


def test_empty_and_single_event():
    assert decode_rally([]).events == 0
    single = decode_rally([(10, 0.9, "far")])
    assert single.contacts == [10] and single.flips == []


def test_a_certain_classifier_still_yields_a_legal_sequence():
    """
    Probabilities of 1.0 must not make a legal path infinitely expensive.

    Without a probability floor, log(0) on the alternative would be infinite and the
    decoder could be forced into an impossible ordering or a crash.
    """
    events = [(10, 1.0, "far"), (30, 1.0, "far"), (50, 1.0, "near")]
    r = decode_rally(events)
    assert r.events == 3
    # The impossible repeat must have been broken somewhere.
    assert r.contacts != [10, 30, 50]


def test_long_alternating_rally_stays_stable():
    """Twenty legal events should decode unchanged, not drift."""
    events = []
    frame = 0
    for i in range(10):
        side = "far" if i % 2 == 0 else "near"
        events.append((frame, 0.93, side)); frame += 20
        events.append((frame, 0.07, side)); frame += 20
    r = decode_rally(events)
    assert r.flips == []
    assert len(r.contacts) == 10 and len(r.bounces) == 10


# ── Discarding spurious candidates ────────────────────────────────────────────
# Relabelling alone measured WORSE than not decoding at all (false positives 7 -> 10 on
# the labelled reference clip), because both rules resolve a conflict by pushing an event
# into the other class, and half the candidates are not events at all. The NOISE state is
# the fix: it lets the decoder drop a candidate instead of promoting it.

def test_deletion_prior_zero_keeps_the_old_behaviour():
    """The NOISE state must be off by default, so callers opt in deliberately."""
    events = [(10, 0.97, "far"), (30, 0.97, "far"), (50, 0.97, "near")]
    r = decode_rally(events, deletion_prior=0.0)
    assert r.discarded == []
    assert r.events == 3


def test_a_spurious_candidate_is_discarded_rather_than_promoted():
    """
    The case that made relabelling harmful.

    Two confident contacts by the far player with a confident bounce between them. The
    ball cannot have stayed on one side, so something here is not real. Discarding the
    middle candidate is a better explanation than insisting it was a contact.
    """
    events = [(10, 0.99, "far"), (30, 0.02, "far"), (50, 0.99, "far")]
    r = decode_rally(events, deletion_prior=0.15)
    assert r.discarded, f"nothing discarded: {r.contacts} {r.bounces}"
    assert r.events == 2


def test_discarding_never_touches_a_legal_rally():
    """A sequence that already obeys the rules must survive a non-zero prior intact."""
    events = [(10, 0.95, "far"), (30, 0.05, "far"), (50, 0.95, "near"), (70, 0.05, "near")]
    r = decode_rally(events, deletion_prior=0.15)
    assert r.discarded == []
    assert r.contacts == [10, 50] and r.bounces == [30, 70]


def test_a_high_prior_discards_everything_which_is_why_it_is_swept():
    """
    The failure mode at the top of the range, asserted so nobody raises the prior casually.

    Above about 0.5 the constant cost of calling a candidate spurious beats the cost of
    any real label, so the whole clip decodes to noise. The sweep against labels showed
    exactly this: recall fell to zero at 0.50.
    """
    events = [(10, 0.99, "far"), (30, 0.01, "far"), (50, 0.99, "near")]
    assert decode_rally(events, deletion_prior=0.9).events == 0


def test_discarded_frames_are_reported_not_silently_dropped():
    """A dropped candidate has to be visible, or the decoder is hiding its own decisions."""
    events = [(10, 0.99, "far"), (30, 0.02, "far"), (50, 0.99, "far")]
    r = decode_rally(events, deletion_prior=0.15)
    assert set(r.discarded) <= {10, 30, 50}
    assert "discarded as spurious" in r.summary()


def test_every_candidate_is_accounted_for_exactly_once():
    """Contacts, bounces and discards must partition the input, with nothing invented."""
    events = [(10, 0.9, "far"), (30, 0.3, "far"), (50, 0.8, "far"),
              (70, 0.2, "near"), (90, 0.6, "near")]
    r = decode_rally(events, deletion_prior=0.15)
    accounted = sorted(r.contacts + r.bounces + r.discarded)
    assert accounted == [f for f, _, _ in events]


def test_an_impossible_dead_end_does_not_delete_the_events_before_it():
    """
    Regression: the no-legal-continuation branch used to start a fresh trail, which
    dropped everything decoded before it. One clip went from 22 events to 1.

    Reachable only with discarding off, because any non-zero deletion_prior always offers
    a legal NOISE continuation. Both rules are dead-ended here: the run of confident
    same-side contacts forces a bounce state, and then another same-side contact arrives
    with nowhere legal to go.
    """
    events = [(10, 0.99, 1), (30, 0.01, 1), (50, 0.99, 1), (70, 0.01, 1),
              (90, 0.99, 1), (110, 0.01, 1), (130, 0.99, 1)]
    r = decode_rally(events, deletion_prior=0.0)
    assert r.events == len(events), (
        f"events were deleted with discarding off: {r.contacts} {r.bounces}"
    )
    assert sorted(r.contacts + r.bounces) == [f for f, _, _ in events]


def test_no_events_are_lost_at_any_prior():
    """Every candidate must be accounted for, whether or not discarding is available."""
    events = [(f, 0.99 if i % 2 == 0 else 0.01, 1) for i, f in enumerate(range(0, 300, 20))]
    for prior in (0.0, 0.02, 0.05, 0.15, 0.3):
        r = decode_rally(events, deletion_prior=prior)
        total = len(r.contacts) + len(r.bounces) + len(r.discarded)
        assert total == len(events), f"prior {prior} lost events: {total} of {len(events)}"


# ─────────────────────────────────────────────────────────────────
# Observability: the decoder's own warning condition must escape the log
# ─────────────────────────────────────────────────────────────────
#
# rally_decode's module docstring says that repeatedly overruling a CONFIDENT classifier
# means the candidate forcing the repair was probably never an event, or the classifier
# is wrong on this footage. On the reference clip the grammar overrules at 90% and 94%,
# so that warning condition genuinely fires. It used to exist only as a debug string, so
# it reached nobody. These pin it into the machine-readable output.

from utils.rally_decode import HIGH_CONFIDENCE_FLIP


def test_diagnostics_report_a_clean_decode():
    """A sequence the grammar accepts unchanged must not raise a warning."""
    events = [(10, 0.95, 1), (30, 0.05, None), (50, 0.95, 2)]

    diagnostics = decode_rally(events, deletion_prior=0.0).diagnostics()

    assert diagnostics["relabelled"] == 0
    assert diagnostics["discarded_as_spurious"] == 0
    assert diagnostics["relabelled_over_high_confidence"] == 0
    assert "warning" not in diagnostics


def test_a_high_confidence_override_is_counted_and_warned():
    """
    Two contacts by the same side in a row is impossible, so the grammar must repair it.
    Both are asserted at 95%, well above the classifier's own 86.4% held-out accuracy,
    so whichever one it relabels is an override of a confident call.
    """
    events = [(10, 0.95, 1), (30, 0.95, 1)]

    decoded = decode_rally(events, deletion_prior=0.0)
    diagnostics = decoded.diagnostics()

    assert diagnostics["relabelled"] >= 1
    assert diagnostics["relabelled_over_high_confidence"] >= 1
    assert diagnostics["high_confidence_frames"]
    assert "warning" in diagnostics
    assert "caution" in diagnostics["warning"].lower()


def test_a_low_confidence_override_is_not_warned():
    """
    The grammar resolving an uncertain call is the intended behaviour and must not
    produce a warning, or the warning becomes noise nobody reads.
    """
    events = [(10, 0.52, 1), (30, 0.52, 1)]

    diagnostics = decode_rally(events, deletion_prior=0.0).diagnostics()

    assert diagnostics["relabelled"] >= 1
    assert diagnostics["relabelled_over_high_confidence"] == 0
    assert "warning" not in diagnostics


def test_flip_confidences_are_recorded_for_every_flip():
    events = [(10, 0.95, 1), (30, 0.95, 1)]
    decoded = decode_rally(events, deletion_prior=0.0)

    assert len(decoded.flip_confidences) == len(decoded.flips)
    for frame, confidence in decoded.flip_confidences:
        assert 0.0 <= confidence <= 1.0


def test_high_confidence_threshold_sits_above_the_classifier_accuracy():
    """
    The point of the threshold is "surer than it has any right to be". The hit/bounce
    classifier measures 86.4% held out, so anything below that is not a confident call.
    """
    assert HIGH_CONFIDENCE_FLIP >= 0.86


def test_diagnostics_are_json_serialisable():
    import json

    json.dumps(decode_rally([(10, 0.95, 1), (30, 0.95, 1)],
                            deletion_prior=0.02).diagnostics())


def test_derive_shot_frames_notes_carry_diagnostics():
    """
    derive_shot_frames returns notes as a list subclass so six existing callers keep
    unpacking a 4-tuple. If that ever becomes a plain list again, main.py silently stops
    reporting decoder diagnostics, which is exactly the failure this feature fixes.
    """
    from utils.hit_bounce_classifier import DecodeNotes

    notes = DecodeNotes(["a flip"])
    notes.diagnostics = {"relabelled": 1}

    assert isinstance(notes, list)
    assert list(notes) == ["a flip"]
    assert notes.diagnostics["relabelled"] == 1
