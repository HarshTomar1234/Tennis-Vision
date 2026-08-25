"""
utils/rally_decode.py
─────────────────────
Relabels a rally's events using the sequence rules of tennis, not just the classifier.

The problem this solves
-----------------------
The hit/bounce classifier looks at one event at a time and measures 86.4% held out, so
roughly one event in seven is wrong. Independently that is a respectable number. In a
sequence it is not, because the errors produce rallies that cannot physically happen: the
rally self-audit reports one player hitting five times in succession on the reference
clip, which no shot selection or rule of tennis produces.

Flagging that is useful. Fixing it is better, and the information needed is already
present: a rally is a grammar. Some orderings are legal, some are impossible, and the
classifier emits a probability rather than a hard label. Choosing the most likely
labelling that obeys the grammar is a decoding problem, solved exactly by Viterbi over a
small state space. It is the same technique that lets a language model repair character
errors in OCR, applied to a sport instead of a sentence.

The grammar
-----------
State is (last label, which side last struck the ball).

    contact(near) -> contact(near)   IMPOSSIBLE, the ball never crossed the net
    contact(far)  -> contact(far)    IMPOSSIBLE, likewise
    contact(X)    -> contact(Y)      legal, a volley taken before the bounce
    contact(X)    -> bounce          legal, the normal case
    bounce        -> contact(X)      legal, a groundstroke
    bounce        -> bounce          IMPOSSIBLE mid-rally, a second bounce ends the point

Which side a contact came from is data, not a choice: it is read from where the event
happened. Only the contact-or-bounce label is decided here.

Relabelling alone is not enough, and the labels say so
------------------------------------------------------
The first version of this decoder could only repair an impossible ordering by FLIPPING a
label. Measured against the labelled reference clip it made things worse: false positives
went from 7 to 10 with no recall gain. The reason is structural rather than a tuning
problem. Both rules resolve a conflict by relabelling, so every repair pushes an event
into the other class, and on a candidate set where half the candidates are not real events
at all (recall 100%, precision 41%) those repairs land on noise.

So the grammar needs a third option: a candidate can be spurious. A candidate labelled
NOISE leaves the rally state untouched, which lets the decoder resolve an impossible
ordering by DISCARDING a candidate instead of promoting it. That is usually the correct
repair, because the thing that made the ordering impossible is normally a candidate that
was never an event.

Discarding is not free: it costs recall, monotonically, and the labelled data shows it. So
`deletion_prior` is set to the smallest value that gets the full coherence benefit rather
than the one that scores best on coherence. At 0.0 the NOISE state is off and this is pure
relabelling.

What this cannot do
-------------------
It cannot invent an event that was never detected, and it does not try: a missing contact
stays missing, and the rally audit still reports it.

It is also honest about cost. Every flip is recorded with the probability it overruled,
so a run that had to fight the classifier hard is visible rather than silent. If the
decoder is flipping many high-confidence events, the classifier or the candidate
generator is wrong in a way that a grammar should not be papering over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Probability floor, so a confident classifier cannot make a legal path infinitely
# expensive and force the decoder into an illegal one.
_FLOOR = 1e-6

# How likely any given candidate is to be spurious. Swept against labels rather than
# chosen, because the wrong value is quietly destructive in both directions.
#
# Three metrics, because no one of them can set this alone. Coherence is the rally audit's
# count of provably-missing events (9 clips, eval/rally_coherence.py); recall is contacts
# against ground truth (40 labelled clips, eval/event_detection_on_real_detections.py);
# shot F1 is the labelled reference clip (eval/shot_frame_accuracy.py).
#
#   prior   coherence   events kept   dataset recall   shot F1
#   off        83           158           75.9%         0.667
#   0.00        -           158           75.9%         0.583   relabel only, no deletion
#   0.02       35           138           75.9%         0.636   <- default
#   0.05       31           136           75.6%         0.636
#   0.10       34           131           74.4%         0.667
#   0.15       30           126           73.3%         0.667
#   0.20        -             -           71.8%           -
#   0.50        -             -            0.0%         0.000   everything is noise
#
# Most of the benefit is relabelling, not deletion. On a 4-clip subset where all of this
# was re-checked, relabelling alone removes 45% of the impossible events while keeping
# every candidate; the first slice of deletion adds another 5 points and costs 9% of them.
#
# Above 0.02 coherence improves only slightly and not monotonically (35, 31, 34, 30) while
# events kept and recall both fall steadily. Going from 0.02 to 0.15 buys a 14% better
# coherence number for 2.6 points of recall and 12 more discarded events, which is the
# wrong way round: a missing contact is worse than an incoherent one, because the audit can
# see incoherence and cannot see absence.
#
# The reference clip alone suggested a plateau up to 0.20 and that is how 0.15 got shipped
# first. That clip saturates early, so its flatness was an artefact; the 40-clip recall
# curve is what showed the cost was real.
MEASURED_DELETION_PRIOR = 0.02

CONTACT = "contact"
BOUNCE = "bounce"
# Not a real event: a candidate the generators proposed and the grammar found no room for.
NOISE = "noise"


@dataclass
class DecodedRally:
    """The relabelled sequence, and what it cost to get there."""

    contacts: list[int] = field(default_factory=list)
    bounces: list[int] = field(default_factory=list)
    flips: list[str] = field(default_factory=list)
    discarded: list[int] = field(default_factory=list)

    @property
    def events(self) -> int:
        return len(self.contacts) + len(self.bounces)

    def summary(self) -> str:
        if not self.events:
            return "no events to decode"
        if not self.flips and not self.discarded:
            return (f"{self.events} events, already a legal rally "
                    f"(the classifier and the rules agree)")
        parts = []
        if self.flips:
            parts.append(f"{len(self.flips)} relabelled")
        if self.discarded:
            parts.append(f"{len(self.discarded)} discarded as spurious")
        joined = " and ".join(parts)
        return (f"{self.events} events, {joined} to make the sequence "
                f"physically possible")


def _legal(prev_label, prev_side, label, side) -> bool:
    """Whether this transition can happen in a real rally."""
    if prev_label is None:
        return True
    if label == BOUNCE:
        # A second bounce ends the point, so it cannot be followed by more play.
        return prev_label != BOUNCE
    # A contact by the side that last struck the ball means the ball never crossed the
    # net. This holds whether or not a bounce intervened: hit, bounce, hit by one player
    # describes a ball that stayed on its own side, which does not continue a rally.
    # Checking this only when the PREVIOUS event was a contact let exactly that sequence
    # through, which is the shape the reference clip actually produces.
    if side is not None and prev_side is not None and side == prev_side:
        return False
    return True


def decode_rally(events, deletion_prior: float = 0.0) -> DecodedRally:
    """
    Choose the most likely labelling of the event sequence that tennis permits.

    Args:
        events: ordered sequence of (frame, p_contact, side), where `p_contact` is the
                classifier's probability that the event is a racket contact and `side`
                identifies which half of the court it happened in. `side` may be None
                when it could not be determined, in which case the same-player rule
                cannot constrain that event and is skipped for it rather than guessed.
        deletion_prior: probability that any given candidate is not a real event. Lets the
                decoder discard a candidate rather than only relabel it, at a cost of
                -log(deletion_prior) nats. 0.0 disables discarding entirely.

    Returns:
        A DecodedRally. The labels are the best legal path; `flips` records every event
        whose label the grammar overruled, and `discarded` every candidate it dropped.
    """
    events = list(events)
    result = DecodedRally()
    if not events:
        return result

    # Viterbi over states of (label, side that last struck the ball).
    # Costs are negative log probabilities, so lower is better and they add.
    paths: dict[tuple, tuple[float, list]] = {(None, None): (0.0, [])}

    for frame, p_contact, side in events:
        p_contact = min(max(float(p_contact), _FLOOR), 1.0 - _FLOOR)
        p_real = 1.0 - deletion_prior
        options = [
            (CONTACT, -math.log(p_real * p_contact)),
            (BOUNCE, -math.log(p_real * (1.0 - p_contact))),
        ]
        if deletion_prior > 0.0:
            options.append((NOISE, -math.log(deletion_prior)))

        nxt: dict[tuple, tuple[float, list]] = {}
        for (prev_label, prev_side), (cost, trail) in paths.items():
            for label, emission in options:
                if label == NOISE:
                    # A spurious candidate leaves the rally untouched, so the state does
                    # not advance. This is what lets the grammar delete rather than flip.
                    key = (prev_label, prev_side)
                    total = cost + emission
                    if key not in nxt or total < nxt[key][0]:
                        nxt[key] = (total, trail + [(frame, NOISE, p_contact)])
                    continue
                if not _legal(prev_label, prev_side, label, side):
                    continue
                # A bounce does not change who last struck the ball, so the constraint
                # survives across it: hit, bounce, hit by one player is still the ball
                # failing to cross the net.
                new_side = side if label == CONTACT else prev_side
                key = (label, new_side)
                total = cost + emission
                if key not in nxt or total < nxt[key][0]:
                    nxt[key] = (total, trail + [(frame, label, p_contact)])
        if not nxt:
            # No legal continuation exists, which means something upstream is wrong. Reset
            # the CONSTRAINT state and carry on, keeping everything decoded so far.
            #
            # This previously started a fresh trail here, which silently deleted every
            # event before this point from the output: one clip decoded 22 events down to
            # 1. It was invisible for a long time because any non-zero deletion_prior
            # always supplies a legal NOISE continuation, so this branch cannot be reached
            # unless discarding is switched off entirely.
            cost, trail = min(paths.values(), key=lambda v: v[0])
            for label, emission in options:
                if label == NOISE:
                    continue
                new_side = side if label == CONTACT else None
                key = (label, new_side)
                total = cost + emission
                if key not in nxt or total < nxt[key][0]:
                    nxt[key] = (total, trail + [(frame, label, p_contact)])
        paths = nxt

    _, best = min(paths.values(), key=lambda v: v[0])

    for frame, label, p_contact in best:
        if label == NOISE:
            result.discarded.append(frame)
            continue
        (result.contacts if label == CONTACT else result.bounces).append(frame)
        preferred = CONTACT if p_contact >= 0.5 else BOUNCE
        if label != preferred:
            confidence = p_contact if preferred == CONTACT else 1.0 - p_contact
            result.flips.append(
                f"f{frame}: {preferred} -> {label} "
                f"(classifier preferred {preferred} at {confidence:.0%}, "
                f"but that ordering cannot happen)"
            )

    result.contacts.sort()
    result.bounces.sort()
    result.discarded.sort()
    return result
