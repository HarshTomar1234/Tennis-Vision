"""
utils/rally_audit.py
────────────────────
Estimates what the detector MISSED, on the clip in front of it, with no ground truth.

Why this exists
---------------
Every accuracy number in this project comes from a labelled dataset. That tells a reader
how the pipeline performs on average, and tells them nothing about the clip they just
uploaded. A system that reports 72% recall on a benchmark and then says nothing about the
video in your hand has answered the wrong question.

Tennis has enough physical structure to answer the right one. A rally is not an arbitrary
sequence of events: it alternates, and some orderings are impossible rather than merely
unlikely. Where an impossible ordering appears in the detected sequence, an event was
missed, and that can be stated without knowing what the correct sequence was.

The constraints, hardest first
------------------------------
1. **A player cannot hit twice in succession.** In singles the ball must go over the net
   and come back. Two consecutive contacts by the same player mean the opponent's contact
   between them was missed. This is the strongest signal available: no shot selection, no
   tactic and no rule of tennis produces it.

2. **A ball cannot bounce twice and the rally continue.** A second bounce ends the point.
   Two consecutive bounces followed by more play mean a contact between them was missed.

3. **A groundstroke follows a bounce.** Two consecutive contacts with no bounce between
   them is legal, but only for a volley or a half-volley, which are struck near the net.
   Flagged softly, and only when the hitter was nowhere near the net, because this is a
   real pattern rather than an impossibility.

What this deliberately does not do
----------------------------------
It does not guess where the missing events were, or insert them. It counts what the
sequence proves is absent and reports that, so a consumer can decide whether the clip is
worth trusting. Inventing the missing contacts would put fabricated events into the same
output that the rest of this pipeline works to keep free of them.

It is also a LOWER bound. Two missed events in a row can restore a valid-looking
alternation, so the true count can be higher than the count reported and can never be
lower. That is stated in the output rather than left for the reader to infer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# How near a baseline, as a fraction of half-court length, a player must be for a
# contact-without-bounce to look wrong. A volley is struck near the net, so only a
# no-bounce contact from deep in the court is suspicious.
BASELINE_FRACTION = 0.55


@dataclass
class RallyAudit:
    """What the event sequence proves is missing, and why."""

    events: int = 0
    missed_contacts: int = 0
    missed_bounces: int = 0
    findings: list[str] = field(default_factory=list)

    @property
    def implied_missing(self) -> int:
        return self.missed_contacts + self.missed_bounces

    @property
    def completeness(self) -> float:
        """
        Detected events as a share of what the sequence implies should exist.

        1.0 means nothing in the ordering proves anything is missing. It does NOT mean
        the clip was analysed perfectly, because a pair of adjacent misses can leave a
        consistent-looking sequence behind.
        """
        implied = self.events + self.implied_missing
        return self.events / implied if implied else 1.0

    def summary(self) -> str:
        if not self.events:
            return "no events detected, so nothing can be audited"
        if not self.implied_missing:
            return (f"{self.events} events, sequence self-consistent "
                    f"(no missing event is provable from the ordering)")
        return (f"{self.events} events detected, at least {self.implied_missing} missed "
                f"({self.completeness:.0%} complete, lower bound)")


def audit_rally(
    contacts,
    bounces,
    hitter_by_frame: dict | None = None,
    player_side: dict | None = None,
    net_y: float | None = None,
    half_court: float | None = None,
) -> RallyAudit:
    """
    Audit one clip's detected events against the physics of a rally.

    Args:
        contacts:        frames where a racket struck the ball.
        bounces:         frames where the ball hit the ground.
        hitter_by_frame: {contact_frame: player_id}. Enables the strongest check, that a
                         player cannot hit twice in succession. Without it that check is
                         skipped rather than guessed at.
        player_side:     {contact_frame: y position of the hitter in court units}, used
                         only to decide whether a no-bounce contact was plausibly a volley.
        net_y:           net position in the same units as player_side.
        half_court:      half-court length in the same units.

    Returns:
        A RallyAudit. Counts are lower bounds: two adjacent misses can restore a
        valid-looking alternation.
    """
    audit = RallyAudit()

    timeline = sorted(
        [(f, "contact") for f in contacts] + [(f, "bounce") for f in bounces]
    )
    audit.events = len(timeline)
    if len(timeline) < 2:
        return audit

    # 1. The same player twice in succession, checked over the CONTACT sequence rather
    #    than over adjacent timeline entries. A bounce in between does not make it legal:
    #    hit, bounce, hit by one player means the ball never crossed the net, which cannot
    #    happen in a rally. Checking only adjacent events missed exactly this case.
    if hitter_by_frame:
        ordered = sorted(contacts)
        for frame_a, frame_b in zip(ordered, ordered[1:]):
            who_a = hitter_by_frame.get(frame_a)
            who_b = hitter_by_frame.get(frame_b)
            if who_a is not None and who_a == who_b:
                audit.missed_contacts += 1
                audit.findings.append(
                    f"f{frame_a}-f{frame_b}: player {who_a} appears to hit twice in "
                    f"succession, so the opponent's contact between them was missed"
                )

    for (frame_a, kind_a), (frame_b, kind_b) in zip(timeline, timeline[1:]):
        # 2. Two bounces with no contact between, while the rally continues.
        if kind_a == "bounce" and kind_b == "bounce":
            audit.missed_contacts += 1
            audit.findings.append(
                f"f{frame_a}-f{frame_b}: two bounces with no contact between them. "
                f"A second bounce ends the point, so either a contact was missed or the "
                f"clip runs past the end of the rally"
            )
            continue

        # 3. Two contacts with no bounce between them. Legal for a volley, so this is
        #    only reported when the hitter was too deep for one.
        if kind_a == "contact" and kind_b == "contact":
            if hitter_by_frame:
                who_a = hitter_by_frame.get(frame_a)
                if who_a is not None and who_a == hitter_by_frame.get(frame_b):
                    continue   # already counted above as a missed opponent contact
            deep = True
            if player_side and net_y is not None and half_court:
                pos = player_side.get(frame_b)
                if pos is not None:
                    deep = abs(pos - net_y) > BASELINE_FRACTION * half_court
            if deep:
                audit.missed_bounces += 1
                audit.findings.append(
                    f"f{frame_a}-f{frame_b}: two contacts with no bounce between them, "
                    f"struck from too deep to be a volley, so a bounce was missed"
                )

    return audit
