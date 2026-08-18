"""
utils/shot_physics.py
─────────────────────
Classifies shots from physical evidence, and reports the evidence alongside the label.

Why this exists
---------------
`ShotClassifier` decides Volley and Smash from court position alone: a player near the
net is volleying, a ball high in the mini-court is a smash. Both are guesses dressed as
measurements - they have never had ground truth, and on real clips they produced six
backhands out of eight shots and smashes in the middle of baseline rallies.

Three of these shots are, however, *physically* defined, in the same way the serve
turned out to be. The serve detector proved the approach: instead of guessing from
position, state the physical fact that distinguishes the shot and test it directly.

  Smash  ball struck above the player's head, but NOT from a baseline.
         (Above the head is what serve and smash share; where you stand is what
         separates them, so this reuses the serve test with the zone inverted.)

  Volley ball struck with NO bounce between it and the previous contact, by a player
         in the forward half of their side. Volleying is *by definition* hitting
         before the bounce, so this is a fact about the event sequence rather than an
         inference from position. The position clause is a guard, not the signal:
         bounce recall is about 80 %, so a missed bounce would otherwise fake a volley
         at the baseline.

  Lob    outgoing flight whose apex is far above net height. Apex comes from the 3-D
         reconstruction, so this is only claimed for segments that passed the physical
         gates - an over-long segment inflates apex, which is precisely the artefact
         those gates reject.

Deliberately NOT included: the drop shot. It is defined by a low, short flight, and
that needs groundstroke speed, which is not yet validated against ground truth (only
serve speed is). Adding it now would be exactly the guess-dressed-as-measurement this
module exists to replace.

Every result carries the evidence that produced it, so a consumer can show why a shot
was called what it was - and so a wrong call is diagnosable instead of mysterious.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# A smash is played from inside the court; a serve from behind the baseline. Expressed
# as a fraction of half-court length measured from the net, so it scales with any court.
SMASH_MAX_DISTANCE_FROM_NET_FRAC = 0.65

# A volleying player stands in the front portion of their own half. Generous, because
# the no-bounce test is the real signal and this only rejects baseline false positives
# created by a missed bounce.
VOLLEY_MAX_DISTANCE_FROM_NET_FRAC = 0.55

# A lob clears the net by a wide margin. Net height is 0.914 m at centre; a normal
# rally ball crosses around 1-2 m. Metres.
LOB_MIN_APEX_M = 4.0


@dataclass
class ShotCall:
    """A shot label together with the evidence that produced it."""
    shot_type: str
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.shot_type)


def is_smash(
    ball_above_head: bool,
    distance_from_net_m: float,
    half_court_length_m: float,
) -> ShotCall | None:
    """
    A ball struck above the head from inside the court rather than from a baseline.

    `ball_above_head` is the same measurement the serve detector makes (ball centre
    above the top of the player's bounding box). What separates the two shots is where
    the player stands, so a serve must be ruled out before calling a smash.
    """
    if not ball_above_head or half_court_length_m <= 0:
        return None
    limit = SMASH_MAX_DISTANCE_FROM_NET_FRAC * half_court_length_m
    if distance_from_net_m > limit:
        return None       # struck from the baseline - a serve, not a smash
    return ShotCall("Smash", [
        "ball struck above the player's head",
        f"struck {distance_from_net_m:.1f} m from the net (inside {limit:.1f} m)",
    ])


def is_volley(
    bounces_since_previous_contact: int,
    distance_from_net_m: float,
    half_court_length_m: float,
) -> ShotCall | None:
    """
    A ball struck before it bounced.

    The no-bounce condition is the definition of a volley, not a proxy for it. The
    distance clause exists only to reject the false positive a missed bounce would
    otherwise create at the back of the court.
    """
    if bounces_since_previous_contact != 0 or half_court_length_m <= 0:
        return None
    limit = VOLLEY_MAX_DISTANCE_FROM_NET_FRAC * half_court_length_m
    if distance_from_net_m > limit:
        return None
    return ShotCall("Volley", [
        "no bounce between this contact and the previous one",
        f"struck {distance_from_net_m:.1f} m from the net (inside {limit:.1f} m)",
    ])


def is_lob(apex_height_m: float | None) -> ShotCall | None:
    """
    An outgoing flight arcing far above net height.

    Only meaningful for a 3-D segment that passed the reconstruction's physical gates;
    a segment spanning several flights reports an inflated apex, and those are already
    rejected upstream rather than reaching here.
    """
    if apex_height_m is None or apex_height_m < LOB_MIN_APEX_M:
        return None
    return ShotCall("Lob", [f"flight apex {apex_height_m:.1f} m "
                            f"(above the {LOB_MIN_APEX_M:.0f} m lob threshold)"])


def classify_from_physics(
    ball_above_head: bool,
    distance_from_net_m: float,
    half_court_length_m: float,
    bounces_since_previous_contact: int | None,
    outgoing_apex_m: float | None,
    is_serve: bool = False,
) -> ShotCall | None:
    """
    Apply the physical tests in order of specificity.

    Returns None when no physical test fires, which means the shot is an ordinary
    groundstroke and its forehand/backhand label must come from body pose. Returning
    None rather than a default is deliberate: this module only speaks when it has
    evidence.
    """
    if is_serve:
        return ShotCall("Serve", ["ball struck above the head from a baseline"])

    smash = is_smash(ball_above_head, distance_from_net_m, half_court_length_m)
    if smash:
        return smash

    if bounces_since_previous_contact is not None:
        volley = is_volley(bounces_since_previous_contact, distance_from_net_m,
                           half_court_length_m)
        if volley:
            return volley

    return is_lob(outgoing_apex_m)
