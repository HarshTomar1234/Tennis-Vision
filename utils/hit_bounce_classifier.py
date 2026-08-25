"""
utils/hit_bounce_classifier.py
───────────────────────────────
Classifies a floor-level ball event (from utils.ball_state.classify_floor_level) as a
CONTACT (player hit) or BOUNCE (court), using ball-trajectory shape alone - no player
position needed.

Why this exists
----------------
The player-proximity heuristic in classify_contact_vs_bounce has a measured ~5/7 ceiling
on our own footage (docs/journal/0003) that turned out to be mostly an artifact of an
incomplete ground truth, not the heuristic itself (docs/journal/0010, 0011) - but while
investigating that, feature exploration on the real 1,034-event TrackNet ground truth
(eval/explore_hit_bounce_features.py) found a much stronger, complementary signal:

  A bounce is a court reflection - mostly preserves horizontal (x) velocity, since the
  ground doesn't impart much sideways force. A hit is a player redirecting the ball, which
  can reverse or sharply change x-direction (cross-court shots, returns). Measured: hits
  flip x-direction 71.8% of the time, bounces only 2.1% of the time.

A 4-feature logistic regression (height, |vertical-velocity change|, |horizontal-velocity
change|, and whether x-direction flipped) trained on that data reaches 86.4% held-out
accuracy on a clip-level split (not event-level, which would leak camera/lighting/player
correlations) - see eval/train_hit_bounce_classifier.py for the full methodology and
eval/journal 0012 for the honest numbers.

That feature set was chosen on end-to-end shot F1, not on this accuracy. A six-feature
variant scores higher here (89.3%) and measurably worse in the pipeline (rally F1 0.600
against 0.824), because the extra features are raw signed velocities that are clean in
the hand-annotated training data and noisy in real TrackNet detections. The comment above
FEATURE_NAMES in the training script has the full table.

This is trajectory-only and needs no player detection, so it works even when player
tracking is unavailable, and combines naturally with the proximity heuristic where player
positions ARE available (main.py can use both and let them agree/disagree as a signal).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .ball_state import CONTACT, BOUNCE

logger = logging.getLogger(__name__)

# Resolved from this file's location, not the working directory. As a bare relative path
# it only worked when the process happened to be started from the repo root: run from
# anywhere else, or installed as a wheel, the weights were "not found", the classifier
# returned None for every event, and the pipeline fell back to the player-proximity
# heuristic without anything in the output saying so.
DEFAULT_WEIGHTS_PATH = str(
    Path(__file__).resolve().parent.parent / "models" / "hit_bounce_classifier.json"
)
EVENT_WINDOW = 4   # frames before/after the event used to compute velocity - must match
                    # the WINDOW constant in eval/explore_hit_bounce_features.py, since
                    # the trained weights assume this exact window size.

_cached_weights: dict | None = None
_cached_path: str | None = None


def _load_weights(path: str = DEFAULT_WEIGHTS_PATH) -> dict | None:
    global _cached_weights, _cached_path
    if _cached_weights is not None and _cached_path == path:
        return _cached_weights

    if not Path(path).exists():
        logger.warning(
            f"Hit/bounce classifier weights not found at '{path}'. "
            f"Train with: python eval/train_hit_bounce_classifier.py"
        )
        return None

    _cached_weights = json.loads(Path(path).read_text())
    _cached_path = path
    return _cached_weights


def compute_event_features(
    positions: list[tuple[float, float] | None],
    event_frame: int,
    window: int = EVENT_WINDOW,
) -> dict | None:
    """
    Extract the trajectory-shape features the classifier needs around `event_frame`.

    Args:
        positions: per-frame (x, y) ball centre, or None where undetected. Index must
                   align with `event_frame`.
        event_frame: the floor-level candidate frame (from classify_floor_level).
        window:    frames of context required on each side (default matches training).

    Returns:
        The six trained features, or None if there isn't enough clean position data
        around this frame to compute velocity on both sides.
    """
    if event_frame < 0 or event_frame >= len(positions) or positions[event_frame] is None:
        return None

    before = [p for p in positions[max(0, event_frame - window):event_frame] if p is not None]
    after  = [p for p in positions[event_frame + 1:event_frame + 1 + window] if p is not None]
    if len(before) < 2 or len(after) < 2:
        return None

    vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
    vy_before = (before[-1][1] - before[0][1]) / (len(before) - 1)
    vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
    vy_after  = (after[-1][1] - after[0][1]) / (len(after) - 1)

    # These must match eval/explore_hit_bounce_features.py exactly, including the
    # convention below: the trained weights assume this encoding, and a mismatch here
    # would be silent - the model would still return confident probabilities from
    # features that no longer mean what it learned.
    return {
        "height_y": positions[event_frame][1],
        "vy_change_mag": abs(vy_after - vy_before),
        "vy_before": vy_before,
        "vy_after": vy_after,
        "vx_change_mag": abs(vx_after - vx_before),
        # 1 when the ball reversed horizontally, which is a racket redirecting it
        # (71.8% of hits, 2.1% of bounces). Below 0.5 px/frame the direction is noise
        # rather than motion, so it is reported as "no flip" - the same collapse
        # training applies when this feature is absent.
        "vx_sign_flip": float(
            abs(vx_before) > 0.5 and abs(vx_after) > 0.5 and (vx_before > 0) != (vx_after > 0)
        ),
    }


def classify_hit_or_bounce(
    features: dict | None,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> tuple[str, float] | None:
    """
    Classify a floor-level event as CONTACT or BOUNCE from its trajectory features.

    Args:
        features: from compute_event_features(); None passes through as None (honest
                  absence, not a guess - matches the convention in pose_shot_classifier).
        weights_path: trained weights, see eval/train_hit_bounce_classifier.py.

    Returns:
        (CONTACT | BOUNCE, probability of the returned label), or None if features are
        missing or weights aren't trained yet.
    """
    if features is None:
        return None

    weights = _load_weights(weights_path)
    if weights is None:
        return None

    # A feature the weights were trained on but the caller did not compute is a bug in
    # this file, not a runtime condition: it means compute_event_features and the trained
    # model have drifted apart. Say so, rather than raising a bare KeyError - and never
    # default it to 0, which would keep returning confident probabilities from an input
    # the model never saw.
    absent = [n for n in weights["feature_names"] if n not in features]
    if absent:
        raise ValueError(
            f"trained weights expect {weights['feature_names']} but these were not "
            f"computed: {absent}. compute_event_features() and the weights in "
            f"{weights_path} are out of sync - retrain, or update the feature computation."
        )

    x = [features[name] for name in weights["feature_names"]]
    mu, sigma, w, b = weights["mu"], weights["sigma"], weights["w"], weights["b"]

    z = b
    for xi, mui, sigi, wi in zip(x, mu, sigma, w):
        z += wi * ((xi - mui) / sigi)

    p_hit = 1.0 / (1.0 + pow(2.718281828, -z))
    if p_hit >= 0.5:
        return CONTACT, p_hit
    return BOUNCE, 1.0 - p_hit


def detect_xvelocity_candidates(
    ball_detections: list[dict],
    min_delta_x: float = 5.0,
    min_spacing: int = 15,
    window: int = EVENT_WINDOW,
) -> list[int]:
    """
    Candidate contact/bounce frames from local peaks in |horizontal-velocity change|,
    complementing (not replacing) trajectory-reversal detection.

    Why this exists: journal 0014 found a structural gap in y-reversal-only detection -
    some real contacts (verified on the reference clip) don't reverse vertical direction
    at all, or reverse too shallowly to separate from noise at any threshold (swept
    min_delta_y down to 1 with no improvement). But the same clip's real shots often DO
    show a sharp change in horizontal velocity (the physical signal behind
    hit_bounce_classifier - journal 0012), even when the vertical trajectory barely moves.

    Tested standalone at dataset scale first (91 real clips): x-velocity alone actually
    recalls WORSE than y-reversal (70.3% vs 76.0% at the best threshold) - it is not a
    good replacement. But the UNION of both signals recalls 87.7%, a genuine +11.7 point
    gain, because they catch different kinds of real events (vertical-redirect shots vs
    horizontal-redirect shots). Use this alongside get_ball_shot_frames, not instead of
    it - feed the union to classify_reversals_by_trajectory, which is what actually
    filters the resulting extra candidates back down to real contacts/bounces.

    Args:
        ball_detections: per-frame {1: [x1,y1,x2,y2]} (same format as get_ball_shot_frames).
        min_delta_x:     minimum |vx change| to count as a candidate peak.
        min_spacing:     minimum frames between two candidates.
        window:          frames of context on each side used to compute velocity.

    Returns:
        Candidate frame indices, sorted, at least min_spacing apart.
    """
    positions: list[tuple[float, float] | None] = []
    for det in ball_detections:
        bbox = det.get(1)
        if bbox is not None:
            positions.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        else:
            positions.append(None)

    n = len(positions)
    scores = [0.0] * n
    for i in range(window, n - window):
        before = [p for p in positions[i - window:i] if p is not None]
        after  = [p for p in positions[i + 1:i + 1 + window] if p is not None]
        if len(before) < 2 or len(after) < 2:
            continue
        vx_before = (before[-1][0] - before[0][0]) / (len(before) - 1)
        vx_after  = (after[-1][0] - after[0][0]) / (len(after) - 1)
        scores[i] = abs(vx_after - vx_before)

    candidates: list[int] = []
    for i in range(n):
        if scores[i] < min_delta_x:
            continue
        lo, hi = max(0, i - min_spacing // 2), min(n, i + min_spacing // 2 + 1)
        if scores[i] == max(scores[lo:hi]):
            if not candidates or i - candidates[-1] >= min_spacing:
                candidates.append(i)

    return candidates


def merge_nearby_candidates(candidates: list[int], min_gap: int = 10) -> list[int]:
    """
    Collapse candidates within min_gap frames of each other into one representative
    frame per cluster (the cluster's median).

    get_ball_shot_frames (y-reversal) and detect_xvelocity_candidates each fire near a
    real event independently, with no knowledge of each other, so their union can put
    several candidates a few frames apart around the same single real contact. On our
    own reference clip this inflated the apparent shot count from 7 (y-reversal only,
    journal 0012) to 22 (union, journal 0015) -- most of the "extra" shots turned out to
    be duplicate detections of the same handful of real events, not new false events
    (see docs/journal/0018's frame-by-frame check). It also made downstream per-frame
    work (pose classification) sensitive to *which* nearby frame got checked: contact
    happens at one instant, so a candidate a few frames off has different wrist
    positions than the true contact frame, and pose can succeed on one cluster member
    while failing on another for the same swing.

    Clustering is bounded, not transitive
    -------------------------------------
    A candidate joins a cluster when it is within min_gap of the cluster's FIRST member,
    so no cluster is ever wider than min_gap frames.

    The original version compared against the cluster's LAST member, which chains: with
    min_gap=10, candidates at frames 0, 10, 20, 30, 40 all collapse into one cluster
    spanning 40 frames and report a single event at frame 20. In a dense rally that is
    not a corner case, it is the normal case, and it silently deleted real contacts.

    Measured across 12 dataset clips, 91 labelled contacts, changing only the linkage:

        linkage    recall   precision      F1
        chained     51.6%      94.0%    0.667
        bounded     68.1%      93.9%    0.790

    16.5 points of recall at no cost in precision, because the events being destroyed
    were real ones. The threshold itself needed no retuning: it was the linkage, not the
    value. Sweeping min_gap from 3 to 10 under bounded linkage moves precision from 75%
    to 94% while recall moves the other way, and 10 sits at the best F1 as well as the
    best precision, which is the trade this project wants.

    Reproduce with `python eval/event_recall_funnel.py --sweep-gap`.

    Args:
        candidates: raw candidate frames (e.g. the union of get_ball_shot_frames and
                    detect_xvelocity_candidates), any order.
        min_gap:    maximum width of a cluster, in frames. Note this is frames rather
                    than seconds, so it is implicitly tied to frame rate; at 30fps it is
                    0.33s, which is shorter than the fastest realistic bounce-to-contact
                    interval. Footage far from 30fps should revisit it.

    Returns:
        One representative frame per cluster, sorted.
    """
    if not candidates:
        return []

    ordered = sorted(candidates)
    clusters: list[list[int]] = [[ordered[0]]]
    for c in ordered[1:]:
        if c - clusters[-1][0] <= min_gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])

    return [cluster[len(cluster) // 2] for cluster in clusters]


def event_contact_probabilities(
    reversal_frames: list[int],
    ball_detections: list[dict],
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> dict[int, float]:
    """
    P(contact) per candidate frame, for frames the trajectory model can decide on.

    Split out of classify_reversals_by_trajectory because constrained decoding
    (utils.rally_decode) needs the probability, not the hard label: a grammar can only
    overrule a classifier if it knows how strongly the classifier objects. Frames with
    no usable trajectory context are absent from the result rather than defaulted.
    """
    positions: list[tuple[float, float] | None] = []
    for det in ball_detections:
        bbox = det.get(1)
        if bbox is not None:
            positions.append(((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0))
        else:
            positions.append(None)

    probabilities: dict[int, float] = {}
    for frame in reversal_frames:
        result = classify_hit_or_bounce(compute_event_features(positions, frame),
                                        weights_path)
        if result is None:
            continue
        label, confidence = result
        probabilities[frame] = confidence if label == CONTACT else 1.0 - confidence
    return probabilities


def classify_reversals_by_trajectory(
    reversal_frames: list[int],
    ball_detections: list[dict],
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> tuple[list[int], list[int]]:
    """
    Drop-in alternative to classify_contact_vs_bounce (utils.ball_state) with the same
    (contacts, bounces) return shape, but using trajectory shape instead of player
    proximity - no player detection needed. See module docstring for why: 84.1% held-out
    accuracy on real data, vs the proximity heuristic's measured ~5/7 ceiling on our own
    footage (which turned out to be mostly an incomplete-ground-truth artifact, but the
    trajectory signal is independently strong regardless - see docs/journal/0012).

    Frames where the classifier can't reach a decision (not enough trajectory context,
    or the weights aren't trained) are dropped from both lists rather than guessed.
    """
    probabilities = event_contact_probabilities(
        reversal_frames, ball_detections, weights_path
    )

    contacts, bounces = [], []
    for frame in reversal_frames:
        p_contact = probabilities.get(frame)
        if p_contact is None:
            continue
        (contacts if p_contact >= 0.5 else bounces).append(frame)

    return contacts, bounces


# The proximity contact/bounce fallback measured about 5 correct in 7 on our own footage
# (see the module docstring), so that is the confidence it carries into decoding. It is a
# weak vote by design: the grammar should be able to overrule it, unlike a calibrated
# trajectory probability.
PROXIMITY_RELIABILITY = 5.0 / 7.0


def striking_side(
    frame: int,
    ball_detections: list[dict],
    player_detections: list[dict],
    max_distance_px: float = float("inf"),
) -> int | None:
    """
    Which player was nearest the ball at this frame, or None if it cannot be said.

    Distance is measured to the player's box rather than to its centre, because a near
    player's box is tall: a ball at their feet is far from the box centre while being
    right on the player. It also uses BOTH axes. Comparing horizontal distance alone is
    close to meaningless in a broadcast view, where both players sit near the centre line
    in x while being metres apart in y, and it attributed nearly every contact to the same
    player until the rally self-audit caught it.

    Args:
        max_distance_px: beyond this, return None rather than name a player. The rally
            grammar needs that, because a wrong side makes it reject a real contact. Leave
            unbounded when a nearest player is wanted regardless of distance, as when
            placing a contact on the mini-court.
    """
    if frame >= len(ball_detections) or frame >= len(player_detections):
        return None
    bbox = ball_detections[frame].get(1)
    players = player_detections[frame]
    if bbox is None or not players:
        return None

    ball_x, ball_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0

    def box_distance(pid):
        x1, y1, x2, y2 = players[pid]
        dx = max(x1 - ball_x, 0.0, ball_x - x2)
        dy = max(y1 - ball_y, 0.0, ball_y - y2)
        return (dx * dx + dy * dy) ** 0.5

    nearest = min(players, key=box_distance)
    return nearest if box_distance(nearest) <= max_distance_px else None


class DecodeNotes(list):
    """
    The decoder's human-readable notes, carrying its machine-readable account alongside.

    A plain list, so every existing caller keeps working: `derive_shot_frames` has six
    of them across main.py and the evals, all unpacking a 4-tuple and most ignoring this
    element entirely. Widening the return signature would have churned five evals that
    are currently correct, for the sake of a diagnostic only main.py reads.

    `.diagnostics` is the dict that reaches summary.json. See
    utils.rally_decode.DecodedRally.diagnostics for what it contains and why it matters:
    the decoder's own docstring warns that repeatedly overruling a confident classifier
    signals an upstream problem, and until now that warning existed only in the log.
    """

    diagnostics: dict = {}


def derive_shot_frames(
    ball_tracker,
    ball_detections: list[dict],
    player_detections: list[dict],
    shot_player_distance_px: int = 300,
    deletion_prior: float | None = None,
):
    """
    Turn ball detections into the pipeline's confirmed shot and bounce frames.

    This exists so the evals grade what the product actually reports. They previously
    stopped at the raw candidate union - the output of the three generators before any
    contact-vs-bounce classification - and scored that as if it were the shot list. On
    the reference clip that union is 25 candidates against 7 real shots, so the eval
    published 28% precision for a stage the pipeline never emits: every bounce in the
    rally was being counted as a false-positive shot. `eval/_ball_source.py` had already
    aligned the *detections* between eval and pipeline; this closes the same gap for
    event derivation, which is where the shot numbers actually come from.

    Three generators feed the union because each is blind to a different event shape:
    y-reversal and x-velocity are hit-shaped by construction (x-velocity recalls only
    12% of bounces), while the bounce generator recalls 83.9%. Merging is what stops
    two generators firing on one real event and inflating the count.

    Args:
        ball_tracker:             supplies `get_ball_shot_frames` (the y-reversal generator).
        ball_detections:          per-frame {ball_id: bbox}, already interpolated.
        player_detections:        per-frame {player_id: bbox}, for the proximity fallback.
        shot_player_distance_px:  proximity threshold for the fallback classifier.
        deletion_prior:           how readily rally decoding may discard a candidate as
                                  spurious. Defaults to the swept value in
                                  utils.rally_decode; pass 0.0 to disable discarding.

    Returns:
        (shot_frames, bounce_frames, raw_reversal_frames, decode_flips). The first three
        are sorted frame lists; `decode_flips` describes every label the rally grammar
        overruled (see utils.rally_decode), and is empty when the classifier's own
        labelling was already a physically possible rally.
    """
    from .ball_state import classify_contact_vs_bounce
    from .bounce_candidates import detect_bounce_candidates
    from .rally_decode import MEASURED_DELETION_PRIOR, decode_rally

    raw_reversals = merge_nearby_candidates(sorted(
        set(ball_tracker.get_ball_shot_frames(ball_detections))
        | set(detect_xvelocity_candidates(ball_detections))
        | set(detect_bounce_candidates(ball_detections))
    ))

    # Probabilities rather than labels, because decoding below needs to know how strongly
    # the classifier holds each opinion, not just which way it leans. Computed once and
    # used for both the fallback split and the decode.
    probabilities = event_contact_probabilities(raw_reversals, ball_detections)

    # Reversals the trajectory model cannot decide on (too little context, e.g. near a
    # clip boundary) fall back to player proximity rather than being silently dropped.
    unclassified = [f for f in raw_reversals if f not in probabilities]
    if unclassified:
        prox_contacts, prox_bounces = classify_contact_vs_bounce(
            unclassified, ball_detections, player_detections,
            shot_player_distance_px=shot_player_distance_px,
        )
    else:
        prox_contacts, prox_bounces = [], []

    # Constrained decoding. Each event above was labelled in isolation, which is why the
    # sequence can come out physically impossible (one player hitting twice with no reply
    # in between). Re-label the sequence as a whole, keeping the most likely labelling
    # that a rally permits. This cannot recover an event that was never detected.
    proximity_labels = {f: True for f in prox_contacts}
    proximity_labels.update({f: False for f in prox_bounces})

    events = []
    for frame in raw_reversals:
        p_contact = probabilities.get(frame)
        if p_contact is None:
            if frame not in proximity_labels:
                continue
            # The proximity fallback has no calibrated probability, so it enters decoding
            # at its measured reliability rather than as a certainty.
            p_contact = (PROXIMITY_RELIABILITY if proximity_labels[frame]
                         else 1.0 - PROXIMITY_RELIABILITY)
        side = striking_side(frame, ball_detections, player_detections,
                             shot_player_distance_px)
        events.append((frame, p_contact, side))

    if deletion_prior is None:
        deletion_prior = MEASURED_DELETION_PRIOR
    decoded = decode_rally(events, deletion_prior=deletion_prior)

    notes = DecodeNotes(decoded.flips)
    if decoded.discarded:
        notes.append(
            f"{len(decoded.discarded)} candidate(s) discarded as spurious at frames "
            f"{decoded.discarded}: no legal place in the rally, and dropping them was a "
            f"better explanation than promoting them"
        )
    notes.diagnostics = decoded.diagnostics()
    return decoded.contacts, decoded.bounces, raw_reversals, notes
