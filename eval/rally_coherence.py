"""
eval/rally_coherence.py
───────────────────────
Does constrained rally decoding make the reported rally physically possible?

Why this eval is unusual
------------------------
Every other eval here needs ground truth, which limits them to the clips somebody sat
down and labelled. This one does not. The rally audit (utils.rally_audit) counts events
that the ORDERING proves are missing, using only the fact that some sequences cannot
happen in tennis: a player cannot hit twice in succession, a ball cannot bounce twice and
the rally continue. That is checkable on any clip, so this measures the decoder on all of
them rather than on the one with labels.

What it does not tell you
-------------------------
A coherent rally is not a correct rally. This metric can be driven to zero by discarding
every candidate, which is exactly what a deletion prior above 0.5 does. So it is only
meaningful read alongside eval/shot_frame_accuracy.py, which grades contacts against
labels and will show the recall those discards cost. Neither number is sufficient alone,
which is why the decoder's default prior was chosen where the first improves and the
second does not move.

Usage:
  python eval/rally_coherence.py                       # all eval clips, default prior
  python eval/rally_coherence.py --prior 0.3
  python eval/rally_coherence.py --sweep               # compare several priors
"""
import argparse
import contextlib
import glob
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils.rally_decode as rally_decode
from court_line_detector import CourtLineDetector
from trackers import PlayerTracker, TrackNetBallTracker
from utils import read_video, select_two_players, stub_path_for_video
from utils.hit_bounce_classifier import derive_shot_frames, striking_side
from utils.rally_audit import audit_rally

DEFAULT_CLIPS = "datasets/eval_clips/*.mp4"
SHOT_DISTANCE_PX = 300


def analyse(path: str, priors: list[float | None]) -> dict:
    """Audit one clip once per decoder setting. `None` means decoding disabled."""
    frames = read_video(path)
    if len(frames) < 20:
        return {}

    ball_tracker = TrackNetBallTracker(model_path="models/tracknet.pt")
    ball = ball_tracker.interpolate_ball_positions(ball_tracker.detect_frames(frames))

    player_tracker = PlayerTracker(model_path="yolov8x")
    people = player_tracker.detect_frames(
        frames, read_from_stub=True,
        stub_path=stub_path_for_video("tracker_stubs/player_detections.pkl", path),
    )
    # Two players only. Passing every detected person puts spectator track ids into the
    # grammar's same-player rule, which is the strongest constraint it has.
    court = CourtLineDetector("models/keypoints_model_geoaug.pth")
    players, _ = select_two_players(player_tracker, people, court.predict(frames[0]))

    results = {}
    for prior in priors:
        # Decoding is disabled by substituting the classifier's own labels, rather than by
        # a flag inside the pipeline, so both arms run through identical code.
        if prior is None:
            contacts, bounces = _undecoded(ball_tracker, ball, players)
        else:
            contacts, bounces, _raw, _notes = derive_shot_frames(
                ball_tracker, ball, players, SHOT_DISTANCE_PX, deletion_prior=prior,
            )
        hitters = {}
        for frame in contacts:
            who = striking_side(frame, ball, players, SHOT_DISTANCE_PX)
            if who is not None:
                hitters[frame] = who
        results[prior] = audit_rally(contacts, bounces, hitter_by_frame=hitters)
    return results


def _undecoded(ball_tracker, ball, players):
    """The classifier's per-event labels, with no grammar applied. The baseline."""
    original = rally_decode.decode_rally

    def passthrough(events, deletion_prior=0.0):
        out = rally_decode.DecodedRally()
        for frame, p_contact, _side in events:
            (out.contacts if p_contact >= 0.5 else out.bounces).append(frame)
        out.contacts.sort()
        out.bounces.sort()
        return out

    rally_decode.decode_rally = passthrough
    try:
        contacts, bounces, _raw, _notes = derive_shot_frames(
            ball_tracker, ball, players, SHOT_DISTANCE_PX
        )
    finally:
        rally_decode.decode_rally = original
    return contacts, bounces


def main():
    ap = argparse.ArgumentParser(description="Rally coherence with and without decoding")
    ap.add_argument("--clips", default=DEFAULT_CLIPS)
    ap.add_argument("--prior", type=float, default=rally_decode.MEASURED_DELETION_PRIOR)
    ap.add_argument("--sweep", action="store_true",
                    help="compare several deletion priors instead of just the default")
    args = ap.parse_args()

    priors: list[float | None] = [None]
    priors += [0.0, 0.02, 0.05, 0.1, 0.15] if args.sweep else [args.prior]

    paths = sorted(glob.glob(args.clips))
    if not paths:
        print(f"No clips matched {args.clips}")
        sys.exit(1)

    totals = {p: [0, 0] for p in priors}          # prior -> [events, implied missing]
    header = "".join(f"{_label(p):>14}" for p in priors)
    print(f"{'clip':<24}{header}")
    print("-" * (24 + 14 * len(priors)))

    for path in paths:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                results = analyse(path, priors)
        except Exception as exc:
            print(f"{os.path.basename(path):<24} failed: {exc}")
            continue
        if not results:
            continue
        cells = ""
        for p in priors:
            audit = results[p]
            totals[p][0] += audit.events
            totals[p][1] += audit.implied_missing
            cells += f"{audit.events:>7}/{audit.implied_missing:<6}"
        print(f"{os.path.basename(path):<24}{cells}")

    print("-" * (24 + 14 * len(priors)))
    cells = "".join(f"{totals[p][0]:>7}/{totals[p][1]:<6}" for p in priors)
    print(f"{'TOTAL events/missing':<24}{cells}")

    base = totals[None][1]
    for p in priors:
        if p is None:
            continue
        drop = (base - totals[p][1]) / base if base else 0.0
        print(f"  prior {p:<5} removes {drop:.0%} of the provably-missing events, "
              f"keeping {totals[p][0]} of {totals[None][0]} detected events")
    print("\nA coherent rally is not a correct one: read this with "
          "eval/shot_frame_accuracy.py, which shows what the discards cost in recall.")


def _label(prior):
    return "off" if prior is None else f"p={prior}"


if __name__ == "__main__":
    main()
