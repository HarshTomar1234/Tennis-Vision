"""
utils/fps_support.py
────────────────────
Says whether a clip's frame rate is one this pipeline has evidence for.

The problem
-----------
Every threshold in the event-detection path is expressed in FRAMES, and every velocity
feature the hit/bounce classifier reads is expressed in PIXELS PER FRAME:

    EVENT_WINDOW = 4                      utils/hit_bounce_classifier.py
    min_spacing = 15                      detect_xvelocity_candidates
    min_spacing = 8                       detect_bounce_candidates
    merge_nearby_candidates(min_gap=10)
    vy_change_mag, vx_change_mag          the two largest classifier weights

None of those is normalised by frame rate. The same physical motion sampled at a
different rate therefore produces a different feature vector, a different candidate set,
and a different contact-versus-bounce split. Not a slightly different one.

The measurement
---------------
The reference clip's ball track (input_video_2, 570 frames, 30 fps) resampled in time
onto other rates, then run through the real candidate generators and the real classifier.
Events per second is the physically meaningful comparison: the rally contains the same
number of contacts however fast the camera sampled it.

    fps    events/s   vs 30fps   contact:bounce
     10      0.63       -57%          9:3
     12      0.74       -50%         11:3
     15      0.84       -43%         10:6
     18      1.05       -29%          9:11
     20      1.16       -21%         14:8
     22      1.32       -11%         16:9
     24      1.37        -7%         15:11
     25      1.37        -7%         15:11
     27      1.42        -4%         15:12
     30      1.47         0%         14:14     <- baseline, and where every
     33      1.68       +14%         16:16        published number was measured
     36      1.79       +21%         14:20
     40      2.00       +36%         13:25
     50      2.21       +50%         15:27
     60      2.42       +64%         11:35
     90      2.74       +86%          7:45

Two failures, in opposite directions. Below the band, real events stop being proposed at
all: at 15 fps nearly half the events per second are gone. Above it, the generators fire
too often AND the classifier's per-frame velocities shrink, so it calls almost everything
a bounce: at 60 fps the split is 11 contacts to 35 bounces on a rally that has roughly 14
of each. Shot counts stop meaning anything well before the pipeline stops producing them.

Honest limit of the method: below 30 fps the resampling discards information, which is
what a slower camera does, so those rows are close to fair. Above 30 fps it INVENTS
intermediate points that a real fast camera would have measured independently, and it
cannot model the sharper motion or reduced blur either. **The high-rate rows are a lower
bound on the disruption, not an estimate of it.** A real 60 fps clip is likely worse than
the table says, not better.

Why this gate and not a fix
---------------------------
The fix is to express the event path in seconds and metres rather than frames and pixels,
and to retrain the classifier on rate-normalised features. That is a training change with
no evidence behind it yet, and it would invalidate every published number in the process.
It is post-launch research.

What ships now is the honest half: detect the condition and say so. A clip outside the
measured band still runs, because refusing to process it would be worse than processing
it with a stated caveat, but nothing about the run claims the accuracy that was measured
inside the band.
"""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED = "supported"
PARTIALLY_SUPPORTED = "partially_supported"
UNSUPPORTED = "unsupported"

# Boundaries taken from rows of the table above rather than chosen for roundness.
#
# 23 to 31: every evaluation clip (23.57 to 29.82 fps) and both input videos (30.0) sit
# here, so this is the range every published number was actually measured on. Across it
# the event rate stays within about 7% of baseline and the contact/bounce balance holds.
SUPPORTED_MIN_FPS = 23.0
SUPPORTED_MAX_FPS = 31.0

# 18 and 50: the last rows where the pipeline still recovers a recognisable rally.
# At 18 fps it finds 29% fewer events per second; at 50 fps it finds 50% more and has
# begun mislabelling contacts as bounces (15:27). Numbers from here are wrong in a known
# direction, which is different from being unusable.
PARTIAL_MIN_FPS = 18.0
PARTIAL_MAX_FPS = 50.0


@dataclass(frozen=True)
class FpsSupport:
    """Whether this frame rate is one the measured record covers, and what to expect."""

    fps: float
    status: str
    reason: str

    @property
    def is_supported(self) -> bool:
        return self.status == SUPPORTED

    def as_dict(self) -> dict:
        """Shape written into summary.json, so a consumer can act on it."""
        return {
            "fps": round(self.fps, 2),
            "status": self.status,
            "measured_range_fps": [SUPPORTED_MIN_FPS, SUPPORTED_MAX_FPS],
            "reason": self.reason,
        }


def assess_fps(fps: float | None) -> FpsSupport:
    """
    Classify a clip's frame rate against the range this pipeline has evidence for.

    Args:
        fps: frames per second from the video header. None or a non-positive value means
             the header could not be read.

    Returns:
        An FpsSupport. Never raises and never refuses to return: the caller decides what
        to do, and the pipeline's choice is to continue with the caveat attached rather
        than to reject the clip.
    """
    if not fps or fps <= 0:
        return FpsSupport(
            fps=0.0,
            status=UNSUPPORTED,
            reason=(
                "Frame rate could not be read from the video header. Every event "
                "threshold in this pipeline is expressed in frames, so without the real "
                "rate there is no way to know whether they apply to this clip."
            ),
        )

    if SUPPORTED_MIN_FPS <= fps <= SUPPORTED_MAX_FPS:
        return FpsSupport(
            fps=fps,
            status=SUPPORTED,
            reason=(
                f"{fps:.1f} fps is inside the {SUPPORTED_MIN_FPS:.0f} to "
                f"{SUPPORTED_MAX_FPS:.0f} fps range every published accuracy number was "
                f"measured on."
            ),
        )

    if fps < SUPPORTED_MIN_FPS:
        detail = (
            "Below the measured range the candidate generators propose fewer events, "
            "because their spacing and window thresholds are counted in frames: at "
            "18 fps the reference clip yields 29% fewer events per second, at 15 fps "
            "43% fewer. Expect an under-count of shots and bounces."
        )
    else:
        detail = (
            "Above the measured range the generators fire more often while the "
            "classifier's per-frame velocities shrink, so contacts get labelled as "
            "bounces: at 50 fps the reference clip yields 50% more events per second "
            "with the contact/bounce split moving from 14:14 to 15:27. Expect an "
            "over-count of events and an under-count of shots. This is measured by "
            "resampling a 30 fps clip, which cannot reproduce a fast camera's sharper "
            "motion, so it is a lower bound on the effect rather than an estimate."
        )

    if PARTIAL_MIN_FPS <= fps <= PARTIAL_MAX_FPS:
        return FpsSupport(
            fps=fps,
            status=PARTIALLY_SUPPORTED,
            reason=(
                f"{fps:.1f} fps is outside the measured {SUPPORTED_MIN_FPS:.0f} to "
                f"{SUPPORTED_MAX_FPS:.0f} fps range. The pipeline still recovers a "
                f"recognisable rally here, and its event counts and anything derived "
                f"from them are wrong in a known direction. {detail}"
            ),
        )

    return FpsSupport(
        fps=fps,
        status=UNSUPPORTED,
        reason=(
            f"{fps:.1f} fps is far outside the measured {SUPPORTED_MIN_FPS:.0f} to "
            f"{SUPPORTED_MAX_FPS:.0f} fps range, beyond the "
            f"{PARTIAL_MIN_FPS:.0f} to {PARTIAL_MAX_FPS:.0f} fps region where the "
            f"pipeline still recovers a usable rally. Event counts, shot counts and "
            f"every speed derived from them should not be treated as measurements. "
            f"{detail}"
        ),
    )
