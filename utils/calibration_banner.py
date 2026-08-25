"""
utils/calibration_banner.py
───────────────────────────
Draws an on-screen warning when a run's court fit failed validation.

Why this is not optional polish
-------------------------------
The pipeline already detects an untrustworthy court fit and records it in
`summary.json` as `court_calibrated: false`. But the rendered video is what people
actually look at, share and screenshot - and until now it drew exactly the same
confident km/h figures, stats panel and mini-court dots whether the court was fitted
correctly or fitted to the crowd. A viewer had no way to tell the two apart.

That is the single most damaging failure mode for a project whose whole claim is that
its numbers are measured: a wrong number presented confidently travels further than a
right one, and it travels as a screenshot with no JSON attached.

Anything derived from the court homography - speeds, distances, mini-court positions -
is meaningless when the fit is wrong. Rather than silently hide those overlays (which
would make the failure invisible in a different way), the run is stamped so the output
carries its own caveat.

The same argument applies to frame rate, which is why that warning lives here too. Every
event threshold in this pipeline is counted in frames, so a clip sampled outside the
range those thresholds were measured on produces a different event set for the same
tennis (see utils/fps_support.py). That is invisible in the rendered video unless the
video says so.
"""
from __future__ import annotations

import cv2

_BANNER_TEXT = "COURT NOT CALIBRATED - speeds and positions are NOT measurements"
_SUBTEXT = "The detected court does not lie on the real court lines. See summary.json"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WARNING_BG = (0, 0, 140)      # dark red, BGR
_CAUTION_BG = (0, 90, 150)     # amber, BGR - a caveat, not a failure
_WARNING_FG = (255, 255, 255)


def _stamp(frames: list, headline: str, detail: str, background) -> list:
    """Draw one warning band across every frame. Mutates and returns `frames`."""
    if not frames:
        return frames

    height, width = frames[0].shape[:2]
    band_height = max(46, int(height * 0.075))
    scale = width / 1280.0

    for frame in frames:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, band_height), background, -1)
        # Semi-transparent so the banner cannot completely hide the action it warns about.
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        (text_width, _), _ = cv2.getTextSize(headline, _FONT, 0.62 * scale, 2)
        cv2.putText(frame, headline,
                    (max(8, (width - text_width) // 2), int(band_height * 0.45)),
                    _FONT, 0.62 * scale, _WARNING_FG, 2, cv2.LINE_AA)

        (detail_width, _), _ = cv2.getTextSize(detail, _FONT, 0.42 * scale, 1)
        cv2.putText(frame, detail,
                    (max(8, (width - detail_width) // 2), int(band_height * 0.82)),
                    _FONT, 0.42 * scale, _WARNING_FG, 1, cv2.LINE_AA)

    return frames


def draw_calibration_warning(frames: list, line_support: float) -> list:
    """
    Stamp every frame with a court-calibration warning.

    Args:
        frames:       output frames, drawn in place.
        line_support: the measured score, shown so the reader can see how far off the
                      fit was rather than just that it failed.
    """
    return _stamp(
        frames,
        _BANNER_TEXT,
        f"{_SUBTEXT}  (line support {line_support:.3f})",
        _WARNING_BG,
    )


def draw_frame_rate_warning(frames: list, fps: float, status: str,
                            supported_range: tuple[float, float]) -> list:
    """
    Stamp every frame with a frame-rate caveat.

    A rendered video is what gets screenshotted and shared, and a shot count from a
    60 fps clip looks exactly like a shot count from a 30 fps one. This is the only
    place the distinction reaches someone who never opens summary.json.

    Args:
        frames:          output frames, drawn in place.
        fps:             the clip's actual frame rate.
        status:          "partially_supported" or "unsupported". A supported clip is not
                         stamped: a banner that appears on every run is one nobody reads.
        supported_range: (min, max) fps the published numbers were measured on.
    """
    if status == "unsupported":
        headline = f"UNSUPPORTED FRAME RATE ({fps:.0f} fps) - event counts are NOT measurements"
        background = _WARNING_BG
    else:
        headline = f"FRAME RATE OUTSIDE MEASURED RANGE ({fps:.0f} fps) - counts are approximate"
        background = _CAUTION_BG

    low, high = supported_range
    detail = (f"Event thresholds are counted in frames and were measured at "
              f"{low:.0f}-{high:.0f} fps. See summary.json")
    return _stamp(frames, headline, detail, background)
