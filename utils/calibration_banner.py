"""
utils/calibration_banner.py
───────────────────────────
Draws an on-screen warning when a run's court fit failed validation.

Why this is not optional polish
-------------------------------
The pipeline already detects an untrustworthy court fit and records it in
`summary.json` as `court_calibrated: false`. But the rendered video is what people
actually look at, share and screenshot — and until now it drew exactly the same
confident km/h figures, stats panel and mini-court dots whether the court was fitted
correctly or fitted to the crowd. A viewer had no way to tell the two apart.

That is the single most damaging failure mode for a project whose whole claim is that
its numbers are measured: a wrong number presented confidently travels further than a
right one, and it travels as a screenshot with no JSON attached.

Anything derived from the court homography — speeds, distances, mini-court positions —
is meaningless when the fit is wrong. Rather than silently hide those overlays (which
would make the failure invisible in a different way), the run is stamped so the output
carries its own caveat.
"""
from __future__ import annotations

import cv2

_BANNER_TEXT = "COURT NOT CALIBRATED - speeds and positions are NOT measurements"
_SUBTEXT = "The detected court does not lie on the real court lines. See summary.json"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WARNING_BG = (0, 0, 140)      # dark red, BGR
_WARNING_FG = (255, 255, 255)


def draw_calibration_warning(frames: list, line_support: float) -> list:
    """
    Stamp every frame with a warning band. Mutates and returns `frames`.

    Args:
        frames:       output frames, drawn in place.
        line_support: the measured score, shown so the reader can see how far off the
                      fit was rather than just that it failed.
    """
    if not frames:
        return frames

    height, width = frames[0].shape[:2]
    band_height = max(46, int(height * 0.075))
    scale = width / 1280.0

    for frame in frames:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, band_height), _WARNING_BG, -1)
        # Semi-transparent so the banner cannot completely hide the action it warns about.
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        (text_width, _), _ = cv2.getTextSize(_BANNER_TEXT, _FONT, 0.62 * scale, 2)
        cv2.putText(frame, _BANNER_TEXT,
                    (max(8, (width - text_width) // 2), int(band_height * 0.45)),
                    _FONT, 0.62 * scale, _WARNING_FG, 2, cv2.LINE_AA)

        detail = f"{_SUBTEXT}  (line support {line_support:.3f})"
        (detail_width, _), _ = cv2.getTextSize(detail, _FONT, 0.42 * scale, 1)
        cv2.putText(frame, detail,
                    (max(8, (width - detail_width) // 2), int(band_height * 0.82)),
                    _FONT, 0.42 * scale, _WARNING_FG, 1, cv2.LINE_AA)

    return frames
