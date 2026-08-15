"""
utils/web_video.py
──────────────────
Produces a browser-playable copy of the annotated video.

Why this is needed
------------------
The pipeline writes AVI with the MPEG-4 Part 2 codec, which OpenCV's VideoWriter
produces reliably across platforms. No browser can play it: Chrome, Firefox, Edge and
Safari support H.264/MP4, WebM and Ogg, and none of them support AVI. The 3-D viewer
referenced that file directly, so its "annotated video" tab showed nothing at all —
silently, because a <video> element with an unsupported source simply stays blank.

Transcoding to H.264 costs a few seconds and makes the whole output shareable: the
viewer, the video and the JSON can be zipped and opened by anyone.

`+faststart` moves the MP4 index to the front of the file so playback can begin before
the whole file is read, which matters when the page is opened over a network share or
served rather than opened locally.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def to_browser_playable(source: str | Path, crf: int = 23) -> Path | None:
    """
    Transcode `source` to an H.264 MP4 beside it and return the new path.

    Args:
        source: the rendered video (typically .avi).
        crf:    x264 quality, lower is better. 23 keeps annotations legible at a size
                that still shares easily.

    Returns:
        Path to the .mp4, or None when ffmpeg is unavailable or the transcode fails.
        Returning None rather than raising is deliberate: a missing web copy should
        cost the viewer its video tab, not the whole analysis run.
    """
    source = Path(source)
    if not source.exists():
        logger.warning(f"Cannot transcode: {source} does not exist")
        return None

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found on PATH — the 3-D viewer's video tab will be "
                       "empty, because browsers cannot play the AVI the pipeline writes")
        return None

    target = source.with_suffix(".mp4")
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(source),
         "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
         "-pix_fmt", "yuv420p",          # required for broad browser compatibility
         "-movflags", "+faststart",
         str(target)],
        capture_output=True, text=True,
    )

    if not target.exists() or target.stat().st_size == 0:
        logger.warning(f"Transcode failed: {result.stderr.strip()[:200]}")
        return None
    return target
