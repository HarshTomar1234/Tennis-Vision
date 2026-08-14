"""
scripts/build_clip_suite.py
───────────────────────────
Builds the evaluation clip suite from a manifest of public match videos.

Why a script and not a folder of files
--------------------------------------
The clips themselves are third-party broadcast footage and are not redistributed
through this repo (see .gitignore). What IS committed is this script and its manifest,
so anyone can rebuild the identical suite and re-run every published per-surface
number. The evidence is reproducible even though the video is not ours to ship.

The manifest records surface and players per clip, which is what makes the results
table meaningful: "8/9 court fits" says little, "15/15 grass, 14/15 clay, 15/15 hard"
says something.

Usage:
    python scripts/build_clip_suite.py --list            # show the manifest
    python scripts/build_clip_suite.py --surface grass   # build one surface
    python scripts/build_clip_suite.py                   # build everything

Requires yt-dlp and ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CLIP_ROOT = Path("datasets/eval_clips")
MANIFEST_PATH = Path("datasets/clip_manifest.json")
SOURCE_CACHE = Path("datasets/source_videos")

CLIP_SECONDS = 15
TARGET_HEIGHT = 720

# YouTube caps downloads at 360p for the default clients because of SABR streaming;
# android_vr still exposes the 720p/1080p adaptive formats. Without this the suite
# would be built at a resolution the ball detector cannot work with.
YT_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=android_vr"]
YT_FORMAT = "136+140/22/18"   # 720p video + audio, falling back to muxed formats


@dataclass(frozen=True)
class ClipSpec:
    """One clip to cut: which video, where, and what it contains."""
    video_id: str
    start_s: int
    surface: str          # grass | clay | hard
    players: str
    note: str = ""        # e.g. "serve + radar overlay", "camera motion"

    @property
    def name(self) -> str:
        return f"{self.surface}_{self.video_id}_{self.start_s:04d}"


def load_manifest() -> list[ClipSpec]:
    if not MANIFEST_PATH.exists():
        sys.exit(f"Manifest not found: {MANIFEST_PATH}")
    raw = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [ClipSpec(**entry) for entry in raw["clips"]]


def ensure_source(video_id: str, cookies_from: str | None = None,
                  pause_s: int = 20, attempts: int = 3) -> Path | None:
    """
    Download a source video once; reuse it for every clip cut from it.

    Downloading several full matches back to back trips YouTube's bot check
    ("Sign in to confirm you're not a bot"), which is a rate limit rather than a
    permanent block: the first sources succeed and later ones fail. Retries with a
    growing pause therefore recover the run, and a pause between fresh downloads
    keeps it from tripping in the first place. `--cookies-from-browser` avoids the
    check entirely but needs the browser closed on Windows (it locks its cookie DB),
    so it is offered rather than required.
    """
    SOURCE_CACHE.mkdir(parents=True, exist_ok=True)
    target = SOURCE_CACHE / f"{video_id}.mp4"
    if target.exists():
        return target

    cookie_args = ["--cookies-from-browser", cookies_from] if cookies_from else []

    for attempt in range(1, attempts + 1):
        print(f"  downloading source {video_id} (attempt {attempt}/{attempts}) ...")
        result = subprocess.run(
            ["yt-dlp", "--no-playlist", *YT_CLIENT_ARGS, *cookie_args,
             "-f", YT_FORMAT, "--merge-output-format", "mp4",
             "-o", str(SOURCE_CACHE / "%(id)s.%(ext)s"),
             f"https://www.youtube.com/watch?v={video_id}"],
            capture_output=True, text=True,
        )
        if target.exists():
            time.sleep(pause_s)   # be a good citizen before the next fresh download
            return target

        error = (result.stderr or "").strip().splitlines()
        last = error[-1] if error else "unknown error"
        rate_limited = "not a bot" in last or "429" in last
        print(f"  attempt {attempt} failed: {last[:140]}")
        if not rate_limited or attempt == attempts:
            break
        backoff = pause_s * attempt * 3
        print(f"  rate limited — waiting {backoff}s before retry")
        time.sleep(backoff)

    return None


def cut_clip(source: Path, spec: ClipSpec, out_dir: Path) -> Path | None:
    """Cut and normalise one clip. Re-encoded so every clip has a clean keyframe start."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{spec.name}.mp4"
    if target.exists():
        return target

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", str(spec.start_s), "-t", str(CLIP_SECONDS), "-i", str(source),
         "-vf", f"scale=-2:{TARGET_HEIGHT}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(target)],
        capture_output=True, text=True,
    )
    if not target.exists():
        print(f"  FAILED cutting {spec.name}: {result.stderr.strip()[:200]}")
        return None
    return target


def main() -> int:
    # Windows consoles default to cp1252, which cannot encode the box-drawing and
    # em-dash characters in this module's docstring — argparse prints it for --help
    # and would raise UnicodeEncodeError before doing anything useful.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", choices=["grass", "clay", "hard"],
                    help="build only one surface")
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    ap.add_argument("--cookies-from-browser", default=None, metavar="BROWSER",
                    help="e.g. chrome/edge/firefox — avoids YouTube's bot check. On "
                         "Windows the browser must be CLOSED; it locks its cookie DB.")
    ap.add_argument("--pause", type=int, default=20, metavar="SECONDS",
                    help="pause between fresh source downloads (default 20)")
    args = ap.parse_args()

    specs = load_manifest()
    if args.surface:
        specs = [s for s in specs if s.surface == args.surface]

    if args.list:
        by_surface: dict[str, int] = {}
        for s in specs:
            by_surface[s.surface] = by_surface.get(s.surface, 0) + 1
        print(f"\n{len(specs)} clips in manifest: " +
              ", ".join(f"{k} {v}" for k, v in sorted(by_surface.items())))
        for s in specs:
            print(f"  {s.name:34s} {s.players:34s} {s.note}")
        return 0

    for tool in ("yt-dlp", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found on PATH — required to build the suite")

    built, failed = [], []
    for spec in specs:
        source = ensure_source(spec.video_id, args.cookies_from_browser, args.pause)
        if source is None:
            failed.append(spec)
            continue
        if cut_clip(source, spec, CLIP_ROOT / spec.surface):
            built.append(spec)
            print(f"  built {spec.surface}/{spec.name}.mp4")
        else:
            failed.append(spec)

    index = CLIP_ROOT / "index.csv"
    with open(index, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "surface", "players", "source_video_id", "start_s", "note"])
        for s in built:
            writer.writerow([f"{s.surface}/{s.name}.mp4", s.surface, s.players,
                             s.video_id, s.start_s, s.note])

    print(f"\nBuilt {len(built)} clips, {len(failed)} failed. Index: {index}")
    if failed:
        print("Failed: " + ", ".join(s.name for s in failed))
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
