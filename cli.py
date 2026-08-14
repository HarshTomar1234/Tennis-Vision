"""
cli.py
──────
Console entry point for the `tennis-vision` command.

Subcommands are thin wrappers over the existing modules rather than reimplementations,
so there is exactly one code path per capability and the CLI cannot drift from what
`python main.py` does.

    tennis-vision analyze clip.mp4 -o output/run.avi
    tennis-vision download-models
    tennis-vision version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

__version__ = "0.1.0"


def _cmd_analyze(argv: list[str]) -> int:
    """Run the full pipeline on one video."""
    parser = argparse.ArgumentParser(
        prog="tennis-vision analyze",
        description="Analyse a tennis video: ball, players, court, shots and stats.",
    )
    parser.add_argument("input", help="path to the input video")
    parser.add_argument("-o", "--output", default=None,
                        help="annotated output video path (default: from config)")
    parser.add_argument("-c", "--config", default="configs/config.yaml",
                        help="config YAML (use configs/dev.yaml to enable caching)")
    parser.add_argument("--no-stubs", action="store_true",
                        help="force fresh detection, ignoring any cached stubs")
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="process only the first N frames (0 = all) — quick check "
                             "on a long video before a full run")
    parser.add_argument("--fast", action="store_true",
                        help="single-frame court keypoints; faster, less camera-robust")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 2

    # main.main() reads sys.argv, so hand it the flags it expects rather than
    # duplicating the pipeline here.
    forwarded = ["main.py", "--input", args.input, "--config", args.config]
    if args.output:
        forwarded += ["--output", args.output]
    if args.max_frames:
        forwarded += ["--max-frames", str(args.max_frames)]
    if args.no_stubs:
        forwarded.append("--no-stubs")
    if args.fast:
        forwarded.append("--fast")
    if args.debug:
        forwarded.append("--debug")

    import main as pipeline

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        pipeline.main()
    finally:
        sys.argv = original_argv
    return 0


def _cmd_download_models(argv: list[str]) -> int:
    """Fetch model weights into models/."""
    argparse.ArgumentParser(
        prog="tennis-vision download-models",
        description="Download the model weights the pipeline needs.",
    ).parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    import download_models

    return download_models.main()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tennis-vision",
        description="Measured, reproducible tennis video analysis from a single camera.",
        epilog="Run 'tennis-vision <command> --help' for command-specific options.",
    )
    parser.add_argument("command", nargs="?", default="help",
                        choices=["analyze", "download-models", "version", "help"],
                        help="what to do")
    args, rest = parser.parse_known_args()

    if args.command == "analyze":
        return _cmd_analyze(rest)
    if args.command == "download-models":
        return _cmd_download_models(rest)
    if args.command == "version":
        print(f"tennis-vision {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
