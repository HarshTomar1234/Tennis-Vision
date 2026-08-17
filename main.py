#!/usr/bin/env python3
"""
Tennis-Vision main pipeline.

Usage:
  python main.py                                 # uses config.yaml defaults
  python main.py --input input_videos/clip.mp4  # override input
  python main.py --no-stubs                      # fresh detection run
  python main.py --fast                          # single-frame keypoints, no ByteTrack
  python main.py --debug                         # verbose log output
"""
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import yaml

import constants
from court_line_detector import CourtLineDetector
from mini_visual_court import MiniCourt
from trackers import BallTracker, PlayerTracker
from utils import (
    MIN_LINE_SUPPORT,
    PoseEstimator,
    ShotClassifier,
    UILayoutManager,
    assess_court_fit,
    classify_contact_vs_bounce,
    classify_floor_level,
    classify_forehand_backhand,
    classify_reversals_by_trajectory,
    convert_pixel_distance_to_meters,
    derive_shot_frames,
    detect_xvelocity_candidates,
    draw_player_stats,
    draw_shot_classifications,
    measure_distance_between_points,
    merge_nearby_candidates,
    peak_speed_kmh_near_frame,
    read_video,
    save_video,
    smooth_trajectories,
    stub_matches_frames,
    stub_path_for_video,
)
from utils.bounce_candidates import detect_bounce_candidates
from utils.calibration_banner import draw_calibration_warning
from utils.serve_detector import detect_serve_frames
from utils.serve_landing import find_serve_landing
from utils.shot_physics import classify_from_physics, is_lob
from utils.trajectory_3d import crosses_net, reconstruct_rally
from utils.viewer_3d import build_viewer
from utils.web_video import to_browser_playable
from utils.serve_speed import bounce_is_in_service_box, find_serve_and_bounce, serve_speed_kmh


# ── Config & CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tennis-Vision: AI-powered tennis match analysis")
    p.add_argument("--input",    "-i", help="Path to input video (overrides config)")
    p.add_argument("--output",   "-o", help="Path to output video (overrides config)")
    p.add_argument("--config",   "-c", default="configs/config.yaml", help="Config YAML path")
    p.add_argument("--no-stubs", action="store_true", help="Disable cached stubs, force fresh detection")
    p.add_argument("--fast",     action="store_true", help="Fast mode: first-frame keypoints, no ByteTrack")
    p.add_argument("--debug",    action="store_true", help="Enable DEBUG log level")
    p.add_argument("--max-frames", type=_positive_int, default=0, metavar="N",
                   help="Process only the first N frames (0 = all). Useful for a quick "
                        "check on a long video before committing to a full run.")
    return p.parse_args()


def _lowest_ball_point_near(
    ball_detections: list, frame: int, window: int = 4
) -> tuple[float, float] | None:
    """
    Ball centre at its lowest point on screen within ±`window` frames of `frame`.

    In image coordinates y grows downward, so the largest y is the ball nearest the
    court surface - the instant of the bounce. Bounce candidates are accurate to a few
    frames, and using a frame where the ball is still airborne sends its floor
    projection far down-court, because the camera ray through a raised ball meets the
    ground well beyond the true landing point.

    Returns None when no ball is detected anywhere in the window.
    """
    best: tuple[float, float] | None = None
    for f in range(max(0, frame - window), min(len(ball_detections), frame + window + 1)):
        bbox = ball_detections[f].get(1)
        if bbox is None:
            continue
        centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        if best is None or centre[1] > best[1]:
            best = centre
    return best


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 0:
        # A negative value would silently slice frames off the END of the video
        # (frames[:-5]) while the log claimed "first -5 frames".
        raise argparse.ArgumentTypeError(f"must be >= 0, got {n}")
    return n


# These must stay in sync with configs/config.yaml. They are not a minimal fallback:
# they are what runs when no YAML is present, which is the case for anyone who installed
# the wheel rather than cloning. When they drifted from config.yaml the result was a
# silently degraded pipeline for exactly those users - the superseded court model
# (median 4.03px against 2.90px, 4 of 9 clips passing against 8), the YOLO ball detector
# instead of TrackNet, and no pose-based shot classification. Nothing errored and nothing
# in the output said the run was worse than the README's measured numbers.
_DEFAULTS: dict = {
    "pipeline": {
        "per_frame_keypoints": True,
        "use_bytetrack": True,
        "shot_classification": True,
        "use_homography": True,
        "use_tracknet": True,
        "use_pose_shots": True,
    },
    "models": {
        "player": "yolov8x",
        "ball": "models/last.pt",
        # The geometrically fine-tuned weights, which is what scripts/download_models.py
        # fetches. models/keypoints_model.pth is the superseded original.
        "court": "models/keypoints_model_geoaug.pth",
        "tracknet": "models/tracknet.pt",
        "pose": "models/pose_landmarker_lite.task",
    },
    "io": {
        "input_video": "input_videos/input_video_2.mp4",
        "output_video": "output/videos/output_video.avi",
        "output_frames_dir": "output/frames",
        "output_stats_dir": "output/stats",
        "log_dir": "logs",
        "player_stub_path": "tracker_stubs/player_detections.pkl",
        "ball_stub_bytetrack_path": "tracker_stubs/ball_detections_tracked.pkl",
        "ball_stub_path": "tracker_stubs/ball_detections.pkl",
        "tracknet_stub_path": "tracker_stubs/ball_detections_tracknet.pkl",
    },
    "stubs": {
        # Off, matching configs/config.yaml and the README. A stub is one video's cached
        # detections; the paths are keyed per clip now, but caching still hides genuine
        # detector changes behind stale results, so a first run should always be real.
        # configs/dev.yaml turns these on for repeated runs on one clip.
        "use_player_stubs": False,
        "use_ball_stubs": False,
    },
    "detection": {
        "player_confidence": 0.7,
        "ball_confidence": 0.6,
        "shot_player_distance_px": 300,
    },
    "shot_classifier": {
        "volley_distance_threshold": 40,
        "smash_height_threshold": 0.7,
        "net_y_position_relative": 0.5,
    },
    "logging": {
        "level": "INFO",
        "write_to_file": True,
    },
}


_BASE_CONFIG = "configs/config.yaml"


def _merge_yaml_into(cfg: dict, path: str) -> None:
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    for section, values in user.items():
        if section in cfg and isinstance(values, dict):
            cfg[section].update(values)
        else:
            cfg[section] = values


def load_config(config_path: str) -> dict:
    """
    Load configuration as three layers: built-in defaults, then configs/config.yaml,
    then the requested file.

    The base config always applies. Before this, an alternate config such as
    configs/dev.yaml only merged over the built-in defaults - so dev.yaml, which
    documents itself as "identical to config.yaml except caching", silently ran the
    OLD court model because the fine-tuned weights are configured in config.yaml's
    models: section, not in the code defaults.
    """
    cfg = deepcopy(_DEFAULTS)
    if os.path.exists(_BASE_CONFIG):
        _merge_yaml_into(cfg, _BASE_CONFIG)
    if config_path and os.path.abspath(config_path) != os.path.abspath(_BASE_CONFIG):
        if os.path.exists(config_path):
            _merge_yaml_into(cfg, config_path)
        else:
            # An explicitly requested config that doesn't exist is a user error, not
            # a situation to paper over with defaults.
            print(f"warning: config file not found: {config_path} - "
                  f"using {_BASE_CONFIG} + built-in defaults", file=sys.stderr)
    return cfg


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(cfg: dict) -> logging.Logger:
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_cfg.get("write_to_file", True):
        log_dir = Path(cfg["io"].get("log_dir", "logs"))
        log_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handlers.append(logging.FileHandler(log_dir / f"run_{stamp}.log", encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    return logging.getLogger("tennis_vision")


# ── Stats output ───────────────────────────────────────────────────────────────

def save_stats(stats_df: pd.DataFrame, output_dir: str, logger: logging.Logger,
               court_fit: tuple[bool, float] | None = None,
               serve_speed_kmh: float = 0.0,
               trajectories_3d: list | None = None):
    """Write full stats CSV + match-summary JSON to output_dir."""
    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out / f"stats_{stamp}.csv"
    stats_df.to_csv(csv_path, index=False)
    logger.info(f"Stats CSV  → {csv_path}")

    last = stats_df.iloc[-1]

    def _safe(col: str, default=0.0):
        return round(float(last.get(col, default)), 1)

    summary = {
        "generated_at": stamp,
        "total_shots_p1": int(last.get("player_1_number_of_shots", 0)),
        "total_shots_p2": int(last.get("player_2_number_of_shots", 0)),
        "avg_shot_speed_p1_kmh": _safe("player_1_average_shot_speed"),
        "avg_shot_speed_p2_kmh": _safe("player_2_average_shot_speed"),
        "avg_player_speed_p1_kmh": _safe("player_1_average_player_speed"),
        "avg_player_speed_p2_kmh": _safe("player_2_average_player_speed"),
    }
    if serve_speed_kmh > 0:
        # Named for what it is: average over the flight, not a radar-equivalent
        # contact speed. Consumers must not present it as the latter.
        summary["serve_avg_flight_speed_kmh"] = round(serve_speed_kmh, 1)
    if court_fit is not None:
        # Consumers must be able to tell a measured speed from one derived off a
        # court fitted to the wrong part of the frame - the numbers look identical.
        is_valid, support = court_fit
        summary["court_calibrated"] = bool(is_valid)
        summary["court_line_support"] = round(float(support), 3)
        if not is_valid:
            summary["warning"] = (
                "Court fit failed validation - speeds, distances and mini-court "
                "positions are derived from an unreliable court and should not be "
                "treated as measurements."
            )
    if trajectories_3d:
        # Speeds here include the vertical component the floor projection discards, so
        # they are not comparable with avg_shot_speed_*_kmh above and are named apart.
        speeds = [t.speed_kmh for t in trajectories_3d]
        summary["shot_speed_3d_kmh"] = {
            "segments": len(speeds),
            "mean": round(sum(speeds) / len(speeds), 1),
            "max": round(max(speeds), 1),
            "note": ("Free-flight reconstruction between floor-anchored events. "
                     "Average over each segment, so below a radar reading at contact; "
                     "drag and spin are not modelled."),
        }

    json_path = out / f"summary_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {json_path}")

    if trajectories_3d:
        # Written separately: the viewer needs the full arcs, and embedding a few
        # thousand points in the summary would drown the numbers a human reads.
        scene_path = out / f"trajectory3d_{stamp}.json"
        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump({
                "segments": [
                    {
                        "start_frame": t.start_frame,
                        "end_frame": t.end_frame,
                        "speed_kmh": round(t.speed_kmh, 1),
                        "apex_height_m": round(t.apex_height_m, 2),
                        "points": [[round(c, 3) for c in p] for p in t.points],
                    }
                    for t in trajectories_3d
                ],
            }, f, indent=2)
        logger.info(f"3-D scene  → {scene_path}")


# ── Pipeline ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI flags override config values
    if args.input:
        cfg["io"]["input_video"] = args.input
    if args.output:
        cfg["io"]["output_video"] = args.output
    if args.no_stubs:
        cfg["stubs"]["use_player_stubs"] = False
        cfg["stubs"]["use_ball_stubs"] = False
    if args.debug:
        cfg["logging"]["level"] = "DEBUG"
    if args.fast:
        cfg["pipeline"]["per_frame_keypoints"] = False
        cfg["pipeline"]["use_bytetrack"] = False

    logger = setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("Tennis-Vision pipeline starting")
    logger.info(f"Input   : {cfg['io']['input_video']}")
    logger.info(f"Output  : {cfg['io']['output_video']}")
    logger.info(f"Config  : {args.config}")
    logger.info(f"Mode    : {'camera-robust' if cfg['pipeline']['per_frame_keypoints'] else 'fast'}")
    logger.info(f"Homogr. : {'on' if cfg['pipeline']['use_homography'] else 'off (approx)'}")
    logger.info("=" * 60)

    # ── 1. Load video ──────────────────────────────────────────────
    logger.info("[1/9] Loading video frames...")
    input_path = cfg["io"]["input_video"]
    video_frames = read_video(input_path)
    if not video_frames:
        logger.error(f"No frames could be read from {input_path}. The file may be "
                     f"missing, empty, or in a codec OpenCV cannot decode.")
        sys.exit(1)

    if args.max_frames and args.max_frames < len(video_frames):
        logger.info(f"  Limiting to first {args.max_frames} frames (--max-frames)")
        video_frames = video_frames[:args.max_frames]

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        logger.warning("Could not read FPS from video header, defaulting to 30")
        fps = 30.0

    logger.info(f"  {len(video_frames)} frames | {fps:.1f} fps | "
                f"{video_frames[0].shape[1]}×{video_frames[0].shape[0]}px")

    # ── 2. Player detection ────────────────────────────────────────
    logger.info("[2/9] Player detection...")
    player_tracker = PlayerTracker(model_path=cfg["models"]["player"])
    use_player_stubs = cfg["stubs"]["use_player_stubs"]
    # Keyed to this clip for the same reason as the ball stub below: a shared cache
    # handed one video's player boxes to another.
    player_stub = stub_path_for_video(cfg["io"]["player_stub_path"], input_path)
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=use_player_stubs,
        stub_path=player_stub,
    )
    source = f"stub ({player_stub})" if use_player_stubs else "fresh YOLO"
    logger.info(f"  Source: {source}")

    # ── 3. Ball detection ──────────────────────────────────────────
    logger.info("[3/9] Ball detection...")
    use_tracknet = cfg["pipeline"].get("use_tracknet", False)

    if use_tracknet:
        import pickle
        from trackers.tracknet_ball_tracker import TrackNetBallTracker
        ball_tracker = TrackNetBallTracker(
            model_path=cfg["models"].get("tracknet", "models/tracknet.pt")
        )
        # Keyed to this clip: a shared stub silently fed one video's ball positions to
        # another. See utils.video_utils.stub_path_for_video.
        tracknet_stub = stub_path_for_video(
            cfg["io"].get("tracknet_stub_path", "tracker_stubs/ball_detections_tracknet.pkl"),
            input_path,
        )
        use_ball_stubs = cfg["stubs"].get("use_ball_stubs", False)

        cached = None
        if use_ball_stubs and Path(tracknet_stub).exists():
            with open(tracknet_stub, "rb") as f:
                cached = pickle.load(f)
            if not stub_matches_frames(cached, video_frames):
                logger.warning(
                    f"  Stub {tracknet_stub} has {len(cached)} frames but this clip has "
                    f"{len(video_frames)}; ignoring it and detecting fresh"
                )
                cached = None

        if cached is not None:
            logger.info(f"  TrackNet v2 - loading from stub ({tracknet_stub})")
            ball_detections = cached
        else:
            logger.info("  TrackNet v2 (temporal heatmap, 3-frame context)")
            ball_detections = ball_tracker.detect_frames(video_frames)
            Path(tracknet_stub).parent.mkdir(exist_ok=True)
            with open(tracknet_stub, "wb") as f:
                pickle.dump(ball_detections, f)
            logger.info(f"  Saved TrackNet detections → {tracknet_stub}")
    else:
        ball_tracker = BallTracker(model_path=cfg["models"]["ball"])
        use_ball_stubs = cfg["stubs"]["use_ball_stubs"]

        if cfg["pipeline"]["use_bytetrack"]:
            logger.info("  ByteTrack mode (Kalman filter + temporal smoothing)")
            ball_detections = ball_tracker.detect_frames_with_tracking(
                video_frames,
                read_from_stub=use_ball_stubs,
                stub_path=cfg["io"]["ball_stub_bytetrack_path"],
            )
        else:
            logger.info("  Standard YOLO mode")
            ball_detections = ball_tracker.detect_frames(
                video_frames,
                read_from_stub=use_ball_stubs,
                stub_path=cfg["io"]["ball_stub_path"],
            )

    raw_detected = sum(1 for d in ball_detections if d.get(1))
    total = len(video_frames)
    logger.info(f"  Raw detections: {raw_detected}/{total} frames "
                f"({100 * raw_detected / total:.1f}%)")

    logger.info("  Interpolating missing positions...")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # ── 4. Court line detection ────────────────────────────────────
    logger.info("[4/9] Court keypoint detection...")
    court_detector = CourtLineDetector(cfg["models"]["court"])

    if cfg["pipeline"]["per_frame_keypoints"]:
        logger.info("  Per-frame mode (camera-robust, slower)...")
        all_court_keypoints = court_detector.predict_all_frames(
            video_frames, smooth=True, window_size=5
        )
        court_keypoints = all_court_keypoints[0]
    else:
        logger.info("  Single-frame mode (fast)...")
        court_keypoints = court_detector.predict(video_frames[0])
        all_court_keypoints = [court_keypoints] * len(video_frames)

    logger.info(f"  {len(all_court_keypoints)} keypoint sets ready")

    court_fit = assess_court_fit(video_frames, all_court_keypoints)
    court_valid, line_support = court_fit
    if court_valid:
        logger.info(f"  Court fit OK (line support {line_support:.3f})")
    else:
        logger.warning(
            f"  COURT FIT FAILED VALIDATION (line support {line_support:.3f} < "
            f"{MIN_LINE_SUPPORT}). The detected keypoints do not lie on painted "
            f"court lines, so speeds, distances and mini-court positions from this "
            f"run are NOT measurements. See utils/court_validity.py."
        )

    # ── 5. Player selection ────────────────────────────────────────
    logger.info("[5/9] Filtering to 2 main players...")
    player_detections = player_tracker.choose_and_filter_players(
        player_detections, court_keypoints
    )

    # Build the id map from every frame, not frame 0: selection already narrowed
    # this to the chosen players, but a player can be absent from the opening
    # frame (replay wipe, off-screen at serve) and reading only frame 0 would
    # silently drop them from the whole run.
    chosen_ids = sorted({track_id for frame in player_detections for track_id in frame})
    player_id_map = {orig: new for new, orig in enumerate(chosen_ids[:2], start=1)}
    logger.info(f"  Player ID mapping: {player_id_map}")

    normalized: list[dict] = []
    for frame in player_detections:
        normalized.append(
            {player_id_map[k]: v for k, v in frame.items() if k in player_id_map}
        )
    player_detections = normalized

    # ── 6. Mini-court setup ────────────────────────────────────────
    logger.info("[6/9] Building mini-court visualization...")
    layout_manager = UILayoutManager(video_frames[0].shape, court_keypoints)
    mini_court = MiniCourt(
        video_frames[0],
        layout_params=layout_manager.get_mini_court_params(),
    )
    stats_params = layout_manager.get_stats_panel_params()
    # Metres per mini-court pixel. Defined here, next to the court it describes, because
    # everything downstream that speaks in real-world units depends on it.
    px_to_m_scale = constants.DOUBLE_LINE_WIDTH / mini_court.get_width_of_mini_court()
    logger.info(f"  Mini-court position: start=({mini_court.start_x}, {mini_court.start_y}) "
                f"size={mini_court.mini_court_width}×{mini_court.mini_court_height}px")

    # ── 7. Shot frames + coordinate mapping ───────────────────────
    logger.info("[7/9] Detecting shot frames + mapping to mini-court...")
    # Candidate generation, merging and the contact-vs-bounce split all live in
    # utils.hit_bounce_classifier.derive_shot_frames so the evals grade this exact
    # logic rather than a re-implementation of it. See that function for why three
    # generators are needed and what each one is blind to.
    shot_dist_px = cfg.get("detection", {}).get("shot_player_distance_px", 300)
    confirmed_shot_frames, bounce_frames, raw_reversal_frames = derive_shot_frames(
        ball_tracker, ball_detections, player_detections, shot_dist_px
    )

    # Floor-level anchors for BALL GEOMETRY: every trajectory reversal (contact or
    # bounce) is a valid homography anchor - the floor transform is correct at floor
    # level regardless of which caused it. In-flight frames interpolate between
    # anchors instead of being projected (wrong - the ball has real height while
    # airborne). See utils.ball_state and docs/journal/0003 for why contact-vs-bounce
    # is NOT needed for this part.
    floor_states = classify_floor_level(raw_reversal_frames, len(video_frames))

    logger.info(
        f"  {len(raw_reversal_frames)} reversals → "
        f"{sum(1 for s in floor_states if s == 'floor_level')} floor-level anchors | "
        f"{len(confirmed_shot_frames)} confirmed shots + {len(bounce_frames)} bounces"
    )
    ball_shot_frames = confirmed_shot_frames

    use_hom = cfg["pipeline"]["use_homography"]
    logger.info(f"  Coordinate mapping: {'homography (perspective-correct, floor-anchored)' if use_hom else 'nearest-keypoint (approximate)'}")

    player_mini_court, _unused_ball_mini_court = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, all_court_keypoints,
        use_homography=use_hom,
    )
    ball_mini_court = mini_court.convert_ball_to_mini_court_coordinates(
        ball_detections, all_court_keypoints, floor_states, use_homography=use_hom,
    )

    # Kalman smoothing (Phase 1, Step 3) - stabilizes the projected dots frame to
    # frame (João's feedback) and gives continuous velocity for the shot-speed stat
    # below, instead of depending on distance between two possibly-noisy shot-frame
    # detections. See docs/journal/0004.
    logger.info("  Smoothing positions with Kalman filter...")
    player_mini_court, _player_velocities = smooth_trajectories(player_mini_court)
    ball_mini_court, ball_velocities       = smooth_trajectories(ball_mini_court)

    # ── 8. Shot classification ─────────────────────────────────────
    shot_classifications: dict = {}
    if cfg["pipeline"]["shot_classification"]:
        logger.info("[8/9] Classifying shots...")
        sc_cfg = cfg.get("shot_classifier", {})
        shot_classifier = ShotClassifier(
            volley_threshold=sc_cfg.get("volley_distance_threshold", 40),
            smash_height_threshold=sc_cfg.get("smash_height_threshold", 0.7),
            net_y_relative=sc_cfg.get("net_y_position_relative", 0.5),
        )
        # Serves are identified from physical evidence (ball struck above the
        # player's head, from a baseline) rather than from position in the sequence.
        court_kp = mini_court.get_court_drawing_keypoints()
        serve_frames = detect_serve_frames(
            ball_shot_frames, ball_detections, player_detections, player_mini_court,
            far_baseline_y=court_kp[1], near_baseline_y=court_kp[5],
        )
        logger.info(f"  Serve detection: {len(serve_frames)} of {len(ball_shot_frames)} "
                    f"contacts carry serve evidence {serve_frames if serve_frames else ''}")

        shot_classifications = shot_classifier.classify_shots(
            player_mini_court, ball_mini_court, ball_shot_frames,
            mini_court.court_drawing_height, serve_frames=serve_frames,
        )

        # Replace the position-guessed Volley/Smash labels with physically evidenced
        # ones where the evidence exists. ShotClassifier decides those two from court
        # position alone, which has never had ground truth and produced smashes in the
        # middle of baseline rallies. See utils/shot_physics.py.
        kp_sp = mini_court.get_court_drawing_keypoints()
        net_y_px = (kp_sp[1] + kp_sp[5]) / 2.0
        half_court_m = abs(kp_sp[5] - net_y_px) * px_to_m_scale
        bounce_lookup = sorted(bounce_frames)
        physics_calls: dict[int, list[str]] = {}

        for frame in sorted(shot_classifications):
            if frame in serve_frames:
                continue   # already evidenced by the serve detector

            players_here = player_detections[frame] if frame < len(player_detections) else {}
            ball_bbox = ball_detections[frame].get(1) if frame < len(ball_detections) else None
            hitter_id = shot_classifications[frame].get("player_id")
            hitter_box = players_here.get(hitter_id)
            if ball_bbox is None or hitter_box is None:
                continue

            above_head = ((ball_bbox[1] + ball_bbox[3]) / 2.0) < hitter_box[1]
            hitter_mini = player_mini_court.get(frame, {}).get(hitter_id)
            if hitter_mini is None:
                continue
            distance_from_net_m = abs(hitter_mini[1] - net_y_px) * px_to_m_scale

            # Bounces strictly between the previous contact and this one. Zero means
            # the ball was struck before it bounced - the definition of a volley.
            previous = [f for f in ball_shot_frames if f < frame]
            bounces_between = (
                sum(1 for b in bounce_lookup if previous[-1] < b < frame)
                if previous else None
            )

            # Lob needs the 3-D apex, which is reconstructed later in the pipeline;
            # it is applied in a second pass once trajectories exist.
            call = classify_from_physics(
                ball_above_head=above_head,
                distance_from_net_m=distance_from_net_m,
                half_court_length_m=half_court_m,
                bounces_since_previous_contact=bounces_between,
                outgoing_apex_m=None,
            )
            if call:
                shot_classifications[frame]["shot_type"] = call.shot_type
                physics_calls[frame] = call.reasons

        # Any Volley/Smash that survives without physical evidence was a position
        # guess. Downgrade it rather than ship a label nothing supports.
        downgraded = 0
        for frame, info in shot_classifications.items():
            if info.get("shot_type") in ("Volley", "Smash") and frame not in physics_calls:
                info["shot_type"] = "Groundstroke"
                downgraded += 1

        if physics_calls or downgraded:
            logger.info(f"  Physics shot evidence: {len(physics_calls)} shot(s) evidenced"
                        f"{', ' + str(downgraded) + ' unevidenced Volley/Smash downgraded' if downgraded else ''}")
            for frame, reasons in sorted(physics_calls.items()):
                logger.debug(f"    f{frame} {shot_classifications[frame]['shot_type']}: "
                             f"{'; '.join(reasons)}")
        # Upgrade forehand/backhand from real body geometry where pose is available.
        # Serve, Volley and Smash keep their existing rules - those are genuine physical
        # signatures (overhead reach, net proximity). Forehand vs backhand was the one
        # label with no real basis in position data, so that is the only one replaced.
        # See utils/pose_shot_classifier.py and docs/journal/0006.
        if cfg["pipeline"].get("use_pose_shots", False):
            pose_estimator = PoseEstimator(
                model_path=cfg["models"].get("pose", "models/pose_landmarker_lite.task")
            )
            if pose_estimator.available:
                upgraded = 0
                for shot_frame, info in shot_classifications.items():
                    if info["shot_type"] not in ("Forehand", "Backhand"):
                        continue   # don't second-guess Serve/Volley/Smash
                    ball_bbox = ball_detections[shot_frame].get(1)
                    player_bbox = player_detections[shot_frame].get(info["player_id"])
                    if ball_bbox is None or player_bbox is None:
                        continue
                    ball_xy = ((ball_bbox[0] + ball_bbox[2]) / 2.0,
                               (ball_bbox[1] + ball_bbox[3]) / 2.0)
                    landmarks = pose_estimator.detect_in_bbox(
                        video_frames[shot_frame], player_bbox
                    )
                    result = classify_forehand_backhand(
                        landmarks, ball_xy,
                        max_contact_distance=2.0 * (player_bbox[2] - player_bbox[0]),
                    )
                    if result is not None:
                        info["shot_type"] = result[0]
                        info["pose_confidence"] = round(result[1], 2)
                        upgraded += 1
                        logger.debug(f"  Frame {shot_frame}: pose OK -> {result[0]} ({result[1]:.2f})")
                    else:
                        has_shoulders = bool(landmarks) and "LEFT_SHOULDER" in landmarks and "RIGHT_SHOULDER" in landmarks
                        has_wrist = bool(landmarks) and ("LEFT_WRIST" in landmarks or "RIGHT_WRIST" in landmarks)
                        if not landmarks:
                            why = "no pose detected in crop"
                        elif not has_shoulders:
                            why = "shoulders missing"
                        elif not has_wrist:
                            why = "wrists missing"
                        else:
                            why = "ambiguous hand or too far from ball"
                        logger.debug(f"  Frame {shot_frame}: pose FAILED ({why}) -- kept '{info['shot_type']}'")
                pose_estimator.close()
                logger.info(
                    f"  Pose-based forehand/backhand: {upgraded} of "
                    f"{len(shot_classifications)} shots upgraded "
                    f"(rest kept position-based - pose unavailable or ambiguous)"
                )
            else:
                logger.info("  Pose model unavailable - keeping position-based labels")

        types = [v["shot_type"] for v in shot_classifications.values()]
        logger.info(f"  {len(shot_classifications)} shots classified: {types}")
    else:
        logger.info("[8/9] Shot classification disabled")

    # ── Build stats DataFrame ──────────────────────────────────────
    logger.info("  Computing player statistics...")
    det_cfg = cfg.get("detection", {})

    # Each running total carries its OWN sample counter. Averaging a total by an
    # unrelated count was producing two wrong numbers at once: player movement accrued
    # on the opponent's shots but was divided by this player's shot count, and shot
    # speed was divided by every shot including those whose speed was rejected as
    # physically implausible. Measured on one run: P1 movement published 9.1 km/h where
    # its own samples give 7.6, P2 published 5.6 where its samples give 6.7.
    player_stats_data: list[dict] = [{
        "frame_num": 0,
        "player_1_number_of_shots": 0, "player_1_total_shot_speed": 0,
        "player_1_shot_speed_samples": 0,
        "player_1_last_shot_speed": 0,  "player_1_total_player_speed": 0,
        "player_1_player_speed_samples": 0, "player_1_last_player_speed": 0,
        "player_2_number_of_shots": 0, "player_2_total_shot_speed": 0,
        "player_2_shot_speed_samples": 0,
        "player_2_last_shot_speed": 0,  "player_2_total_player_speed": 0,
        "player_2_player_speed_samples": 0, "player_2_last_player_speed": 0,
    }]


    for idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[idx]
        end_frame   = ball_shot_frames[idx + 1]
        duration_s  = (end_frame - start_frame) / fps  # real FPS, not hardcoded 24
        if duration_s <= 0:
            continue

        ball_start = ball_mini_court[start_frame].get(1)
        if ball_start is None:
            continue

        # Ball shot speed = peak Kalman velocity near the contact frame - matches how
        # real speed guns measure it (at/near contact), not averaged over the whole
        # flight between two shot-frame detections. See docs/journal/0004.
        ball_speed_kmh = peak_speed_kmh_near_frame(
            ball_velocities, frame=start_frame, entity_id=1, window=5,
            px_to_m_scale=px_to_m_scale, fps=fps,
            max_realistic_kmh=constants.MAX_REALISTIC_BALL_SPEED_KMH,
        )

        player_pos = player_mini_court[start_frame]
        if not player_pos:
            continue
        shooter_id = min(
            player_pos.keys(),
            key=lambda pid: measure_distance_between_points(player_pos[pid], ball_start),
        )
        opponent_id = 1 if shooter_id == 2 else 2

        opp_start = player_mini_court[start_frame].get(opponent_id)
        opp_end   = player_mini_court[end_frame].get(opponent_id)
        opp_speed_kmh = 0.0
        if opp_start and opp_end:
            opp_dist_px = measure_distance_between_points(opp_start, opp_end)
            opp_dist_m  = convert_pixel_distance_to_meters(
                opp_dist_px, constants.DOUBLE_LINE_WIDTH, mini_court.get_width_of_mini_court()
            )
            opp_speed_kmh = opp_dist_m / duration_s * 3.6

        row = deepcopy(player_stats_data[-1])
        # Delay display by 3 frames so stats appear after visible racket contact,
        # not at the y-reversal detection point which can be slightly early.
        row["frame_num"] = start_frame + 3
        row[f"player_{shooter_id}_number_of_shots"]   += 1
        if ball_speed_kmh > 0:
            # 0.0 means "no valid speed" (no data, or filtered as physically
            # unrealistic -- see peak_speed_kmh_near_frame). Skip it rather than
            # let a bad reading corrupt the running average; last_shot_speed keeps
            # its previous value instead of showing a fabricated number.
            row[f"player_{shooter_id}_total_shot_speed"]  += ball_speed_kmh
            row[f"player_{shooter_id}_shot_speed_samples"] += 1
            row[f"player_{shooter_id}_last_shot_speed"]    = ball_speed_kmh
        row[f"player_{opponent_id}_total_player_speed"] += opp_speed_kmh
        row[f"player_{opponent_id}_player_speed_samples"] += 1
        row[f"player_{opponent_id}_last_player_speed"]  = opp_speed_kmh

        if cfg["pipeline"]["shot_classification"] and start_frame in shot_classifications:
            row[f"player_{shooter_id}_shot_type"] = shot_classifications[start_frame]["shot_type"]

        player_stats_data.append(row)
        shot_label = shot_classifications.get(start_frame, {}).get("shot_type", "?")
        logger.debug(f"  Shot {idx + 1}: P{shooter_id} | {ball_speed_kmh:.1f} km/h | {shot_label}")

    frames_df = pd.DataFrame({"frame_num": range(len(video_frames))})
    stats_df  = pd.merge(frames_df, pd.DataFrame(player_stats_data),
                         on="frame_num", how="left").ffill()

    for pid in (1, 2):
        shot_n = stats_df[f"player_{pid}_shot_speed_samples"].replace(0, 1)
        move_n = stats_df[f"player_{pid}_player_speed_samples"].replace(0, 1)
        stats_df[f"player_{pid}_average_shot_speed"]   = stats_df[f"player_{pid}_total_shot_speed"] / shot_n
        stats_df[f"player_{pid}_average_player_speed"] = stats_df[f"player_{pid}_total_player_speed"] / move_n

    logger.info(f"  P1: {int(stats_df['player_1_number_of_shots'].iloc[-1])} shots, "
                f"avg {stats_df['player_1_average_shot_speed'].iloc[-1]:.1f} km/h")
    logger.info(f"  P2: {int(stats_df['player_2_number_of_shots'].iloc[-1])} shots, "
                f"avg {stats_df['player_2_average_shot_speed'].iloc[-1]:.1f} km/h")

    # Serve speed, measured from floor-anchored geometry only (server's feet at
    # contact, ball's first bounce) - the one speed in this pipeline that never
    # touches an airborne ball's floor projection. See utils/serve_speed.py.
    # The landing is found by the serve's own physics rather than taken from the
    # generic bounce detector, which lands a few frames late on serves - and a few
    # frames is decisive. Measured: the generic candidate sat after the ball had
    # already bounced and risen, projecting to -31 m on a 23.7 m court and yielding
    # 366 km/h. See utils/serve_landing.py.
    serve_speed = 0.0
    serve_reject = ""
    serve_contact = None
    serve_landing = None
    serve_frames_found = [f for f, info in shot_classifications.items()
                          if str(info.get("shot_type", "")).lower() == "serve"]
    if serve_frames_found and court_valid:
        serve_contact = min(serve_frames_found)
        server_id = shot_classifications[serve_contact].get("player_id")
        contact_pos = player_mini_court.get(serve_contact, {}).get(server_id)

        kp_mc = mini_court.get_court_drawing_keypoints()
        origin_x, origin_y = kp_mc[0], kp_mc[1]
        _h_cache_sl: dict = {}

        def _project(frame: int, image_point):
            kp = (all_court_keypoints[min(frame, len(all_court_keypoints) - 1)]
                  if cfg["pipeline"]["per_frame_keypoints"] else court_keypoints)
            key = tuple(kp)
            if key not in _h_cache_sl:
                _h_cache_sl[key] = mini_court.compute_homography(kp)
            H = _h_cache_sl[key]
            if H is None:
                return None
            mx, my = mini_court.apply_homography(H, image_point)
            return ((mx - origin_x) * px_to_m_scale, (my - origin_y) * px_to_m_scale)

        if contact_pos is not None:
            server_y_m = (contact_pos[1] - origin_y) * px_to_m_scale
            found = find_serve_landing(
                serve_contact, ball_detections, _project,
                server_y_m=server_y_m,
                net_y_m=((kp_mc[1] + kp_mc[5]) / 2.0 - origin_y) * px_to_m_scale,
                service_line_far_m=(kp_mc[17] - origin_y) * px_to_m_scale,
                service_line_near_m=(kp_mc[21] - origin_y) * px_to_m_scale,
                fps=fps,
            )
            if found is None:
                serve_reject = " (landing never observed inside the service box)"
            else:
                landing_frame, landing_court = found
                serve_landing = landing_frame
                contact_court = ((contact_pos[0] - origin_x) * px_to_m_scale,
                                 server_y_m)
                serve_speed = serve_speed_kmh(
                    contact_court, landing_court, serve_contact, landing_frame,
                    px_to_m_scale=1.0,   # positions are already metres
                    fps=fps,
                    max_realistic_kmh=constants.MAX_REALISTIC_BALL_SPEED_KMH,
                )
        else:
            serve_reject = " (no court position for the server at contact)"
    elif serve_frames_found and not court_valid:
        serve_reject = " (court fit failed validation)"

    if serve_speed > 0:
        logger.info(f"  Serve: {serve_speed:.1f} km/h (average over flight, "
                    f"contact f{serve_contact} → landing f{serve_landing})")
    else:
        logger.info(f"  Serve: not measurable{serve_reject or ' (no serve detected)'}")

    # ── 3-D trajectory reconstruction ──────────────────────────────
    # This is the fix for rally speeds, not a visualisation extra: speeds derived from
    # the floor projection of an airborne ball are geometrically wrong, and a free-flight
    # reconstruction between floor-anchored events recovers the vertical component the
    # projection discards. See utils/trajectory_3d.py.
    #
    # Endpoint positions are computed here from scratch rather than reusing
    # ball_mini_court, for two reasons found in review:
    #   1. At a CONTACT the ball is 0.9-2.6 m in the air, so its floor projection is
    #      displaced along the camera ray - the very error this module removes. The
    #      floor-valid measurement at a contact is the hitting player's FEET.
    #   2. ball_mini_court positions are clamped to the drawing panel's bounds, which
    #      is right for pixels on screen and wrong for physics input: a wide bounce
    #      snapped to the panel edge silently shortens the segment.
    # Positions are origin-referenced to the court's far-left corner so the emitted
    # JSON is genuine court-frame metres, as the viewer expects.
    bounce_set = set(bounce_frames)
    event_frames_3d = sorted(set(ball_shot_frames) | bounce_set)
    trajectories_3d = []
    if court_valid:
        court_kp_draw = mini_court.get_court_drawing_keypoints()
        origin_x, origin_y = court_kp_draw[0], court_kp_draw[1]
        court_width_m = (court_kp_draw[2] - court_kp_draw[0]) * px_to_m_scale
        court_length_m = (court_kp_draw[5] - court_kp_draw[1]) * px_to_m_scale
        # A ball can land out, but not in the stands. An event projecting further than
        # this beyond the lines is a misclassified event (an airborne point projected
        # along the camera ray), and one such endpoint corrupts speed AND apex - seen
        # live: a serve read 366 km/h from a landing that projected ~5x too far.
        out_margin_m = 5.0
        _h_cache_3d: dict = {}
        ball_positions_m = {}
        hitter_by_frame: dict[int, int] = {}
        for frame in event_frames_3d:
            kp = (all_court_keypoints[min(frame, len(all_court_keypoints) - 1)]
                  if cfg["pipeline"]["per_frame_keypoints"] else court_keypoints)
            key = tuple(kp)
            if key not in _h_cache_3d:
                _h_cache_3d[key] = mini_court.compute_homography(kp)
            H = _h_cache_3d[key]
            if H is None:
                continue

            if frame in bounce_set:
                # A bounce is on the floor, so the ball's own projection is valid there
                # - but ONLY at the instant it actually touches. The candidate frame is
                # accurate to a few frames, and a ball caught still descending is metres
                # in the air, which the floor homography throws far down-court. Measured:
                # a serve landing projected to y = -36 m (court is 23.7 m long), which
                # produced a 366 km/h reading.
                #
                # The touch instant is where the ball is LOWEST on screen (max image y),
                # so search a small window for it rather than trusting the candidate.
                point = _lowest_ball_point_near(ball_detections, frame, window=4)
                if point is None:
                    continue
            else:
                # A contact is airborne: use the hitting player's feet instead.
                players = player_detections[frame] if frame < len(player_detections) else {}
                bbox = ball_detections[frame].get(1) if frame < len(ball_detections) else None
                if not players or bbox is None:
                    continue
                ball_x = (bbox[0] + bbox[2]) / 2.0
                hitter = min(players, key=lambda pid: abs(
                    (players[pid][0] + players[pid][2]) / 2.0 - ball_x))
                px1, _, px2, py2 = players[hitter]
                point = ((px1 + px2) / 2.0, float(py2))
                hitter_by_frame[frame] = hitter

            mx, my = mini_court.apply_homography(H, point)
            x_m = (mx - origin_x) * px_to_m_scale
            y_m = (my - origin_y) * px_to_m_scale
            kind = "bounce" if frame in bounce_set else "contact"
            if (-out_margin_m <= x_m <= court_width_m + out_margin_m
                    and -out_margin_m <= y_m <= court_length_m + out_margin_m):
                ball_positions_m[frame] = (x_m, y_m)
                logger.debug(f"    f{frame:<5} {kind:<7} court=({x_m:6.1f}, {y_m:6.1f}) m")
            else:
                # Logged rather than dropped in silence: an event landing in the stands
                # is the signature of a misclassified event, and knowing WHICH events
                # fail is how the underlying detector gets fixed.
                logger.debug(f"    f{frame:<5} {kind:<7} court=({x_m:6.1f}, {y_m:6.1f}) m "
                             f"REJECTED - outside court +{out_margin_m:.0f} m "
                             f"(court is {court_width_m:.1f} x {court_length_m:.1f} m)")

        shot_type_by_frame = {f: info.get("shot_type") for f, info in shot_classifications.items()}
        trajectories_3d = reconstruct_rally(
            event_frames_3d, ball_positions_m, shot_type_by_frame, bounce_set, fps,
        )
        # The same player cannot hit the ball twice in a row - the rules require a
        # bounce or the opponent between. A contact→contact segment with one hitter at
        # both ends therefore proves an event was missed between them, and its
        # feet-to-feet "flight" (the distance one player shuffled) is not a ball
        # trajectory. Seen live: serve→re-serve read 17 km/h over 1.24 s.
        # Read from the mini-court directly rather than a variable defined inside the
        # shot-classification branch: with shot_classification disabled (a shipped
        # config option) that variable never exists, and the pipeline crashed here
        # AFTER completing and logging every stat, before writing any output.
        net_y_court_m = ((court_kp_draw[1] + court_kp_draw[5]) / 2.0 - origin_y) * px_to_m_scale
        kept = []
        for t in trajectories_3d:
            # A bounce-to-bounce span has no racket contact at either end, so no shot
            # was played across it. It is the ball bouncing on after a point ended, or
            # the interpolator bridging dead time. Measured on input_video_2: 2 of 16
            # segments, both at ~28 km/h, well below any struck ball.
            if t.start_frame in bounce_set and t.end_frame in bounce_set:
                logger.debug(f"    reject f{t.start_frame}->f{t.end_frame}: "
                             f"bounce to bounce, no racket contact ({t.speed_kmh:.0f} km/h)")
                continue

            both_contacts = (t.start_frame in hitter_by_frame
                             and t.end_frame in hitter_by_frame)
            if both_contacts:
                if hitter_by_frame[t.start_frame] == hitter_by_frame[t.end_frame]:
                    logger.debug(f"    reject f{t.start_frame}->f{t.end_frame}: "
                                 f"same player at both ends (missed event between)")
                    continue
                # Different players stand on opposite sides, so their shots must be
                # separated by a net crossing. One that is not proves the opponent's
                # shot went undetected and two same-side events were joined.
                if not crosses_net(t.start[:2], t.end[:2], net_y_court_m):
                    logger.debug(f"    reject f{t.start_frame}->f{t.end_frame}: "
                                 f"two different players' contacts that never cross "
                                 f"the net ({t.speed_kmh:.0f} km/h)")
                    continue
            kept.append(t)
        trajectories_3d = kept

    if trajectories_3d:
        speeds_3d = [t.speed_kmh for t in trajectories_3d]
        logger.info(f"  3-D reconstruction: {len(trajectories_3d)} flight segments | "
                    f"speed {min(speeds_3d):.0f}-{max(speeds_3d):.0f} km/h "
                    f"(mean {sum(speeds_3d) / len(speeds_3d):.0f}) | "
                    f"apex {max(t.apex_height_m for t in trajectories_3d):.1f} m")
    elif not court_valid:
        # Same refuse-don't-guess rule as serve speed: a reconstruction on an invalid
        # court fit would be confidently wrong, so none is attempted.
        logger.info("  3-D reconstruction: skipped (court fit failed validation)")

    # Lob is the one physical shot test that needs the 3-D apex, so it runs here rather
    # than with the others. Only gated segments reach this point: an over-long segment
    # reports an inflated apex and is rejected upstream, which is exactly what would
    # otherwise turn an ordinary rally ball into a "lob".
    lobs = 0
    for trajectory in trajectories_3d:
        info = shot_classifications.get(trajectory.start_frame)
        if info is None or info.get("shot_type") in ("Serve", "Smash", "Volley"):
            continue
        call = is_lob(trajectory.apex_height_m)
        if call:
            info["shot_type"] = call.shot_type
            lobs += 1
            logger.debug(f"    f{trajectory.start_frame} Lob: {'; '.join(call.reasons)}")
    if lobs:
        logger.info(f"  Lob detection: {lobs} shot(s) identified by flight apex")

    # Interactive 3-D viewer. Written next to the output video so the page can
    # reference it by relative path; the two files travel together.
    viewer_spec = None
    if trajectories_3d:
        viewer_spec = (trajectories_3d,
                       {f: i.get("shot_type") for f, i in shot_classifications.items()},
                       court_valid)
    else:
        logger.info("  3-D reconstruction: no reconstructable flight segments")

    save_stats(stats_df, cfg["io"].get("output_stats_dir", "output/stats"), logger,
               court_fit=court_fit, serve_speed_kmh=serve_speed,
               trajectories_3d=trajectories_3d)

    # ── 9. Render output video ─────────────────────────────────────
    logger.info("[9/9] Rendering output video...")
    output_frames = video_frames.copy()

    logger.debug("  Filtering player detections by confidence...")
    player_detections = player_tracker.filter_by_confidence(
        player_detections, det_cfg.get("player_confidence", 0.7)
    )
    logger.debug("  Filtering ball detections by confidence...")
    ball_detections = ball_tracker.filter_by_confidence(
        ball_detections, det_cfg.get("ball_confidence", 0.6)
    )

    logger.debug("  Drawing player bounding boxes...")
    output_frames = player_tracker.draw_bboxes(output_frames, player_detections, thickness=2)

    logger.debug("  Drawing ball bounding boxes...")
    output_frames = ball_tracker.draw_bboxes(
        output_frames, ball_detections, color=(0, 255, 255), thickness=2
    )

    logger.debug("  Drawing player stats panel...")
    output_frames = draw_player_stats(output_frames, stats_df, stats_params)

    logger.debug("  Drawing court keypoints...")
    if cfg["pipeline"]["per_frame_keypoints"]:
        output_frames = court_detector.draw_keypoints_on_video_dynamic(
            output_frames, all_court_keypoints, point_color=(0, 140, 255), radius=5
        )
    else:
        output_frames = court_detector.draw_keypoints_on_video(
            output_frames, court_keypoints, point_color=(0, 140, 255), radius=5
        )

    logger.debug("  Drawing mini court + player/ball positions...")
    output_frames = mini_court.draw_mini_court(output_frames)
    output_frames = mini_court.draw_ball_trajectory(output_frames, ball_mini_court)
    output_frames = mini_court.draw_points_on_mini_court(
        output_frames, player_mini_court, color=(0, 255, 0), draw_trail=True, label=None
    )
    output_frames = mini_court.draw_points_on_mini_court(
        output_frames, ball_mini_court, color=(0, 255, 255), label=None
    )

    logger.debug("  Adding per-frame overlays...")
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if i in {sf + 3 for sf in ball_shot_frames}:
            cv2.putText(frame, "BALL SHOT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if cfg["pipeline"]["shot_classification"]:
        logger.debug("  Adding shot classification overlays...")
        output_frames = draw_shot_classifications(
            output_frames, shot_classifications, ball_shot_frames
        )

    if not court_valid:
        # The rendered video is what gets watched and screenshotted, and before this
        # banner existed it looked identical whether the court was fitted correctly or
        # fitted to the crowd. Drawn LAST so no panel can paint over the warning.
        logger.debug("  Stamping calibration warning...")
        output_frames = draw_calibration_warning(output_frames, line_support)

    # Save output
    output_path = cfg["io"]["output_video"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if save_video(output_frames, output_path, fps=fps):
        logger.info(f"Output video → {output_path}")
    else:
        alt = output_path.replace(".avi", "_fallback.mp4")
        logger.warning(f"AVI save failed - retrying as {alt}...")
        if save_video(output_frames, alt, fps=fps):
            logger.info(f"Output video → {alt}")
        else:
            logger.error("All video save attempts failed")

    if viewer_spec is not None:
        # Built after the video so it can reference a browser-playable copy. The
        # pipeline writes AVI/MPEG-4 Part 2, which OpenCV produces reliably but no
        # browser can play - the viewer's video tab was silently blank because a
        # <video> element with an unsupported source just shows nothing.
        trajectories, shot_labels, fit_ok = viewer_spec
        web_video = to_browser_playable(output_path)
        if web_video is None:
            logger.warning("  Viewer video tab will be empty: no browser-playable copy "
                           "could be produced (is ffmpeg installed?)")
        viewer_path = build_viewer(
            trajectories,
            Path(output_path).with_suffix(".html"),
            fps=fps,
            video_path=web_video.name if web_video else None,
            shot_types=shot_labels,
            court_valid=fit_ok,
        )
        logger.info(f"3-D viewer  → {viewer_path}  (open in any browser)")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
