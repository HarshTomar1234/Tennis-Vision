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
    PoseEstimator,
    ShotClassifier,
    UILayoutManager,
    classify_contact_vs_bounce,
    classify_floor_level,
    classify_forehand_backhand,
    convert_pixel_distance_to_meters,
    draw_player_stats,
    draw_shot_classifications,
    measure_distance_between_points,
    peak_speed_kmh_near_frame,
    read_video,
    save_video,
    smooth_trajectories,
)


# ── Config & CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tennis-Vision: AI-powered tennis match analysis")
    p.add_argument("--input",    "-i", help="Path to input video (overrides config)")
    p.add_argument("--output",   "-o", help="Path to output video (overrides config)")
    p.add_argument("--config",   "-c", default="configs/config.yaml", help="Config YAML path")
    p.add_argument("--no-stubs", action="store_true", help="Disable cached stubs, force fresh detection")
    p.add_argument("--fast",     action="store_true", help="Fast mode: first-frame keypoints, no ByteTrack")
    p.add_argument("--debug",    action="store_true", help="Enable DEBUG log level")
    return p.parse_args()


_DEFAULTS: dict = {
    "pipeline": {
        "per_frame_keypoints": True,
        "use_bytetrack": True,
        "shot_classification": True,
        "use_homography": True,
    },
    "models": {
        "player": "yolov8x",
        "ball": "models/last.pt",
        "court": "models/keypoints_model.pth",
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
    },
    "stubs": {
        "use_player_stubs": True,
        "use_ball_stubs": False,
    },
    "detection": {
        "player_confidence": 0.7,
        "ball_confidence": 0.6,
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


def load_config(config_path: str) -> dict:
    """Load YAML config and deep-merge over built-in defaults."""
    cfg = deepcopy(_DEFAULTS)
    if config_path and os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for section, values in user.items():
            if section in cfg and isinstance(values, dict):
                cfg[section].update(values)
            else:
                cfg[section] = values
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

def save_stats(stats_df: pd.DataFrame, output_dir: str, logger: logging.Logger):
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
    json_path = out / f"summary_{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary JSON → {json_path}")


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
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=use_player_stubs,
        stub_path=cfg["io"]["player_stub_path"],
    )
    source = f"stub ({cfg['io']['player_stub_path']})" if use_player_stubs else "fresh YOLO"
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
        tracknet_stub = cfg["io"].get("tracknet_stub_path", "tracker_stubs/ball_detections_tracknet.pkl")
        use_ball_stubs = cfg["stubs"].get("use_ball_stubs", False)

        if use_ball_stubs and Path(tracknet_stub).exists():
            logger.info(f"  TrackNet v2 — loading from stub ({tracknet_stub})")
            with open(tracknet_stub, "rb") as f:
                ball_detections = pickle.load(f)
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

    # ── 5. Player selection ────────────────────────────────────────
    logger.info("[5/9] Filtering to 2 main players...")
    player_detections = player_tracker.choose_and_filter_players(
        player_detections, court_keypoints
    )

    first_frame_ids = sorted(player_detections[0].keys())
    player_id_map = {orig: new for new, orig in enumerate(first_frame_ids[:2], start=1)}
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
    logger.info(f"  Mini-court position: start=({mini_court.start_x}, {mini_court.start_y}) "
                f"size={mini_court.mini_court_width}×{mini_court.mini_court_height}px")

    # ── 7. Shot frames + coordinate mapping ───────────────────────
    logger.info("[7/9] Detecting shot frames + mapping to mini-court...")
    raw_reversal_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Floor-level anchors for BALL GEOMETRY: every trajectory reversal (contact or
    # bounce) is a valid homography anchor — the floor transform is correct at floor
    # level regardless of which caused it. In-flight frames interpolate between
    # anchors instead of being projected (wrong — the ball has real height while
    # airborne). See utils.ball_state and docs/journal/0003 for why contact-vs-bounce
    # is NOT needed for this part.
    floor_states = classify_floor_level(raw_reversal_frames, len(video_frames))

    # Best-effort CONTACT vs BOUNCE split for shot counting / stats only — a
    # player-proximity heuristic with a measured ~5/7 ceiling on the reference clip
    # (position data alone cannot cleanly separate them; see journal 0003). Not
    # ground truth — used for "who hit the ball and when", not for geometry validity.
    shot_dist_px = cfg.get("detection", {}).get("shot_player_distance_px", 300)
    confirmed_shot_frames, bounce_frames = classify_contact_vs_bounce(
        raw_reversal_frames, ball_detections, player_detections,
        shot_player_distance_px=shot_dist_px,
    )
    logger.info(
        f"  {len(raw_reversal_frames)} y-reversals → {len(floor_states) and sum(1 for s in floor_states if s == 'floor_level')} "
        f"floor-level anchors | {len(confirmed_shot_frames)} confirmed shots + "
        f"{len(bounce_frames)} bounces (best-effort, player within {shot_dist_px}px)"
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

    # Kalman smoothing (Phase 1, Step 3) — stabilizes the projected dots frame to
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
        shot_classifications = shot_classifier.classify_shots(
            player_mini_court, ball_mini_court, ball_shot_frames,
            mini_court.court_drawing_height,
        )
        # Upgrade forehand/backhand from real body geometry where pose is available.
        # Serve, Volley and Smash keep their existing rules — those are genuine physical
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
                pose_estimator.close()
                logger.info(
                    f"  Pose-based forehand/backhand: {upgraded} of "
                    f"{len(shot_classifications)} shots upgraded "
                    f"(rest kept position-based — pose unavailable or ambiguous)"
                )
            else:
                logger.info("  Pose model unavailable — keeping position-based labels")

        types = [v["shot_type"] for v in shot_classifications.values()]
        logger.info(f"  {len(shot_classifications)} shots classified: {types}")
    else:
        logger.info("[8/9] Shot classification disabled")

    # ── Build stats DataFrame ──────────────────────────────────────
    logger.info("  Computing player statistics...")
    det_cfg = cfg.get("detection", {})

    player_stats_data: list[dict] = [{
        "frame_num": 0,
        "player_1_number_of_shots": 0, "player_1_total_shot_speed": 0,
        "player_1_last_shot_speed": 0,  "player_1_total_player_speed": 0,
        "player_1_last_player_speed": 0,
        "player_2_number_of_shots": 0, "player_2_total_shot_speed": 0,
        "player_2_last_shot_speed": 0,  "player_2_total_player_speed": 0,
        "player_2_last_player_speed": 0,
    }]

    px_to_m_scale = constants.DOUBLE_LINE_WIDTH / mini_court.get_width_of_mini_court()

    for idx in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[idx]
        end_frame   = ball_shot_frames[idx + 1]
        duration_s  = (end_frame - start_frame) / fps  # real FPS, not hardcoded 24
        if duration_s <= 0:
            continue

        ball_start = ball_mini_court[start_frame].get(1)
        if ball_start is None:
            continue

        # Ball shot speed = peak Kalman velocity near the contact frame — matches how
        # real speed guns measure it (at/near contact), not averaged over the whole
        # flight between two shot-frame detections. See docs/journal/0004.
        ball_speed_kmh = peak_speed_kmh_near_frame(
            ball_velocities, frame=start_frame, entity_id=1, window=5,
            px_to_m_scale=px_to_m_scale, fps=fps,
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
        row[f"player_{shooter_id}_total_shot_speed"]  += ball_speed_kmh
        row[f"player_{shooter_id}_last_shot_speed"]    = ball_speed_kmh
        row[f"player_{opponent_id}_total_player_speed"] += opp_speed_kmh
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
        n = stats_df[f"player_{pid}_number_of_shots"].replace(0, 1)
        stats_df[f"player_{pid}_average_shot_speed"]   = stats_df[f"player_{pid}_total_shot_speed"] / n
        stats_df[f"player_{pid}_average_player_speed"] = stats_df[f"player_{pid}_total_player_speed"] / n

    logger.info(f"  P1: {int(stats_df['player_1_number_of_shots'].iloc[-1])} shots, "
                f"avg {stats_df['player_1_average_shot_speed'].iloc[-1]:.1f} km/h")
    logger.info(f"  P2: {int(stats_df['player_2_number_of_shots'].iloc[-1])} shots, "
                f"avg {stats_df['player_2_average_shot_speed'].iloc[-1]:.1f} km/h")

    save_stats(stats_df, cfg["io"].get("output_stats_dir", "output/stats"), logger)

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

    # Save output
    output_path = cfg["io"]["output_video"]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if save_video(output_frames, output_path):
        logger.info(f"Output video → {output_path}")
    else:
        alt = output_path.replace(".avi", "_fallback.mp4")
        logger.warning(f"AVI save failed — retrying as {alt}...")
        if save_video(output_frames, alt):
            logger.info(f"Output video → {alt}")
        else:
            logger.error("All video save attempts failed")

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
