"""
mini_visual_court/mini_court.py
───────────────────────────────
Top-down 2-D court overlay rendered in a corner of every output frame.
Handles both coordinate mapping (video frame → mini-court) and rendering.

Coordinate mapping pipeline
----------------------------
Primary  : cv2.findHomography (14-point RANSAC) — perspective-correct.
Fallback : nearest keypoint + normalised linear offset — used only when
           RANSAC finds fewer than 4 inliers (severely noisy detections).

Speed calculation accuracy
---------------------------
Every position is in mini-court pixel space.  The physical scale factor is
    1 px  =  DOUBLE_LINE_WIDTH / court_drawing_width  metres
applied identically to X and Y because the drawing is laid out with the same
_meters_to_px helper in both axes, preserving the court's true aspect ratio.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    measure_distance_between_points,
    measure_xy_distance,
)

logger = logging.getLogger(__name__)


# ── Module-level visual constants (BGR) ───────────────────────────────────────
_COURT_FILL   = (176, 127,  89)   # terracotta hard-court surface
_LINE_COLOR   = (  0,   0,   0)   # black court lines
_NET_COLOR    = ( 30,  30, 180)   # dark-red net
_KP_OUTLINE   = (  0,   0,   0)
_KP_FILL      = (  0,   0, 255)   # red keypoint markers
_PLAYER_BG    = (  0,   0,   0)
_PLAYER_FILL  = (  0, 180,   0)   # green player dots
_BALL_BG      = (  0,   0,   0)
_BALL_FILL    = (128,   0, 128)   # purple ball dot


class MiniCourt:
    """Top-down 2-D tennis court overlay with perspective-correct coordinate mapping."""

    _MIN_COURT_W    = 180     # minimum drawing width  (px)
    _MIN_COURT_H    = 360     # minimum drawing height (px)
    _PADDING        = 15      # gap between background box edge and court lines (px)
    _WIDTH_FRAC     = 0.18    # default width as fraction of video frame width

    def __init__(
        self,
        frame: np.ndarray,
        mini_court_width:  int | None  = None,
        mini_court_height: int | None  = None,
        layout_params:     dict | None = None,
    ) -> None:
        """
        Args:
            frame:             First video frame — used to derive default dimensions.
            mini_court_width:  Fixed pixel width (ignored when layout_params provided).
            mini_court_height: Fixed pixel height (ignored when layout_params provided).
            layout_params:     Dict from UILayoutManager.get_mini_court_params().
                               Expected keys: width, height, start_x, start_y.
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        self.padding_court = self._PADDING

        if layout_params is not None:
            self.mini_court_width         = layout_params["width"]
            self.mini_court_height        = layout_params["height"]
            self.drawing_rectangle_width  = layout_params["width"]
            self.drawing_rectangle_height = layout_params["height"]
            self.buffer                   = 10
            self._layout_start_x          = layout_params["start_x"]
            self._layout_start_y          = layout_params["start_y"]
            self._use_layout_position     = True
        else:
            w = mini_court_width  or max(int(self.frame_width * self._WIDTH_FRAC), self._MIN_COURT_W)
            h = mini_court_height or max(int(w * 2.2), self._MIN_COURT_H)
            self.mini_court_width         = w
            self.mini_court_height        = h
            self.drawing_rectangle_width  = w + 30
            self.drawing_rectangle_height = h + 30
            self.buffer                   = max(25, int(min(self.frame_width, self.frame_height) * 0.035))
            self._use_layout_position     = False

        self._set_canvas_position(frame)
        self._set_court_position()
        self._set_drawing_keypoints()
        self._set_court_lines()

    # ── Private init helpers ──────────────────────────────────────────────────

    def _set_canvas_position(self, frame: np.ndarray) -> None:
        """Compute the background rectangle's top-left / bottom-right pixel bounds."""
        if self._use_layout_position:
            self.start_x = self._layout_start_x
            self.start_y = self._layout_start_y
        else:
            # Default: top-right corner of the video frame
            self.start_x = frame.shape[1] - self.buffer - self.drawing_rectangle_width
            self.start_y = self.buffer
        self.end_x = self.start_x + self.drawing_rectangle_width
        self.end_y = self.start_y + self.drawing_rectangle_height

    def _set_court_position(self) -> None:
        """Derive court-line bounds inside the background rectangle."""
        self.court_start_x      = self.start_x + self.padding_court
        self.court_start_y      = self.start_y + self.padding_court
        self.court_end_x        = self.end_x   - self.padding_court
        self.court_end_y        = self.end_y   - self.padding_court
        self.court_drawing_width  = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y
        # Extended play zone: dots may appear anywhere in the background area
        self.playing_area_start_y = self.start_y
        self.playing_area_end_y   = self.end_y

    def _meters_to_px(self, meters: float) -> float:
        """Convert real-world metres to mini-court pixels (uniform X/Y scale)."""
        return convert_meters_to_pixel_distance(
            meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width
        )

    def _set_drawing_keypoints(self) -> None:
        """
        Compute the 14 reference keypoints (28 floats) that define the court diagram.

        Flat-array layout: [x0,y0, x1,y1, ..., x13,y13].
        Index map (top = far baseline, bottom = near baseline):

          0  outer TL   1  outer TR
          2  outer BL   3  outer BR
          4  singles TL  5  singles BL
          6  singles TR  7  singles BR
          8  svc far-L   9  svc far-R
         10  svc near-L 11  svc near-R
         12  centre-T (far)  13  centre-T (near)
        """
        cx, cy = float(self.court_start_x), float(self.court_start_y)
        w      = float(self.court_drawing_width)
        full_h = self._meters_to_px(constants.HALF_COURT_LINE_HEIGHT * 2)
        alley  = self._meters_to_px(constants.DOUBLE_ALLY_DIFFERENCE)
        svc_h  = self._meters_to_px(constants.NO_MANS_LAND_HEIGHT)
        svc_w  = self._meters_to_px(constants.SINGLE_LINE_WIDTH)

        p = [0.0] * 28

        # Outer doubles corners (0-3)
        p[0],  p[1]  = cx,      cy
        p[2],  p[3]  = cx + w,  cy
        p[4],  p[5]  = cx,      cy + full_h
        p[6],  p[7]  = cx + w,  cy + full_h

        # Singles sideline endpoints (4-7)
        p[8],  p[9]  = cx + alley,     cy
        p[10], p[11] = cx + alley,     cy + full_h
        p[12], p[13] = cx + w - alley, cy
        p[14], p[15] = cx + w - alley, cy + full_h

        # Service-box inner corners (8-11)
        p[16], p[17] = cx + alley,         cy + svc_h
        p[18], p[19] = cx + alley + svc_w, cy + svc_h
        p[20], p[21] = cx + alley,         cy + full_h - svc_h
        p[22], p[23] = cx + alley + svc_w, cy + full_h - svc_h

        # Centre-T marks (12-13)
        p[24], p[25] = cx + alley + svc_w / 2, cy + svc_h
        p[26], p[27] = cx + alley + svc_w / 2, cy + full_h - svc_h

        self.drawing_key_points = [int(v) for v in p]

    def _set_court_lines(self) -> None:
        """Pairs of keypoint indices that define each line segment to draw."""
        self.lines = [
            (0, 2),    # left outer sideline
            (1, 3),    # right outer sideline
            (0, 1),    # far baseline
            (2, 3),    # near baseline
            (4, 5),    # left singles sideline
            (6, 7),    # right singles sideline
            (8, 9),    # far service line
            (10, 11),  # near service line
            (12, 13),  # centre service line
        ]

    # ── Public geometry accessors ─────────────────────────────────────────────

    def get_start_point_of_mini_court(self) -> tuple[int, int]:
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self) -> int:
        return self.court_drawing_width

    def get_court_drawing_keypoints(self) -> list[int]:
        return self.drawing_key_points

    # ── Homography ────────────────────────────────────────────────────────────

    def compute_homography(self, video_keypoints: np.ndarray) -> np.ndarray | None:
        """
        Fit a 3×3 perspective homography: video-frame space → mini-court space.

        Source points: the 14 court keypoints detected by the ResNet-50 detector.
        Destination  : the 14 pre-computed drawing_key_points.
        RANSAC (10 px reprojection threshold) discards noisy detections.

        Returns the 3×3 matrix H, or None when fewer than 4 inliers survive.
        The caller falls back to nearest-keypoint approximation on None.
        """
        try:
            src = np.array(video_keypoints,        dtype=np.float32).reshape(-1, 1, 2)
            dst = np.array(self.drawing_key_points, dtype=np.float32).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)
            inliers = int(mask.sum()) if mask is not None else 0
            if H is None or inliers < 4:
                logger.debug("Homography rejected: %d/14 inliers", inliers)
                return None
            logger.debug("Homography OK: %d/14 inliers", inliers)
            return H
        except Exception as exc:  # noqa: BLE001
            logger.debug("Homography failed: %s", exc)
            return None

    def apply_homography(
        self, H: np.ndarray, point: tuple[float, float]
    ) -> tuple[float, float]:
        """Map one (x, y) from video-frame space to mini-court space via H."""
        pt  = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def _fallback_mapping(
        self,
        vx: float, vy: float,
        video_kp: np.ndarray,
        y_scale: float = 1.0,
    ) -> tuple[float, float]:
        """
        Approximate mapping used only when homography is unavailable.

        Finds the geometrically closest video keypoint, then applies a linear
        offset (normalised by frame dimensions) to the corresponding mini-court
        keypoint.  Less accurate than homography — cannot correct perspective.

        y_scale: compensates for aspect-ratio differences between the video
                 perspective and the orthographic drawing (1.4 for player feet,
                 1.0 for the ball centre).
        """
        n = len(video_kp) // 2
        dists = np.hypot(
            vx - video_kp[0::2][:n],
            vy - video_kp[1::2][:n],
        )
        k     = int(np.argmin(dists))
        dx    = (vx - video_kp[k * 2])     / self.frame_width  * self.mini_court_width
        dy    = (vy - video_kp[k * 2 + 1]) / self.frame_height * self.mini_court_height * y_scale
        return self.drawing_key_points[k * 2] + dx, self.drawing_key_points[k * 2 + 1] + dy

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes:   list[dict],
        ball_boxes:     list[dict],
        court_keypoints,
        use_homography: bool = True,
    ) -> tuple[dict, dict]:
        """
        Map every player and ball bounding box to mini-court pixel coordinates.

        Args:
            player_boxes:    List (one entry per frame) of {player_id: [x1,y1,x2,y2]}.
            ball_boxes:      List (one entry per frame) of {ball_id:   [x1,y1,x2,y2]}.
            court_keypoints: Flat keypoint array (all frames share it) OR a list of
                             per-frame arrays for camera-robust mode.
            use_homography:  Attempt RANSAC homography first when True.

        Returns:
            (player_positions, ball_positions) — dicts with the same outer
            structure as the inputs, but values are (x, y) mini-court coords.
        """
        # Detect per-frame vs. single-frame keypoint input
        per_frame = (
            isinstance(court_keypoints, list)
            and len(court_keypoints) > 0
            and hasattr(court_keypoints[0], "__len__")
            and not isinstance(court_keypoints[0], (int, float))
        )

        # Cache homography by keypoint tuple to avoid recomputing identical frames.
        _h_cache: dict[tuple, np.ndarray | None] = {}

        out_players: dict[int, dict] = {}
        out_ball:    dict[int, dict] = {}

        for frame_num in range(len(player_boxes)):
            kp = court_keypoints[min(frame_num, len(court_keypoints) - 1)] if per_frame else court_keypoints

            out_players[frame_num] = {}
            out_ball[frame_num]    = {}

            H: np.ndarray | None = None
            if use_homography:
                key = tuple(kp)
                if key not in _h_cache:
                    _h_cache[key] = self.compute_homography(kp)
                H = _h_cache[key]

            # ── Players: foot position = bottom-centre of bounding box ────────
            for pid, bbox in player_boxes[frame_num].items():
                try:
                    fx, fy = (bbox[0] + bbox[2]) / 2.0, float(bbox[3])
                    if H is not None:
                        mx, my = self.apply_homography(H, (fx, fy))
                    else:
                        mx, my = self._fallback_mapping(fx, fy, kp, y_scale=1.4)
                    out_players[frame_num][pid] = (
                        float(np.clip(mx, self.start_x, self.end_x)),
                        float(np.clip(my, self.playing_area_start_y, self.playing_area_end_y)),
                    )
                except (ValueError, TypeError, IndexError):
                    out_players[frame_num][pid] = (
                        self.start_x + self.mini_court_width  // 2,
                        self.start_y + self.mini_court_height // 2,
                    )

            # ── Ball: centre of bounding box ──────────────────────────────────
            for bid, bbox in (ball_boxes[frame_num] if frame_num < len(ball_boxes) else {}).items():
                try:
                    bx = (bbox[0] + bbox[2]) / 2.0
                    by = (bbox[1] + bbox[3]) / 2.0
                    if H is not None:
                        mx, my = self.apply_homography(H, (bx, by))
                    else:
                        mx, my = self._fallback_mapping(bx, by, kp)
                    out_ball[frame_num][bid] = (
                        int(np.clip(mx, self.start_x, self.end_x)),
                        int(np.clip(my, self.playing_area_start_y, self.playing_area_end_y)),
                    )
                except (ValueError, TypeError, IndexError):
                    out_ball[frame_num][bid] = (
                        self.start_x + self.mini_court_width  // 2,
                        self.start_y + self.mini_court_height // 2,
                    )

        return out_players, out_ball

    # ── Rendering ─────────────────────────────────────────────────────────────

    def draw_mini_court(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Render the background, court surface, lines, and keypoints on every frame."""
        for frame in frames:
            self._draw_background(frame)
            self._draw_court_surface(frame)
        return frames

    def _draw_background(self, frame: np.ndarray) -> None:
        """Semi-transparent white panel behind the mini-court area."""
        roi   = frame[self.start_y:self.end_y, self.start_x:self.end_x]
        white = np.full_like(roi, 255)
        frame[self.start_y:self.end_y, self.start_x:self.end_x] = (
            cv2.addWeighted(roi, 0.5, white, 0.5, 0)
        )

    def _draw_court_surface(self, frame: np.ndarray) -> None:
        """Fill court with surface colour, then draw lines, net, and keypoint markers."""
        kp = self.drawing_key_points

        # Court surface fill — outer corners in correct (non-self-intersecting) winding
        surface  = np.zeros_like(frame)
        corners  = np.array(
            [[kp[0], kp[1]], [kp[2], kp[3]], [kp[6], kp[7]], [kp[4], kp[5]]],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.fillPoly(surface, [corners], _COURT_FILL)
        mask = np.any(surface != 0, axis=-1)
        frame[mask] = cv2.addWeighted(frame, 0.1, surface, 0.9, 0)[mask]

        # Court lines
        for a, b in self.lines:
            cv2.line(
                frame,
                (int(kp[a * 2]), int(kp[a * 2 + 1])),
                (int(kp[b * 2]), int(kp[b * 2 + 1])),
                _LINE_COLOR, 2,
            )

        # Net (midpoint between far and near baselines)
        net_y = int((kp[1] + kp[5]) / 2)
        cv2.line(frame, (int(kp[0]), net_y), (int(kp[2]), net_y), _NET_COLOR, 3)

        # Keypoint markers
        for i in range(0, len(kp), 2):
            cx, cy = int(kp[i]), int(kp[i + 1])
            cv2.circle(frame, (cx, cy), 5, _KP_OUTLINE, -1)
            cv2.circle(frame, (cx, cy), 4, _KP_FILL,    -1)

    def draw_points_on_mini_court(
        self,
        frames:     list[np.ndarray],
        positions:  dict[int, dict],
        color:      tuple[int, int, int] = (0, 255, 0),
        draw_trail: bool = False,
        label:      str | None = None,
    ) -> list[np.ndarray]:
        """
        Render player or ball dots on the mini-court overlay.

        Distinguishes players from the ball via the colour sentinel:
          color=(0,255,0)   → green player circles
          color=(0,255,255) → purple ball dots
        """
        is_ball = (color == (0, 255, 255))

        for frame_num, frame in enumerate(frames):
            if frame_num not in positions:
                continue
            for _, pos in positions[frame_num].items():
                try:
                    x, y = int(pos[0]), int(pos[1])
                    if is_ball:
                        cv2.circle(frame, (x, y), 9, _BALL_BG,   -1)
                        cv2.circle(frame, (x, y), 7, _BALL_FILL, -1)
                    else:
                        cv2.circle(frame, (x, y), 10, _PLAYER_BG,   -1)
                        cv2.circle(frame, (x, y),  8, _PLAYER_FILL, -1)
                except (ValueError, TypeError):
                    continue

        return frames

    def draw_ball_trajectory(
        self,
        frames:    list[np.ndarray],
        positions: dict[int, dict],
    ) -> list[np.ndarray]:
        """Trajectory trail rendering — not yet implemented."""
        return frames
