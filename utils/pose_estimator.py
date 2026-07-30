"""
utils/pose_estimator.py
───────────────────────
Thin wrapper over MediaPipe Tasks PoseLandmarker, scoped to what shot classification
needs: upper-body landmarks for a single known player, in original-frame pixel space.

Why crop-then-detect
--------------------
A broadcast tennis frame contains ball kids, line judges, and a full crowd. Running pose
on the whole frame would find dozens of people and leave us guessing which is the player.
We already know exactly where each player is (their tracked bbox), so we crop to that box
and detect one pose inside it. Fewer false positives, much less compute.

Why only on contact frames
--------------------------
The label we want is per-shot, not per-frame. Running pose on all 570 frames is wasted
work — the caller passes only the frames where a shot was detected.

API note: mediapipe 1.0.0 removed the legacy `mp.solutions.pose` interface. This uses the
current Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), which requires a
downloaded `.task` model bundle:
  curl -L -o models/pose_landmarker_lite.task \\
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Landmark names this module exposes. MediaPipe labels these anatomically (LEFT_WRIST is
# the person's own left wrist regardless of which way they face the camera), which is
# what makes the body-relative shot geometry work — see utils/pose_shot_classifier.py.
UPPER_BODY_LANDMARKS = (
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW",    "RIGHT_ELBOW",
    "LEFT_WRIST",    "RIGHT_WRIST",
    "LEFT_HIP",      "RIGHT_HIP",
)


class PoseEstimator:
    """
    Detects one pose inside a player's bounding box and returns landmarks in
    original-frame pixel coordinates.

    Usage:
        est = PoseEstimator("models/pose_landmarker_lite.task")
        landmarks = est.detect_in_bbox(frame, player_bbox)   # dict or None
        est.close()
    """

    def __init__(
        self,
        model_path: str = "models/pose_landmarker_lite.task",
        min_visibility: float = 0.5,
        bbox_padding: float = 0.15,
    ):
        """
        Args:
            model_path:     path to the MediaPipe .task bundle.
            min_visibility: landmarks below this visibility score are treated as missing
                            rather than trusted — a low-visibility wrist is a guess, and
                            a guessed wrist produces a guessed shot label.
            bbox_padding:   fraction of bbox size to expand the crop by. A tracked box is
                            often tight around the torso and clips an extended hitting
                            arm, which is exactly the landmark we care about most.
        """
        self.model_path     = model_path
        self.min_visibility = min_visibility
        self.bbox_padding   = bbox_padding
        self._landmarker    = None
        self._landmark_index: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not Path(self.model_path).exists():
            logger.warning(
                f"Pose model not found at '{self.model_path}'. Pose-based shot "
                f"classification will be skipped. Download with:\n"
                f"  curl -L -o {self.model_path} https://storage.googleapis.com/"
                f"mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/"
                f"pose_landmarker_lite.task"
            )
            return

        try:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._landmark_index = {
                name: int(getattr(vision.PoseLandmark, name))
                for name in UPPER_BODY_LANDMARKS
            }
            logger.info(f"PoseEstimator loaded ({self.model_path})")
        except Exception as exc:  # noqa: BLE001 — a missing/broken model must not kill the run
            logger.error(f"Failed to load pose model: {exc}")
            self._landmarker = None

    @property
    def available(self) -> bool:
        """False when the model could not be loaded — callers should fall back."""
        return self._landmarker is not None

    def _pad_bbox(self, bbox, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        pad_x = (x2 - x1) * self.bbox_padding
        pad_y = (y2 - y1) * self.bbox_padding
        return (
            max(0,        int(x1 - pad_x)),
            max(0,        int(y1 - pad_y)),
            min(frame_w,  int(x2 + pad_x)),
            min(frame_h,  int(y2 + pad_y)),
        )

    def detect_in_bbox(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> dict[str, tuple[float, float, float]] | None:
        """
        Detect a pose inside `bbox` and return upper-body landmarks as
        (x_px, y_px, z_px) in ORIGINAL frame pixel coordinates.

        Why z matters here
        ------------------
        A tennis player turns side-on to hit, which collapses the shoulder axis in image
        space — measured as low as 2.8px wide on this clip's groundstrokes, where a
        2-3px landmark error flips the left/right sign outright. MediaPipe's z (depth
        relative to the hip midpoint) grows exactly when image-x shrinks: on a side-on
        frame here, shoulder dx was 0.037 while dz was 0.607. Carrying z lets the shot
        classifier build the body axis in the horizontal (x, z) plane, which stays
        well-conditioned at every body orientation.

        z is reported by MediaPipe on roughly the same scale as *normalised* x, so it is
        multiplied by the crop width to put all three components in comparable pixel
        units.

        Returns None when the model is unavailable, the crop is degenerate, no pose is
        found, or every landmark falls below `min_visibility`. Returning None is
        deliberate: an absent label is more useful downstream than a fabricated one.
        """
        if self._landmarker is None or frame is None or bbox is None:
            return None

        import mediapipe as mp

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = self._pad_bbox(bbox, frame_w, frame_h)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Pose detection failed on crop: {exc}")
            return None

        if not result.pose_landmarks:
            return None

        pose = result.pose_landmarks[0]
        landmarks: dict[str, tuple[float, float, float]] = {}

        for name, idx in self._landmark_index.items():
            if idx >= len(pose):
                continue
            lm = pose[idx]
            if getattr(lm, "visibility", 1.0) < self.min_visibility:
                continue
            # MediaPipe returns coordinates normalised to the crop — map x/y back to the
            # original frame so callers can compare against ball/court positions, and
            # scale z by crop width so all three components share pixel-like units.
            landmarks[name] = (
                x1 + lm.x * crop_w,
                y1 + lm.y * crop_h,
                float(getattr(lm, "z", 0.0)) * crop_w,
            )

        return landmarks or None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
