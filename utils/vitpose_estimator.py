"""
utils/vitpose_estimator.py
───────────────────────────
Fallback pose estimator using ViTPose (HuggingFace transformers), used only when the
primary MediaPipe estimator finds no wrists at all -- typically motion-blur frames
during a fast swing (e.g. frames 166/167 on our reference clip, journal 0014/0018/0019).
See docs/journal/0021 for the comparison that motivated this.

Why fallback-only, not a replacement
-------------------------------------
ViTPose's keypoints are 2-D only (no depth), unlike MediaPipe's (x, y, z). The
forehand/backhand classifier's side-on collapse fix (utils/pose_shot_classifier.py)
specifically needs that z to stay reliable when a player turns side-on to hit -- exactly
the situation a 2-D-only estimate can't safely resolve on its own. So this is only
reached when MediaPipe has nothing at all to offer; classify_forehand_backhand refuses
(rather than guesses) when the resulting 2-D shoulder axis is too narrow to trust
(MIN_2D_ONLY_AXIS_PX). On our hardest measured frames ViTPose's own confidence ran
0.08-0.29 -- low, real signal, not a confident detection either.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HF ViTPose (COCO 17-keypoint) label -> this codebase's anatomical names, matching
# utils/pose_estimator.py's UPPER_BODY_LANDMARKS so both sources are interchangeable to
# classify_forehand_backhand.
_LABEL_MAP = {
    "L_Shoulder": "LEFT_SHOULDER",
    "R_Shoulder": "RIGHT_SHOULDER",
    "L_Elbow":    "LEFT_ELBOW",
    "R_Elbow":    "RIGHT_ELBOW",
    "L_Wrist":    "LEFT_WRIST",
    "R_Wrist":    "RIGHT_WRIST",
    "L_Hip":      "LEFT_HIP",
    "R_Hip":      "RIGHT_HIP",
}


class ViTPoseEstimator:
    """
    Detects one pose inside a player's bounding box using ViTPose and returns
    upper-body landmarks as (x_px, y_px) -- no depth, see module docstring.

    Usage:
        est = ViTPoseEstimator()
        landmarks = est.detect_in_bbox(frame, player_bbox)   # dict or None
    """

    def __init__(
        self,
        model_name: str = "usyd-community/vitpose-base-simple",
        min_score: float = 0.05,
    ):
        """
        Args:
            model_name: HuggingFace checkpoint id.
            min_score:  keypoints below this confidence are treated as missing. Measured
                        on frame 167 of our reference clip (the exact motion-blur case
                        this fallback exists for): shoulders/elbows/wrists scored
                        0.08-0.30, while genuine noise (nose/eyes) scored <0.03. 0.3
                        would reject nearly every real example this was built for --
                        the real safety net against a bad low-confidence read is the
                        downstream geometry in classify_forehand_backhand
                        (MIN_2D_ONLY_AXIS_PX, ambiguity_ratio, max_contact_distance),
                        which is grounded in measured physical evidence, not this
                        per-keypoint score. This threshold only needs to separate
                        "the model saw some structure" from "the model saw nothing."
        """
        self.min_score = min_score
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._load(model_name)

    def _load(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, VitPoseForPoseEstimation

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = VitPoseForPoseEstimation.from_pretrained(model_name).to(self._device)
            self._model.eval()
            logger.info(f"ViTPoseEstimator loaded ({model_name}, device={self._device})")
        except Exception as exc:  # noqa: BLE001 — a missing/broken model must not kill the run
            logger.error(f"Failed to load ViTPose: {exc}")
            self._processor = None
            self._model = None

    @property
    def available(self) -> bool:
        """False when the model could not be loaded — callers should skip the fallback."""
        return self._model is not None

    def detect_in_bbox(
        self,
        frame: np.ndarray,
        bbox: list[float],
    ) -> dict[str, tuple[float, float]] | None:
        """
        Detect a pose inside `bbox` and return upper-body landmarks as (x_px, y_px) in
        ORIGINAL frame pixel coordinates -- no z, see module docstring.

        Unlike PoseEstimator.detect_in_bbox, this does not crop-then-detect: ViTPose
        takes the full image plus a box (center_x, center_y, w, h) directly.

        Returns None when the model is unavailable, the bbox is degenerate, or nothing
        scores above min_score.
        """
        if self._model is None or frame is None or bbox is None:
            return None

        import torch

        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cx, cy, w, h = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
        boxes = [[[cx, cy, w, h]]]

        try:
            inputs = self._processor(rgb, boxes=boxes, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            result = self._processor.post_process_pose_estimation(outputs, boxes=boxes)[0][0]
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"ViTPose detection failed on bbox: {exc}")
            return None

        id2label = self._model.config.id2label
        landmarks: dict[str, tuple[float, float]] = {}
        for idx, vitpose_name in id2label.items():
            name = _LABEL_MAP.get(vitpose_name)
            if name is None:
                continue
            score = float(result["scores"][idx])
            if score < self.min_score:
                continue
            kp = result["keypoints"][idx]
            landmarks[name] = (float(kp[0]), float(kp[1]))

        return landmarks or None
