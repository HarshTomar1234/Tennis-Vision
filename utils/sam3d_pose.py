"""
utils/sam3d_pose.py
───────────────────
Optional pose backend using SAM 3D Body, interface-compatible with PoseEstimator.

Why this exists
---------------
MediaPipe does not fail to find the player. Measured across THETIS backhand clips it finds
a pose on 100% of frames, and then omits the landmarks of the occluded arm:

    class              frames  no pose  most-missing
    backhand_volley       117        0  right elbow 68%, right wrist 58%
    forehand_volley       120        0  left elbow   9%, left wrist   5%
    backhand              142        0  right elbow 47%, right wrist 44%
    forehand_flat         146        0  left elbow  25%, left wrist  23%

On a backhand the missing arm is the one holding the racket, roughly half the time.
Forehand versus backhand is decided by where that arm is, so this is upstream of every
classifier built on those landmarks: the geometric rule scores 54% and a trained
classifier scores 53.6% on broadcast, and both are reading a hand that is often not there.

SAM 3D Body predicts a whole-body mesh rather than independent landmarks, so an occluded
wrist is inferred from the body instead of dropped. Measured on the exact frames MediaPipe
gave up on, it returned keypoints on 6 of 6 (eval/sam3d_occluded_arm_test.py).

Cost, and why this is contact-frames-only
-----------------------------------------
1.58 s per frame on a GTX 1050 Ti in float32. A 570-frame clip with two players would be
about 30 minutes, which is not viable. But pose is only needed where a shot happens,
roughly 15 contacts per clip:

    per frame        1,140 inferences    ~30 minutes
    contact frames      30 inferences    ~47 seconds

So this is a sparse backend by design, not a replacement for per-frame tracking.

Licensing, which is why this is optional and not a dependency
--------------------------------------------------------------
The weights are under Meta's SAM License, not MIT. That licence grants free use,
modification and derivative works, and it requires that anyone REDISTRIBUTING the
materials pass the same terms along with them.

This project is MIT, so it must not bundle the weights: doing so would have our licence
make a promise about Meta's weights that we have no standing to make. The weights are
therefore fetched by the user from Hugging Face, where they accept Meta's terms directly.
It is the same arrangement already used for the TrackNet weights, and it costs the user
one command.

Loading notes, all of which cost time to discover
--------------------------------------------------
  - load_sam_3d_body returns (model, config), not a model
  - mhr_path must be passed explicitly, or the pose head is built with an empty filename
  - the method is process_one_image, not predict, and it returns a LIST, one per person
  - do NOT wrap inference in torch.autocast: the MHR head is a TorchScript module and
    raises NotImplementedError inside an autocast context
  - do NOT cast the weights either. float16 fails because preprocessing feeds bfloat16
    into float16 biases; bfloat16 fails because a head expects float32. Plain float32
    is the configuration that runs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_DIR = "models/sam3d_body"

# Indices into the mhr70 keypoint format, read from
# sam_3d_body.metadata.mhr70.pose_info rather than assumed.
MHR70_INDEX = {
    "LEFT_SHOULDER": 5,
    "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,
    "RIGHT_ELBOW": 8,
    "RIGHT_WRIST": 41,
    "LEFT_WRIST": 62,
}


class Sam3dPoseEstimator:
    """
    Drop-in alternative to PoseEstimator, backed by SAM 3D Body.

    Exposes `available` and `detect_in_bbox(frame, bbox)` with the same contract, so a
    caller can swap backends without knowing which is running. Landmarks come back as
    (x_px, y_px, z) in original frame coordinates, with z from the predicted 3-D
    keypoints so the body-relative geometry stays well conditioned when a player turns
    side-on.
    """

    def __init__(self, weights_dir: str = DEFAULT_WEIGHTS_DIR, code_dir: str | None = None):
        self._estimator = None
        self._weights = Path(weights_dir)
        self._code_dir = code_dir or os.environ.get("SAM3D_BODY_CODE", "")
        self._load()

    def _load(self) -> None:
        ckpt = self._weights / "model.ckpt"
        mhr = self._weights / "assets" / "mhr_model.pt"
        if not ckpt.exists() or not mhr.exists():
            logger.info(
                "SAM 3D Body weights not found in %s. This backend is optional; see "
                "README 'Optional: SAM 3D Body pose backend' for the two commands that "
                "fetch it.", self._weights
            )
            return

        try:
            import sys

            import torch

            if self._code_dir and self._code_dir not in sys.path:
                sys.path.insert(0, self._code_dir)
            from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, cfg = load_sam_3d_body(
                checkpoint_path=str(ckpt), device=device, mhr_path=str(mhr)
            )
            model.eval()
            self._estimator = SAM3DBodyEstimator(model, cfg)
            self._torch = torch
            logger.info("SAM 3D Body pose backend loaded on %s", device)
        except Exception as exc:  # noqa: BLE001 - an optional backend must not break a run
            logger.warning(
                "SAM 3D Body backend unavailable (%s: %s). Falling back to MediaPipe.",
                type(exc).__name__, str(exc)[:160],
            )
            self._estimator = None

    @property
    def available(self) -> bool:
        return self._estimator is not None

    def detect_in_bbox(self, frame, bbox) -> dict[str, tuple[float, float, float]] | None:
        """
        Landmarks for the person in `bbox`, or None when nothing usable was produced.

        Returning None rather than a partial guess is deliberate and matches
        PoseEstimator: an absent label is more useful downstream than a fabricated one.
        """
        if self._estimator is None or frame is None or bbox is None:
            return None

        import cv2

        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        try:
            with self._torch.no_grad():
                out = self._estimator.process_one_image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    bboxes=np.array([[x1, y1, x2, y2]], dtype=np.float32),
                    inference_type="body",
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("SAM 3D Body inference failed: %s", str(exc)[:160])
            return None

        person = out[0] if isinstance(out, list) and out else out
        if not isinstance(person, dict):
            return None

        kp2d = self._to_numpy(person.get("pred_keypoints_2d"))
        kp3d = self._to_numpy(person.get("pred_keypoints_3d"))
        if kp2d is None or kp2d.shape[0] <= max(MHR70_INDEX.values()):
            return None

        landmarks = {}
        for name, idx in MHR70_INDEX.items():
            x, y = float(kp2d[idx][0]), float(kp2d[idx][1])
            z = float(kp3d[idx][2]) if kp3d is not None and kp3d.shape[0] > idx else 0.0
            landmarks[name] = (x, y, z)
        return landmarks

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        arr = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else None

    def close(self) -> None:
        self._estimator = None
