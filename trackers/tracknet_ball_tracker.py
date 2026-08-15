"""
trackers/tracknet_ball_tracker.py
──────────────────────────────────
TrackNet v2 ball tracker - drop-in replacement for BallTracker.

Architecture: yastrebksv/TrackNet (BallTrackerNet)
  VGG-style encoder + decoder, 9-channel input (3 consecutive RGB frames),
  256-channel output collapsed via argmax + HoughCircles to a ball (x, y).
  Temporal context handles motion blur → 60%+ detection rate on tennis.

── Download pretrained weights ──────────────────────────────────────────────
  pip install gdown
  gdown --id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O models/tracknet.pt

── Activate in config.yaml ──────────────────────────────────────────────────
  pipeline:
    use_tracknet: true
  models:
    tracknet: "models/tracknet.pt"
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Network (exact architecture from yastrebksv/TrackNet) ─────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    """
    Input : (B, 9, H, W)  - 3 consecutive RGB frames stacked channel-wise
    Output: (B, 256, H*W) - spatial activation map (testing=False)
                          - softmax over 256 channels (testing=True)
    Ball position = argmax(dim=1) → postprocess with HoughCircles.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1  = ConvBlock(9,   64)
        self.conv2  = ConvBlock(64,  64)
        self.pool1  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3  = ConvBlock(64,  128)
        self.conv4  = ConvBlock(128, 128)
        self.pool2  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5  = ConvBlock(128, 256)
        self.conv6  = ConvBlock(256, 256)
        self.conv7  = ConvBlock(256, 256)
        self.pool3  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8  = ConvBlock(256, 512)
        self.conv9  = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        self.ups1   = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        self.ups2   = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        self.ups3   = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64,  64)
        self.conv18 = ConvBlock(64,  out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def forward(self, x: torch.Tensor, testing: bool = False) -> torch.Tensor:
        b = x.size(0)
        x = self.conv2(self.conv1(x))
        x = self.conv4(self.conv3(self.pool1(x)))
        x = self.conv7(self.conv6(self.conv5(self.pool2(x))))
        x = self.conv10(self.conv9(self.conv8(self.pool3(x))))
        x = self.conv13(self.conv12(self.conv11(self.ups1(x))))
        x = self.conv15(self.conv14(self.ups2(x)))
        x = self.conv18(self.conv17(self.conv16(self.ups3(x))))
        out = x.reshape(b, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ── Tracker ────────────────────────────────────────────────────────────────────

class TrackNetBallTracker:
    """
    Drop-in replacement for BallTracker using TrackNet v2.

    Same public interface as BallTracker:
      detect_frames, interpolate_ball_positions, get_ball_shot_frames,
      draw_bboxes, filter_by_confidence
    """

    INPUT_HEIGHT = 360
    INPUT_WIDTH  = 640

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path     = model_path
        self.conf_threshold = conf_threshold   # kept for API compatibility

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"TrackNetBallTracker: device={self.device}")

        self.model: BallTrackerNet | None = None
        self._load_model()

    # ── Model loading ──────────────────────────────────────────────

    def _load_model(self):
        if not Path(self.model_path).exists():
            logger.warning(
                f"TrackNet weights not found at '{self.model_path}'.\n"
                "  Download:  pip install gdown && "
                "gdown --id 1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl -O models/tracknet.pt\n"
                "  Then set:  pipeline.use_tracknet: true  in config.yaml"
            )
            return

        try:
            self.model = BallTrackerNet().to(self.device)
            state = torch.load(self.model_path, map_location=self.device)

            # Handle checkpoint wrappers from different save conventions
            if isinstance(state, dict):
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in state:
                        state = state[key]
                        break

            expected = set(self.model.state_dict().keys())
            loaded   = set(state.keys())
            missing  = expected - loaded

            if missing:
                logger.warning(
                    f"TrackNet: partial weight match - "
                    f"{len(expected - missing)}/{len(expected)} keys loaded. "
                    f"First missing: {sorted(missing)[:3]}"
                )
                self.model.load_state_dict(state, strict=False)
            else:
                self.model.load_state_dict(state)

            self.model.eval()
            logger.info(f"TrackNet weights loaded from '{self.model_path}'")
        except Exception as exc:
            logger.error(f"Failed to load TrackNet weights: {exc}")
            self.model = None

    # ── Inference helpers ──────────────────────────────────────────

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Stack 3 consecutive frames → (1, 9, H_in, W_in) tensor."""
        resized = [
            cv2.resize(f, (self.INPUT_WIDTH, self.INPUT_HEIGHT)).astype(np.float32) / 255.0
            for f in frames
        ]
        stacked = np.concatenate([r.transpose(2, 0, 1) for r in resized], axis=0)
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def _output_to_bbox(self, model_out: np.ndarray,
                         orig_h: int, orig_w: int) -> list[float] | None:
        """
        Convert (H_in * W_in,) argmax map → [x1,y1,x2,y2] in original coords.
        Ball = centroid of pixels where argmax != 0 (non-background channel wins).
        Requires a minimum cluster size to avoid noise.
        """
        ball_mask = model_out.reshape((self.INPUT_HEIGHT, self.INPUT_WIDTH)) != 0
        ys, xs = np.where(ball_mask)
        if len(xs) < 5:   # too sparse - noise
            return None
        cx_in = xs.mean()
        cy_in = ys.mean()
        cx = int(cx_in * orig_w / self.INPUT_WIDTH)
        cy = int(cy_in * orig_h / self.INPUT_HEIGHT)
        half = 12
        return [cx - half, cy - half, cx + half, cy + half]

    # ── Public interface ───────────────────────────────────────────

    def detect_frames(self, frames: list[np.ndarray],
                       read_from_stub: bool = False,
                       stub_path: str | None = None) -> list[dict]:
        """
        Detect ball in every frame.
        Returns List[Dict] matching BallTracker format: {1: [x1,y1,x2,y2]}.
        """
        if self.model is None:
            logger.warning("TrackNet model not loaded - returning empty detections")
            return [{} for _ in frames]

        orig_h, orig_w = frames[0].shape[:2]
        detections: list[dict] = []

        logger.info(f"Running TrackNet on {len(frames)} frames...")
        with torch.no_grad():
            for i, frame in enumerate(frames):
                prev_frame = frames[max(0, i - 1)]
                next_frame = frames[min(len(frames) - 1, i + 1)]

                tensor = self._preprocess([prev_frame, frame, next_frame])
                out    = self.model(tensor)                              # (1, 256, H*W)
                output = out.argmax(dim=1).detach().cpu().numpy()       # (1, H*W)

                bbox = self._output_to_bbox(output[0], orig_h, orig_w)
                detections.append({1: bbox} if bbox is not None else {})

                if (i + 1) % 100 == 0:
                    det_so_far = sum(1 for d in detections if d.get(1))
                    logger.debug(
                        f"  Frame {i+1}/{len(frames)} - "
                        f"running detection rate: {100*det_so_far/(i+1):.1f}%"
                    )

        detected = sum(1 for d in detections if d.get(1))
        logger.info(
            f"TrackNet: {detected}/{len(frames)} frames detected "
            f"({100*detected/len(frames):.1f}%)"
        )
        return detections

    def detect_frames_with_tracking(self, frames: list[np.ndarray],
                                     read_from_stub: bool = False,
                                     stub_path: str | None = None) -> list[dict]:
        """TrackNet already uses temporal context - no separate tracker needed."""
        return self.detect_frames(frames, read_from_stub, stub_path)

    def interpolate_ball_positions(
        self,
        detections:    list[dict],
        smooth_window: int = 3,
    ) -> list[dict]:
        """
        Fill detection gaps with linear interpolation, then apply a rolling
        median to reduce per-frame jitter caused by fast-moving balls.

        smooth_window: frames over which the median is computed (set to 1 to
                       disable smoothing).  A 3-frame window removes single-frame
                       position spikes without noticeably shifting the trajectory.
        """
        rows = [
            {"frame": i, "x": (d[1][0] + d[1][2]) / 2.0, "y": (d[1][1] + d[1][3]) / 2.0}
            if d.get(1) else {"frame": i, "x": np.nan, "y": np.nan}
            for i, d in enumerate(detections)
        ]
        df = pd.DataFrame(rows).set_index("frame")
        df[["x", "y"]] = df[["x", "y"]].interpolate().bfill().ffill()

        if smooth_window > 1:
            df["x"] = df["x"].rolling(smooth_window, center=True, min_periods=1).median()
            df["y"] = df["y"].rolling(smooth_window, center=True, min_periods=1).median()

        half = 12
        return [
            {1: [cx - half, cy - half, cx + half, cy + half]}
            for cx, cy in ((float(r["x"]), float(r["y"])) for _, r in df.iterrows())
        ]

    def get_ball_shot_frames(
        self,
        detections:  list[dict],
        min_spacing: int   = 15,   # minimum frames between shots (~0.5 s at 30 fps)
        min_delta_y: float = 6.0,  # minimum y-change to count as a real reversal (px)
    ) -> list[int]:
        """
        Detect frames where the ball reverses vertical direction (a shot was hit).

        Compares the average y-position over a context window before and after each
        candidate frame.  This is more robust than the single-frame adjacent check,
        which fires on interpolation noise and produces consecutive 1-frame "shots".

        After finding all candidates, a minimum-spacing pass keeps only the first
        frame of each dense burst, ensuring the shot count and inter-shot durations
        are physically realistic.
        """
        y = np.array([
            (d[1][1] + d[1][3]) / 2.0 if d.get(1) else np.nan
            for d in detections
        ])
        n      = len(y)
        window = max(3, min_spacing // 3)   # frames to average on each side

        candidates: list[int] = []
        for i in range(window, n - window):
            y_prev = y[i - window : i]
            y_next = y[i + 1     : i + window + 1]

            prev_valid = y_prev[~np.isnan(y_prev)]
            next_valid = y_next[~np.isnan(y_next)]
            if len(prev_valid) == 0 or len(next_valid) == 0:
                continue

            avg_prev = float(prev_valid.mean())
            avg_next = float(next_valid.mean())
            curr     = float(y[i]) if not np.isnan(y[i]) else (avg_prev + avg_next) / 2.0

            delta_before = curr - avg_prev
            delta_after  = avg_next - curr

            # Direction must flip AND the magnitude must be non-trivial
            if delta_before * delta_after < 0 and abs(delta_before) >= min_delta_y:
                candidates.append(i)

        # Keep only the first frame of each burst that is closer than min_spacing
        shot_frames: list[int] = []
        for frame in candidates:
            if not shot_frames or frame - shot_frames[-1] >= min_spacing:
                shot_frames.append(frame)

        return shot_frames

    def draw_bboxes(self, frames: list[np.ndarray], detections: list[dict],
                     color: tuple[int, int, int] = (0, 255, 255),
                     thickness: int = 2) -> list[np.ndarray]:
        for frame, det in zip(frames, detections):
            if det.get(1):
                x1, y1, x2, y2 = [int(v) for v in det[1]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frames

    def filter_by_confidence(self, detections: list[dict],
                              threshold: float) -> list[dict]:
        """Confidence is baked into detection - pass through unchanged."""
        return detections
