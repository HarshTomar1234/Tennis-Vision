# Tennis Detection and Analysis System

<div align="center">
  <img src="frame_images/tennis_analysis_quarter_frame53.png" width="800" alt="Tennis Analysis System">
  <p><em>Computer Vision-based Tennis Match Analysis with Camera-Robust Tracking</em></p>
</div>

## Overview

This project implements a computer vision system for tennis match analysis. It detects
players and the ball, tracks their movements, analyzes shots, and produces per-player
statistics (shot speed, movement speed, shot type). This branch (`sprint/ball-geometry`)
is active development on top of the camera-robust V2 pipeline — ball trajectory geometry,
a real perspective homography, pose-based shot classification, and a trained hit/bounce
classifier.

Every number in this README is from an eval script in `eval/`, cited by name so it can be
reproduced — see [Measured Results](#measured-results). Where something hasn't been
measured yet, it's marked as such rather than estimated.

## Quickstart

```bash
git clone https://github.com/HarshTomar1234/Tennis-Vision.git
cd Tennis-Vision

pip install -r requirements.txt

# Runs configs/config.yaml's default input video, writes to output/videos/output_video.avi
python main.py
```

## Features

- **Player Detection and Tracking** — YOLOv8x with a 6-criteria scoring system to
  distinguish players from line judges, ball boys, and umpires
- **Ball Detection and Trajectory** — TrackNet, with floor-anchored coordinate mapping
  (the homography is only valid when the ball is at floor level — contact or bounce — so
  in-flight positions interpolate between anchors instead of being projected directly)
  and a Kalman filter for smoothing and continuous velocity estimation
- **Court Line Detection** — ResNet-50 regression, 14 keypoints, per-frame (handles
  camera pan/tilt) with temporal smoothing
- **Shot Classification** — rule-based (serve/forehand/backhand/volley/smash from
  position and trajectory), with pose-based forehand/backhand upgrade via MediaPipe
  (body-relative geometry: does the hitting arm cross the shoulder midline in the
  horizontal plane, so it's handedness-, facing-, and side-on-agnostic)
- **Hit vs. Bounce Classification** — trained logistic regression on ball-trajectory
  shape (height, vertical/horizontal velocity change), no player position needed
- **Mini Court Visualization** — bird's-eye view with real ball trajectory trail
- **Real Perspective Homography** — `cv2.findHomography`, not nearest-keypoint approximation

## Directory Structure

```
Tennis-Vision/
├── configs/                 # config.yaml — all tunable parameters, no magic numbers in code
├── constants/                # Court dimensions, physical plausibility bounds
├── court_line_detector/      # ResNet-50 court keypoint regression
├── eval/                     # Accuracy/plausibility eval scripts — every claim in this
│                              #   README traces back to one of these
├── input_videos/             # Input tennis match videos
├── mini_visual_court/        # Mini-court coordinate mapping + trajectory drawing
├── models/                   # Trained model weights (large weights gitignored — see
│                              #   Installation; models/hit_bounce_classifier.json is
│                              #   small and committed)
├── notes/                    # CV concept write-ups (homography, Kalman filtering, SORT,
│                              #   DeepSORT re-ID, temporal smoothing, shot detection)
├── tests/                    # pytest unit tests (76 passing — see Measured Results)
├── tools/                    # label_shots.py — keyboard-driven contact/bounce labeling tool
├── trackers/                 # tracknet_ball_tracker.py (production), player_tracker.py,
│                              #   ball_tracker.py (legacy YOLO ball tracker, superseded)
├── utils/
│   ├── ball_state.py               # floor-level/in-flight classification
│   ├── hit_bounce_classifier.py    # trained hit/bounce classifier + candidate generation
│   ├── kalman_smoother.py          # position smoothing, peak-velocity estimation
│   ├── pose_estimator.py           # MediaPipe Tasks PoseLandmarker wrapper
│   ├── pose_shot_classifier.py     # pose-based forehand/backhand
│   ├── shot_classifier.py          # rule-based shot type classification
│   └── ui_layout_manager.py        # resolution-adaptive UI positioning
├── main.py                   # Pipeline entry point (argparse + config.yaml)
└── requirements.txt
```

## Installation

1. Clone and install dependencies:
   ```bash
   git clone https://github.com/HarshTomar1234/Tennis-Vision.git
   cd Tennis-Vision
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Model weights (gitignored — download separately):
   - Player detection: YOLOv8x auto-downloads via `ultralytics` on first run
   - Court keypoints: `models/keypoints_model.pth` (ResNet-50, place manually)
   - Ball detection (TrackNet): `models/tracknet.pt` — see download command in
     `configs/config.yaml`
   - Pose (forehand/backhand): `models/pose_landmarker_lite.task` — see download
     command in `configs/config.yaml`

## Usage

```bash
python main.py                          # uses configs/config.yaml defaults
python main.py --input path/to/clip.mp4 --output out.avi
python main.py --config my_config.yaml  # override any parameter
python main.py --no-stubs               # force fresh detection (skip cached stubs)
python main.py --fast                   # first-frame keypoints only, no per-frame tracking
python main.py --debug                  # DEBUG log level
```

All tunable parameters (which pipeline stages run, detection thresholds, shot classifier
thresholds, I/O paths) live in `configs/config.yaml` — no magic numbers in code.

## Measured Results

Every number below is from a named eval script, reproducible with one command. Dated
because these are still-moving numbers on an active sprint branch, not final claims.

### Ball tracking (as of 2026-08-01)

| Metric | Result | Script |
|---|---|---|
| Raw ball detection rate (TrackNet) | 82.5% (470/570 frames) | pipeline log |
| Shot-frame accuracy | 7/7 matched, mean offset 4.9 frames — EXCELLENT | `eval/shot_frame_accuracy.py` |
| Ball speed plausibility | 21/21 in physical range (36.6–181.6 km/h) — PASS | `eval/speed_accuracy.py` |
| Player speed plausibility | 21/21 in physical range | `eval/speed_accuracy.py` |

### Contact/bounce event detection, at real dataset scale (91 clips, TrackNet's own
training data — same lineage as `models/tracknet.pt`, not a foreign benchmark)

| Configuration | Recall | Precision | Script |
|---|---|---|---|
| Real-model detection noise (3 clips, 55 events) | 92.7% | 94.9% | `eval/retest_real_model_noise.py` |
| y-reversal only, through trained classifier (91 clips) | 75.8% | 88.9% | `eval/retest_union_candidates_full_pipeline.py` |
| **y-reversal + x-velocity union, through trained classifier** (production config) | **87.6%** | **90.3%** | `eval/retest_union_candidates_full_pipeline.py` |

### Hit vs. bounce classification

Trajectory-only logistic regression (ball height, vertical/horizontal velocity change),
no player position needed — 84.1% held-out accuracy, trained on 820 events / tested on
214 held-out events (clip-level split, not event-level, to avoid leaking
camera/lighting/player correlations). See `models/hit_bounce_classifier.json` and
`eval/train_hit_bounce_classifier.py`.

### Pose-based shot classification

On the current reference clip: 6 of 21 shots upgraded from position-based to
pose-verified forehand/backhand (the rest keep the position-based rule when pose is
unavailable or ambiguous — reported, not hidden).

### Test suite

76 unit + integration tests passing (`pytest tests/`), covering ball-state
classification, Kalman smoothing (including the physical speed-plausibility gate),
mini-court coordinate mapping, trajectory drawing, pose-based shot classification, and
the hit/bounce classifier.

### Not yet measured

Player detection accuracy and court keypoint accuracy have no eval script against
ground truth yet — they're visually verified working (see the journal in `docs/`, local
only) but not claimed as measured numbers here. Per-shot-type accuracy (forehand vs.
backhand vs. serve, individually) also isn't separately measured — the pose-based
forehand/backhand result above is the closest verified proxy.

## Technical Notes

Deeper CV concept write-ups in `notes/`:

- `01_homography_basics.md` — Court coordinate transformation
- `02_kalman_filter.md` — Ball trajectory smoothing
- `03_temporal_smoothing.md` — Keypoint jitter reduction
- `04_sort_tracker.md` — Multi-object tracking
- `05_deepsort_reid.md` — Re-identification concepts
- `06_shot_detection.md` — Shot classification methodology
- `camera_robust_notes.py` — Complete camera-robust implementation guide

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

- YOLOv8 (Ultralytics) — player detection
- TrackNet — ball detection
- MediaPipe — pose estimation
- OpenCV — image processing, homography, Kalman filtering
- PyTorch — deep learning components
- ResNet-50 — court keypoint regression

## License

MIT License — see [LICENSE](LICENSE).
