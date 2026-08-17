# Tennis Detection and Analysis System

<div align="center">
  <img src="frame_images/tennis_analysis_quarter_frame53.png" width="800" alt="Tennis Analysis System">
  <p><em>Computer Vision-based Tennis Match Analysis with Camera-Robust Tracking</em></p>
</div>

## Overview

This project implements a computer vision system for tennis match analysis. It detects
players and the ball, tracks their movements, analyzes shots, and produces per-player
statistics (shot speed, movement speed, shot type). This branch (`sprint/ball-geometry`)
is active development on top of the camera-robust V2 pipeline - ball trajectory geometry,
a real perspective homography, pose-based shot classification, and a trained hit/bounce
classifier.

Every number in this README is from an eval script in `eval/`, cited by name so it can be
reproduced - see [Measured Results](#measured-results). Where something hasn't been
measured yet, it's marked as such rather than estimated.

## Quickstart

```bash
git clone https://github.com/HarshTomar1234/Tennis-Vision.git
cd Tennis-Vision

pip install -e .                        # or: pip install -r requirements.txt
tennis-vision download-models           # fetches model weights (~140 MB)

# Analyse the bundled sample clip
tennis-vision analyze input_videos/input_video_2.mp4 -o output/demo.avi
```

`tennis-vision download-models` prints one manual step: the TrackNet ball-detection
weights belong to their original author and are fetched from that project rather than
rehosted here. Everything else downloads automatically.

Add `--max-frames 60` for a quick check before committing to a full run.

## Features

- **Player Detection and Tracking** - YOLOv8x with a 6-criteria scoring system to
  distinguish players from line judges, ball boys, and umpires
- **Ball Detection and Trajectory** - TrackNet, with floor-anchored coordinate mapping
  (the homography is only valid when the ball is at floor level - contact or bounce - so
  in-flight positions interpolate between anchors instead of being projected directly)
  and a Kalman filter for smoothing and continuous velocity estimation
- **Court Line Detection** - ResNet-50 regression, 14 keypoints, per-frame (handles
  camera pan/tilt) with temporal smoothing
- **Shot Classification** - rule-based (serve/forehand/backhand/volley/smash from
  position and trajectory), with pose-based forehand/backhand upgrade via MediaPipe
  (body-relative geometry: does the hitting arm cross the shoulder midline in the
  horizontal plane, so it's handedness-, facing-, and side-on-agnostic)
- **Hit vs. Bounce Classification** - trained logistic regression on ball-trajectory
  shape (height, vertical/horizontal velocity change, horizontal direction reversal),
  no player position needed
- **Mini Court Visualization** - bird's-eye view with real ball trajectory trail
- **Real Perspective Homography** - `cv2.findHomography`, not nearest-keypoint approximation

## Directory Structure

```
Tennis-Vision/
├── configs/                 # config.yaml - all tunable parameters, no magic numbers in code
├── constants/                # Court dimensions, physical plausibility bounds
├── court_line_detector/      # ResNet-50 court keypoint regression
├── eval/                     # Accuracy/plausibility eval scripts - every claim in this
│                              #   README traces back to one of these
├── input_videos/             # Input tennis match videos
├── mini_visual_court/        # Mini-court coordinate mapping + trajectory drawing
├── models/                   # Trained model weights (large weights gitignored - see
│                              #   Installation; models/hit_bounce_classifier.json is
│                              #   small and committed)
├── notes/                    # CV concept write-ups (homography, Kalman filtering, SORT,
│                              #   DeepSORT re-ID, temporal smoothing, shot detection)
├── tests/                    # pytest unit + integration tests (176 passing)
├── tools/                    # label_shots.py - keyboard-driven contact/bounce labeling tool
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

2. Model weights (gitignored - fetched by script, not committed):
   ```bash
   python scripts/download_models.py     # same as: tennis-vision download-models
   ```

   | Weight | Size | Source |
   |---|---|---|
   | Court keypoints (`keypoints_model_geoaug.pth`) | 95 MB | [Coddieharsh/tennis-court-keypoints](https://huggingface.co/Coddieharsh/tennis-court-keypoints) - automatic |
   | Pose (`pose_landmarker_lite.task`) | 6 MB | Google MediaPipe - automatic |
   | Ball detection (`tracknet.pt`) | 43 MB | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) - **one manual command**, printed by the script |
   | Player detection (YOLOv8) | - | auto-downloads via `ultralytics` on first run |

   The court model is our fine-tune, published with a model card recording provenance
   and per-surface accuracy. TrackNet's weights are the upstream author's and their
   licence is unstated, so we point at the original rather than redistribute them.

3. Optional - caching for repeated runs on one clip:
   ```bash
   tennis-vision analyze clip.mp4 -c configs/dev.yaml
   ```
   Detection caches are **off by default**: a cache holds one specific video's
   detections, so using it on a different clip produces confident nonsense.

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
thresholds, I/O paths) live in `configs/config.yaml` - no magic numbers in code.

## Measured Results

Every number below names the eval script that produced it. Dated, because these are
still-moving numbers on an active sprint branch, not final claims.

**Reproducibility, stated honestly:** scripts marked 📦 need a third-party dataset
(7+ GB, not redistributable - see `datasets/README.md` for the source). Scripts marked
✅ run against what ships in this repo plus the downloadable weights.

### Ball tracking (as of 2026-08-17)

| Metric | Result | Script |
|---|---|---|
| Ball *detection rate* (a position was output, **not** an accuracy) | 88.6% (16 clips) | 📦 `eval/ball_localization_accuracy.py` |
| Ball *localization* error vs ground truth | median **5.4px**, 90th 18.0px (at 360x640) | 📦 `eval/ball_localization_accuracy.py` |
| Ball located within 5px of ground truth | 46.8% of outputs, 42.5% of visible-ball frames | 📦 `eval/ball_localization_accuracy.py` |
| Shot-frame recall, reference clip | 7/7 shots found, mean offset 7.4 frames | `eval/shot_frame_accuracy.py` |
| Shot-frame precision, reference clip | 58.3% (5 false positives in 12 reported), F1 0.74 | `eval/shot_frame_accuracy.py` |
| Ball speed *plausibility* (a range check, **not** accuracy) | 21/21 within physical bounds | ✅ `eval/speed_accuracy.py` |
| Player speed *plausibility* (range check) | 21/21 within physical bounds | ✅ `eval/speed_accuracy.py` |

> ⚠️ **Detection rate is not accuracy, and the difference here is large.** 89.6% is how
> often the detector output a ball position. Only 40.9% of visible-ball frames are
> located within 5px of the hand-labelled position, and 82.9% within 20px. Both numbers
> are true and they measure different things. Most projects publish the first and let it
> be read as the second.
>
> In practice: the detector reliably finds roughly where the ball is, and is not
> pixel-precise. That is adequate for trajectory shape, bounce timing and speed over a
> flight, and marginal for exact landing coordinates. Measured against the original
> TrackNet dataset's own hand-labelled ground truth, 20 clips, 2,365 frames.

> ⚠️ Those two rows check that speeds are *physically possible*, not that they are
> *correct*. Rally speeds are currently **systematically low** - see Limitations.

> ⚠️ Those two shot-frame rows come from **one clip with 7 labelled shots**, which is too
> small to draw conclusions from. They are listed because that clip is the reproducible
> demo, not because 7 events settle anything. The dataset-scale event numbers below (76
> labelled contacts, 10 clips) are the ones to trust, and they disagree with this clip on
> precision. Shot detection finds every real shot here and over-reports: 12 for a rally
> containing 7, the extras being bounces the classifier lets through. Treat the shot
> **count** as an upper bound.

### Contact/bounce event detection, at real dataset scale (91 clips, TrackNet's own
training data - same lineage as `models/tracknet.pt`, not a foreign benchmark)

Two different questions, with two very different answers. Both are reported because the
gap between them is the honest measure of how much ball-detection noise costs.

**Given perfect ball positions** (the dataset's own hand-labelled coordinates fed straight
into the candidate generators). This isolates the generators and is an upper bound, not
shipped behaviour:

| Configuration | Recall | Precision | Script |
|---|---|---|---|
| y-reversal only, through trained classifier (91 clips) | 75.8% | 88.9% | 📦 `eval/retest_union_candidates_full_pipeline.py` |
| y-reversal + x-velocity union, through trained classifier (91 clips) | 87.6% | 90.3% | 📦 `eval/retest_union_candidates_full_pipeline.py` |

**Running real TrackNet detection end to end**, which is what the pipeline actually does
(10 clips, 76 labelled contacts, trajectory classifier only, no player proximity):

| Postprocessing | Recall | Precision | F1 | Mean offset | Script |
|---|---|---|---|---|---|
| mean of all responding pixels (previous) | 48.7% | 92.5% | 0.638 | 3.6 frames | 📦 `eval/event_detection_on_real_detections.py` |
| **largest connected component** (production config) | **51.3%** | **92.9%** | **0.661** | **3.2 frames** | 📦 `eval/event_detection_on_real_detections.py` |

> ⚠️ **Event recall on real detections is roughly half the upper bound: 51.3% against
> 87.6%.** Precision holds up (92.9%), so the events reported are overwhelmingly real, but
> around half the contacts in a rally are missed. The cause is ball-detection noise, not
> the candidate generators: a 5.8px median localization error both manufactures reversals
> and buries real ones. Improving ball localization is therefore the highest-value work
> for every event-derived number, which is why it is first on the roadmap.

### Hit vs. bounce classification

Trajectory-only logistic regression (ball height, vertical/horizontal velocity change,
and whether the ball reversed horizontally), no player position needed - 86.4% held-out
accuracy, trained on 820 events / tested on 214 held-out events (clip-level split, not
event-level, to avoid leaking camera/lighting/player correlations). See
`models/hit_bounce_classifier.json` and `eval/train_hit_bounce_classifier.py`.

The feature set was chosen on end-to-end shot F1, not on this accuracy, and the two
disagree. A six-feature variant adding raw signed velocities scores higher here (89.3%)
and clearly worse in the pipeline (rally F1 0.600 against 0.824), because those features
are clean in the hand-annotated training data and noisy in real TrackNet detections. The
full comparison table is in the training script.

### Pose-based shot classification

On the current reference clip: 6 of 21 shots upgraded from position-based to
pose-verified forehand/backhand (the rest keep the position-based rule when pose is
unavailable or ambiguous - reported, not hidden).

### Test suite

146 unit + integration tests passing (`pytest tests/`), covering ball-state
classification, Kalman smoothing (including the physical speed-plausibility gate),
mini-court coordinate mapping, trajectory drawing, pose-based shot classification, and
the hit/bounce classifier. The end-to-end smoke test runs genuine fresh detection -
it depends on no cached artefacts, so it fails for everyone if the pipeline breaks.

### Court keypoint accuracy (2026-08-09)

Held-out validation split of the TennisCourtDetector dataset, 2,211 images:

| Metric | Base weights | Fine-tuned (shipped) | Script |
|---|---|---|---|
| Median keypoint error | 4.03 px | **2.90 px** | 📦 `eval/court_keypoint_accuracy.py` |
| Images with all 14 keypoints within 25 px | 96.8 % | **98.3 %** | 📦 same |
| Real clips passing the court-validity gate | 4/9 | **8/9** | 📦 `eval/court_validity_calibration.py` |

Per-surface median error is near-identical (hard 3.90 px, clay 4.58 px, grass/green
4.65 px), so surface is not a weakness; camera framing is. Validated on Wimbledon
grass the model had never seen.

---

## Limitations

Stated plainly, because the point of this project is that its numbers are honest.

**Wrong today:**
- **Rally shot speeds are systematically low.** The ball is projected through the
  *floor* homography while it is airborne ~90 % of the time, so derived speeds are
  wrong by an amount that varies with camera geometry. The fix is 3-D trajectory
  reconstruction (roadmap below), not a tuning change.
- **Serve speed is implemented but unvalidated.** It uses only floor-valid geometry
  (server's feet, ball's bounce), and refuses to report rather than guess when its
  physical gates fail - but it has not yet been confirmed against radar ground truth.

**Unmeasured:**
- Volley and Smash labels come from position rules with **no ground truth**. Serve is
  now detected from physical evidence; those two are not.
- The learned shot classifier (73.4 % on unseen subjects, 6 classes) is trained on
  THETIS *indoor demonstration* footage and is **not wired into the pipeline** -
  transfer to broadcast video is unvalidated.
- Player detection accuracy has no ground-truth eval script.

**Out of scope right now:**
- **Ground-level cameras fail.** Validated on broadcast and elevated fixed-camera
  footage only. The validity gate flags these rather than reporting wrong numbers.
- **Doubles and amateur footage are untested** - every evaluation clip is broadcast
  singles.

## Roadmap

- 3-D ball trajectory reconstruction - the fix for rally speeds *and* the basis for a
  3-D rally viewer. These are the same problem: a correct 3-D trajectory is what makes
  a speed correct.
- Validate the temporal shot classifier on broadcast footage, then wire it in.
- Ground truth for Volley/Smash, so they can be claimed or dropped.

## Technical Notes

Deeper CV concept write-ups in `notes/`:

- `01_homography_basics.md` - Court coordinate transformation
- `02_kalman_filter.md` - Ball trajectory smoothing
- `03_temporal_smoothing.md` - Keypoint jitter reduction
- `04_sort_tracker.md` - Multi-object tracking
- `05_deepsort_reid.md` - Re-identification concepts
- `06_shot_detection.md` - Shot classification methodology
- `camera_robust_notes.py` - Complete camera-robust implementation guide

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Credits

- YOLOv8 (Ultralytics) - player detection
- TrackNet - ball detection
- MediaPipe - pose estimation
- OpenCV - image processing, homography, Kalman filtering
- PyTorch - deep learning components
- ResNet-50 - court keypoint regression

## License

MIT License - see [LICENSE](LICENSE).
